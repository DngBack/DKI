import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import torch

from methods import (
    build_naive_messages,
    get_decoder_layers,
)
from dki_kv import (
    DKIKVController,
    build_dki_kv_memory,
    patch_qwen3_attention_layers,
    run_dki_kv_inference,
)
from qwen_runner import QwenVLRunner
from schema_eval import (
    align_to_schema,
    extract_largest_valid_json,
    pretty_json,
    schema_metrics,
    value_match_metrics,
)


def _load_demo_text_from_json(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _load_demo_json_obj(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_demo_items(demo_pdf: str, demo_json: str, shots: int) -> List[Dict[str, str]]:
    text = _load_demo_text_from_json(demo_json)
    return [{"id": f"demo_{i+1}", "image": demo_pdf, "text": text} for i in range(shots)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--demo-pdf",
        default="data/samples/giay_gui_tien_tiet_kiem/giay_gui_tien_tiet_kiem_1.pdf",
    )
    parser.add_argument(
        "--demo-json",
        default="data/samples/giay_gui_tien_tiet_kiem/giay_gui_tien_tiet_kiem_1.json",
    )
    parser.add_argument(
        "--query-pdf",
        default="data/test/samples/giay_gui_tien_tiet_kiem/giay_gui_tien_tiet_kiem_test_1.pdf",
    )
    parser.add_argument("--query-gt-json", default="", help="Optional GT JSON for quantitative value matching")
    parser.add_argument("--shots", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--top-n", type=int, default=32)
    parser.add_argument("--top-n-text", type=int, default=16)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument(
        "--target-layers",
        default="mid",
        help='Layer spec: "mid" or comma list like "14,18,22"',
    )
    parser.add_argument("--output", default="draft/test_dki/results/compare_log.json")
    args = parser.parse_args()

    for p in [args.demo_pdf, args.demo_json, args.query_pdf]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input file: {p}")
    if args.query_gt_json and not os.path.exists(args.query_gt_json):
        raise FileNotFoundError(f"Missing query GT JSON: {args.query_gt_json}")

    demo_items = _make_demo_items(args.demo_pdf, args.demo_json, args.shots)
    demo_schema_obj = _load_demo_json_obj(args.demo_json)
    runner = QwenVLRunner(
        model_name=args.model_name,
        attn_implementation="eager",
        max_pixels=args.max_pixels,
    )

    print("=" * 80)
    print("Running naive multimodal few-shot")
    print(f"model={args.model_name} shots={args.shots}")
    print("=" * 80)
    naive_messages = build_naive_messages(demo_items, args.query_pdf, demo_schema_obj)
    naive_result = runner.generate(naive_messages, max_new_tokens=args.max_new_tokens)
    naive_json = extract_largest_valid_json(naive_result.text)
    naive_aligned = align_to_schema(demo_schema_obj, naive_json if naive_json is not None else {})
    naive_schema_stats = schema_metrics(demo_schema_obj, naive_json, naive_aligned)
    print("[NAIVE] elapsed_sec:", round(naive_result.elapsed_sec, 3))
    print("[NAIVE] output:\n", naive_result.text)
    if naive_json is not None:
        print("[NAIVE] parsed_json:\n", pretty_json(naive_json))
    print("[NAIVE] aligned_json:\n", pretty_json(naive_aligned))
    print("[NAIVE] schema_metrics:", naive_schema_stats)

    print("\n" + "=" * 80)
    print("Running DKI-KV (demo pass -> KV extract -> attention KV inject)")
    print("=" * 80)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    num_layers = len(get_decoder_layers(runner.model))
    if args.target_layers.strip().lower() == "mid":
        target_layers = [num_layers // 2]
    else:
        target_layers = [int(x.strip()) for x in args.target_layers.split(",") if x.strip()]
    memory_by_layer = build_dki_kv_memory(
        runner,
        demo_items,
        target_layers=target_layers,
        schema_obj=demo_schema_obj,
        top_n=args.top_n,
        top_n_text=args.top_n_text,
    )
    controller = DKIKVController()
    controller.set_memory(memory_by_layer, beta=args.beta)
    restore_dki_patch = patch_qwen3_attention_layers(runner.model, controller, target_layers=target_layers)
    dki_result = run_dki_kv_inference(
        runner,
        query_image=args.query_pdf,
        schema_obj=demo_schema_obj,
        controller=controller,
        max_new_tokens=args.max_new_tokens,
    )
    restore_dki_patch()
    dki_json = extract_largest_valid_json(dki_result.text)
    dki_aligned = align_to_schema(demo_schema_obj, dki_json if dki_json is not None else {})
    dki_schema_stats = schema_metrics(demo_schema_obj, dki_json, dki_aligned)

    print("[DKI] target_layers:", target_layers)
    print("[DKI] top_n:", args.top_n)
    print("[DKI] beta:", args.beta)
    print("[DKI] elapsed_sec:", round(dki_result.elapsed_sec, 3))
    print("[DKI] output:\n", dki_result.text)
    if dki_json is not None:
        print("[DKI] parsed_json:\n", pretty_json(dki_json))
    print("[DKI] aligned_json:\n", pretty_json(dki_aligned))
    print("[DKI] schema_metrics:", dki_schema_stats)

    naive_value_stats = None
    dki_value_stats = None
    if args.query_gt_json:
        query_gt_obj = _load_demo_json_obj(args.query_gt_json)
        naive_value_stats = value_match_metrics(query_gt_obj, naive_aligned, demo_schema_obj)
        dki_value_stats = value_match_metrics(query_gt_obj, dki_aligned, demo_schema_obj)
        print("[NAIVE] value_match_metrics:", naive_value_stats)
        print("[DKI] value_match_metrics:", dki_value_stats)

    out_obj: Dict[str, Any] = {
        "config": {
            "model_name": args.model_name,
            "shots": args.shots,
            "demo_pdf": args.demo_pdf,
            "demo_json": args.demo_json,
            "query_pdf": args.query_pdf,
            "query_gt_json": args.query_gt_json,
            "target_layers": target_layers,
            "top_n": args.top_n,
            "top_n_text": args.top_n_text,
            "beta": args.beta,
            "max_new_tokens": args.max_new_tokens,
            "max_pixels": args.max_pixels,
        },
        "naive": {
            "elapsed_sec": naive_result.elapsed_sec,
            "output": naive_result.text,
            "parsed_json": naive_json,
            "aligned_json": naive_aligned,
            "schema_metrics": naive_schema_stats,
            "value_match_metrics": naive_value_stats,
        },
        "dki_kv": {
            "elapsed_sec": dki_result.elapsed_sec,
            "output": dki_result.text,
            "parsed_json": dki_json,
            "aligned_json": dki_aligned,
            "schema_metrics": dki_schema_stats,
            "value_match_metrics": dki_value_stats,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"\nSaved compare log to: {out_path}")


if __name__ == "__main__":
    main()
