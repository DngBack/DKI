import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import torch

from dyna_prefix import build_or_load_prefix_memory, run_dyna_prefix_inference
from methods import build_naive_messages, get_decoder_layers
from qwen_runner import QwenVLRunner
from schema_eval import (
    align_to_schema,
    extract_largest_valid_json,
    pretty_json,
    schema_metrics,
    value_match_metrics,
)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _demo_items(demo_pdf: str, demo_json: str, shots: int) -> List[Dict[str, str]]:
    demo_obj = _load_json(demo_json)
    demo_text = json.dumps(demo_obj, ensure_ascii=False, indent=2)
    return [{"id": f"demo_{i+1}", "image": demo_pdf, "text": demo_text} for i in range(shots)]


def _resolve_target_layers(spec: str, num_layers: int) -> List[int]:
    raw = spec.strip().lower()
    if raw == "mid":
        return [num_layers // 2]
    if raw == "triad":
        return sorted(set([num_layers // 3, num_layers // 2, (2 * num_layers) // 3]))
    if "|" in raw:
        return [int(x.strip()) for x in raw.split("|") if x.strip()]
    if "," in raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return [int(raw)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--demo-pdf", default="data/samples/giay_gui_tien_tiet_kiem/giay_gui_tien_tiet_kiem_1.pdf")
    parser.add_argument("--demo-json", default="data/samples/giay_gui_tien_tiet_kiem/giay_gui_tien_tiet_kiem_1.json")
    parser.add_argument("--query-pdf", default="data/test/samples/giay_gui_tien_tiet_kiem/giay_gui_tien_tiet_kiem_test_1.pdf")
    parser.add_argument("--query-gt-json", default="")
    parser.add_argument("--shots", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--target-layers", default="mid")
    parser.add_argument("--top-n", type=int, default=32)
    parser.add_argument("--top-n-text", type=int, default=16)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--prefix-cache", default="draft/test_dki/results/prefix_cache.pt")
    parser.add_argument("--rebuild-prefix", action="store_true")
    parser.add_argument("--output", default="draft/test_dki/results/compare_dyna_prefix.json")
    args = parser.parse_args()

    for p in [args.demo_pdf, args.demo_json, args.query_pdf]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input file: {p}")
    if args.query_gt_json and not os.path.exists(args.query_gt_json):
        raise FileNotFoundError(f"Missing query GT JSON: {args.query_gt_json}")

    demo_schema_obj = _load_json(args.demo_json)
    demos = _demo_items(args.demo_pdf, args.demo_json, args.shots)
    runner = QwenVLRunner(
        model_name=args.model_name,
        attn_implementation="eager",
        max_pixels=args.max_pixels,
    )
    num_layers = len(get_decoder_layers(runner.model))
    target_layers = _resolve_target_layers(args.target_layers, num_layers)

    print("=" * 80)
    print("Running naive multimodal few-shot")
    print("=" * 80)
    naive_messages = build_naive_messages(demos, args.query_pdf, demo_schema_obj)
    naive_result = runner.generate(naive_messages, max_new_tokens=args.max_new_tokens)
    naive_json = extract_largest_valid_json(naive_result.text)
    naive_aligned = align_to_schema(demo_schema_obj, naive_json if naive_json is not None else {})
    naive_schema_stats = schema_metrics(demo_schema_obj, naive_json, naive_aligned)
    print("[NAIVE] elapsed_sec:", round(naive_result.elapsed_sec, 3))
    print("[NAIVE] aligned_json:\n", pretty_json(naive_aligned))

    print("\n" + "=" * 80)
    print("Running DynaOCR-Prefix test (support -> prefix cache -> query)")
    print("=" * 80)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    memory_by_layer = build_or_load_prefix_memory(
        runner=runner,
        demos=demos,
        schema_obj=demo_schema_obj,
        target_layers=target_layers,
        top_n=args.top_n,
        top_n_text=args.top_n_text,
        cache_path=args.prefix_cache,
        rebuild_cache=args.rebuild_prefix,
    )
    dyna_result = run_dyna_prefix_inference(
        runner=runner,
        query_image=args.query_pdf,
        schema_obj=demo_schema_obj,
        target_layers=target_layers,
        memory_by_layer=memory_by_layer,
        beta=args.beta,
        max_new_tokens=args.max_new_tokens,
    )
    dyna_json = extract_largest_valid_json(dyna_result.text)
    dyna_aligned = align_to_schema(demo_schema_obj, dyna_json if dyna_json is not None else {})
    dyna_schema_stats = schema_metrics(demo_schema_obj, dyna_json, dyna_aligned)
    print("[DYNA] target_layers:", target_layers)
    print("[DYNA] elapsed_sec:", round(dyna_result.elapsed_sec, 3))
    print("[DYNA] aligned_json:\n", pretty_json(dyna_aligned))

    naive_value_stats = None
    dyna_value_stats = None
    if args.query_gt_json:
        gt = _load_json(args.query_gt_json)
        naive_value_stats = value_match_metrics(gt, naive_aligned, demo_schema_obj)
        dyna_value_stats = value_match_metrics(gt, dyna_aligned, demo_schema_obj)
        print("[NAIVE] value_match_metrics:", naive_value_stats)
        print("[DYNA] value_match_metrics:", dyna_value_stats)

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
            "prefix_cache": args.prefix_cache,
            "rebuild_prefix": args.rebuild_prefix,
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
        "dyna_prefix": {
            "elapsed_sec": dyna_result.elapsed_sec,
            "output": dyna_result.text,
            "parsed_json": dyna_json,
            "aligned_json": dyna_aligned,
            "schema_metrics": dyna_schema_stats,
            "value_match_metrics": dyna_value_stats,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"\nSaved compare log to: {out_path}")


if __name__ == "__main__":
    main()
