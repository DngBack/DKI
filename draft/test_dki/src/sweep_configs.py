import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from dki_kv import DKIKVController, build_dki_kv_memory, patch_qwen3_attention_layers, run_dki_kv_inference
from methods import build_naive_messages, get_decoder_layers
from qwen_runner import QwenVLRunner
from run_compare import _load_demo_json_obj, _load_demo_text_from_json
from schema_eval import align_to_schema, extract_largest_valid_json, schema_metrics, value_match_metrics


def _make_demo_items(demo_pdf: str, demo_json: str, shots: int) -> List[Dict[str, str]]:
    text = _load_demo_text_from_json(demo_json)
    return [{"id": f"demo_{i+1}", "image": demo_pdf, "text": text} for i in range(shots)]


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _resolve_layer_spec(spec: str, num_layers: int) -> List[int]:
    spec = spec.strip().lower()
    if spec == "mid":
        return [num_layers // 2]
    if spec == "triad":
        m = num_layers // 2
        return sorted(set([max(0, m - 4), m, min(num_layers - 1, m + 4)]))
    return _parse_int_list(spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--demo-pdf", default="data/samples/giay_gui_tien_tiet_kiem/giay_gui_tien_tiet_kiem_1.pdf")
    parser.add_argument("--demo-json", default="data/samples/giay_gui_tien_tiet_kiem/giay_gui_tien_tiet_kiem_1.json")
    parser.add_argument("--query-pdf", default="data/test/samples/giay_gui_tien_tiet_kiem/giay_gui_tien_tiet_kiem_test_1.pdf")
    parser.add_argument("--query-gt-json", default="")
    parser.add_argument("--shots", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--top-n-list", default="4,8,12")
    parser.add_argument("--top-n-text", type=int, default=16)
    parser.add_argument("--beta-list", default="0.05,0.1,0.2,0.3")
    parser.add_argument(
        "--layer-spec-list",
        default="mid,triad",
        help='Comma-delimited tokens among "mid", "triad" or explicit lists like "16|20|24".',
    )
    parser.add_argument("--output", default="draft/test_dki/results/sweep_results.json")
    args = parser.parse_args()

    demo_items = _make_demo_items(args.demo_pdf, args.demo_json, args.shots)
    schema_obj = _load_demo_json_obj(args.demo_json)
    query_gt_obj = _load_demo_json_obj(args.query_gt_json) if args.query_gt_json else None

    runner = QwenVLRunner(
        model_name=args.model_name,
        attn_implementation="eager",
        max_pixels=args.max_pixels,
    )

    print("=" * 80)
    print("Baseline NAIVE (run once)")
    print("=" * 80)
    naive_messages = build_naive_messages(demo_items, args.query_pdf, schema_obj)
    naive_result = runner.generate(naive_messages, max_new_tokens=args.max_new_tokens)
    naive_json = extract_largest_valid_json(naive_result.text)
    naive_aligned = align_to_schema(schema_obj, naive_json if naive_json is not None else {})
    naive_schema = schema_metrics(schema_obj, naive_json, naive_aligned)
    naive_value = value_match_metrics(query_gt_obj, naive_aligned, schema_obj) if query_gt_obj else None

    num_layers = len(get_decoder_layers(runner.model))
    top_n_list = _parse_int_list(args.top_n_list)
    beta_list = _parse_float_list(args.beta_list)
    layer_specs_raw = [x.strip() for x in args.layer_spec_list.split(",") if x.strip()]
    expanded_layer_specs: List[List[int]] = []
    for spec in layer_specs_raw:
        if "|" in spec:
            expanded_layer_specs.append([int(x) for x in spec.split("|") if x])
        else:
            expanded_layer_specs.append(_resolve_layer_spec(spec, num_layers))

    trials = []
    total = len(top_n_list) * len(beta_list) * len(expanded_layer_specs)
    idx = 0
    for top_n in top_n_list:
        for beta in beta_list:
            for layers in expanded_layer_specs:
                idx += 1
                print(f"\n[{idx}/{total}] DKI trial top_n={top_n} beta={beta} layers={layers}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                start = time.time()
                memory_by_layer = build_dki_kv_memory(
                    runner,
                    demo_items,
                    target_layers=layers,
                    schema_obj=schema_obj,
                    top_n=top_n,
                    top_n_text=args.top_n_text,
                )
                controller = DKIKVController()
                controller.set_memory(memory_by_layer, beta=beta)
                restore = patch_qwen3_attention_layers(runner.model, controller, target_layers=layers)
                dki_result = run_dki_kv_inference(
                    runner,
                    query_image=args.query_pdf,
                    schema_obj=schema_obj,
                    controller=controller,
                    max_new_tokens=args.max_new_tokens,
                )
                restore()
                dki_json = extract_largest_valid_json(dki_result.text)
                dki_aligned = align_to_schema(schema_obj, dki_json if dki_json is not None else {})
                dki_schema = schema_metrics(schema_obj, dki_json, dki_aligned)
                dki_value = value_match_metrics(query_gt_obj, dki_aligned, schema_obj) if query_gt_obj else None
                trial = {
                    "top_n": top_n,
                    "beta": beta,
                    "layers": layers,
                    "elapsed_sec": dki_result.elapsed_sec,
                    "wall_sec": time.time() - start,
                    "schema_metrics": dki_schema,
                    "value_match_metrics": dki_value,
                    "parsed_json_exists": dki_json is not None,
                }
                trials.append(trial)

    def score_key(t: Dict[str, Any]):
        value_rate = t["value_match_metrics"]["leaf_exact_match_rate"] if t["value_match_metrics"] else -1.0
        schema_rate = t["schema_metrics"]["leaf_non_empty_rate"]
        speed_gain = naive_result.elapsed_sec - t["elapsed_sec"]
        return (value_rate, schema_rate, speed_gain)

    trials_sorted = sorted(trials, key=score_key, reverse=True)
    summary = {
        "config": vars(args),
        "naive": {
            "elapsed_sec": naive_result.elapsed_sec,
            "schema_metrics": naive_schema,
            "value_match_metrics": naive_value,
        },
        "best_trial": trials_sorted[0] if trials_sorted else None,
        "trials": trials_sorted,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("SWEEP DONE")
    print(f"saved: {out_path}")
    if trials_sorted:
        print("best:", trials_sorted[0])


if __name__ == "__main__":
    main()
