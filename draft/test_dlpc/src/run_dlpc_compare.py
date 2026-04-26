import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

CURRENT_DIR = Path(__file__).resolve().parent
DKI_SRC_DIR = CURRENT_DIR.parents[1] / "test_dki" / "src"
if str(DKI_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DKI_SRC_DIR))


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _demo_items(demo_pdf: str, demo_json: str, shots: int) -> List[Dict[str, str]]:
    demo_obj = _load_json(demo_json)
    demo_text = json.dumps(demo_obj, ensure_ascii=False, indent=2)
    return [{"id": f"demo_{i+1}", "image": demo_pdf, "text": demo_text} for i in range(shots)]


def _make_schema_skeleton(obj: Any):
    if isinstance(obj, dict):
        return {k: _make_schema_skeleton(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_schema_skeleton(obj[0])] if obj else []
    return ""


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
    parser.add_argument("--target-layers", default="30|31")
    parser.add_argument("--top-n-visual", type=int, default=48)
    parser.add_argument("--top-n-text", type=int, default=24)
    parser.add_argument("--prefix-len", type=int, default=64)
    parser.add_argument("--compressor-mode", default="separate", choices=["cross", "separate"])
    parser.add_argument(
        "--schema-only-prefix",
        action="store_true",
        help="Build DLPC prefix from schema skeleton instead of support answer values.",
    )
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--value-blend-lambda", type=float, default=0.01)
    parser.add_argument("--value-beta-scale", type=float, default=0.1)
    parser.add_argument("--key-phase-tokens", type=int, default=48)
    parser.add_argument(
        "--no-positional-reset-prefix",
        action="store_true",
        help="Disable prefix key centering (positional reset).",
    )
    parser.add_argument(
        "--inject-value-prefix",
        action="store_true",
        help="Inject prefix into Value branch too (off by default for tokenization safety).",
    )
    parser.add_argument(
        "--field-gated-injection",
        action="store_true",
        help="Heuristic JSON-state gating: inject only when currently decoding a value span.",
    )
    parser.add_argument("--prefix-cache", default="draft/test_dlpc/results/prefix_cache.pt")
    parser.add_argument("--rebuild-prefix", action="store_true")
    parser.add_argument("--output", default="draft/test_dlpc/results/compare_dlpc.json")
    args = parser.parse_args()

    from dlpc import (  # noqa: WPS433
        build_dlpc_prefix_memory,
        load_prefix_cache,
        run_dlpc_inference,
        save_prefix_cache,
    )
    from methods import build_naive_messages, get_decoder_layers  # noqa: WPS433
    from qwen_runner import QwenVLRunner  # noqa: WPS433
    from schema_eval import (  # noqa: WPS433
        align_to_schema,
        extract_largest_valid_json,
        pretty_json,
        schema_metrics,
        value_match_metrics,
    )

    for p in [args.demo_pdf, args.demo_json, args.query_pdf]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input file: {p}")
    if args.query_gt_json and not os.path.exists(args.query_gt_json):
        raise FileNotFoundError(f"Missing query GT JSON: {args.query_gt_json}")

    demo_schema_obj = _load_json(args.demo_json)
    demos = _demo_items(args.demo_pdf, args.demo_json, args.shots)
    if args.schema_only_prefix:
        schema_skeleton_text = json.dumps(_make_schema_skeleton(demo_schema_obj), ensure_ascii=False, indent=2)
        for demo in demos:
            demo["prefix_text"] = schema_skeleton_text
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
    print("Running DLPC test (compress support -> deep KV inject)")
    print("=" * 80)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    use_cache = bool(args.prefix_cache and (not args.rebuild_prefix) and Path(args.prefix_cache).exists())
    if use_cache:
        prefix_text_mode = "schema" if args.schema_only_prefix else "answer"
        memory_by_layer = load_prefix_cache(
            cache_path=args.prefix_cache,
            model_name=runner.model.config._name_or_path,
            schema_obj=demo_schema_obj,
            target_layers=target_layers,
            top_n_visual=args.top_n_visual,
            top_n_text=args.top_n_text,
            prefix_len=args.prefix_len,
            compressor_mode=args.compressor_mode,
            prefix_text_mode=prefix_text_mode,
        )
    else:
        memory_by_layer = build_dlpc_prefix_memory(
            wrapper=runner,
            demos=demos,
            target_layers=target_layers,
            schema_obj=demo_schema_obj,
            top_n_visual=args.top_n_visual,
            top_n_text=args.top_n_text,
            prefix_len=args.prefix_len,
            compressor_mode=args.compressor_mode,
        )
        if args.prefix_cache:
            prefix_text_mode = "schema" if args.schema_only_prefix else "answer"
            save_prefix_cache(
                cache_path=args.prefix_cache,
                model_name=runner.model.config._name_or_path,
                schema_obj=demo_schema_obj,
                target_layers=target_layers,
                top_n_visual=args.top_n_visual,
                top_n_text=args.top_n_text,
                prefix_len=args.prefix_len,
                compressor_mode=args.compressor_mode,
                prefix_text_mode=prefix_text_mode,
                memory_by_layer=memory_by_layer,
            )

    dlpc_result, dlpc_prefix_mass_logs = run_dlpc_inference(
        wrapper=runner,
        query_image=args.query_pdf,
        schema_obj=demo_schema_obj,
        target_layers=target_layers,
        memory_by_layer=memory_by_layer,
        beta=args.beta,
        value_blend_lambda=args.value_blend_lambda,
        inject_value_prefix=args.inject_value_prefix,
        value_beta_scale=args.value_beta_scale,
        key_phase_tokens=args.key_phase_tokens,
        positional_reset_prefix=(not args.no_positional_reset_prefix),
        field_gated_injection=args.field_gated_injection,
        max_new_tokens=args.max_new_tokens,
    )
    dlpc_json = extract_largest_valid_json(dlpc_result.text)
    dlpc_aligned = align_to_schema(demo_schema_obj, dlpc_json if dlpc_json is not None else {})
    dlpc_schema_stats = schema_metrics(demo_schema_obj, dlpc_json, dlpc_aligned)
    print("[DLPC] target_layers:", target_layers)
    print("[DLPC] elapsed_sec:", round(dlpc_result.elapsed_sec, 3))
    print("[DLPC] aligned_json:\n", pretty_json(dlpc_aligned))

    naive_value_stats = None
    dlpc_value_stats = None
    if args.query_gt_json:
        gt = _load_json(args.query_gt_json)
        naive_value_stats = value_match_metrics(gt, naive_aligned, demo_schema_obj)
        dlpc_value_stats = value_match_metrics(gt, dlpc_aligned, demo_schema_obj)
        print("[NAIVE] value_match_metrics:", naive_value_stats)
        print("[DLPC] value_match_metrics:", dlpc_value_stats)

    out_obj: Dict[str, Any] = {
        "config": {
            "model_name": args.model_name,
            "shots": args.shots,
            "demo_pdf": args.demo_pdf,
            "demo_json": args.demo_json,
            "query_pdf": args.query_pdf,
            "query_gt_json": args.query_gt_json,
            "target_layers": target_layers,
            "top_n_visual": args.top_n_visual,
            "top_n_text": args.top_n_text,
            "prefix_len": args.prefix_len,
            "compressor_mode": args.compressor_mode,
            "schema_only_prefix": args.schema_only_prefix,
            "beta": args.beta,
            "value_blend_lambda": args.value_blend_lambda,
            "value_beta_scale": args.value_beta_scale,
            "key_phase_tokens": args.key_phase_tokens,
            "positional_reset_prefix": (not args.no_positional_reset_prefix),
            "inject_value_prefix": args.inject_value_prefix,
            "field_gated_injection": args.field_gated_injection,
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
        "dlpc": {
            "elapsed_sec": dlpc_result.elapsed_sec,
            "output": dlpc_result.text,
            "parsed_json": dlpc_json,
            "aligned_json": dlpc_aligned,
            "schema_metrics": dlpc_schema_stats,
            "value_match_metrics": dlpc_value_stats,
            "prefix_mass_logs": dlpc_prefix_mass_logs,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"\nSaved compare log to: {out_path}")


if __name__ == "__main__":
    main()

