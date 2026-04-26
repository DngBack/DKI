#!/usr/bin/env python3
"""Run SEF-ICL v1 on a query document (text evidence + field ICL + verifier + naive fallback)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="SEF-ICL v1: evidence-first field extraction with Qwen-VL")
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
    parser.add_argument("--query-gt-json", default="", help="Optional ground-truth JSON for field EM")
    parser.add_argument("--shots", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--naive-max-new-tokens", type=int, default=1024)
    parser.add_argument("--field-max-new-tokens", type=int, default=384)
    parser.add_argument("--extract-max-new-tokens", type=int, default=256)
    parser.add_argument("--verify-max-new-tokens", type=int, default=384)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--demo-evidence-from-image",
        action="store_true",
        help="Run evidence proposal on each demo image when building field demos (slow, richer)",
    )
    parser.add_argument("--output", default="sef_icl/results/sef_run.json")
    args = parser.parse_args()

    import bootstrap

    bootstrap.ensure_dki_src_path()

    from metrics import pretty_json
    from qwen_runner import QwenVLRunner
    from sef_pipeline import _demo_items_from_files, _load_json_obj, attach_value_metrics_if_gt, run_sef_icl_v1

    for p in [args.demo_pdf, args.demo_json, args.query_pdf]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input file: {p}")
    if args.query_gt_json and not os.path.exists(args.query_gt_json):
        raise FileNotFoundError(f"Missing query GT JSON: {args.query_gt_json}")

    schema_obj = _load_json_obj(args.demo_json)
    demo_items = _demo_items_from_files(args.demo_pdf, args.demo_json, args.shots)
    gt_obj = _load_json_obj(args.query_gt_json) if args.query_gt_json else None

    runner = QwenVLRunner(
        model_name=args.model_name,
        attn_implementation="eager",
        max_pixels=args.max_pixels,
    )

    out = run_sef_icl_v1(
        runner,
        demo_items,
        args.query_pdf,
        schema_obj,
        naive_max_new_tokens=args.naive_max_new_tokens,
        field_max_new_tokens=args.field_max_new_tokens,
        extract_max_new_tokens=args.extract_max_new_tokens,
        verify_max_new_tokens=args.verify_max_new_tokens,
        confidence_threshold=args.confidence_threshold,
        demo_evidence_from_image=args.demo_evidence_from_image,
    )
    attach_value_metrics_if_gt(out, schema_obj, gt_obj)
    out["config"] = {
        "model_name": args.model_name,
        "shots": args.shots,
        "demo_pdf": args.demo_pdf,
        "demo_json": args.demo_json,
        "query_pdf": args.query_pdf,
        "query_gt_json": args.query_gt_json,
        "max_pixels": args.max_pixels,
        "naive_max_new_tokens": args.naive_max_new_tokens,
        "field_max_new_tokens": args.field_max_new_tokens,
        "extract_max_new_tokens": args.extract_max_new_tokens,
        "verify_max_new_tokens": args.verify_max_new_tokens,
        "confidence_threshold": args.confidence_threshold,
        "demo_evidence_from_image": args.demo_evidence_from_image,
    }

    print("=" * 80)
    print("NAIVE (full-document few-shot)")
    print("=" * 80)
    print("elapsed_sec:", round(out["naive"]["elapsed_sec"], 3))
    print("schema_metrics:", out["naive"]["schema_metrics"])
    if out["naive"].get("value_match_metrics"):
        print("value_match_metrics:", out["naive"]["value_match_metrics"])
    print("aligned_json:\n", pretty_json(out["naive"]["aligned_json"]))

    print("\n" + "=" * 80)
    print("SEF-ICL v1")
    print("=" * 80)
    print("total_elapsed_sec:", round(out["sef"]["total_elapsed_sec"], 3))
    print("field_passes_elapsed_sec:", round(out["sef"]["field_passes_elapsed_sec"], 3))
    print("schema_metrics:", out["sef"]["schema_metrics"])
    if out["sef"].get("value_match_metrics"):
        print("value_match_metrics:", out["sef"]["value_match_metrics"])
    print("aligned_json:\n", pretty_json(out["sef"]["aligned_json"]))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved full log to: {out_path}")


if __name__ == "__main__":
    main()
