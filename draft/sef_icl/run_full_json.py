#!/usr/bin/env python3
"""Baseline: naive multimodal few-shot full JSON only (no SEF stages)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_json_obj(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _demo_items(demo_pdf: str, demo_json: str, shots: int):
    text = json.dumps(_load_json_obj(demo_json), ensure_ascii=False, indent=2)
    return [{"id": f"demo_{i+1}", "image": demo_pdf, "text": text} for i in range(shots)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Naive full-JSON few-shot with Qwen-VL")
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
    parser.add_argument("--query-gt-json", default="")
    parser.add_argument("--shots", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--output", default="sef_icl/results/naive_full_json.json")
    args = parser.parse_args()

    import bootstrap

    bootstrap.ensure_dki_src_path()

    from methods import build_naive_messages
    from json_utils import extract_largest_valid_json
    from metrics import align_to_schema, pretty_json, schema_metrics, value_match_metrics
    from qwen_runner import QwenVLRunner

    for p in [args.demo_pdf, args.demo_json, args.query_pdf]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input file: {p}")
    if args.query_gt_json and not os.path.exists(args.query_gt_json):
        raise FileNotFoundError(f"Missing query GT JSON: {args.query_gt_json}")

    schema_obj = _load_json_obj(args.demo_json)
    demos = _demo_items(args.demo_pdf, args.demo_json, args.shots)
    runner = QwenVLRunner(
        model_name=args.model_name,
        attn_implementation="eager",
        max_pixels=args.max_pixels,
    )
    msgs = build_naive_messages(demos, args.query_pdf, schema_obj)
    res = runner.generate(msgs, max_new_tokens=args.max_new_tokens)
    pred = extract_largest_valid_json(res.text)
    aligned = align_to_schema(schema_obj, pred if pred is not None else {})
    stats = schema_metrics(schema_obj, pred, aligned)
    out = {
        "config": vars(args),
        "elapsed_sec": res.elapsed_sec,
        "output_text": res.text,
        "parsed_json": pred,
        "aligned_json": aligned,
        "schema_metrics": stats,
    }
    if args.query_gt_json:
        gt = _load_json_obj(args.query_gt_json)
        out["value_match_metrics"] = value_match_metrics(gt, aligned, schema_obj)

    print("elapsed_sec:", round(res.elapsed_sec, 3))
    print("schema_metrics:", stats)
    if out.get("value_match_metrics"):
        print("value_match_metrics:", out["value_match_metrics"])
    print("aligned_json:\n", pretty_json(aligned))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
