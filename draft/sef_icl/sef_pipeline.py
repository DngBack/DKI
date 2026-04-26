"""SEF-ICL v1: text evidence + field-level ICL + verifier + naive fallback."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import bootstrap

bootstrap.ensure_dki_src_path()

from methods import build_naive_messages  # noqa: E402

import field_demos
import schema_tools as st
from evidence import propose_evidence
from field_extract import extract_field_value
from fusion import fuse_field_value
from json_utils import extract_largest_valid_json
from metrics import align_to_schema, schema_metrics, value_match_metrics
from verifier import verify_field


def _load_json_obj(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _demo_items_from_files(demo_pdf: str, demo_json: str, shots: int) -> List[Dict[str, str]]:
    text = json.dumps(_load_json_obj(demo_json), ensure_ascii=False, indent=2)
    return [{"id": f"demo_{i+1}", "image": demo_pdf, "text": text} for i in range(shots)]


def run_sef_icl_v1(
    runner,
    demo_items: List[Dict[str, str]],
    query_image: str,
    schema_obj: Dict[str, Any],
    *,
    naive_max_new_tokens: int = 1024,
    field_max_new_tokens: int = 384,
    extract_max_new_tokens: int = 256,
    verify_max_new_tokens: int = 384,
    confidence_threshold: float = 0.5,
    demo_evidence_from_image: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    naive_messages = build_naive_messages(demo_items, query_image, schema_obj)
    naive_result = runner.generate(naive_messages, max_new_tokens=naive_max_new_tokens)
    naive_json = extract_largest_valid_json(naive_result.text)
    naive_aligned = align_to_schema(schema_obj, naive_json if naive_json is not None else {})
    naive_schema_stats = schema_metrics(schema_obj, naive_json, naive_aligned)

    fields = st.flatten_schema_paths(schema_obj)
    path_to_value: Dict[str, str] = {}
    per_field: List[Dict[str, Any]] = []
    field_time_sec = 0.0

    for fp in fields:
        t_field = time.time()
        ev = propose_evidence(runner, query_image, fp, max_new_tokens=field_max_new_tokens)
        demo_block = field_demos.build_field_demo_lines(
            demo_items,
            fp,
            runner=runner,
            demo_evidence_from_image=demo_evidence_from_image,
            max_new_tokens_evidence=field_max_new_tokens,
        )
        ex = extract_field_value(
            runner,
            query_image,
            fp,
            ev.evidence_text or "(no evidence text)",
            demo_block,
            max_new_tokens=extract_max_new_tokens,
        )
        vr = verify_field(
            runner,
            query_image,
            fp,
            ev.evidence_text,
            ex.value,
            max_new_tokens=verify_max_new_tokens,
        )
        naive_leaf = st.get_by_path(naive_aligned, fp)
        naive_str = "" if naive_leaf is None else str(naive_leaf)
        fus = fuse_field_value(
            fp,
            naive_str,
            vr,
            confidence_threshold=confidence_threshold,
            require_supported=True,
        )
        path_to_value[fp] = fus.final_value
        field_time_sec += time.time() - t_field
        per_field.append(
            {
                "field_path": fp,
                "evidence": {
                    "evidence_text": ev.evidence_text,
                    "nearby_labels": ev.nearby_labels,
                    "confidence": ev.confidence,
                    "raw": ev.raw_text,
                },
                "extract": {"value": ex.value, "raw": ex.raw_text},
                "verify": {
                    "supported": vr.supported,
                    "corrected_value": vr.corrected_value,
                    "reason": vr.reason,
                    "confidence": vr.confidence,
                    "raw": vr.raw_text,
                },
                "fusion": {"final_value": fus.final_value, "source": fus.source},
            }
        )

    assembled = st.assemble_json(schema_obj, path_to_value)
    sef_aligned = align_to_schema(schema_obj, assembled)
    sef_schema_stats = schema_metrics(schema_obj, assembled, sef_aligned)

    total_elapsed = time.time() - t0
    return {
        "naive": {
            "elapsed_sec": naive_result.elapsed_sec,
            "output_text": naive_result.text,
            "parsed_json": naive_json,
            "aligned_json": naive_aligned,
            "schema_metrics": naive_schema_stats,
        },
        "sef": {
            "aligned_json": sef_aligned,
            "schema_metrics": sef_schema_stats,
            "per_field": per_field,
            "field_passes_elapsed_sec": field_time_sec,
            "total_elapsed_sec": total_elapsed,
            "config": {
                "confidence_threshold": confidence_threshold,
                "demo_evidence_from_image": demo_evidence_from_image,
            },
        },
    }


def attach_value_metrics_if_gt(
    out: Dict[str, Any],
    schema_obj: Dict[str, Any],
    gt_obj: Dict[str, Any] | None,
) -> None:
    if not gt_obj:
        return
    out["naive"]["value_match_metrics"] = value_match_metrics(
        gt_obj, out["naive"]["aligned_json"], schema_obj
    )
    out["sef"]["value_match_metrics"] = value_match_metrics(gt_obj, out["sef"]["aligned_json"], schema_obj)
