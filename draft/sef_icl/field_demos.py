"""Phase B: convert document demos into field-level text demonstrations."""
from __future__ import annotations

from typing import Any, Dict, List

import schema_tools as st


DemoSpec = Dict[str, str]  # {"image": path, "text": full_json_string}


def value_at_path(json_obj: Dict[str, Any], path: str) -> str:
    v = st.get_by_path(json_obj, path)
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return ""
    return str(v)


def heuristic_demo_evidence(field_path: str, value: str) -> str:
    leaf = st.leaf_key_from_path(field_path)
    if value.strip():
        return f'Form region associated with labels like "{leaf}"; demo labeled value: {value}'
    return f'Form region associated with labels like "{leaf}" (demo had empty value).'


def build_demo_evidence_for_field(
    runner,
    demo_image: str,
    field_path: str,
    max_new_tokens: int = 384,
) -> str:
    """Optional: run evidence proposal on a demo document for richer field demos."""
    from evidence import propose_evidence

    ep = propose_evidence(runner, demo_image, field_path, max_new_tokens=max_new_tokens)
    if ep.evidence_text.strip():
        return ep.evidence_text.strip()
    return heuristic_demo_evidence(field_path, "")


def build_field_demo_lines(
    demos: List[DemoSpec],
    field_path: str,
    runner=None,
    demo_evidence_from_image: bool = False,
    max_new_tokens_evidence: int = 384,
) -> str:
    lines: List[str] = []
    for i, demo in enumerate(demos, start=1):
        import json

        try:
            j = json.loads(demo["text"])
        except json.JSONDecodeError:
            j = {}
        val = value_at_path(j, field_path)
        if demo_evidence_from_image and runner is not None:
            ev = build_demo_evidence_for_field(
                runner, demo["image"], field_path, max_new_tokens=max_new_tokens_evidence
            )
        else:
            ev = heuristic_demo_evidence(field_path, val)
        lines.append(f"Example {i}:\n  Field: {field_path}\n  Evidence: {ev}\n  Answer: {val}\n")
    return "\n".join(lines).strip()
