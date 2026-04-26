"""Phase D: field-level value extraction conditioned on evidence + demos."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import prompts


@dataclass
class FieldExtractResult:
    field_path: str
    value: str
    raw_text: str


def extract_field_value(
    runner,
    query_image: str,
    field_path: str,
    evidence_text: str,
    field_examples_block: str,
    max_new_tokens: int = 256,
) -> FieldExtractResult:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": prompts.EXTRACT_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": query_image},
                {
                    "type": "text",
                    "text": prompts.extract_user_prompt(field_path, evidence_text, field_examples_block),
                },
            ],
        },
    ]
    res = runner.generate(messages, max_new_tokens=max_new_tokens)
    val = _strip_value_line(res.text)
    return FieldExtractResult(field_path=field_path, value=val, raw_text=res.text)


def _strip_value_line(text: str) -> str:
    s = text.strip()
    if not s:
        return ""
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""
    first = lines[0]
    for prefix in ("Value:", "Answer:", "Output:", "Field value:"):
        if first.lower().startswith(prefix.lower()):
            first = first[len(prefix) :].strip()
    if (first.startswith('"') and first.endswith('"')) or (first.startswith("'") and first.endswith("'")):
        first = first[1:-1]
    return first.strip()
