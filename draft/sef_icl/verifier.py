"""Phase E: self-verification for a single field."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import prompts


@dataclass
class VerifyResult:
    field_path: str
    supported: bool
    corrected_value: str
    reason: str
    confidence: float
    raw_text: str


def verify_field(
    runner,
    query_image: str,
    field_path: str,
    evidence_text: str,
    predicted_value: str,
    max_new_tokens: int = 384,
) -> VerifyResult:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": prompts.VERIFY_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": query_image},
                {
                    "type": "text",
                    "text": prompts.verify_user_prompt(field_path, predicted_value, evidence_text),
                },
            ],
        },
    ]
    res = runner.generate(messages, max_new_tokens=max_new_tokens)
    parsed = _parse_verify_json(res.text)
    supported = bool(parsed.get("supported", False))
    corrected = str(parsed.get("corrected_value") or "").strip()
    if not corrected:
        corrected = predicted_value
    reason = str(parsed.get("reason") or "")
    try:
        conf = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    return VerifyResult(
        field_path=field_path,
        supported=supported,
        corrected_value=corrected,
        reason=reason,
        confidence=conf,
        raw_text=res.text,
    )


def _parse_verify_json(text: str) -> Dict[str, Any]:
    from json_utils import parse_json_object

    obj = parse_json_object(text)
    return obj if obj else {}
