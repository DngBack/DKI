"""Phase C: evidence proposal on query (text-first v1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import prompts


@dataclass
class EvidenceProposal:
    field_path: str
    evidence_text: str
    nearby_labels: List[str]
    confidence: float
    raw_text: str


def propose_evidence(
    runner,
    query_image: str,
    field_path: str,
    max_new_tokens: int = 384,
) -> EvidenceProposal:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": prompts.EVIDENCE_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": query_image},
                {"type": "text", "text": prompts.evidence_user_prompt(field_path)},
            ],
        },
    ]
    res = runner.generate(messages, max_new_tokens=max_new_tokens)
    parsed = _parse_evidence_json(res.text)
    return EvidenceProposal(
        field_path=field_path,
        evidence_text=parsed.get("evidence_text") or "",
        nearby_labels=list(parsed.get("nearby_labels") or []),
        confidence=float(parsed.get("confidence") or 0.0),
        raw_text=res.text,
    )


def _parse_evidence_json(text: str) -> Dict[str, Any]:
    from json_utils import parse_json_object

    obj = parse_json_object(text)
    if not obj:
        return {"evidence_text": "", "nearby_labels": [], "confidence": 0.0}
    return obj
