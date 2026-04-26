"""Prompt strings for SEF-ICL v1 (text evidence + field ICL + verifier)."""
from __future__ import annotations

EVIDENCE_SYSTEM = (
    "You are reading a Vietnamese banking form image. "
    "Locate textual evidence that would support filling one target field. "
    "Reply with one JSON object only, no markdown."
)

VERIFY_SYSTEM = (
    "You verify whether an extracted field value is directly supported by given evidence "
    "from a Vietnamese banking form. Reply with one JSON object only, no markdown."
)

EXTRACT_SYSTEM = (
    "You extract a single field value for structured OCR from a Vietnamese banking form. "
    "Follow the few-shot style of the examples. Output only the field value as plain text, "
    "no JSON, no quotes unless part of the value."
)


def evidence_user_prompt(field_path: str) -> str:
    return (
        f"Target field path:\n{field_path}\n\n"
        "Find the text evidence in the image that supports this field.\n"
        "Return JSON only with this shape:\n"
        '{\n  "evidence_text": "",\n  "nearby_labels": [],\n  "confidence": 0.0\n}\n'
    )


def extract_user_prompt(field_path: str, evidence_text: str, field_examples_block: str) -> str:
    return (
        f"Target field:\n{field_path}\n\n"
        f"Evidence from the document:\n{evidence_text}\n\n"
        f"Few-shot examples (evidence -> value):\n{field_examples_block}\n\n"
        "Now extract the value for the target field from the evidence. "
        "Return the field value only (single line)."
    )


def verify_user_prompt(field_path: str, predicted_value: str, evidence_text: str) -> str:
    return (
        f"Field:\n{field_path}\n\n"
        f"Predicted value:\n{predicted_value}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Is the predicted value directly supported by the evidence?\n"
        "Return JSON only:\n"
        '{\n  "supported": true,\n  "corrected_value": "",\n  "reason": "",\n  "confidence": 0.0\n}\n'
    )
