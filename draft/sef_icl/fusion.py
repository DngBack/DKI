"""Fuse SEF field output with naive full-document fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from verifier import VerifyResult


@dataclass
class FusionDecision:
    field_path: str
    final_value: str
    source: str  # "sef" | "naive"
    verify: Optional[VerifyResult] = None


def fuse_field_value(
    field_path: str,
    naive_value: str,
    verify: VerifyResult,
    confidence_threshold: float,
    require_supported: bool = True,
) -> FusionDecision:
    ok = verify.confidence >= confidence_threshold
    if require_supported:
        ok = ok and verify.supported
    if ok:
        return FusionDecision(
            field_path=field_path,
            final_value=str(verify.corrected_value or ""),
            source="sef",
            verify=verify,
        )
    return FusionDecision(
        field_path=field_path,
        final_value=naive_value,
        source="naive",
        verify=verify,
    )
