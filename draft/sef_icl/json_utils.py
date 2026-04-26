"""JSON extraction helpers (delegates to schema_eval)."""
from __future__ import annotations

from typing import Any, Dict

import bootstrap

bootstrap.ensure_dki_src_path()

from schema_eval import extract_largest_valid_json  # noqa: E402

__all__ = ["extract_largest_valid_json", "parse_json_object"]


def parse_json_object(text: str) -> Dict[str, Any] | None:
    obj = extract_largest_valid_json(text)
    if isinstance(obj, dict):
        return obj
    return None
