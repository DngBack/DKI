"""Evaluation metrics bridging to schema_eval."""
from __future__ import annotations

from typing import Any, Dict

import bootstrap

bootstrap.ensure_dki_src_path()

from schema_eval import (  # noqa: E402
    align_to_schema,
    pretty_json,
    schema_metrics,
    value_match_metrics,
)

__all__ = [
    "align_to_schema",
    "pretty_json",
    "schema_metrics",
    "value_match_metrics",
]
