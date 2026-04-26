"""Ensure draft/test_dki/src is on sys.path for shared Qwen + schema_eval code."""
from __future__ import annotations

import sys
from pathlib import Path


def ensure_dki_src_path() -> None:
    root = Path(__file__).resolve().parent.parent
    src = root / "draft" / "test_dki" / "src"
    if src.is_dir():
        p = str(src)
        if p not in sys.path:
            # Append so `sef_icl/` (often inserted first by entry scripts) wins local imports
            # like `qwen_runner.py` while `methods` / `schema_eval` still resolve from this src.
            sys.path.append(p)
