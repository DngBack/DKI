"""Load QwenVLRunner from draft/test_dki/src (avoid name shadowing with this file)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

def _load_dki_qwen_runner():
    src = Path(__file__).resolve().parent.parent / "draft" / "test_dki" / "src" / "qwen_runner.py"
    spec = importlib.util.spec_from_file_location("dki_qwen_runner", src)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_mod = _load_dki_qwen_runner()
GenerationResult = _mod.GenerationResult
QwenVLRunner = _mod.QwenVLRunner

__all__ = ["GenerationResult", "QwenVLRunner"]
