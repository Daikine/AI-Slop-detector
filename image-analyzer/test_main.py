"""Unit tests for image-analyzer local heuristics."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "image_analyzer_main",
    Path(__file__).resolve().parent / "main.py",
)
_mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_mod)


def test_empty_image_is_safe():
    result = _mod._local_analyze(b"", "empty")
    assert result["score"] == 0
    assert result["features"]["qr_codes"] == []
    assert "no image" in result["reason"]
