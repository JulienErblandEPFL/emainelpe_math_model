"""Tests for ``scripts.train_sft._resolve_resume``.

Exercises the resume-from-checkpoint resolver: ``None`` passthrough,
``"latest"`` numeric-suffix sort (the bug we shipped in Stage 3 and fixed
post-smoke), missing-checkpoint error, and explicit-path passthrough.
"""
from __future__ import annotations

import pytest

from scripts.train_sft import _resolve_resume


def _mkckpts(root, *steps):
    for s in steps:
        (root / f"checkpoint-{s}").mkdir()


def test_resume_none_returns_none(tmp_path):
    assert _resolve_resume(None, tmp_path) is None


def test_resume_latest_sorts_by_integer_step_not_lexicographic(tmp_path):
    """Lexicographic order would put 9500 > 16000 because '9' > '1' at the
    first differing character. The fix sorts by int(suffix) instead."""
    _mkckpts(tmp_path, 500, 9500, 16000)
    resolved = _resolve_resume("latest", tmp_path)
    assert resolved.endswith("/checkpoint-16000")


def test_resume_latest_handles_single_checkpoint(tmp_path):
    _mkckpts(tmp_path, 500)
    resolved = _resolve_resume("latest", tmp_path)
    assert resolved.endswith("/checkpoint-500")


def test_resume_latest_raises_when_no_checkpoints(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        _resolve_resume("latest", tmp_path)


def test_resume_explicit_path_returned_unchanged(tmp_path):
    explicit = str(tmp_path / "checkpoint-12345")
    assert _resolve_resume(explicit, tmp_path) == explicit
