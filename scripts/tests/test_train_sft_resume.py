"""Tests for ``scripts.train_sft._resolve_resume`` plus the v4
``--init-from-adapter`` CLI / validation path (added 2026-05-13).

Exercises the resume-from-checkpoint resolver: ``None`` passthrough,
``"latest"`` numeric-suffix sort (the bug we shipped in Stage 3 and fixed
post-smoke), missing-checkpoint error, and explicit-path passthrough.

Also tests:
  - ``--init-from-adapter`` flag parsing.
  - Mutual exclusion with ``--resume`` (argparse-level error).
  - ``validate_init_adapter_config`` pure helper: refuses adapters whose
    r / alpha / target_modules diverge from the locked configs/lora.yaml.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.train_sft import (
    _parse_args,
    _resolve_resume,
    validate_init_adapter_config,
)


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


# =============================================================================
# --init-from-adapter — CLI + LoRA config validation. Added for v4
# (2026-05-13). Mutually exclusive with --resume.
# =============================================================================

_LOCKED_LORA_YAML = {
    "lora": {
        "r": 32,
        "alpha": 64,
        "dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    },
    "max_seq_length": 4096,
}


def _required_args(*extra: str) -> list[str]:
    """Build a minimal argv that satisfies the three required SFT flags
    plus whatever extra flags the test wants to exercise."""
    base = [
        "--train-file", "/tmp/train.jsonl",
        "--eval-file", "/tmp/eval.jsonl",
        "--output-dir", "/tmp/out",
    ]
    return base + list(extra)


def test_init_from_adapter_flag_parses():
    """--init-from-adapter resolves to a Path argument on args."""
    args = _parse_args(_required_args(
        "--init-from-adapter", "/scratch/Julien/runs/v3/final",
    ))
    assert args.init_from_adapter == Path("/scratch/Julien/runs/v3/final")
    # --resume must default to None (not mutually pre-set).
    assert args.resume is None


def test_init_from_adapter_default_is_none():
    """Without the flag, init_from_adapter defaults to None — preserves
    the v1/v2/v3 training path byte-stable."""
    args = _parse_args(_required_args())
    assert args.init_from_adapter is None


def test_init_from_adapter_resume_mutually_exclusive():
    """Argparse enforces mutual exclusion: passing both --resume and
    --init-from-adapter must exit non-zero with a clear message."""
    with pytest.raises(SystemExit):
        _parse_args(_required_args(
            "--resume", "latest",
            "--init-from-adapter", "/some/v3/final",
        ))


def test_init_from_adapter_lora_config_passes_on_match():
    """A correctly-configured adapter_config.json (r=32, alpha=64, the
    locked 7-module set) must NOT raise."""
    adapter_cfg = {
        "r": 32,
        "lora_alpha": 64,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    }
    # No raise.
    validate_init_adapter_config(adapter_cfg, _LOCKED_LORA_YAML)


def test_init_from_adapter_lora_config_validation_bad_rank():
    """An adapter with r=16 (instead of locked r=32) must be refused
    with a clear message naming both the actual and expected rank.
    Loading such an adapter would silently break the Phase 3 merge."""
    adapter_cfg = {
        "r": 16,  # WRONG — locked is 32
        "lora_alpha": 64,
        "target_modules": _LOCKED_LORA_YAML["lora"]["target_modules"],
    }
    with pytest.raises(RuntimeError) as excinfo:
        validate_init_adapter_config(adapter_cfg, _LOCKED_LORA_YAML)
    msg = str(excinfo.value)
    assert "r=16" in msg  # actual
    assert "r=32" in msg  # expected
    assert "init-from-adapter" in msg  # which flag is at fault


def test_init_from_adapter_lora_config_validation_bad_alpha():
    """An adapter with alpha=128 (instead of locked alpha=64) is
    refused — alpha mismatch silently rescales the merged delta in
    Phase 3."""
    adapter_cfg = {
        "r": 32,
        "lora_alpha": 128,  # WRONG
        "target_modules": _LOCKED_LORA_YAML["lora"]["target_modules"],
    }
    with pytest.raises(RuntimeError) as excinfo:
        validate_init_adapter_config(adapter_cfg, _LOCKED_LORA_YAML)
    msg = str(excinfo.value)
    assert "lora_alpha=128" in msg
    assert "alpha=64" in msg


def test_init_from_adapter_lora_config_validation_bad_target_modules():
    """An adapter with a different target_modules set is refused —
    even a single missing/extra module breaks merge compatibility."""
    # Attention-only adapter (missing the 3 MLP modules).
    adapter_cfg = {
        "r": 32,
        "lora_alpha": 64,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    with pytest.raises(RuntimeError) as excinfo:
        validate_init_adapter_config(adapter_cfg, _LOCKED_LORA_YAML)
    msg = str(excinfo.value)
    assert "target_modules" in msg
    assert "Missing in adapter" in msg
    # The 3 MLP modules are flagged as missing.
    for missing in ("gate_proj", "up_proj", "down_proj"):
        assert missing in msg
