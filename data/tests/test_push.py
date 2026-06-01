"""CPU-only tests for scripts/push.py.

Tests the preflight surface (required args, generation_config.json
presence, HF auth) without performing any real upload. The push_to_hub
call itself needs HF credentials and is exercised manually.
"""
from __future__ import annotations

import json

import pytest

from scripts.push import (
    default_commit_message,
    has_hf_auth,
    main,
    verify_model_dir,
    _parse_args,
)


# =============================================================================
# default_commit_message
# =============================================================================

def test_default_commit_message_is_generic():
    msg = default_commit_message()
    assert "eval_loss" not in msg
    assert isinstance(msg, str)
    assert msg.strip()


# =============================================================================
# CLI surface
# =============================================================================

def test_parse_args_requires_model_dir_and_hf_repo():
    with pytest.raises(SystemExit):
        _parse_args([])


def test_parse_args_requires_hf_repo_when_only_model_dir_given(tmp_path):
    with pytest.raises(SystemExit):
        _parse_args(["--model-dir", str(tmp_path)])


def test_parse_args_accepts_required_pair(tmp_path):
    args = _parse_args([
        "--model-dir", str(tmp_path),
        "--hf-repo", "your-org/your-repo",
    ])
    assert args.model_dir == tmp_path
    assert args.hf_repo == "your-org/your-repo"
    assert args.commit_message is None  # default → default_commit_message()


# =============================================================================
# verify_model_dir — generation_config.json presence is mandatory
# =============================================================================

def _good_model_dir(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"temperature": 0.4}), encoding="utf-8"
    )
    return tmp_path


def test_verify_model_dir_passes_when_generation_config_present(tmp_path):
    d = _good_model_dir(tmp_path / "out")
    verify_model_dir(d)  # must not raise


def test_verify_model_dir_fails_when_directory_missing(tmp_path):
    with pytest.raises(RuntimeError, match="does not exist"):
        verify_model_dir(tmp_path / "no_such_dir")


def test_verify_model_dir_fails_when_generation_config_missing(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    with pytest.raises(RuntimeError, match="generation_config.json"):
        verify_model_dir(d)


# =============================================================================
# main() — fails loud without HF auth (no network calls hit)
# =============================================================================

def test_main_fails_when_generation_config_missing(tmp_path, capsys):
    """No generation_config.json → exit code 2 BEFORE auth or upload."""
    d = tmp_path / "out"
    d.mkdir()
    rc = main([
        "--model-dir", str(d),
        "--hf-repo", "your-org/your-repo",
    ])
    assert rc == 2


def test_main_fails_when_no_hf_auth(tmp_path, monkeypatch):
    """generation_config.json present but no HF_TOKEN and no cached login →
    exit code 3 BEFORE attempting any upload."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    # Force has_hf_auth() to report False by stubbing HfFolder.get_token.
    try:
        from huggingface_hub import HfFolder
        monkeypatch.setattr(HfFolder, "get_token", staticmethod(lambda: None))
    except ImportError:
        pass

    d = _good_model_dir(tmp_path / "out")
    rc = main([
        "--model-dir", str(d),
        "--hf-repo", "your-org/your-repo",
    ])
    assert rc == 3


def test_has_hf_auth_true_when_hf_token_env_set(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    assert has_hf_auth() is True
