"""CPU-only tests for scripts/merge_and_push.py.

Exercise only the pure helpers (gen-config writer, chat-template byte
diff, file-existence preflight). The actual merge, smoke inference, and
HF push need a GPU + credentials and are out of scope here. Should run
in well under 30s on a laptop.
"""
from __future__ import annotations

import json

import pytest

from scripts.merge_and_push import (
    GENERATION_CONFIG_REQUIRED_KEYS,
    PreflightError,
    QWEN3_BOS_TOKEN_ID,
    QWEN3_EOS_TOKEN_IDS,
    QWEN3_PAD_TOKEN_ID,
    TRANSFORMERS_VERSION,
    build_generation_config,
    chat_templates_byte_match,
    default_commit_message,
    read_saved_chat_template,
    run_file_preflight,
    write_generation_config,
)


# =============================================================================
# generation_config.json schema writer
# =============================================================================

def test_build_generation_config_uses_cli_sampling_values():
    cfg = build_generation_config(temperature=0.7, top_p=0.8, top_k=40)
    assert cfg["temperature"] == 0.7
    assert cfg["top_p"] == 0.8
    assert cfg["top_k"] == 40


def test_build_generation_config_locks_qwen3_token_ids_and_do_sample():
    """do_sample, BOS/EOS/pad, and transformers_version are NOT CLI-controlled.
    Pinning them here means a future change to the schema fails this test
    instead of silently shipping a checkpoint that the CI samples wrong."""
    cfg = build_generation_config(temperature=0.3, top_p=0.95, top_k=20)
    assert cfg["do_sample"] is True
    assert cfg["bos_token_id"] == QWEN3_BOS_TOKEN_ID
    assert cfg["pad_token_id"] == QWEN3_PAD_TOKEN_ID
    assert cfg["eos_token_id"] == list(QWEN3_EOS_TOKEN_IDS)
    assert cfg["transformers_version"] == TRANSFORMERS_VERSION


def test_build_generation_config_has_all_required_keys():
    cfg = build_generation_config(temperature=0.3, top_p=0.95, top_k=20)
    for key in GENERATION_CONFIG_REQUIRED_KEYS:
        assert key in cfg


def test_write_generation_config_writes_valid_json_with_cli_values(tmp_path):
    target = write_generation_config(
        tmp_path, temperature=0.42, top_p=0.9, top_k=15
    )
    assert target == tmp_path / "generation_config.json"
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["temperature"] == 0.42
    assert data["top_p"] == 0.9
    assert data["top_k"] == 15
    # And the locked fields are still there:
    assert data["do_sample"] is True
    assert data["eos_token_id"] == list(QWEN3_EOS_TOKEN_IDS)


def test_write_generation_config_creates_output_dir_if_missing(tmp_path):
    nested = tmp_path / "does" / "not" / "yet" / "exist"
    write_generation_config(nested, temperature=0.3, top_p=0.95, top_k=20)
    assert (nested / "generation_config.json").is_file()


# =============================================================================
# chat-template byte diff
# =============================================================================

def test_chat_templates_byte_match_returns_true_when_identical(tmp_path):
    a = tmp_path / "a.jinja"
    b = tmp_path / "b.jinja"
    a.write_text("{%- set enable_thinking = true %}\nhello\n", encoding="utf-8")
    b.write_text("{%- set enable_thinking = true %}\nhello\n", encoding="utf-8")
    assert chat_templates_byte_match(a, b)


def test_chat_templates_byte_match_returns_false_on_one_byte_drift(tmp_path):
    a = tmp_path / "a.jinja"
    b = tmp_path / "b.jinja"
    a.write_text("{%- set enable_thinking = true %}\n", encoding="utf-8")
    b.write_text("{%- set enable_thinking = false %}\n", encoding="utf-8")
    assert not chat_templates_byte_match(a, b)


def test_chat_templates_byte_match_catches_trailing_newline_drift(tmp_path):
    """The whole point of this check is catching invisible diffs that
    HF tokenizer.save_pretrained sometimes introduces (e.g. stripping
    a trailing newline)."""
    a = tmp_path / "a.jinja"
    b = tmp_path / "b.jinja"
    a.write_text("hello\n", encoding="utf-8")
    b.write_text("hello", encoding="utf-8")
    assert not chat_templates_byte_match(a, b)


# =============================================================================
# read_saved_chat_template
# =============================================================================

def test_read_saved_chat_template_prefers_jinja_sidecar(tmp_path):
    (tmp_path / "chat_template.jinja").write_text("FROM SIDECAR", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "FROM CONFIG"}), encoding="utf-8"
    )
    assert read_saved_chat_template(tmp_path) == "FROM SIDECAR"


def test_read_saved_chat_template_falls_back_to_tokenizer_config(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "FROM CONFIG"}), encoding="utf-8"
    )
    assert read_saved_chat_template(tmp_path) == "FROM CONFIG"


def test_read_saved_chat_template_raises_when_neither_present(tmp_path):
    with pytest.raises(PreflightError, match="chat template"):
        read_saved_chat_template(tmp_path)


# =============================================================================
# run_file_preflight — synthetic-dir pass + each individual failure
# =============================================================================

LOCKED_TEMPLATE_TEXT = "{%- set enable_thinking = true %}\nLOCKED\n"


def _good_dir(tmp_path):
    """Build a synthetic merged-checkpoint dir that should pass preflight."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps({"_name_or_path": "x"}))
    (tmp_path / "model.safetensors").write_bytes(b"\x00\x00")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({}))
    (tmp_path / "chat_template.jinja").write_text(
        LOCKED_TEMPLATE_TEXT, encoding="utf-8"
    )
    write_generation_config(
        tmp_path, temperature=0.3, top_p=0.95, top_k=20
    )
    return tmp_path


def _locked_template_file(tmp_path):
    p = tmp_path / "locked.jinja"
    p.write_text(LOCKED_TEMPLATE_TEXT, encoding="utf-8")
    return p


def test_run_file_preflight_passes_on_complete_synthetic_dir(tmp_path):
    out = _good_dir(tmp_path / "out")
    locked = _locked_template_file(tmp_path)
    run_file_preflight(out, locked)  # must not raise


def test_run_file_preflight_fails_when_output_dir_missing(tmp_path):
    locked = _locked_template_file(tmp_path)
    with pytest.raises(PreflightError, match="does not exist"):
        run_file_preflight(tmp_path / "no_such_dir", locked)


def test_run_file_preflight_fails_when_config_json_missing(tmp_path):
    out = _good_dir(tmp_path / "out")
    (out / "config.json").unlink()
    locked = _locked_template_file(tmp_path)
    with pytest.raises(PreflightError, match="config.json"):
        run_file_preflight(out, locked)


def test_run_file_preflight_fails_when_no_safetensors(tmp_path):
    out = _good_dir(tmp_path / "out")
    (out / "model.safetensors").unlink()
    locked = _locked_template_file(tmp_path)
    with pytest.raises(PreflightError, match="safetensors"):
        run_file_preflight(out, locked)


def test_run_file_preflight_fails_when_tokenizer_files_missing(tmp_path):
    out = _good_dir(tmp_path / "out")
    (out / "tokenizer.json").unlink()
    locked = _locked_template_file(tmp_path)
    with pytest.raises(PreflightError, match="tokenizer"):
        run_file_preflight(out, locked)


def test_run_file_preflight_fails_when_generation_config_missing(tmp_path):
    out = _good_dir(tmp_path / "out")
    (out / "generation_config.json").unlink()
    locked = _locked_template_file(tmp_path)
    with pytest.raises(PreflightError, match="generation_config.json"):
        run_file_preflight(out, locked)


def test_run_file_preflight_fails_when_generation_config_is_malformed(tmp_path):
    out = _good_dir(tmp_path / "out")
    (out / "generation_config.json").write_text("not json")
    locked = _locked_template_file(tmp_path)
    with pytest.raises(PreflightError, match="not valid JSON"):
        run_file_preflight(out, locked)


def test_run_file_preflight_fails_when_generation_config_is_missing_a_key(tmp_path):
    out = _good_dir(tmp_path / "out")
    bad = {k: 0 for k in GENERATION_CONFIG_REQUIRED_KEYS if k != "do_sample"}
    (out / "generation_config.json").write_text(json.dumps(bad))
    locked = _locked_template_file(tmp_path)
    with pytest.raises(PreflightError, match="missing required keys"):
        run_file_preflight(out, locked)


def test_run_file_preflight_fails_on_chat_template_byte_drift(tmp_path):
    """The silent-failure mode flagged in CLAUDE.md: any drift in the
    locked Jinja silently breaks thinking mode at CI time."""
    out = _good_dir(tmp_path / "out")
    (out / "chat_template.jinja").write_text(
        LOCKED_TEMPLATE_TEXT + "extra\n", encoding="utf-8"
    )
    locked = _locked_template_file(tmp_path)
    with pytest.raises(PreflightError, match="chat_template differs"):
        run_file_preflight(out, locked)


def test_run_file_preflight_accepts_chat_template_in_tokenizer_config(tmp_path):
    """Some transformers versions embed chat_template inside
    tokenizer_config.json instead of writing a sidecar .jinja. The
    preflight must accept either."""
    out = _good_dir(tmp_path / "out")
    (out / "chat_template.jinja").unlink()
    (out / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": LOCKED_TEMPLATE_TEXT}), encoding="utf-8"
    )
    locked = _locked_template_file(tmp_path)
    run_file_preflight(out, locked)  # must not raise


# =============================================================================
# default_commit_message
# =============================================================================

def test_default_commit_message_includes_adapter_basename_and_eval_loss(tmp_path):
    msg = default_commit_message(tmp_path / "runs" / "some-run-name" / "final")
    assert "final" in msg
    assert "eval_loss=0.3803" in msg
