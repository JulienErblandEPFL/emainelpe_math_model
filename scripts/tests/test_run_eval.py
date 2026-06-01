"""Tests for ``scripts/run_eval.py``.

CPU-only. Never imports ``vllm``, ``torch``, ``transformers.AutoTokenizer``,
or ``AutoConfig`` — those live in runtime helpers we don't unit-test.

Coverage map:

- ``load_eval_jsonl`` — JSONL parsing, blank-line skip, malformed-line raise
- ``normalize_input_row`` — both supported schemas (validation_samples and
  data_out/eval.jsonl messages-shape) plus error cases
- ``build_generations_dump`` — shape sanity and uniform-n enforcement
- ``write_generations_jsonl`` — round-trip
- ``_check_max_model_len`` — comparison logic only (AutoConfig load uncovered)
- ``load_generation_config_from_model_dir`` — returns None for HF-id strings,
  parses real files, swallows malformed JSON with a warning
- ``resolve_sampling_params`` — three-tier priority (CLI > gen_config > fallback)
  and override-warning emission
- ``format_summary`` — output shape
- One integration test: pass our generations-dump shape into the real
  ``evaluate.score.score_generations`` at production n=8 with mixed
  correct/incorrect, asserting numerically correct pass@1 / pass@8.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_eval import (
    CI_MAX_MODEL_LEN,
    CI_MAX_NEW_TOKENS,
    FALLBACK_TEMPERATURE,
    FALLBACK_TOP_K,
    FALLBACK_TOP_P,
    LEGACY_MAX_MODEL_LEN,
    LEGACY_MAX_NEW_TOKENS,
    _check_max_model_len,
    build_generations_dump,
    format_summary,
    load_eval_jsonl,
    load_generation_config_from_model_dir,
    normalize_input_row,
    resolve_context_caps,
    resolve_sampling_params,
    write_generations_jsonl,
)


# =============================================================================
# load_eval_jsonl
# =============================================================================

def test_load_eval_jsonl_skips_blank_lines(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    assert load_eval_jsonl(f) == [{"a": 1}, {"b": 2}]


def test_load_eval_jsonl_raises_on_invalid_json(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_eval_jsonl(f)


def test_load_eval_jsonl_returns_empty_for_empty_file(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text("", encoding="utf-8")
    assert load_eval_jsonl(f) == []


# =============================================================================
# normalize_input_row
# =============================================================================

def test_normalize_passes_validation_samples_schema_through():
    row = {"prompt": "What is 2+2?", "answer": "4"}
    assert normalize_input_row(row) == {"prompt": "What is 2+2?", "answer": "4"}


def test_normalize_coerces_non_string_answer_to_string():
    """validation_samples answers are typically strings, but the JSON spec
    permits numbers; coercing keeps downstream score_generations happy
    (its _gold() also coerces with str())."""
    row = {"prompt": "Q", "answer": 42}
    assert normalize_input_row(row) == {"prompt": "Q", "answer": "42"}


def test_normalize_extracts_from_messages_schema():
    """data_out/eval.jsonl rows carry an assistant turn shaped as
    ``<think>...</think>\\n\\n\\boxed{ANS}``. The extractor must recover ANS."""
    row = {"messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>\n2+2=4\n</think>\n\n\\boxed{4}"},
    ]}
    assert normalize_input_row(row) == {"prompt": "What is 2+2?", "answer": "4"}


def test_normalize_messages_handles_nested_braces():
    """The boxed answer may itself contain braces (LaTeX expressions);
    relies on evaluate.extract_answer's brace-balanced extractor."""
    row = {"messages": [
        {"role": "user", "content": "Compute."},
        {"role": "assistant", "content": "<think>...</think>\n\n\\boxed{\\frac{1}{2}}"},
    ]}
    out = normalize_input_row(row)
    assert out["answer"] == "\\frac{1}{2}"


def test_normalize_messages_raises_on_no_boxed():
    row = {"messages": [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "The answer is 4."},
    ]}
    with pytest.raises(ValueError, match=r"no \\boxed"):
        normalize_input_row(row)


def test_normalize_messages_raises_on_wrong_user_role():
    row = {"messages": [
        {"role": "system", "content": "..."},
        {"role": "assistant", "content": "\\boxed{4}"},
    ]}
    with pytest.raises(ValueError, match="user"):
        normalize_input_row(row)


def test_normalize_messages_raises_on_wrong_assistant_role():
    row = {"messages": [
        {"role": "user", "content": "Q"},
        {"role": "user", "content": "\\boxed{4}"},
    ]}
    with pytest.raises(ValueError, match="assistant"):
        normalize_input_row(row)


def test_normalize_messages_raises_on_too_short():
    row = {"messages": [{"role": "user", "content": "Q"}]}
    with pytest.raises(ValueError, match="too short"):
        normalize_input_row(row)


def test_normalize_raises_on_unknown_schema():
    row = {"foo": "bar"}
    with pytest.raises(ValueError, match="unrecognized"):
        normalize_input_row(row)


# =============================================================================
# build_generations_dump
# =============================================================================

def test_build_generations_dump_attaches_completions():
    items = [{"prompt": "Q1", "answer": "A1"}]
    comps = [["c1", "c2"]]
    rows = build_generations_dump(items, comps)
    assert rows == [{"prompt": "Q1", "answer": "A1", "completions": ["c1", "c2"]}]


def test_build_generations_dump_preserves_order():
    items = [
        {"prompt": "Q1", "answer": "A1"},
        {"prompt": "Q2", "answer": "A2"},
    ]
    comps = [["x"], ["y"]]
    rows = build_generations_dump(items, comps)
    assert rows[0]["prompt"] == "Q1" and rows[0]["completions"] == ["x"]
    assert rows[1]["prompt"] == "Q2" and rows[1]["completions"] == ["y"]


def test_build_generations_dump_handles_empty_items():
    assert build_generations_dump([], []) == []


def test_build_generations_dump_raises_on_count_mismatch():
    with pytest.raises(ValueError, match="items count"):
        build_generations_dump([{"prompt": "Q", "answer": "A"}], [])


def test_build_generations_dump_raises_on_uniform_n_violation():
    """score_generations refuses non-uniform completion counts; we surface
    that earlier so a generation crash isn't blamed on vLLM."""
    items = [
        {"prompt": "Q1", "answer": "A1"},
        {"prompt": "Q2", "answer": "A2"},
    ]
    comps = [["x", "y"], ["z"]]
    with pytest.raises(ValueError, match="completions"):
        build_generations_dump(items, comps)


# =============================================================================
# write_generations_jsonl
# =============================================================================

def test_write_generations_jsonl_roundtrips(tmp_path):
    rows = [
        {"prompt": "Q1", "answer": "A1", "completions": ["c1", "c2"]},
        {"prompt": "Q2", "answer": "A2", "completions": ["c3", "c4"]},
    ]
    path = tmp_path / "out.jsonl"
    write_generations_jsonl(rows, path)
    assert load_eval_jsonl(path) == rows


def test_write_generations_jsonl_creates_parent_dir(tmp_path):
    path = tmp_path / "nested" / "dirs" / "out.jsonl"
    write_generations_jsonl([{"a": 1}], path)
    assert path.exists()


def test_write_generations_jsonl_preserves_unicode(tmp_path):
    """ensure_ascii=False is used so the file is human-readable for
    non-Latin prompts (e.g. multilingual eval set)."""
    rows = [{"prompt": "你好", "answer": "答", "completions": ["café"]}]
    path = tmp_path / "out.jsonl"
    write_generations_jsonl(rows, path)
    assert "你好" in path.read_text(encoding="utf-8")


# =============================================================================
# _check_max_model_len
# =============================================================================

def test_check_max_model_len_passes_when_ceiling_high():
    _check_max_model_len(positional_ceiling=40960, max_model_len=20480)


def test_check_max_model_len_passes_when_equal():
    _check_max_model_len(positional_ceiling=20480, max_model_len=20480)


def test_check_max_model_len_raises_when_ceiling_low():
    with pytest.raises(RuntimeError, match="max_position_embeddings"):
        _check_max_model_len(positional_ceiling=4096, max_model_len=20480)


def test_check_max_model_len_error_mentions_both_numbers():
    """The error must surface both numbers so the operator can decide
    whether to lower --max-model-len or pick a different model."""
    with pytest.raises(RuntimeError) as exc:
        _check_max_model_len(positional_ceiling=4096, max_model_len=20480)
    msg = str(exc.value)
    assert "4096" in msg
    assert "20480" in msg


# =============================================================================
# resolve_context_caps
# =============================================================================

def test_resolve_context_caps_no_args_returns_ci_caps():
    """No flags, no overrides → CI-faithful 4096 / 4096. This is the
    headline default-flip: a fresh ``python scripts/run_eval.py``
    invocation is now calibrated against what CI will see, not the
    legacy permissive 20480/16384."""
    mml, mnt = resolve_context_caps()
    assert mml == CI_MAX_MODEL_LEN == 4096
    assert mnt == CI_MAX_NEW_TOKENS == 4096


def test_resolve_context_caps_default_mode_is_ci():
    """Explicit ``legacy_mode=False`` (the default), no overrides:
    matches the README's combined max_model_len cap."""
    mml, mnt = resolve_context_caps(
        legacy_mode=False, max_model_len_arg=None, max_new_tokens_arg=None
    )
    assert mml == CI_MAX_MODEL_LEN == 4096
    assert mnt == CI_MAX_NEW_TOKENS == 4096


def test_resolve_context_caps_legacy_mode():
    """--no-ci-mode (legacy escape hatch): 20480 / 16384. Tracks
    docs/project_description.pdf page 3 (max_new_tokens=16384)."""
    mml, mnt = resolve_context_caps(
        legacy_mode=True, max_model_len_arg=None, max_new_tokens_arg=None
    )
    assert mml == LEGACY_MAX_MODEL_LEN == 20480
    assert mnt == LEGACY_MAX_NEW_TOKENS == 16384


def test_resolve_context_caps_explicit_max_model_len_wins_in_ci_mode():
    mml, mnt = resolve_context_caps(
        legacy_mode=False, max_model_len_arg=8192, max_new_tokens_arg=None
    )
    assert mml == 8192
    assert mnt == CI_MAX_NEW_TOKENS


def test_resolve_context_caps_explicit_max_new_tokens_wins_in_ci_mode():
    mml, mnt = resolve_context_caps(
        legacy_mode=False, max_model_len_arg=None, max_new_tokens_arg=2048
    )
    assert mml == CI_MAX_MODEL_LEN
    assert mnt == 2048


def test_resolve_context_caps_explicit_overrides_apply_in_legacy_mode():
    mml, mnt = resolve_context_caps(
        legacy_mode=True, max_model_len_arg=10000, max_new_tokens_arg=8000
    )
    assert mml == 10000
    assert mnt == 8000


def test_resolve_context_caps_independent_overrides():
    """Overriding one cap must not affect the resolution of the other."""
    mml, mnt = resolve_context_caps(
        legacy_mode=False, max_model_len_arg=10000, max_new_tokens_arg=8000
    )
    assert (mml, mnt) == (10000, 8000)


# =============================================================================
# load_generation_config_from_model_dir
# =============================================================================

def test_load_generation_config_returns_none_for_hf_id():
    """Bare HF-id strings ('Qwen/Qwen3-1.7B') aren't local dirs."""
    assert load_generation_config_from_model_dir("Qwen/Qwen3-1.7B") is None


def test_load_generation_config_returns_none_when_missing(tmp_path):
    assert load_generation_config_from_model_dir(str(tmp_path)) is None


def test_load_generation_config_parses_json(tmp_path):
    (tmp_path / "generation_config.json").write_text(
        '{"temperature": 0.6, "top_p": 0.9, "top_k": 50}',
        encoding="utf-8",
    )
    assert load_generation_config_from_model_dir(str(tmp_path)) == {
        "temperature": 0.6, "top_p": 0.9, "top_k": 50,
    }


def test_load_generation_config_returns_none_on_malformed_json(tmp_path, caplog):
    (tmp_path / "generation_config.json").write_text(
        "this is not json", encoding="utf-8"
    )
    with caplog.at_level("WARNING"):
        result = load_generation_config_from_model_dir(str(tmp_path))
    assert result is None
    assert any("malformed" in r.message.lower() for r in caplog.records)


# =============================================================================
# resolve_sampling_params
# =============================================================================

def _ns(**overrides):
    """argparse.Namespace stub with sensible CI-contract defaults."""
    base = dict(
        temperature=None, top_p=None, top_k=None,
        n=8, max_new_tokens=16384, seed=42,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_resolve_sampling_params_no_override_uses_fallback(caplog):
    with caplog.at_level("INFO"):
        params = resolve_sampling_params(_ns(), gen_config_dict=None)
    assert params["temperature"] == FALLBACK_TEMPERATURE
    assert params["top_p"] == FALLBACK_TOP_P
    assert params["top_k"] == FALLBACK_TOP_K
    assert params["n"] == 8
    assert params["max_tokens"] == 16384
    assert params["seed"] == 42
    assert any("Stage-4 fallback" in r.message for r in caplog.records)
    assert not any(r.levelname == "WARNING" for r in caplog.records)


def test_resolve_sampling_params_uses_generation_config_when_present(caplog):
    gc = {"temperature": 0.6, "top_p": 0.9, "top_k": 50}
    with caplog.at_level("INFO"):
        params = resolve_sampling_params(_ns(), gen_config_dict=gc)
    assert params["temperature"] == 0.6
    assert params["top_p"] == 0.9
    assert params["top_k"] == 50
    assert any("generation_config.json" in r.message for r in caplog.records)


def test_resolve_sampling_params_partial_gen_config_falls_back_per_field():
    """generation_config.json only sets temperature; top_p and top_k stay
    on the Stage-4 fallback. This is the realistic Stage 5 case where the
    pushed config might omit some fields."""
    gc = {"temperature": 0.6}
    params = resolve_sampling_params(_ns(), gen_config_dict=gc)
    assert params["temperature"] == 0.6
    assert params["top_p"] == FALLBACK_TOP_P
    assert params["top_k"] == FALLBACK_TOP_K


def test_resolve_sampling_params_warns_on_each_override(caplog):
    args = _ns(temperature=0.7, top_p=None, top_k=15)
    with caplog.at_level("WARNING"):
        params = resolve_sampling_params(args, gen_config_dict=None)
    assert params["temperature"] == 0.7
    assert params["top_p"] == FALLBACK_TOP_P
    assert params["top_k"] == 15
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 2
    assert any("temperature" in w.message for w in warnings)
    assert any("top_k" in w.message for w in warnings)


def test_resolve_sampling_params_cli_overrides_gen_config(caplog):
    """CLI is the highest-priority tier; it must beat generation_config.json."""
    gc = {"temperature": 0.6, "top_p": 0.9, "top_k": 50}
    args = _ns(temperature=0.1)
    with caplog.at_level("WARNING"):
        params = resolve_sampling_params(args, gen_config_dict=gc)
    assert params["temperature"] == 0.1
    assert params["top_p"] == 0.9
    assert params["top_k"] == 50


def test_resolve_sampling_params_threads_locked_ci_contract_fields():
    """n, max_tokens, seed are taken straight from args; not subject to
    fallback resolution."""
    args = _ns(n=4, max_new_tokens=512, seed=7)
    params = resolve_sampling_params(args, gen_config_dict=None)
    assert params["n"] == 4
    assert params["max_tokens"] == 512
    assert params["seed"] == 7


def test_resolve_sampling_params_with_loaded_generation_config_roundtrip(tmp_path):
    """End-to-end: write a generation_config.json on disk, load it via the
    real loader, hand it to the resolver. Catches any drift between the
    loader's output shape and the resolver's input expectations."""
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"temperature": 0.6, "top_p": 0.9, "top_k": 50}),
        encoding="utf-8",
    )
    gc = load_generation_config_from_model_dir(str(tmp_path))
    params = resolve_sampling_params(_ns(), gen_config_dict=gc)
    assert params["temperature"] == 0.6
    assert params["top_p"] == 0.9
    assert params["top_k"] == 50


# =============================================================================
# format_summary
# =============================================================================

def test_format_summary_renders_pass_at_k():
    fake = {
        "metrics": {"pass@1": 0.4, "pass@8": 0.7},
        "n_problems": 10,
        "n_completions": 8,
        "benchmark_method": "boxed",
    }
    s = format_summary(fake)
    assert "pass@1=0.4000" in s
    assert "pass@8=0.7000" in s
    assert "n_problems=10" in s
    assert "n_completions=8" in s
    assert "method=boxed" in s


# =============================================================================
# Integration: feed our dump shape into the real evaluate.score scorer
# =============================================================================

def test_score_generations_accepts_our_dump_shape_n8():
    """Production case: build a 4-problem dump at n=8 with mixed
    correct/incorrect completions and verify pass@1 and pass@8 are
    numerically correct under the unbiased Chen-et-al estimator.

    Per-problem c counts and pass@k contributions:
      Problem 0: 8/8 correct → c=8, pass@1=8/8=1.000, pass@8=1.0
      Problem 1: 0/8 correct → c=0, pass@1=0/8=0.000, pass@8=0.0
      Problem 2: 4/8 correct → c=4, pass@1=4/8=0.500, pass@8=1.0 (any-of-8)
      Problem 3: 1/8 correct → c=1, pass@1=1/8=0.125, pass@8=1.0

    Aggregate: pass@1 = (1.0 + 0.0 + 0.5 + 0.125) / 4 = 0.40625
              pass@8 = (1.0 + 0.0 + 1.0 + 1.0) / 4 = 0.75

    This catches schema-mismatch bugs against the production n=8 contract,
    not just the "any non-empty" path.
    """
    from evaluate.score import score_generations

    correct = "Some reasoning. \\boxed{4}"
    wrong = "Some reasoning. \\boxed{5}"

    dump = [
        {"prompt": "Q0", "answer": "4", "completions": [correct] * 8},
        {"prompt": "Q1", "answer": "4", "completions": [wrong] * 8},
        {"prompt": "Q2", "answer": "4", "completions": [correct] * 4 + [wrong] * 4},
        {"prompt": "Q3", "answer": "4", "completions": [correct] + [wrong] * 7},
    ]
    result = score_generations(dump, method="boxed")

    assert result["n_problems"] == 4
    assert result["n_completions"] == 8
    assert result["benchmark_method"] == "boxed"
    assert result["metrics"]["pass@1"] == pytest.approx(0.40625)
    assert result["metrics"]["pass@8"] == pytest.approx(0.75)
