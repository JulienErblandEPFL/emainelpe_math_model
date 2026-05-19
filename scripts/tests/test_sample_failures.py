"""CPU-only unit tests for scripts/sample_failures.

All tests target pure helpers — no torch / transformers / vllm imports.
Heavy runtime helpers are reused from ``scripts.eval_local`` and tested
there; this file covers only the *new* logic in sample_failures:
JSONL parsing (delegated to eval_local but exercised end-to-end here),
threshold semantics, output row construction, and summary formatting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.sample_failures import (
    build_failure_rows,
    format_failure_summary,
    is_failure,
    resolve_sampling_params,
    write_failures_jsonl,
)
from scripts.eval_local import load_eval_jsonl, normalize_input_row


# -----------------------------------------------------------------------------
# Test 1 — JSONL parsing: the prompt-set loader must accept both
# {prompt, answer} and {messages: [...]} schemas (the two real-world
# inputs to sample_failures).
# -----------------------------------------------------------------------------

def test_prompt_set_jsonl_parsing(tmp_path: Path):
    path = tmp_path / "problems.jsonl"
    rows = [
        {"prompt": "What is 2+2?", "answer": "4"},
        {"prompt": "What is 3*5?", "answer": "15"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    raw = load_eval_jsonl(path)
    assert len(raw) == 2

    normalized = [normalize_input_row(r) for r in raw]
    assert normalized[0]["prompt"] == "What is 2+2?"
    assert normalized[0]["answer"] == "4"
    assert normalized[1]["answer"] == "15"


# -----------------------------------------------------------------------------
# Test 2 — failure threshold semantics. Strict default (threshold=0)
# fires only on c==0. A non-zero threshold catches "rare-correct"
# problems. Out-of-range inputs raise.
# -----------------------------------------------------------------------------

def test_is_failure_threshold_logic():
    # threshold=0.0 (strict): only c==0 is a failure.
    assert is_failure(c=0, n=4, threshold=0.0) is True
    assert is_failure(c=1, n=4, threshold=0.0) is False
    assert is_failure(c=4, n=4, threshold=0.0) is False

    # threshold=0.25: c<=1 out of 4 is a failure.
    assert is_failure(c=0, n=4, threshold=0.25) is True
    assert is_failure(c=1, n=4, threshold=0.25) is True   # 1/4 == 0.25, <=
    assert is_failure(c=2, n=4, threshold=0.25) is False  # 0.5 > 0.25

    # threshold=1.0: every problem is a failure (degenerate).
    assert is_failure(c=4, n=4, threshold=1.0) is True

    # Argument validation.
    with pytest.raises(ValueError, match="n must be positive"):
        is_failure(c=0, n=0, threshold=0.0)
    with pytest.raises(ValueError, match="threshold must be in"):
        is_failure(c=0, n=4, threshold=1.5)


# -----------------------------------------------------------------------------
# Test 3 — build_failure_rows emits the correct keys, only includes
# failed problems, and carries through prompt / answer / completions.
# -----------------------------------------------------------------------------

def test_build_failure_rows_filters_and_tags_correctly():
    items = [
        {"prompt": "P_solved", "answer": "1"},
        {"prompt": "P_failed", "answer": "2"},
        {"prompt": "P_mixed",  "answer": "3"},
    ]
    completions = [
        ["solved_1", "solved_2", "solved_3", "solved_4"],
        ["bad_1", "bad_2", "bad_3", "bad_4"],
        ["mix_1", "mix_2", "mix_3", "mix_4"],
    ]
    detailed = [
        {"c": 4, "n": 4, "completions": []},   # all correct → not a failure
        {"c": 0, "n": 4, "completions": []},   # zero correct → failure (strict)
        {"c": 2, "n": 4, "completions": []},   # half correct → not a failure
    ]

    rows = build_failure_rows(
        detailed_results=detailed,
        items=items,
        completions_per_item=completions,
        threshold=0.0,
        model_tag="v5",
    )
    assert len(rows) == 1

    row = rows[0]
    assert row["prompt"] == "P_failed"
    assert row["answer"] == "2"
    assert row["v5_pass_at_n"] == 0.0
    assert row["v5_completions"] == ["bad_1", "bad_2", "bad_3", "bad_4"]

    # Threshold=0.5 should also include P_mixed (c/n = 0.5 <= 0.5).
    rows_lenient = build_failure_rows(
        detailed_results=detailed,
        items=items,
        completions_per_item=completions,
        threshold=0.5,
        model_tag="v5",
    )
    assert {r["prompt"] for r in rows_lenient} == {"P_failed", "P_mixed"}
    # Per-problem pass@n preserved as float.
    mixed = next(r for r in rows_lenient if r["prompt"] == "P_mixed")
    assert mixed["v5_pass_at_n"] == 0.5


# -----------------------------------------------------------------------------
# Test 4 — format_failure_summary produces the expected one-liner.
# -----------------------------------------------------------------------------

def test_format_failure_summary():
    s = format_failure_summary(
        n_total=10, n_failures=3, threshold=0.0, n_samples=4,
    )
    assert "n_problems=10" in s
    assert "n_failures=3" in s
    assert "rate=0.300" in s
    assert "threshold=0" in s
    assert "n=4" in s

    # Empty-input edge case: rate should be 0.0 (no div-by-zero).
    s_empty = format_failure_summary(
        n_total=0, n_failures=0, threshold=0.0, n_samples=4,
    )
    assert "rate=0.000" in s_empty


# -----------------------------------------------------------------------------
# Test 5 — failures.jsonl write round-trip preserves the dict shape
# and the keys, and creates the parent directory if needed.
# -----------------------------------------------------------------------------

def test_write_failures_jsonl_roundtrips(tmp_path: Path):
    rows = [
        {
            "prompt": "Find min x s.t. xyz=3.",
            "answer": "(3+\\sqrt{6})^{-1/3}",
            "v5_pass_at_n": 0.0,
            "v5_completions": ["bad attempt 1", "bad attempt 2"],
        },
        {
            "prompt": "Compute 2+2.",
            "answer": "4",
            "v5_pass_at_n": 0.25,
            "v5_completions": ["wrong", "wrong", "\\boxed{4}", "wrong"],
        },
    ]
    target = tmp_path / "nested" / "failures.jsonl"  # parent doesn't exist yet
    write_failures_jsonl(rows, target)

    assert target.is_file()
    reloaded = [json.loads(line) for line in target.read_text().splitlines()]
    assert reloaded == rows


# -----------------------------------------------------------------------------
# Bonus — sampling-params resolution: defaults match the v5 contract,
# CLI overrides take precedence over generation_config.json.
# -----------------------------------------------------------------------------

def test_resolve_sampling_params_prefers_cli_then_gen_config():
    args = argparse.Namespace(
        temperature=0.4, top_p=None, top_k=None,
        n=4, max_new_tokens=4096, seed=42,
    )

    # No generation_config.json → falls back to eval_local's FALLBACK_TOP_P / K.
    params = resolve_sampling_params(args, gen_config_dict=None)
    assert params["temperature"] == 0.4
    assert params["top_p"] == 0.95
    assert params["top_k"] == 20
    assert params["n"] == 4
    assert params["seed"] == 42

    # generation_config.json supplies top_p / top_k.
    gen_cfg = {"temperature": 0.4, "top_p": 0.9, "top_k": 50}
    params = resolve_sampling_params(args, gen_config_dict=gen_cfg)
    assert params["top_p"] == 0.9
    assert params["top_k"] == 50

    # CLI override beats both.
    args.top_p = 0.5
    params = resolve_sampling_params(args, gen_config_dict=gen_cfg)
    assert params["top_p"] == 0.5
