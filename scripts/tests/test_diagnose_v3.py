"""Tests for ``scripts/diagnose_v3.py`` — v3 SFT failure-mode diagnostic.

CPU-only. The module under test gates ``vllm``/``transformers``/``datasets``
imports inside runtime helpers (``_build_llm``, ``_self_check_chat_template``,
``_hf_load_math_test``), so importing the module itself is laptop-safe.

Coverage map:

- ``detect_repetition`` — sliding-window counter; positive and negative cases
- ``classify_failure_mode`` — priority order (6 labels) + the two specific
  bug-shape edge cases called out in the spec
- Loaders — ``load_validation_problems``, ``load_indist_problems``,
  ``normalize_math_test_row`` (both HF-500 and competition_math schemas)
- ``aggregate_per_problem`` and ``aggregate_target_summary`` (including the
  zero-correct edge and the per-subject/per-level branches)
- CLI parsing — ``--target`` / ``--limit`` defaults + ``--force``
- ``MATH_SUBJECTS`` invariant: exactly the 7 canonical names
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.diagnose_v3 import (
    ALL_FAILURE_MODES,
    ALL_TARGETS,
    FM_CORRECT,
    FM_NO_BOX,
    FM_OTHER,
    FM_REPETITION,
    FM_TRUNCATED,
    FM_WRONG_BOX,
    MATH_LEVELS,
    MATH_SUBJECTS,
    PER_PROBLEM_FM_KEYS,
    TRUNCATION_TOKEN_THRESHOLD,
    _assert_math_test_schema,
    aggregate_per_problem,
    aggregate_target_summary,
    classify_failure_mode,
    detect_repetition,
    format_full_summary,
    load_indist_problems,
    load_validation_problems,
    normalize_math_test_row,
    _parse_args,
    _resolve_targets,
)


# =============================================================================
# detect_repetition
# =============================================================================

def test_detect_repetition_false_on_short_text():
    assert detect_repetition("abc") is False


def test_detect_repetition_false_on_diverse_text():
    text = "The quick brown fox jumps over the lazy dog. " * 2 + "Some unique tail."
    # 100-char windows repeat only twice → below the 3-count threshold.
    assert detect_repetition(text) is False


def test_detect_repetition_true_on_obvious_loop():
    chunk = "x" * 100
    text = chunk * 3 + " trailing"
    assert detect_repetition(text) is True


def test_detect_repetition_true_on_periodic_text():
    """A 100-char chunk repeated 3+ times must trip detection (the spec's
    canonical loop-detection signal). Period equals window width so every
    sliding window sees the chunk 3 times exactly."""
    chunk = "abcdefghij" * 10  # exactly 100 chars
    text = chunk * 3
    assert detect_repetition(text) is True


# =============================================================================
# classify_failure_mode — priority order + spec'd edge cases
# =============================================================================

_GOLD = "42"
_LONG_PREAMBLE = "x" * (TRUNCATION_TOKEN_THRESHOLD + 100)  # ensures n_tokens passes


def test_classify_correct_when_box_matches():
    text = "Some reasoning. \\boxed{42}"
    label, extracted = classify_failure_mode(text, _GOLD, completion_token_len=200)
    assert label == FM_CORRECT
    assert extracted == "42"


def test_classify_repetition_beats_correct_per_priority_order():
    """Priority order: repetition (1) > correct (2). A completion that loops
    AND happens to box the correct answer is still classified as
    ``repetition``.

    Rationale: this diagnostic is meant to surface broken generation
    behavior so v4 data design can target it. A model that loops out of
    control and accidentally lands on the right \\boxed{} value is still
    broken — counting it as ``correct`` would mask the underlying
    generation pathology in the failure-mode breakdown. The pass@k metrics
    are reported alongside (and they still credit the correct answer if
    that's what we eventually want to grade on), but the failure-mode view
    treats the loop as the dominant signal.
    """
    chunk = "abcdefghij" * 10  # 100 chars
    text = chunk * 3 + " \\boxed{42}"
    assert detect_repetition(text) is True  # sanity
    label, _ = classify_failure_mode(text, _GOLD, completion_token_len=200)
    assert label == FM_REPETITION


def test_classify_repetition_beats_wrong_box():
    """Spec test: completion with a \\boxed{} inside a repeating loop is
    classified as repetition, NOT wrong_box."""
    sentence = "abcdefghij" * 10  # 100 chars
    text = sentence * 3 + " \\boxed{99}"  # wrong answer, but looping
    label, _ = classify_failure_mode(text, _GOLD, completion_token_len=200)
    assert label == FM_REPETITION


def test_classify_repetition_beats_truncated():
    """Spec test: a truncated repeating completion is classified as
    repetition (priority over truncated)."""
    sentence = "abcdefghij" * 10  # 100 chars
    text = sentence * 5  # repeats > 3 times, no box
    label, _ = classify_failure_mode(
        text, _GOLD, completion_token_len=TRUNCATION_TOKEN_THRESHOLD + 10,
    )
    assert label == FM_REPETITION


def test_classify_no_box_when_completion_has_no_box():
    text = "The answer is forty-two but I forgot to box it"
    label, extracted = classify_failure_mode(text, _GOLD, completion_token_len=50)
    assert label == FM_NO_BOX
    assert extracted is None


def _unique_filler(n_words: int) -> str:
    """Build a long text with no 100-char window repeating — used by tests
    that need long completions WITHOUT tripping the repetition detector."""
    return " ".join(f"word{i:04d}" for i in range(n_words))


def test_classify_truncated_when_long_and_no_box_near_end():
    """Has a box early, then long unique reasoning cut off without a closing
    box → truncated. The filler is unique-by-word so repetition does NOT fire."""
    body_filler = _unique_filler(1200)  # ~9.6k chars, no repeating windows
    text = "\\boxed{nope}" + body_filler
    assert len(text[-800:]) == 800
    assert "\\boxed" not in text[-800:]
    assert detect_repetition(text) is False  # sanity: not a loop
    label, _ = classify_failure_mode(
        text, _GOLD, completion_token_len=TRUNCATION_TOKEN_THRESHOLD + 5,
    )
    assert label == FM_TRUNCATED


def test_classify_wrong_box_when_short_and_wrong():
    text = "I think the answer is forty-one. \\boxed{41}"
    label, extracted = classify_failure_mode(text, _GOLD, completion_token_len=80)
    assert label == FM_WRONG_BOX
    assert extracted == "41"


def test_classify_does_not_truncate_when_box_in_tail():
    """Box near the end + long unique-content completion → NOT truncated
    (wrong_box instead). Body is unique-by-word so repetition does NOT fire."""
    body_filler = _unique_filler(1200)
    text = body_filler + " \\boxed{41}"
    assert detect_repetition(text) is False  # sanity
    label, _ = classify_failure_mode(
        text, _GOLD, completion_token_len=TRUNCATION_TOKEN_THRESHOLD + 5,
    )
    assert label == FM_WRONG_BOX


# =============================================================================
# Loaders — validation, indist, math_test
# =============================================================================

def test_load_validation_problems_parses_schema(tmp_path):
    f = tmp_path / "math.jsonl"
    f.write_text(
        json.dumps({"prompt": "What is 2+2?", "answer": "4"}) + "\n"
        + json.dumps({"prompt": "What is 3+3?", "answer": "6"}) + "\n",
        encoding="utf-8",
    )
    rows = load_validation_problems(f)
    assert len(rows) == 2
    assert rows[0] == {
        "problem_id": "validation_0",
        "problem": "What is 2+2?",
        "gold_answer": "4",
        "subject": None,
        "level": None,
    }
    assert rows[1]["problem_id"] == "validation_1"


def test_load_validation_problems_skips_blank_lines(tmp_path):
    f = tmp_path / "math.jsonl"
    f.write_text(
        json.dumps({"prompt": "Q1", "answer": "1"}) + "\n"
        + "\n"
        + json.dumps({"prompt": "Q2", "answer": "2"}) + "\n",
        encoding="utf-8",
    )
    rows = load_validation_problems(f)
    assert [r["problem_id"] for r in rows] == ["validation_0", "validation_1"]


def test_load_indist_problems_extracts_messages_schema(tmp_path):
    f = tmp_path / "eval.jsonl"
    f.write_text(
        json.dumps({
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "<think>obvious</think>\n\\boxed{4}"},
            ],
        }) + "\n",
        encoding="utf-8",
    )
    rows = load_indist_problems(f)
    assert len(rows) == 1
    assert rows[0]["problem"] == "What is 2+2?"
    assert rows[0]["gold_answer"] == "4"
    assert rows[0]["problem_id"] == "indist_0"
    assert rows[0]["subject"] is None and rows[0]["level"] is None


def test_load_indist_problems_drops_rows_without_box(tmp_path):
    f = tmp_path / "eval.jsonl"
    f.write_text(
        json.dumps({
            "messages": [
                {"role": "user", "content": "good"},
                {"role": "assistant", "content": "<think>ok</think>\n\\boxed{1}"},
            ],
        }) + "\n"
        + json.dumps({
            "messages": [
                {"role": "user", "content": "bad"},
                {"role": "assistant", "content": "I forgot to box my answer"},
            ],
        }) + "\n",
        encoding="utf-8",
    )
    rows = load_indist_problems(f)
    assert len(rows) == 1
    assert rows[0]["problem"] == "good"


def test_normalize_math_test_row_huggingface_h4_schema():
    """HuggingFaceH4/MATH-500: {problem, solution, answer, subject, level:int}."""
    raw = {
        "problem": "Compute 2+2.",
        "solution": "Adding gives \\boxed{4}",
        "answer": "4",
        "subject": "Algebra",
        "level": 1,
    }
    norm = normalize_math_test_row(raw, idx=7)
    assert norm == {
        "problem_id": "math_test_7",
        "problem": "Compute 2+2.",
        "gold_answer": "4",
        "subject": "Algebra",
        "level": "Level 1",
    }


def test_normalize_math_test_row_competition_math_schema():
    """hendrycks/competition_math: {problem, solution, type, level: 'Level X'}."""
    raw = {
        "problem": "Compute 2+2.",
        "solution": "Trivially \\boxed{4}.",
        "type": "Prealgebra",
        "level": "Level 2",
    }
    norm = normalize_math_test_row(raw, idx=0)
    assert norm == {
        "problem_id": "math_test_0",
        "problem": "Compute 2+2.",
        "gold_answer": "4",
        "subject": "Prealgebra",
        "level": "Level 2",
    }


def test_normalize_math_test_row_handles_subject_variant():
    """Some forks use 'Counting and Probability' instead of '& Probability'."""
    raw = {
        "problem": "p", "solution": "\\boxed{1}",
        "type": "Counting and Probability", "level": 3,
    }
    norm = normalize_math_test_row(raw, idx=0)
    assert norm is not None
    assert norm["subject"] == "Counting & Probability"


def test_normalize_math_test_row_returns_none_when_no_gold():
    raw = {"problem": "p", "solution": "no box here", "level": 1, "type": "Algebra"}
    assert normalize_math_test_row(raw, idx=0) is None


def test_normalize_math_test_row_returns_none_when_no_problem():
    raw = {"problem": "", "answer": "5", "level": 1, "type": "Algebra"}
    assert normalize_math_test_row(raw, idx=0) is None


def test_math_test_schema_assertion():
    """Fail-fast schema check for the HF MATH-test loader.

    Positive case: a well-formed row (HF MATH-500 or competition_math
    shape) passes silently.

    Negative case: a row missing a required field group raises
    ``RuntimeError`` with a clear, actionable message — naming the
    dataset path, the expected fields, the found fields, and the missing
    groups. Catches the bug class where a HF fallback path returns a
    different schema and the loader silently produces garbage.
    """
    # Positive: HF MATH-500 shape (carries both `answer` and `solution`).
    _assert_math_test_schema("HuggingFaceH4/MATH-500", {
        "problem": "Compute 2+2.",
        "solution": "It's \\boxed{4}.",
        "answer": "4",
        "subject": "Algebra",
        "level": 1,
    })
    # Positive: competition_math shape (uses `type` instead of `subject`).
    _assert_math_test_schema("hendrycks/competition_math", {
        "problem": "Compute 2+2.",
        "solution": "\\boxed{4}",
        "type": "Prealgebra",
        "level": "Level 1",
    })

    # Negative: missing subject/type group.
    bad_row = {"problem": "Compute 2+2.", "solution": "\\boxed{4}"}
    with pytest.raises(RuntimeError) as excinfo:
        _assert_math_test_schema("fake/unknown-MATH-fork", bad_row)
    msg = str(excinfo.value)
    assert "fake/unknown-MATH-fork" in msg
    assert "subject (one of)" in msg
    assert "Expected fields" in msg
    assert "Found fields" in msg
    assert "Missing" in msg

    # Negative: missing both `solution` and `answer` (no gold).
    with pytest.raises(RuntimeError, match="gold .one of."):
        _assert_math_test_schema(
            "fake/no-gold",
            {"problem": "p", "subject": "Algebra", "level": 1},
        )


def test_math_subjects_are_exactly_the_seven_canonical_names():
    """Spec requirement: subjects are exactly the 7 canonical names."""
    assert MATH_SUBJECTS == (
        "Algebra",
        "Counting & Probability",
        "Geometry",
        "Intermediate Algebra",
        "Number Theory",
        "Prealgebra",
        "Precalculus",
    )
    assert len(MATH_SUBJECTS) == 7


def test_math_levels_cover_1_through_5():
    assert MATH_LEVELS == ("Level 1", "Level 2", "Level 3", "Level 4", "Level 5")


# =============================================================================
# Aggregation
# =============================================================================

def _make_completion_row(problem_id: str, idx: int, fm: str, correct: bool) -> dict:
    return {
        "problem_id": problem_id,
        "completion_idx": idx,
        "completion_text": "<dummy>",
        "extracted_answer": "42" if correct else "0",
        "is_correct": correct,
        "failure_mode": fm,
    }


def test_aggregate_per_problem_counts_correct_and_failure_modes():
    rows = [
        _make_completion_row("p", 0, FM_CORRECT, True),
        _make_completion_row("p", 1, FM_CORRECT, True),
        _make_completion_row("p", 2, FM_WRONG_BOX, False),
        _make_completion_row("p", 3, FM_NO_BOX, False),
    ]
    out = aggregate_per_problem(
        problem_id="p", target="validation",
        subject=None, level=None,
        problem="Q?", gold_answer="42",
        per_completion_rows=rows,
    )
    assert out["n_completions"] == 4
    assert out["n_correct"] == 2
    assert out["solve_rate"] == 0.5
    assert out["failure_modes"] == {
        "no_box": 1, "wrong_box": 1, "truncated": 0, "repetition": 0, "other": 0,
    }


def test_aggregate_per_problem_zero_completions_safe():
    out = aggregate_per_problem(
        problem_id="p", target="validation",
        subject=None, level=None,
        problem="Q?", gold_answer="42",
        per_completion_rows=[],
    )
    assert out["n_completions"] == 0
    assert out["solve_rate"] == 0.0


def test_aggregate_target_summary_zero_correct_edge():
    """Edge: a target with zero correct completions still emits valid metrics."""
    per_problem = [
        {
            "problem_id": "validation_0",
            "target": "validation",
            "subject": None, "level": None,
            "problem": "Q", "gold_answer": "1",
            "n_completions": 8, "n_correct": 0,
            "solve_rate": 0.0,
            "failure_modes": {k: 0 for k in PER_PROBLEM_FM_KEYS},
        },
    ]
    per_completion = [
        _make_completion_row("validation_0", i, FM_WRONG_BOX, False)
        for i in range(8)
    ]
    s = aggregate_target_summary("validation", per_problem, per_completion, n_completions=8)
    assert s["metrics"]["pass@1"] == 0.0
    assert s["metrics"]["pass@8"] == 0.0
    assert s["failure_mode_distribution"]["wrong_box"] == 8
    # Every label keyed (zero if absent).
    for k in ALL_FAILURE_MODES:
        assert k in s["failure_mode_distribution"]


def test_aggregate_target_summary_pass_at_k_uses_n_completions():
    """All correct → pass@1 = pass@n = 1.0."""
    rows = [
        {
            "problem_id": f"p{i}", "target": "indist",
            "subject": None, "level": None,
            "problem": "Q", "gold_answer": "1",
            "n_completions": 4, "n_correct": 4, "solve_rate": 1.0,
            "failure_modes": {k: 0 for k in PER_PROBLEM_FM_KEYS},
        }
        for i in range(3)
    ]
    comps = [
        _make_completion_row(f"p{i}", c, FM_CORRECT, True)
        for i in range(3) for c in range(4)
    ]
    s = aggregate_target_summary("indist", rows, comps, n_completions=4)
    assert s["metrics"]["pass@1"] == pytest.approx(1.0)
    assert s["metrics"]["pass@4"] == pytest.approx(1.0)
    # Not the math_test target: no per_subject / per_level.
    assert "per_subject" not in s
    assert "per_level" not in s


def test_aggregate_target_summary_math_test_per_subject_and_level():
    """math_test target emits per_subject + per_level breakdowns."""
    rows = [
        {
            "problem_id": "math_test_0", "target": "math_test",
            "subject": "Algebra", "level": "Level 1",
            "problem": "Q", "gold_answer": "1",
            "n_completions": 4, "n_correct": 4, "solve_rate": 1.0,
            "failure_modes": {k: 0 for k in PER_PROBLEM_FM_KEYS},
        },
        {
            "problem_id": "math_test_1", "target": "math_test",
            "subject": "Geometry", "level": "Level 5",
            "problem": "Q", "gold_answer": "1",
            "n_completions": 4, "n_correct": 0, "solve_rate": 0.0,
            "failure_modes": {k: 0 for k in PER_PROBLEM_FM_KEYS} | {"no_box": 4},
        },
    ]
    comps = [
        _make_completion_row("math_test_0", i, FM_CORRECT, True) for i in range(4)
    ] + [
        _make_completion_row("math_test_1", i, FM_NO_BOX, False) for i in range(4)
    ]
    s = aggregate_target_summary("math_test", rows, comps, n_completions=4)
    assert "per_subject" in s
    assert s["per_subject"]["Algebra"]["pass@1"] == pytest.approx(1.0)
    assert s["per_subject"]["Algebra"]["n_problems"] == 1
    assert s["per_subject"]["Geometry"]["pass@1"] == pytest.approx(0.0)
    assert s["per_subject"]["Geometry"]["failure_modes"]["no_box"] == 4
    assert "per_level" in s
    assert s["per_level"]["Level 1"]["pass@1"] == pytest.approx(1.0)
    assert s["per_level"]["Level 5"]["pass@1"] == pytest.approx(0.0)


def test_format_full_summary_contains_section_headers():
    """Smoke test that the formatted block has the spec headers."""
    summaries = {
        "validation": {
            "target": "validation", "n_problems": 1, "n_completions": 8,
            "metrics": {"pass@1": 0.25, "pass@8": 0.5},
            "failure_mode_distribution": {k: 0 for k in ALL_FAILURE_MODES},
        },
    }
    per_problem = {"validation": [
        {"problem_id": "validation_0", "n_correct": 0, "n_completions": 8},
    ]}
    out = format_full_summary(summaries, per_problem)
    assert "=== v3 DIAGNOSTIC SUMMARY ===" in out
    assert "Validation (N=1, n=8):" in out
    assert "pass@1: 0.250" in out
    assert "pass@8: 0.500" in out


# =============================================================================
# CLI
# =============================================================================

def test_parse_args_defaults_to_all_targets():
    args = _parse_args(["--model", "x"])
    assert args.target == "all"
    assert args.limit is None
    assert args.force is False


def test_parse_args_rejects_invalid_target():
    with pytest.raises(SystemExit):
        _parse_args(["--model", "x", "--target", "bogus"])


def test_parse_args_accepts_each_valid_target():
    for t in (*ALL_TARGETS, "all"):
        args = _parse_args(["--model", "x", "--target", t])
        assert args.target == t


def test_parse_args_limit_parses_int():
    args = _parse_args(["--model", "x", "--limit", "5"])
    assert args.limit == 5


def test_parse_args_force_flag():
    args = _parse_args(["--model", "x", "--force"])
    assert args.force is True


def test_resolve_targets_expands_all():
    assert _resolve_targets("all") == list(ALL_TARGETS)


def test_resolve_targets_returns_singleton_for_specific():
    assert _resolve_targets("validation") == ["validation"]
    assert _resolve_targets("math_test") == ["math_test"]
