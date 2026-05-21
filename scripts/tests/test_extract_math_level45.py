"""CPU-only unit tests for scripts/extract_math_level45.

All tests target pure helpers — no ``datasets`` import. The lazy
runtime import lives inside ``main()`` and is not exercised here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.extract_math_level45 import (
    DEFAULT_SUBJECTS,
    build_problem_row,
    extract_last_boxed,
    format_summary,
    keep_row,
    normalize_level,
    parse_level_filter,
    write_problems_jsonl,
)


# -----------------------------------------------------------------------------
# Test 1 — \boxed{} extraction. Covers simple, nested, last-of-many,
# missing-box, and malformed (unclosed brace) inputs.
# -----------------------------------------------------------------------------

def test_extract_last_boxed_simple():
    assert extract_last_boxed(r"The answer is \boxed{42}.") == "42"
    assert extract_last_boxed(r"\boxed{x+1}") == "x+1"


def test_extract_last_boxed_nested_braces():
    # Regex \boxed\{(.+?)\} would stop at the first '}' and yield
    # '\frac{1' here; the brace-balanced extractor returns the full
    # nested payload.
    assert extract_last_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_last_boxed(r"\boxed{\sqrt{3}+\sqrt{2}}") == r"\sqrt{3}+\sqrt{2}"


def test_extract_last_boxed_picks_last():
    # MATH solutions sometimes have intermediate \boxed{} markers; the
    # extractor must return the LAST one (the gold answer).
    text = r"First try: \boxed{wrong}. After fixing: \boxed{right}."
    assert extract_last_boxed(text) == "right"


def test_extract_last_boxed_missing_returns_none():
    assert extract_last_boxed("No box in this solution.") is None
    assert extract_last_boxed("") is None
    assert extract_last_boxed(None) is None  # type: ignore[arg-type]


def test_extract_last_boxed_unclosed_returns_none():
    # Unterminated brace — depth never returns to zero.
    assert extract_last_boxed(r"\boxed{42") is None
    assert extract_last_boxed(r"\boxed{\frac{1}{2}") is None


# -----------------------------------------------------------------------------
# Test 2 — Level normalization. Accepts int, "5", "Level 5"; rejects
# "Level ?" and empty.
# -----------------------------------------------------------------------------

def test_normalize_level_accepts_known_forms():
    assert normalize_level(5) == "Level 5"
    assert normalize_level("5") == "Level 5"
    assert normalize_level("Level 5") == "Level 5"
    assert normalize_level("  Level 4  ") == "Level 4"


def test_normalize_level_rejects_unknown_forms():
    assert normalize_level(None) is None
    assert normalize_level("") is None
    assert normalize_level("Level ?") is None
    assert normalize_level("?") is None
    assert normalize_level("hard") is None


# -----------------------------------------------------------------------------
# Test 3 — Levels filter parsing.
# -----------------------------------------------------------------------------

def test_parse_level_filter_default_and_variants():
    assert parse_level_filter("4,5") == ("Level 4", "Level 5")
    assert parse_level_filter("5") == ("Level 5",)
    assert parse_level_filter("1, 2, 3") == ("Level 1", "Level 2", "Level 3")
    # "Level 5" tokens are also acceptable for forgiving CLI use.
    assert parse_level_filter("Level 4, Level 5") == ("Level 4", "Level 5")


def test_parse_level_filter_rejects_bad_input():
    with pytest.raises(ValueError, match="not a valid MATH level"):
        parse_level_filter("4,?")
    with pytest.raises(ValueError, match="empty filter"):
        parse_level_filter(",,")


# -----------------------------------------------------------------------------
# Test 4 — keep_row filter: levels AND subjects.
# -----------------------------------------------------------------------------

def test_keep_row_levels_filter():
    levels = ("Level 4", "Level 5")
    assert keep_row(
        {"problem": "...", "level": "Level 5", "type": "Algebra"},
        levels_filter=levels, subjects_filter=None,
    ) is True
    assert keep_row(
        {"problem": "...", "level": "Level 3", "type": "Algebra"},
        levels_filter=levels, subjects_filter=None,
    ) is False
    assert keep_row(
        {"problem": "...", "level": "Level ?", "type": "Algebra"},
        levels_filter=levels, subjects_filter=None,
    ) is False


def test_keep_row_subjects_filter():
    levels = ("Level 5",)
    # 'Intermediate Algebra' matches the 'intermediate_algebra' slug
    # via the normalization in keep_row.
    assert keep_row(
        {"level": "Level 5", "type": "Intermediate Algebra"},
        levels_filter=levels,
        subjects_filter=("intermediate_algebra",),
    ) is True
    # 'Algebra' should NOT match when the filter is intermediate_algebra
    # only.
    assert keep_row(
        {"level": "Level 5", "type": "Algebra"},
        levels_filter=levels,
        subjects_filter=("intermediate_algebra",),
    ) is False
    # 'Counting & Probability' → counting_and_probability slug.
    assert keep_row(
        {"level": "Level 5", "type": "Counting & Probability"},
        levels_filter=levels,
        subjects_filter=("counting_and_probability",),
    ) is True


def test_keep_row_prefers_subject_over_type():
    # Spec says "the dataset uses both, prefer 'subject'". When both
    # are present, 'subject' wins.
    levels = ("Level 5",)
    row = {"level": "Level 5", "subject": "Algebra", "type": "Geometry"}
    assert keep_row(
        row, levels_filter=levels, subjects_filter=("algebra",),
    ) is True
    assert keep_row(
        row, levels_filter=levels, subjects_filter=("geometry",),
    ) is False


# -----------------------------------------------------------------------------
# Test 5 — build_problem_row: happy path, extraction failure, "Level ?",
# missing fields.
# -----------------------------------------------------------------------------

def test_build_problem_row_happy_path():
    raw = {
        "problem": "What is 2+2?",
        "solution": r"Adding gives \boxed{4}.",
        "type": "Algebra",
        "level": "Level 5",
    }
    row = build_problem_row(raw)
    assert row == {
        "prompt": "What is 2+2?",
        "answer": "4",
        "subject": "Algebra",
        "level": "Level 5",
    }


def test_build_problem_row_nested_box():
    raw = {
        "problem": "Simplify.",
        "solution": r"We get \boxed{\frac{1}{2}}.",
        "type": "Algebra",
        "level": "Level 5",
    }
    row = build_problem_row(raw)
    assert row is not None
    assert row["answer"] == r"\frac{1}{2}"


def test_build_problem_row_no_box_returns_none(caplog):
    raw = {
        "problem": "What is 2+2?",
        "solution": "The answer is four.",
        "type": "Algebra",
        "level": "Level 5",
    }
    with caplog.at_level("WARNING", logger="extract_math_level45"):
        assert build_problem_row(raw) is None
    assert any("no \\boxed" in rec.message for rec in caplog.records)


def test_build_problem_row_level_question_mark_returns_none():
    raw = {
        "problem": "...",
        "solution": r"\boxed{0}",
        "type": "Algebra",
        "level": "Level ?",
    }
    assert build_problem_row(raw) is None


def test_build_problem_row_missing_fields_returns_none():
    # Empty problem.
    assert build_problem_row({
        "problem": "", "solution": r"\boxed{0}",
        "type": "Algebra", "level": "Level 5",
    }) is None
    # Missing subject + type.
    assert build_problem_row({
        "problem": "x", "solution": r"\boxed{0}", "level": "Level 5",
    }) is None
    # Empty solution.
    assert build_problem_row({
        "problem": "x", "solution": "",
        "type": "Algebra", "level": "Level 5",
    }) is None


# -----------------------------------------------------------------------------
# Test 6 — JSONL write/read round-trip.
# -----------------------------------------------------------------------------

def test_write_problems_jsonl_roundtrip(tmp_path: Path):
    rows = [
        {"prompt": "p1", "answer": "1", "subject": "Algebra", "level": "Level 5"},
        {"prompt": "p2", "answer": r"\frac{1}{2}",
         "subject": "Counting & Probability", "level": "Level 4"},
    ]
    path = tmp_path / "nested" / "problems.jsonl"
    write_problems_jsonl(rows, path)
    assert path.exists()

    parsed = [json.loads(line) for line in path.read_text().splitlines()]
    assert parsed == rows


# -----------------------------------------------------------------------------
# Test 7 — Summary formatter shows per-subject and per-level breakdowns
# and the total.
# -----------------------------------------------------------------------------

def test_format_summary_counts():
    rows = [
        {"subject": "Algebra", "level": "Level 5",
         "prompt": "...", "answer": "1"},
        {"subject": "Algebra", "level": "Level 4",
         "prompt": "...", "answer": "2"},
        {"subject": "Precalculus", "level": "Level 5",
         "prompt": "...", "answer": "3"},
    ]
    out = format_summary(rows)
    assert "total=3" in out
    assert "'Level 4': 1" in out
    assert "'Level 5': 2" in out
    assert "'Algebra': 2" in out
    assert "'Precalculus': 1" in out


# -----------------------------------------------------------------------------
# Test 8 — DEFAULT_SUBJECTS locks the 7-subject contract shared with
# data/prepare_sft.MATH_TRAIN_SUBJECTS. Drifting either side without
# the other would break v4-mix and this script's defaults.
# -----------------------------------------------------------------------------

def test_default_subjects_match_prepare_sft():
    from data.prepare_sft import MATH_TRAIN_SUBJECTS
    assert DEFAULT_SUBJECTS == MATH_TRAIN_SUBJECTS
