"""CPU-only unit tests for data/prepare_sft.py.

Uses synthetic data only — no Hugging Face dataset downloads. Should run
in well under 1s on a laptop.
"""
from __future__ import annotations

import json
import random

import pytest

from prepare_sft import (
    apply_per_question_cap,
    build_pipeline,
    extract_last_boxed,
    format_response,
    make_example,
    write_jsonl,
)


class TestExtractLastBoxed:
    def test_simple_integer_answer(self):
        result = extract_last_boxed(r"The answer is \boxed{42}.")
        assert result is not None
        before, ans = result
        assert ans == "42"
        assert before == "The answer is "

    def test_nested_frac(self):
        result = extract_last_boxed(r"Therefore \boxed{\frac{1}{2}}")
        assert result is not None
        _, ans = result
        assert ans == r"\frac{1}{2}"

    def test_nested_frac_with_arithmetic(self):
        result = extract_last_boxed(r"\boxed{\frac{a+b}{c+d}}")
        assert result is not None
        _, ans = result
        assert ans == r"\frac{a+b}{c+d}"

    def test_multiple_boxed_uses_last(self):
        text = r"First we got \boxed{x}, then \boxed{y}"
        result = extract_last_boxed(text)
        assert result is not None
        before, ans = result
        assert ans == "y"
        assert before == r"First we got \boxed{x}, then "

    def test_no_boxed_returns_none(self):
        assert extract_last_boxed("the answer has no box at all") is None

    def test_set_notation_with_escaped_braces(self):
        # Set-builder notation inside the box: \{ and \} must NOT close the
        # outer brace.
        result = extract_last_boxed(r"\boxed{\{x : x > 0\}}")
        assert result is not None
        _, ans = result
        assert ans == r"\{x : x > 0\}"

    def test_unbalanced_box_returns_none(self):
        assert extract_last_boxed(r"\boxed{abc def") is None

    def test_whitespace_between_boxed_and_brace(self):
        # `\boxed {x}` is rare but legal LaTeX; the regex tolerates it.
        result = extract_last_boxed(r"\boxed {7}")
        assert result is not None
        _, ans = result
        assert ans == "7"


class TestFormatResponse:
    def test_basic_wrapping(self):
        out = format_response("Step 1.\nStep 2.", "42")
        assert out == "<think>\nStep 1.\nStep 2.\n</think>\n\n\\boxed{42}"

    def test_strips_outer_whitespace_in_reasoning(self):
        out = format_response("   reasoning here   \n", "x")
        assert out == "<think>\nreasoning here\n</think>\n\n\\boxed{x}"


class TestMakeExample:
    def test_chat_message_structure(self):
        ex = make_example("What is 2+2?", "Add them.", "4")
        assert ex == {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": "<think>\nAdd them.\n</think>\n\n\\boxed{4}",
                },
            ]
        }


class TestApplyPerQuestionCap:
    def test_under_cap_keeps_all(self):
        rows = [
            {"query": "Q1", "response": "A1"},
            {"query": "Q1", "response": "A2"},
            {"query": "Q2", "response": "B1"},
        ]
        out = apply_per_question_cap(rows, cap=4, rng=random.Random(0))
        assert len(out) == 3

    def test_caps_a_single_question(self):
        rows = [{"query": "Q1", "response": f"A{i}"} for i in range(10)]
        out = apply_per_question_cap(rows, cap=4, rng=random.Random(0))
        assert len(out) == 4
        assert all(r["query"] == "Q1" for r in out)

    def test_caps_independently_across_questions(self):
        rows = [{"query": "Q1", "response": f"A{i}"} for i in range(10)] + [
            {"query": "Q2", "response": f"B{i}"} for i in range(2)
        ]
        out = apply_per_question_cap(rows, cap=4, rng=random.Random(0))
        assert len(out) == 4 + 2
        assert sum(1 for r in out if r["query"] == "Q1") == 4
        assert sum(1 for r in out if r["query"] == "Q2") == 2

    def test_invalid_cap_raises(self):
        with pytest.raises(ValueError):
            apply_per_question_cap([], cap=0, rng=random.Random(0))


class TestWriteJsonl:
    def test_one_example_per_line_and_valid_json(self, tmp_path):
        examples = [
            {
                "messages": [
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "A1"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Q2"},
                    {"role": "assistant", "content": "A2"},
                ]
            },
        ]
        out_path = tmp_path / "subdir" / "out.jsonl"  # also tests dir creation
        n = write_jsonl(examples, out_path)
        assert n == 2

        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        for line, expected in zip(lines, examples):
            obj = json.loads(line)
            assert obj == expected


class TestBuildPipeline:
    """End-to-end test of the filter → cap → subsample → split flow."""

    def _row(self, q, r):
        return {"query": q, "response": r}

    def test_drops_no_box_rows_and_caps_and_splits(self):
        # Q1 has 6 valid + 1 missing-box; Q2 has 3 valid; Q3 has 1 valid.
        # Per-question cap of 4 should reduce Q1 to 4. Total valid post-cap:
        # 4 + 3 + 1 = 8. Eval size = 2 → train = 6, eval = 2.
        rows = (
            [self._row("Q1", rf"reasoning {i} \boxed{{a{i}}}") for i in range(6)]
            + [self._row("Q1", "no box here")]
            + [self._row("Q2", rf"r{i} \boxed{{b{i}}}") for i in range(3)]
            + [self._row("Q3", r"final \boxed{c}")]
        )
        train, eval_ = build_pipeline(
            rows,
            n_samples=100,
            per_question_cap=4,
            eval_size=2,
            max_response_chars=8000,
            seed=42,
        )
        assert len(train) == 6
        assert len(eval_) == 2
        all_examples = train + eval_
        # Every example is a valid chat dict with the expected structure.
        for ex in all_examples:
            assert list(ex.keys()) == ["messages"]
            assert ex["messages"][0]["role"] == "user"
            assert ex["messages"][1]["role"] == "assistant"
            assistant = ex["messages"][1]["content"]
            assert assistant.startswith("<think>\n")
            assert "</think>\n\n\\boxed{" in assistant
            assert assistant.endswith("}")

    def test_drops_overlong_responses(self):
        rows = [
            self._row("Q1", r"short \boxed{a}"),
            self._row("Q2", "x" * 9000 + r" \boxed{b}"),  # over 8000 chars
        ]
        train, eval_ = build_pipeline(
            rows,
            n_samples=10,
            per_question_cap=4,
            eval_size=0,
            max_response_chars=8000,
            seed=42,
        )
        assert len(train) == 1
        assert len(eval_) == 0
