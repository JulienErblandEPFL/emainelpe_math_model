"""CPU-only unit tests for data/prepare_sft.py.

Uses synthetic data only — no Hugging Face dataset downloads. Should run
in well under 1s on a laptop.
"""
from __future__ import annotations

import json
import random

import pytest

from prepare_sft import (
    V4_MAX_FORMATTED_TOKENS_DEFAULT,
    apply_per_question_cap,
    build_pipeline,
    compose_math_train_buckets,
    dedup_by_problem_text,
    extract_last_boxed,
    format_response,
    make_example,
    normalize_math_train_row,
    normalize_numinamath_row,
    normalize_openmathinstruct_row,
    normalize_problem_text,
    oversample_with_replacement,
    resolve_n_samples,
    strip_trailing_preamble,
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
            min_reasoning_chars=0,
            max_answer_chars=10_000,
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
            min_reasoning_chars=0,
            max_answer_chars=10_000,
            seed=42,
        )
        assert len(train) == 1
        assert len(eval_) == 0


class TestStripTrailingPreamble:
    """Trailing-fragment cleanup applied to text_before_boxed.

    The strip is conservative: it only removes orphan math-mode delimiters
    and a small set of literal answer-preamble phrases anchored to the END
    of the text. Patterns mid-text are never touched.
    """

    def test_strips_trailing_dollar(self):
        assert strip_trailing_preamble("reasoning $") == "reasoning"

    def test_strips_trailing_double_dollar(self):
        assert strip_trailing_preamble("reasoning $$") == "reasoning"

    def test_strips_trailing_latex_display_open(self):
        # Literal text in DART responses is backslash + '['. Use a raw
        # string so the test asserts on 2 chars (\[), not a Python escape.
        assert strip_trailing_preamble(r"reasoning \[") == "reasoning"

    def test_strips_trailing_latex_inline_open(self):
        assert strip_trailing_preamble(r"reasoning \(") == "reasoning"

    def test_strips_the_answer_is_colon(self):
        assert strip_trailing_preamble("step 1. The answer is:") == "step 1."

    def test_strips_the_answer_is_no_colon(self):
        assert strip_trailing_preamble("step 1. The answer is") == "step 1."

    def test_strips_final_answer_colon(self):
        assert strip_trailing_preamble("computed. Final answer:") == "computed."

    def test_strips_answer_colon(self):
        assert strip_trailing_preamble("done. Answer:") == "done."

    def test_iterative_strip_compound(self):
        # The layered case: rstrip + '$' + rstrip + 'The answer is:' + rstrip.
        assert strip_trailing_preamble("reasoning. The answer is: $") == "reasoning."

    def test_case_insensitive_phrase(self):
        assert strip_trailing_preamble("step 1. THE ANSWER IS:") == "step 1."

    def test_does_not_strip_mid_reasoning(self):
        # Phrase appears mid-text, not at end. Leave it alone.
        text = "the answer is X. Now we verify."
        assert strip_trailing_preamble(text) == text

    def test_does_not_strip_therefore(self):
        # 'Therefore' is intentionally NOT in our pattern set. It can be a
        # legitimate end-of-reasoning conclusion.
        assert strip_trailing_preamble("...therefore") == "...therefore"

    def test_returns_empty_when_only_preamble(self):
        # Strip cascade can fully consume the text. Downstream
        # min_reasoning_chars filter handles the empty-reasoning case.
        assert strip_trailing_preamble("The answer is: $") == ""

    def test_real_dart_pattern(self):
        # Regression test: the exact pattern from the RCP diagnostic. A
        # real DART row has prose before "The answer is", and that prose
        # must survive the strip while the trailing fragment is removed.
        text_before, answer = extract_last_boxed(
            r"Step 1: 1 + 1 = 2. Step 2: subtract 2. The answer is: $\boxed{0}$"
        )
        assert text_before == "Step 1: 1 + 1 = 2. Step 2: subtract 2. The answer is: $"
        assert answer == "0"
        cleaned = strip_trailing_preamble(text_before)
        assert cleaned == "Step 1: 1 + 1 = 2. Step 2: subtract 2."


class TestBuildPipelinePurity:
    """Two purity filters layered on top of the existing \\boxed{} filter:

    - min_reasoning_chars: drops rows whose CLEANED reasoning is too short.
    - max_answer_chars: drops rows whose boxed answer is too long.

    Pipeline ordering: extract_last_boxed -> strip_trailing_preamble ->
    min_reasoning_chars filter. The strip happens BEFORE the length check
    so e.g. "reasoning $" gets cleaned to "reasoning" before being measured.
    """

    def _row(self, q, r):
        return {"query": q, "response": r}

    def test_min_reasoning_chars_drops_short(self):
        # Reasoning is ~12 chars; threshold 150 drops it.
        rows = [self._row("Q1", r"a tiny step \boxed{42}")]
        train, eval_ = build_pipeline(
            rows,
            n_samples=100,
            per_question_cap=4,
            eval_size=0,
            max_response_chars=8000,
            min_reasoning_chars=150,
            max_answer_chars=200,
            seed=42,
        )
        assert len(train) == 0

    def test_min_reasoning_chars_keeps_at_lower_threshold(self):
        # Same row, threshold 10 keeps it.
        rows = [self._row("Q1", r"a tiny step \boxed{42}")]
        train, eval_ = build_pipeline(
            rows,
            n_samples=100,
            per_question_cap=4,
            eval_size=0,
            max_response_chars=8000,
            min_reasoning_chars=10,
            max_answer_chars=200,
            seed=42,
        )
        assert len(train) == 1

    def test_max_answer_chars_drops_long_answer(self):
        long_answer = "x" * 300
        rows = [
            self._row(
                "Q1",
                "valid reasoning here " * 10 + r"\boxed{" + long_answer + r"}",
            )
        ]
        train, eval_ = build_pipeline(
            rows,
            n_samples=100,
            per_question_cap=4,
            eval_size=0,
            max_response_chars=8000,
            min_reasoning_chars=10,
            max_answer_chars=200,
            seed=42,
        )
        assert len(train) == 0

    def test_strip_runs_in_pipeline(self):
        # Real DART-style row: prose with "The answer is: $\boxed{42}$" suffix.
        # The strip should remove the orphan "$" and "The answer is:" so the
        # think block is clean.
        long_prose = "Step 1: do thing. Step 2: do other thing. " * 3
        rows = [self._row("Q1", long_prose + r"The answer is: $\boxed{42}$")]
        train, eval_ = build_pipeline(
            rows,
            n_samples=100,
            per_question_cap=4,
            eval_size=0,
            max_response_chars=8000,
            min_reasoning_chars=50,
            max_answer_chars=200,
            seed=42,
        )
        assert len(train) == 1
        assistant = train[0]["messages"][1]["content"]
        think_block = assistant.split("</think>")[0]
        assert "$" not in think_block
        assert "The answer is" not in think_block
        assert assistant.endswith(r"\boxed{42}")


class TestNormalizeOpenMathInstructRow:
    """v2 D2: append \\boxed{expected_answer} so OMI2 rows pass through the
    existing DART pipeline unchanged. Last-box-wins guarantees the appended
    answer is what extract_last_boxed recovers."""

    def _row(self, problem, solution, answer, source="math"):
        return {
            "problem": problem,
            "generated_solution": solution,
            "expected_answer": answer,
            "problem_source": source,
        }

    def test_basic_row_normalizes_to_dart_shape(self):
        out = normalize_openmathinstruct_row(
            self._row("What is 2+2?", "Add them together.", "4")
        )
        assert set(out.keys()) == {"query", "response"}
        assert out["query"] == "What is 2+2?"
        assert out["response"] == "Add them together.\n\\boxed{4}"

    def test_appended_box_is_extractable_as_gold(self):
        out = normalize_openmathinstruct_row(
            self._row("Compute pi", "Some reasoning here.", r"\frac{22}{7}")
        )
        result = extract_last_boxed(out["response"])
        assert result is not None
        before, ans = result
        assert ans == r"\frac{22}{7}"
        assert before == "Some reasoning here.\n"

    def test_solution_with_internal_box_keeps_appended_as_last(self):
        out = normalize_openmathinstruct_row(
            self._row(
                "Compute",
                r"First I tried \boxed{wrong_intermediate}, then corrected.",
                "5",
            )
        )
        before, ans = extract_last_boxed(out["response"])
        assert ans == "5"
        assert "wrong_intermediate" in before

    def test_strips_trailing_whitespace_in_solution(self):
        out = normalize_openmathinstruct_row(
            self._row("Q", "reasoning   \n\n  ", "7")
        )
        assert out["response"] == "reasoning\n\\boxed{7}"

    def test_handles_non_string_expected_answer(self):
        out = normalize_openmathinstruct_row(
            self._row("Q", "reasoning", 42)
        )
        assert out["response"].endswith(r"\boxed{42}")


class TestBuildPipelineWithOmi2Rows:
    """Integration check for D1+D2: feed OMI2-normalized rows into the
    existing build_pipeline. The 'reuse, don't duplicate' invariant lives
    here — if any of these break, the synthesize-fake-DART-row strategy
    has stopped working and we'd need to refactor."""

    def _omi2(self, problem, solution, answer):
        return normalize_openmathinstruct_row({
            "problem": problem,
            "generated_solution": solution,
            "expected_answer": answer,
            "problem_source": "math",
        })

    def test_omi2_rows_produce_canonical_chat_format(self):
        rows = [
            self._omi2(
                f"Problem {i}",
                "Step 1: do something. Step 2: arrive at the answer. " * 4,
                str(i),
            )
            for i in range(5)
        ]
        train, _ = build_pipeline(
            rows,
            n_samples=10,
            per_question_cap=4,
            eval_size=0,
            max_response_chars=8000,
            min_reasoning_chars=50,
            max_answer_chars=200,
            seed=42,
        )
        assert len(train) == 5
        for ex in train:
            assistant = ex["messages"][1]["content"]
            assert assistant.startswith("<think>\n")
            assert "</think>\n\n\\boxed{" in assistant
            assert assistant.endswith("}")

    def test_omi2_per_question_cap_applies_via_query(self):
        """D3: 'unique problem' for OMI2 = unique 'problem' string. After
        normalization that's the 'query' field, so apply_per_question_cap
        works without any modification — 10 augmented solutions for the
        same problem with cap=4 reduces to 4."""
        rows = [
            self._omi2(
                "Same augmented problem",
                f"reasoning attempt {i}. " * 8,
                str(i),
            )
            for i in range(10)
        ]
        train, _ = build_pipeline(
            rows,
            n_samples=100,
            per_question_cap=4,
            eval_size=0,
            max_response_chars=8000,
            min_reasoning_chars=50,
            max_answer_chars=200,
            seed=42,
        )
        assert len(train) == 4


class TestTokenFilterInBuildPipeline:
    """v2 D4: token-length cap on the formatted message sequence. The filter
    is opt-in via two kwargs; when either is None it's a no-op (preserves
    v1 byte-stable behavior).

    Tests inject a fake tokenize_fn so the heavy ``transformers`` import
    stays in main() only and laptop tests run in <1s.
    """

    def _row(self, q, r):
        return {"query": q, "response": r}

    def _make_fake_token_counter(self, mapping):
        def _count(messages):
            content = messages[1]["content"]
            for prefix, n in mapping.items():
                if prefix in content:
                    return n
            return 1
        return _count

    def test_token_filter_drops_overlong_rows(self):
        rows = [
            self._row("Q1", r"reasoning A " * 30 + r"\boxed{a}"),
            self._row("Q2", r"reasoning B " * 30 + r"\boxed{b}"),
        ]
        tokenize_fn = self._make_fake_token_counter({
            "\\boxed{a}": 1000,
            "\\boxed{b}": 5000,
        })
        train, _ = build_pipeline(
            rows,
            n_samples=100,
            per_question_cap=4,
            eval_size=0,
            max_response_chars=8000,
            min_reasoning_chars=10,
            max_answer_chars=200,
            seed=42,
            max_formatted_tokens=3500,
            tokenize_fn=tokenize_fn,
        )
        assert len(train) == 1
        assistant = train[0]["messages"][1]["content"]
        assert assistant.endswith(r"\boxed{a}")

    def test_token_filter_disabled_when_kwargs_none(self):
        """Backward-compat: existing tests don't pass either of the new
        kwargs, and behavior must be identical to v1."""
        rows = [self._row("Q1", r"reasoning here. " * 30 + r"\boxed{a}")]
        train, _ = build_pipeline(
            rows,
            n_samples=100, per_question_cap=4, eval_size=0,
            max_response_chars=8000, min_reasoning_chars=10,
            max_answer_chars=200, seed=42,
        )
        assert len(train) == 1

    def test_token_filter_only_one_arg_set_is_no_op(self):
        """Defensive: passing only max_formatted_tokens (without
        tokenize_fn) or vice versa must NOT silently apply a degenerate
        filter. Both must be set for the filter to fire."""
        rows = [self._row("Q1", r"reasoning here. " * 30 + r"\boxed{a}")]

        train_a, _ = build_pipeline(
            rows, n_samples=100, per_question_cap=4, eval_size=0,
            max_response_chars=8000, min_reasoning_chars=10,
            max_answer_chars=200, seed=42,
            max_formatted_tokens=10,
        )
        assert len(train_a) == 1

        tokenize_fn = self._make_fake_token_counter({"a": 999_999})
        train_b, _ = build_pipeline(
            rows, n_samples=100, per_question_cap=4, eval_size=0,
            max_response_chars=8000, min_reasoning_chars=10,
            max_answer_chars=200, seed=42,
            tokenize_fn=tokenize_fn,
        )
        assert len(train_b) == 1

    def test_token_filter_at_cap_passes(self):
        """Boundary: rows with token count exactly == max_formatted_tokens
        pass the filter (predicate is ``<=``)."""
        rows = [self._row("Q1", r"reasoning here. " * 30 + r"\boxed{a}")]
        tokenize_fn = self._make_fake_token_counter({"\\boxed{a}": 3500})
        train, _ = build_pipeline(
            rows, n_samples=100, per_question_cap=4, eval_size=0,
            max_response_chars=8000, min_reasoning_chars=10,
            max_answer_chars=200, seed=42,
            max_formatted_tokens=3500, tokenize_fn=tokenize_fn,
        )
        assert len(train) == 1

    def test_token_filter_runs_before_per_question_cap(self):
        """Pipeline ordering invariant. Token filter runs BEFORE
        apply_per_question_cap so the cap operates on already-shrunk rows.
        5 same-query rows where 3 are oversized → token filter leaves 2 →
        cap of 4 keeps both."""
        rows = [
            self._row("Q1", f"reasoning {i}. " * 30 + r"\boxed{" + str(i) + r"}")
            for i in range(5)
        ]
        tokenize_fn = self._make_fake_token_counter({
            "\\boxed{0}": 100,
            "\\boxed{1}": 100,
            "\\boxed{2}": 9999,
            "\\boxed{3}": 9999,
            "\\boxed{4}": 9999,
        })
        train, _ = build_pipeline(
            rows, n_samples=100, per_question_cap=4, eval_size=0,
            max_response_chars=8000, min_reasoning_chars=10,
            max_answer_chars=200, seed=42,
            max_formatted_tokens=3500, tokenize_fn=tokenize_fn,
        )
        assert len(train) == 2


class TestResolveNSamples:
    """CLI semantics: --n-samples and --train-size are mutually exclusive
    flags with different meanings:
      --n-samples X    → X total post-filter rows, BEFORE the train/eval split
      --train-size X   → X rows in train.jsonl (translates to
                         n_samples = X + eval_size internally)
    """

    def test_neither_set_uses_default(self):
        assert resolve_n_samples(
            n_samples_arg=None, train_size_arg=None, eval_size=500,
        ) == 50_000

    def test_n_samples_only_passes_through(self):
        assert resolve_n_samples(
            n_samples_arg=50_000, train_size_arg=None, eval_size=500,
        ) == 50_000

    def test_train_size_translates_to_n_samples_plus_eval(self):
        assert resolve_n_samples(
            n_samples_arg=None, train_size_arg=50_000, eval_size=500,
        ) == 50_500

    def test_both_set_raises_value_error(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            resolve_n_samples(
                n_samples_arg=50_000, train_size_arg=50_000, eval_size=500,
            )

    def test_train_size_with_zero_eval_split_works(self):
        """Edge case: --eval-size 0 is legitimate when the operator wants
        a single train file with no held-out slice."""
        assert resolve_n_samples(
            n_samples_arg=None, train_size_arg=1000, eval_size=0,
        ) == 1000


class TestMixedSourceShuffle:
    """v2 D1: the mixed mode runs both pipelines, concatenates, shuffles
    with a fixed seed, then writes. We test the shuffle invariant by
    passing distinguishable synthetic content per source.

    The full mixing flow lives in main() (which calls build_pipeline twice
    + concatenate + shuffle); to test it without main()'s dataset-loading
    branch, we model the same logic directly with synthetic rows.
    """

    def _dart_rows(self, n):
        return [
            {"query": f"DART_Q{i}", "response": f"DART reasoning {i}. "
             r"More reasoning here. Final step. \boxed{" + str(i) + r"}"}
            for i in range(n)
        ]

    def _omi2_rows(self, n):
        return [
            normalize_openmathinstruct_row({
                "problem": f"OMI2_P{i}",
                "generated_solution": f"OMI2 reasoning {i}. "
                "More reasoning here. Final step.",
                "expected_answer": str(i),
                "problem_source": "math",
            })
            for i in range(n)
        ]

    def _shared_kwargs(self):
        return dict(
            per_question_cap=4,
            max_response_chars=8000,
            min_reasoning_chars=20,
            max_answer_chars=200,
            seed=42,
        )

    def test_mixed_output_contains_both_sources_in_proportion(self):
        """50/50 mix at n_samples=10 each → exactly 10 from each source."""
        dart_train, _ = build_pipeline(
            self._dart_rows(40), n_samples=10, eval_size=0,
            **self._shared_kwargs(),
        )
        omi_train, _ = build_pipeline(
            self._omi2_rows(40), n_samples=10, eval_size=0,
            **self._shared_kwargs(),
        )
        merged = dart_train + omi_train
        rng = random.Random(42)
        rng.shuffle(merged)

        n_dart = sum(
            1 for ex in merged if ex["messages"][0]["content"].startswith("DART_")
        )
        n_omi = sum(
            1 for ex in merged if ex["messages"][0]["content"].startswith("OMI2_")
        )
        assert n_dart == 10
        assert n_omi == 10

    def test_mixed_shuffle_interleaves_sources(self):
        """After shuffle, sources should be interleaved rather than block-
        concatenated. With seed=42 and 20 rows, the first 5 must contain
        BOTH source labels — not all-DART-then-all-OMI2 ordering.

        Operational invariant: TRL doesn't reshuffle epochs by default,
        so a block-concatenated train.jsonl would let the model see all
        DART before any OMI2 in epoch 1, biasing early gradients."""
        dart_train, _ = build_pipeline(
            self._dart_rows(40), n_samples=10, eval_size=0,
            **self._shared_kwargs(),
        )
        omi_train, _ = build_pipeline(
            self._omi2_rows(40), n_samples=10, eval_size=0,
            **self._shared_kwargs(),
        )
        merged = dart_train + omi_train
        rng = random.Random(42)
        rng.shuffle(merged)

        first_five_prefixes = {
            ex["messages"][0]["content"].split("_")[0] for ex in merged[:5]
        }
        assert "DART" in first_five_prefixes
        assert "OMI2" in first_five_prefixes

    def test_mixed_shuffle_is_deterministic_across_runs(self):
        """Pin the seed-determinism contract: identical inputs + same seed
        = byte-identical output. Required for v2 reproducibility."""
        dart_train, _ = build_pipeline(
            self._dart_rows(40), n_samples=10, eval_size=0,
            **self._shared_kwargs(),
        )
        omi_train, _ = build_pipeline(
            self._omi2_rows(40), n_samples=10, eval_size=0,
            **self._shared_kwargs(),
        )

        merged_a = dart_train + omi_train
        random.Random(42).shuffle(merged_a)

        merged_b = dart_train + omi_train
        random.Random(42).shuffle(merged_b)

        assert merged_a == merged_b


# =============================================================================
# v3 (2026-05-09): pure --source openmathinstruct path. The four individual
# properties (D2 normalization, D3 cap, D4 token filter, v1-compatible chat
# schema) are each covered by tests above; this single integration test
# exercises them TOGETHER on the same synthetic input, mirroring the v3
# invocation from the operator's three-way SFT ablation plan.
# =============================================================================


class TestOpenMathInstructEndToEnd:
    """End-to-end smoke for ``--source openmathinstruct`` (v3 pure-OMI2 path).

    Models main()'s openmathinstruct branch without invoking main() itself —
    that branch's only main()-specific behavior is loading the dataset (which
    we replace with synthetic rows) and resolving the auto-default token cap
    (which we apply explicitly here). The auto-default value 3500 is pinned
    in ``data/prepare_sft.py:448``; this test hardcodes the same number with
    a comment so a future change to the constant trips the test rather than
    silently diverging."""

    # Auto-default for --source openmathinstruct from prepare_sft.py:448.
    # Must stay in sync with that constant.
    OMI2_AUTO_TOKEN_CAP = 3500

    def _omi2_raw(self, problem, solution, answer, source="math"):
        """Build a synthetic raw OMI2 row in the schema HF actually returns."""
        return {
            "problem": problem,
            "generated_solution": solution,
            "expected_answer": answer,
            "problem_source": source,
        }

    def _make_token_counter(self, long_marker):
        """Returns a tokenize_fn that returns a high token count for any
        message whose assistant content contains long_marker, else a low one.
        Lets the test simulate the 3500-token cap firing on specific rows
        without actually loading the Qwen3 tokenizer."""
        def _count(messages):
            content = messages[1]["content"]
            return 9999 if long_marker in content else 100
        return _count

    def test_v3_pure_omi2_pipeline(self):
        """Single integration test covering all four v3 properties:

        1. Token filter (3500) auto-default applies — long row dropped.
        2. Per-problem cap (4) applies — duplicated problem reduced to 4.
        3. Output schema is v1-compatible — {messages: [user, assistant]}
           with the canonical <think>...</think>\\boxed{} format.
        4. No DART data leaks — every output row's user content matches
           the OMI2-only synthetic input.

        Reasoning chunks chosen so all rows clear min_reasoning_chars=20
        and max_response_chars=8000.
        """
        # 5 distinct OMI2 problems (clean rows that should pass).
        clean_rows = [
            self._omi2_raw(
                f"OMI2_problem_{i}",
                f"Step 1: think. Step 2: verify. Step 3: solve. iteration {i}.",
                str(i),
            )
            for i in range(5)
        ]
        # 5 augmented solutions for the SAME problem (cap should drop to 4).
        duplicate_rows = [
            self._omi2_raw(
                "OMI2_problem_DUP",
                f"Augmented attempt {i}: chain of thought goes here. Done.",
                str(100 + i),
            )
            for i in range(5)
        ]
        # 1 row with very long content marked TOO_LONG (token filter drops).
        long_row = self._omi2_raw(
            "OMI2_problem_LONG",
            "Long verbose solution. TOO_LONG_MARKER. Step 1. Step 2. Step 3.",
            "999",
        )

        raw = clean_rows + duplicate_rows + [long_row]
        normalized = [normalize_openmathinstruct_row(r) for r in raw]

        tokenize_fn = self._make_token_counter("TOO_LONG_MARKER")

        # Mirrors what main() does for --source openmathinstruct: pass-through
        # default per_question_cap=4, the auto-default 3500-token filter, and
        # the standard min_reasoning_chars / max_answer_chars knobs.
        train, eval_ = build_pipeline(
            normalized,
            n_samples=100,
            per_question_cap=4,
            eval_size=0,
            max_response_chars=8000,
            min_reasoning_chars=20,
            max_answer_chars=200,
            seed=42,
            max_formatted_tokens=self.OMI2_AUTO_TOKEN_CAP,
            tokenize_fn=tokenize_fn,
        )

        # Property 1: token filter dropped the long row. Net: 5 clean + 4
        # capped duplicates = 9 rows survive.
        assert len(train) == 9, (
            f"expected 9 rows after token filter + cap, got {len(train)}"
        )

        # Property 2: per-problem cap reduced the 5 duplicates to 4.
        n_dup = sum(
            1 for ex in train
            if ex["messages"][0]["content"] == "OMI2_problem_DUP"
        )
        assert n_dup == 4, f"per-problem cap broken: {n_dup} duplicates kept"

        # Property 3: every row matches v1's chat schema. Same assertions as
        # the existing TestBuildPipeline.test_drops_no_box_rows_and_caps_and_splits
        # check, applied to v3 output.
        for ex in train:
            assert list(ex.keys()) == ["messages"]
            assert ex["messages"][0]["role"] == "user"
            assert ex["messages"][1]["role"] == "assistant"
            assistant = ex["messages"][1]["content"]
            assert assistant.startswith("<think>\n")
            assert "</think>\n\n\\boxed{" in assistant
            assert assistant.endswith("}")

        # Property 4: no DART data leak — every user prompt comes from the
        # synthetic OMI2 input (queries all start with 'OMI2_problem_').
        for ex in train:
            user_content = ex["messages"][0]["content"]
            assert user_content.startswith("OMI2_problem_"), (
                f"unexpected non-OMI2 prompt leaked into v3 output: "
                f"{user_content!r}"
            )
        # The long-row prompt MUST NOT survive (token filter dropped it).
        long_prompts = [
            ex for ex in train
            if ex["messages"][0]["content"] == "OMI2_problem_LONG"
        ]
        assert long_prompts == [], (
            "token filter failed to drop the marked-long row: "
            f"{len(long_prompts)} survived"
        )


# =============================================================================
# v4-mix tests (added 2026-05-13). Cover the pure helpers and CLI-level
# auto-default for --max-formatted-tokens. Heavy paths (the
# build_pipeline + load_dataset + normalize chain in main()) are exercised
# by the existing OMI2 / mixed tests above; here we focus on what's NEW.
# =============================================================================

def _v4_math_row(problem: str, gold: str, subject: str, level: int | str) -> dict:
    """Mock one Hendrycks MATH-train row (the shape normalize_math_train_row
    consumes). The solution carries an explicit \\boxed{gold} so the
    normalizer's extract path is exercised."""
    return {
        "problem": problem,
        "solution": (
            f"This is the working for {problem}. Step by step. "
            r"Therefore the answer is \boxed{" + gold + r"}."
        ),
        "type": subject,
        "level": level,
    }


def test_v4_mix_composition_respects_counts():
    """compose_math_train_buckets samples each bucket to its requested
    count. With a small synthetic dataset the four bucket totals must
    equal the sum of requested counts BEFORE dedup runs."""
    rows = []
    # 3 IntAlg problems, 2 Precalc, 2 Algebra L5, 2 Algebra L2.
    for i in range(3):
        rows.append({"query": f"IA_{i}", "response": "r", "subject": "Intermediate Algebra", "level": "Level 3"})
    for i in range(2):
        rows.append({"query": f"PC_{i}", "response": "r", "subject": "Precalculus", "level": "Level 4"})
    for i in range(2):
        rows.append({"query": f"AL_L5_{i}", "response": "r", "subject": "Algebra", "level": "Level 5"})
    for i in range(2):
        rows.append({"query": f"AL_L2_{i}", "response": "r", "subject": "Algebra", "level": "Level 2"})

    rng = random.Random(42)
    out = compose_math_train_buckets(
        rows=rows,
        intermediate_algebra_count=10,
        precalculus_count=8,
        level45_count=6,
        level13_count=5,
        rng=rng,
    )
    # Sum of bucket targets — pre-dedup, this is exact.
    assert len(out) == 10 + 8 + 6 + 5


def test_v4_mix_oversampling_handles_small_source():
    """When the source pool is smaller than the bucket target,
    oversample_with_replacement samples with replacement (some problems
    appear multiple times) and does NOT raise. This is the IntAlg/Precalc
    case in real data (~1.3k / 750 unique problems vs targets of 12k / 7k)."""
    small_pool = [
        {"query": f"only_{i}", "response": "r"}
        for i in range(3)
    ]
    rng = random.Random(0)
    out = oversample_with_replacement(small_pool, target_count=30, rng=rng)
    assert len(out) == 30
    # With 3 unique problems sampled 30 times, every output row's query is
    # one of the originals.
    assert all(r["query"].startswith("only_") for r in out)
    # At least one duplicate exists (otherwise oversample didn't replicate).
    queries = [r["query"] for r in out]
    assert len(set(queries)) < len(queries)


def test_v4_mix_oversampling_empty_pool_returns_empty():
    """Robustness: oversample on an empty pool must not crash."""
    assert oversample_with_replacement([], target_count=10, rng=random.Random(0)) == []


def test_v4_mix_deduplication():
    """Cross-source dedup (2026-05-13 update): when the same problem
    appears in OMI2 AND MATH-train, the final combined mix has it
    EXACTLY ONCE, with the first-occurrence source winning.

    The concat order in main() is OMI2 → MATH → NuminaMath, so OMI2's
    Llama3.1-405B teacher CoT takes precedence over MATH-train's plain
    Hendrycks solution for any overlapping problem. This is the
    quality-ordering choice baked into the v4-mix flow.
    """
    # Simulate a problem that appears in BOTH OMI2 and MATH-train,
    # with cosmetically different whitespace/LaTeX (dedup normalizes
    # them to the same key).
    omi2_rows = [
        {"query": "Find the value of x.", "response": "OMI2 rich CoT"},
        {"query": "Solve 2+2.", "response": "OMI2 only"},
    ]
    math_rows = [
        {"query": "FIND THE VALUE OF X.", "response": "MATH plain solution"},
        {"query": "Compute $\\pi$.", "response": "MATH only"},
    ]
    # The order matters: OMI2 first.
    combined = omi2_rows + math_rows
    deduped = dedup_by_problem_text(combined)
    # 4 inputs → 3 unique problems after cross-source dedup.
    assert len(deduped) == 3
    # OMI2 won the contested "Find the value of x" problem.
    queries_to_responses = {r["query"]: r["response"] for r in deduped}
    assert queries_to_responses.get("Find the value of x.") == "OMI2 rich CoT"
    # OMI2-only and MATH-only problems both survive.
    assert "OMI2 only" in queries_to_responses.values()
    assert "MATH only" in queries_to_responses.values()


def test_v4_mix_within_bucket_oversampling_preserved():
    """A single bucket with target=10 from a pool of 2 unique problems
    produces 10 rows (with duplicates). This tests the BUCKET COMPOSITION
    stage — within-bucket oversampling is preserved at compose output,
    BEFORE any dedup runs.

    The downstream dedup at cross-source concat will collapse these
    duplicates, but inside the bucket the oversample distribution
    survives. This is the input the diagnostic-driven multipliers
    operate on.
    """
    pool = [
        {"query": "P_0", "response": "r0"},
        {"query": "P_1", "response": "r1"},
    ]
    rng = random.Random(42)
    out = oversample_with_replacement(pool, target_count=10, rng=rng)
    # Exactly 10 rows — oversampling hit the target count.
    assert len(out) == 10
    # Both source problems contribute at least once.
    queries = [r["query"] for r in out]
    assert "P_0" in queries
    assert "P_1" in queries
    # Duplicates exist (oversample with replacement from a 2-pool).
    assert len(set(queries)) < len(queries)


def test_v4_mix_within_bucket_oversampling_then_cross_bucket_dedup():
    """Full flow: bucket A oversamples problem X 5 times, bucket B
    contains X once. After cross-bucket dedup_by_problem_text on the
    concat, the final mix has X EXACTLY ONCE, with bucket A's first
    occurrence winning (B's copy is dropped because it comes later in
    concat order).

    This pins the interaction between within-bucket oversampling
    (preserved at compose) and the strict cross-source/cross-bucket
    dedup (collapses everything to unique-problem-text).
    """
    # Bucket A: 5 oversampled copies of X plus one Y.
    bucket_a = [
        {"query": "X", "response": "from_A_1"},
        {"query": "X", "response": "from_A_2"},
        {"query": "X", "response": "from_A_3"},
        {"query": "X", "response": "from_A_4"},
        {"query": "X", "response": "from_A_5"},
        {"query": "Y", "response": "from_A_Y"},
    ]
    # Bucket B: another X plus a unique Z.
    bucket_b = [
        {"query": "X", "response": "from_B"},
        {"query": "Z", "response": "from_B_Z"},
    ]
    combined = bucket_a + bucket_b
    deduped = dedup_by_problem_text(combined)
    # 8 inputs (6 + 2) → 3 unique problems.
    assert len(deduped) == 3
    # X appears exactly once, kept from bucket A's first occurrence.
    x_rows = [r for r in deduped if r["query"] == "X"]
    assert len(x_rows) == 1
    assert x_rows[0]["response"] == "from_A_1"
    # B's copy of X is dropped.
    assert all(r["response"] != "from_B" for r in deduped)
    # Y (A-only) and Z (B-only) both survive.
    queries = {r["query"] for r in deduped}
    assert {"X", "Y", "Z"} == queries


def test_v4_mix_auto_max_formatted_tokens(monkeypatch, tmp_path, capsys):
    """When --source v4-mix is selected, max_formatted_tokens auto-
    defaults to V4_MAX_FORMATTED_TOKENS_DEFAULT (2900), NOT the v2/v3
    default of 3500. Operators must explicitly override with
    --max-formatted-tokens to disable the OOM safety cap.

    We exercise the resolution logic indirectly by reading the constant
    AND asserting the v2/v3 default differs (3500 → 2900 is the
    documented tightening for v4).
    """
    # The constant is what main() uses for the auto-default.
    assert V4_MAX_FORMATTED_TOKENS_DEFAULT == 2900
    # And it's tighter than the v2/v3 default of 3500 (locked check —
    # if someone widens this without thinking, OOM-fix regression alarm).
    assert V4_MAX_FORMATTED_TOKENS_DEFAULT < 3500


def test_v4_mix_boxed_answer_appending():
    """normalize_math_train_row extracts the gold from the solution via
    the team evaluate/ module's answer extraction, then APPENDS
    \\boxed{gold} to the response. This guarantees extract_last_boxed
    (used later in build_pipeline) finds the gold deterministically,
    even when the source solution had a malformed or differently-placed
    boxed expression.
    """
    raw = _v4_math_row(
        problem="What is 7+5?",
        gold="12",
        subject="Prealgebra",
        level=1,
    )
    norm = normalize_math_train_row(raw)
    assert norm is not None
    # Subject + level propagate.
    assert norm["subject"] == "Prealgebra"
    assert norm["level"] == "Level 1"
    # The response ends with an appended boxed answer.
    assert norm["response"].rstrip().endswith(r"\boxed{12}")
    # And the query is the original problem.
    assert norm["query"] == "What is 7+5?"


def test_v4_mix_normalize_math_train_handles_separate_answer_field():
    """Some MATH forks (HF MATH-500 variants) carry the answer in a
    separate 'answer' field instead of a \\boxed{} in the solution. The
    normalizer must accept either shape — falling back to the answer
    field when the solution has no extractable box."""
    raw = {
        "problem": "Compute 2+2.",
        "solution": "The result is straightforward — 4.",
        "answer": "4",
        "type": "Algebra",
        "level": 1,
    }
    norm = normalize_math_train_row(raw)
    assert norm is not None
    assert norm["response"].rstrip().endswith(r"\boxed{4}")


def test_v4_mix_normalize_math_train_handles_subject_variant():
    """'Counting and Probability' (some forks) canonicalizes to
    'Counting & Probability' to match diagnose_v3.MATH_SUBJECTS."""
    raw = _v4_math_row(
        problem="p", gold="1",
        subject="Counting and Probability", level=2,
    )
    norm = normalize_math_train_row(raw)
    assert norm is not None
    assert norm["subject"] == "Counting & Probability"


def test_v4_mix_normalize_numinamath_filters_via_callsite():
    """normalize_numinamath_row drops rows with no extractable gold.
    The olympiad-source filter is applied at the call site in main()
    (pre-filter), not inside the normalizer — verified here by feeding
    a syntactically valid row and asserting it normalizes.
    """
    raw = {
        "problem": "Olympiad problem.",
        "solution": r"The answer is \boxed{1729}.",
        "source": "olympiads",
    }
    norm = normalize_numinamath_row(raw)
    assert norm is not None
    assert norm["query"] == "Olympiad problem."
    assert norm["response"].rstrip().endswith(r"\boxed{1729}")

    # A row without a parseable gold gets dropped.
    bad = {"problem": "p", "solution": "no box here", "source": "olympiads"}
    assert normalize_numinamath_row(bad) is None
