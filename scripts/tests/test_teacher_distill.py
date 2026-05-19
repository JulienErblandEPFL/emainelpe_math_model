"""CPU-only unit tests for scripts/teacher_distill.

All tests target pure helpers — no torch / vllm imports. The vLLM
generation path is exercised only on a GPU pod; this suite confirms
the SFT-row schema, the two-stage filter (per-problem min-correct +
per-sample correct/well-formed), and the source-tag derivation.
"""
from __future__ import annotations

import json
from pathlib import Path

from data.prepare_sft import make_example
from scripts.teacher_distill import (
    build_distill_row,
    derive_source_tag,
    format_progress_summary,
    should_keep_problem,
)
from scripts.teacher_smoke import is_well_formatted


# -----------------------------------------------------------------------------
# Test 1 — output row schema is SFT-compatible. The "messages" field
# must have exactly the same shape that data/prepare_sft.make_example
# produces, so distillation rows can be concatenated with OMI2/DART
# rows without a schema-conversion pass.
# -----------------------------------------------------------------------------

def test_output_schema_matches_prepare_sft_messages_shape():
    item = {"prompt": "What is 2+2?", "answer": "4"}
    completion = "<think>\n2+2=4\n</think>\n\n\\boxed{4}"

    distill_row = build_distill_row(
        problem_idx=7,
        sample_idx=1,
        item=item,
        teacher_solution=completion,
        pass_at_n=0.5,
        source="teacher_qwen3_32b_awq",
    )

    # Required SFT field.
    assert "messages" in distill_row
    msgs = distill_row["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "What is 2+2?"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == completion.strip()

    # Trace fields (minimal, no extraction blobs).
    assert distill_row["source"] == "teacher_qwen3_32b_awq"
    assert distill_row["problem_idx"] == 7
    assert distill_row["sample_idx"] == 1
    assert distill_row["teacher_pass_at_n"] == 0.5

    # The 'messages' list structure must match what prepare_sft emits
    # (same keys, same role names, same ordering).
    canonical = make_example("What is 2+2?", "2+2=4", "4")
    assert list(distill_row["messages"][0].keys()) == list(canonical["messages"][0].keys())
    assert [m["role"] for m in distill_row["messages"]] == \
           [m["role"] for m in canonical["messages"]]

    # No extraction / scoring metadata leaked into the row.
    forbidden_keys = {"extracted", "correct", "completions", "detailed_results"}
    assert forbidden_keys.isdisjoint(distill_row.keys())


def test_distill_row_jsonl_roundtrips(tmp_path: Path):
    """A built row must serialize and deserialize without loss — the
    output writer just calls json.dumps and stitches newlines."""
    row = build_distill_row(
        problem_idx=0, sample_idx=0,
        item={"prompt": "P", "answer": "A"},
        teacher_solution="<think>r</think>\\boxed{A}",
        pass_at_n=1.0,
        source="teacher_foo",
    )
    path = tmp_path / "distill.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    reloaded = json.loads(path.read_text().splitlines()[0])
    assert reloaded == row


# -----------------------------------------------------------------------------
# Test 2 — per-problem filter: should_keep_problem fires only when the
# teacher hit min_correct or more correct samples.
# -----------------------------------------------------------------------------

def test_should_keep_problem_min_correct():
    # Default min_correct=1: zero correct → drop, >=1 → keep.
    assert should_keep_problem(c=0, min_correct=1) is False
    assert should_keep_problem(c=1, min_correct=1) is True
    assert should_keep_problem(c=4, min_correct=1) is True

    # Strict min_correct=2 (rare-correct dropping): require both samples.
    assert should_keep_problem(c=0, min_correct=2) is False
    assert should_keep_problem(c=1, min_correct=2) is False
    assert should_keep_problem(c=2, min_correct=2) is True

    # min_correct=0 keeps every problem (degenerate but valid).
    assert should_keep_problem(c=0, min_correct=0) is True


# -----------------------------------------------------------------------------
# Test 3 — per-sample filter: a teacher sample is kept only if BOTH
# correct AND well-formatted. The combined filter composition is what
# the main loop applies before emitting an SFT row.
# -----------------------------------------------------------------------------

def test_per_sample_filter_requires_correct_and_well_formatted():
    """Simulate the two per-sample checks the main loop applies."""
    completions = [
        "<think>r</think>\\boxed{4}",              # correct + well-formatted → KEEP
        "no think tag, answer is 4",               # not well-formatted → drop
        "<think>r</think>\\boxed{5}",              # well-formatted but wrong → drop
        "<think>unclosed \\boxed{4}",              # correct but malformed → drop
    ]
    correctness = [True, True, False, True]

    # Re-create the loop's keeper decision exactly.
    keeper_idxs = [
        i for i, (comp, ok) in enumerate(zip(completions, correctness))
        if ok and is_well_formatted(comp)
    ]
    # Only completion 0 satisfies BOTH gates.
    assert keeper_idxs == [0]


# -----------------------------------------------------------------------------
# Test 4 — source-tag derivation: HF model id → snake_case slug.
# -----------------------------------------------------------------------------

def test_derive_source_tag():
    assert derive_source_tag("Qwen/Qwen3-32B-AWQ") == "teacher_qwen3_32b_awq"
    assert derive_source_tag("Qwen/Qwen2.5-Math-72B-Instruct-AWQ") == \
           "teacher_qwen2_5_math_72b_instruct_awq"
    # No org prefix: handled cleanly.
    assert derive_source_tag("local-finetune") == "teacher_local_finetune"


# -----------------------------------------------------------------------------
# Test 5 — progress-summary format: includes processed-count,
# percentage, teacher pass rate, and keeper count.
# -----------------------------------------------------------------------------

def test_format_progress_summary():
    s = format_progress_summary(
        n_processed=50, n_total=200,
        total_attempts=100, total_correct=30,
        total_keepers=42,
    )
    assert "progress 50/200" in s
    assert "25.0%" in s
    assert "teacher_pass_rate=0.300" in s
    assert "(30/100)" in s
    assert "keepers=42" in s

    # Empty-counters edge case: no div-by-zero, no "nan".
    s_zero = format_progress_summary(
        n_processed=0, n_total=200,
        total_attempts=0, total_correct=0,
        total_keepers=0,
    )
    assert "teacher_pass_rate=0.000" in s_zero
    assert "nan" not in s_zero
