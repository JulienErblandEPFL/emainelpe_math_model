"""CPU-only unit tests for scripts/teacher_smoke.

All tests target pure helpers — no torch / vllm imports. The vLLM
load + AWQ generation path is only exercised on a real GPU pod; this
suite confirms the data plumbing around it.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.teacher_smoke import (
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_QUANTIZATION,
    DEFAULT_TEACHER,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    SYSTEM_PROMPT,
    build_chat_messages,
    build_dump_row,
    is_well_formatted,
    summarize,
    write_jsonl,
)


# -----------------------------------------------------------------------------
# Test 1 — prompt formatting: [system, user] messages with the
# Qwen3-thinking-mode system prompt. The prompt does NOT mention
# <think>/</think> — Qwen3's tokenizer chat template enables thinking
# mode natively, so the model emits those tags without instruction.
# -----------------------------------------------------------------------------

def test_build_chat_messages_system_then_user():
    msgs = build_chat_messages("What is 2+2?")

    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert msgs[1] == {"role": "user", "content": "What is 2+2?"}

    # System prompt enforces the \boxed{} terminator the CI scorer
    # extracts; it deliberately does NOT instruct on <think> tags,
    # since Qwen3 emits those automatically in thinking mode.
    assert "\\boxed{}" in SYSTEM_PROMPT
    assert "<think>" not in SYSTEM_PROMPT
    assert "</think>" not in SYSTEM_PROMPT


# -----------------------------------------------------------------------------
# Test 1b — module-level defaults match the Qwen3-32B-AWQ thinking-mode
# contract from the spec. These constants are the single source of
# truth for the CLI defaults; this test guards against accidental
# drift if someone tweaks the module without updating the docstring.
# -----------------------------------------------------------------------------

def test_defaults_match_qwen3_thinking_mode_contract():
    assert DEFAULT_TEACHER == "Qwen/Qwen3-32B-AWQ"
    assert DEFAULT_TEMPERATURE == 0.6
    assert DEFAULT_TOP_P == 0.95
    assert DEFAULT_TOP_K == 20
    assert DEFAULT_PRESENCE_PENALTY == 1.5
    assert DEFAULT_QUANTIZATION == "awq"


# -----------------------------------------------------------------------------
# Test 2 — well-formed detection: BOTH <think>...</think> AND \boxed{...}
# must be present. Multiple edge cases.
# -----------------------------------------------------------------------------

def test_is_well_formatted_requires_both_patterns():
    # Both present, multiline reasoning.
    good_multiline = (
        "<think>\n step 1\n step 2: simplify\n</think>\n"
        "The answer is \\boxed{42}."
    )
    assert is_well_formatted(good_multiline) is True

    # Both inline on one line.
    assert is_well_formatted("<think>quick</think> \\boxed{7}") is True

    # Missing closing </think> → not well-formatted.
    assert is_well_formatted("<think>step 1 \\boxed{4}") is False

    # Missing \boxed{} → not well-formatted.
    assert is_well_formatted("<think>reasoning</think> answer=4") is False

    # \boxed{} only (raw answer, no think block) → not well-formatted.
    assert is_well_formatted("\\boxed{4}") is False

    # Empty string → not well-formatted.
    assert is_well_formatted("") is False

    # Empty <think></think> still counts (regex matches non-greedy zero).
    # This is intentional — the smoke test asks "does the teacher use
    # the format we want?", not "did it write useful tokens inside?".
    assert is_well_formatted("<think></think>\\boxed{x}") is True


# -----------------------------------------------------------------------------
# Test 3 — generation post-processing: build_dump_row derives
# n_well_formatted from the completions and preserves the schema the
# spec requires.
# -----------------------------------------------------------------------------

def test_build_dump_row_counts_and_schema():
    item = {"prompt": "P", "answer": "4"}
    completions = [
        "<think>r</think>\\boxed{4}",     # well-formatted, correct
        "no think tag, answer is 4",      # not well-formatted
        "<think>r</think>\\boxed{5}",     # well-formatted, wrong
        "<think>open only \\boxed{4}",    # missing </think>, not well-formed
    ]
    row = build_dump_row(item, completions, n_correct=1)

    # Schema is exactly what the spec asks for.
    assert set(row.keys()) == {
        "prompt", "answer", "teacher_completions",
        "n_correct", "n_well_formatted",
    }
    assert row["prompt"] == "P"
    assert row["answer"] == "4"
    assert row["teacher_completions"] == completions
    assert row["n_correct"] == 1
    # 2 of 4 completions have BOTH a closed <think> pair AND a \boxed{.
    assert row["n_well_formatted"] == 2

    # build_dump_row must not mutate its inputs.
    completions.append("mutation check")
    row2 = build_dump_row(item, ["x"], n_correct=0)
    assert row2["teacher_completions"] == ["x"]


# -----------------------------------------------------------------------------
# Test 4 — summarize: pass and format rates over the full N×n grid,
# with the empty-input case guarded against div-by-zero.
# -----------------------------------------------------------------------------

def test_summarize_computes_rates():
    rows = [
        {
            "prompt": "p1", "answer": "1",
            "teacher_completions": ["x", "y"],
            "n_correct": 2, "n_well_formatted": 2,
        },
        {
            "prompt": "p2", "answer": "2",
            "teacher_completions": ["x", "y"],
            "n_correct": 0, "n_well_formatted": 1,
        },
    ]
    s = summarize(rows, n_samples=2)
    # 4 total generations; 2 correct → 0.500; 3 well-formatted → 0.750.
    assert "n_problems=2" in s
    assert "n_generations=4" in s
    assert "pass_rate=0.500" in s
    assert "format_rate=0.750" in s

    # Empty rows: rates collapse to 0.0 cleanly (no div-by-zero).
    s_empty = summarize([], n_samples=2)
    assert "n_problems=0" in s_empty
    assert "pass_rate=0.000" in s_empty
    assert "format_rate=0.000" in s_empty


# -----------------------------------------------------------------------------
# Test 5 — JSONL output round-trip; parent directory is created.
# -----------------------------------------------------------------------------

def test_write_jsonl_roundtrips(tmp_path: Path):
    rows = [
        {
            "prompt": "Find min x s.t. xyz=3.",
            "answer": "(3+\\sqrt{6})^{-1/3}",
            "teacher_completions": [
                "<think>solving...</think>\\boxed{(3+\\sqrt{6})^{-1/3}}",
                "no think, answer omitted",
            ],
            "n_correct": 1,
            "n_well_formatted": 1,
        },
    ]
    target = tmp_path / "nested" / "teacher_smoke.jsonl"  # parent missing
    write_jsonl(rows, target)

    assert target.is_file()
    reloaded = [json.loads(line) for line in target.read_text().splitlines()]
    assert reloaded == rows
