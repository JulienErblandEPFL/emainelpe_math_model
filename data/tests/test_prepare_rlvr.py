"""Tests for ``data/prepare_rlvr.py`` — RLVR prompt-set curation.

CPU-only. The heavy ML imports (``vllm``, ``transformers``) live inside
``main()`` and are not exercised here. We test:

  - difficulty_filter   (decision D3b, 2026-05-09): empty, all-easy,
                        all-hard, in-band, exact-edge, invalid band.
  - extract_prompt_and_gold (Stage-1 ``messages`` → (prompt, gold)).
  - validate_pool_row   schema rejection of malformed rows.
  - load_pool_jsonl     end-to-end JSONL load + max_rows cap.
  - write_jsonl         round-trip on a tmp file.
  - solve_rate          arithmetic + zero-rollout guard.
  - build_scored_row    (2026-05-12 fix): output rows MUST contain a
                        chat-templated prompt — defense against the
                        degenerate-rollout incident.
  - CLI parsing         required defaults, mutually-consistent flags.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prepare_rlvr import (
    CHAT_TEMPLATE_OPEN_MARKER,
    DIFFICULTY_HI,
    DIFFICULTY_LO,
    THINK_PREFIX,
    _parse_args,
    build_scored_row,
    difficulty_filter,
    extract_prompt_and_gold,
    load_pool_jsonl,
    solve_rate,
    validate_pool_row,
    write_jsonl,
)


# =============================================================================
# difficulty_filter — the proposal's [0.2, 0.8] band, edge cases.
# =============================================================================

def test_difficulty_filter_empty_input_returns_empty():
    assert difficulty_filter([]) == []


def test_difficulty_filter_all_too_easy_drops_everything():
    rows = [{"solve_rate": r} for r in (0.9, 0.95, 1.0)]
    assert difficulty_filter(rows) == []


def test_difficulty_filter_all_too_hard_drops_everything():
    rows = [{"solve_rate": r} for r in (0.0, 0.05, 0.1)]
    assert difficulty_filter(rows) == []


def test_difficulty_filter_keeps_in_band():
    rows = [
        {"solve_rate": 0.0,  "id": "too_hard"},
        {"solve_rate": 0.2,  "id": "exact_lo"},
        {"solve_rate": 0.5,  "id": "middle"},
        {"solve_rate": 0.8,  "id": "exact_hi"},
        {"solve_rate": 1.0,  "id": "too_easy"},
    ]
    kept_ids = [r["id"] for r in difficulty_filter(rows)]
    assert kept_ids == ["exact_lo", "middle", "exact_hi"]


def test_difficulty_filter_custom_band():
    rows = [{"solve_rate": r} for r in (0.05, 0.15, 0.4, 0.6, 0.95)]
    kept = difficulty_filter(rows, lo=0.1, hi=0.5)
    assert [r["solve_rate"] for r in kept] == [0.15, 0.4]


def test_difficulty_filter_rejects_invalid_band():
    """``hi < lo`` is a config bug; refuse rather than silently drop all."""
    with pytest.raises(ValueError):
        difficulty_filter([], lo=0.8, hi=0.2)
    with pytest.raises(ValueError):
        difficulty_filter([], lo=-0.1, hi=0.5)
    with pytest.raises(ValueError):
        difficulty_filter([], lo=0.0, hi=1.5)


def test_difficulty_filter_band_is_configurable():
    """Rescue-config support: --difficulty-lo and --difficulty-hi CLI flags
    must parse and end-to-end drive the filter to a tighter band.

    The rescue plan tightens the band from the default [0.2, 0.8] to
    [0.35, 0.65] so curated prompts cluster around 50% solve rate — the
    sweet spot where GRPO has the strongest per-prompt reward variance.
    This test confirms the flag→args→filter chain works without code
    changes (the flags already exist on prepare_rlvr.py since 2026-05-09).
    """
    # 1. CLI parsing: the flags exist and pick up custom values.
    args = _parse_args([
        "--difficulty-lo", "0.35",
        "--difficulty-hi", "0.65",
    ])
    assert args.difficulty_lo == 0.35
    assert args.difficulty_hi == 0.65

    # 2. End-to-end: filter respects the parsed band.
    rows = [
        {"solve_rate": 0.10, "id": "below"},
        {"solve_rate": 0.30, "id": "below_new_lo"},
        {"solve_rate": 0.35, "id": "exact_lo"},
        {"solve_rate": 0.50, "id": "middle"},
        {"solve_rate": 0.65, "id": "exact_hi"},
        {"solve_rate": 0.75, "id": "above_new_hi"},
        {"solve_rate": 0.95, "id": "above"},
    ]
    kept_ids = [
        r["id"] for r in difficulty_filter(rows, lo=args.difficulty_lo, hi=args.difficulty_hi)
    ]
    assert kept_ids == ["exact_lo", "middle", "exact_hi"]

    # 3. Defaults still reflect the proposal's [0.2, 0.8] band.
    default_args = _parse_args([])
    assert default_args.difficulty_lo == DIFFICULTY_LO
    assert default_args.difficulty_hi == DIFFICULTY_HI


# =============================================================================
# solve_rate — empirical c/n with the n=k=8 degeneracy explained in module
# docstring.
# =============================================================================

def test_solve_rate_arithmetic():
    assert solve_rate(0, 8) == 0.0
    assert solve_rate(4, 8) == 0.5
    assert solve_rate(8, 8) == 1.0


def test_solve_rate_zero_rollouts_raises():
    with pytest.raises(ValueError):
        solve_rate(0, 0)


# =============================================================================
# extract_prompt_and_gold — pulls (user prompt, boxed gold) out of a Stage 1
# row. This is the bridge between Stage 1's training format and Stage 7's
# RLVR format.
# =============================================================================

def test_extract_prompt_and_gold_happy_path():
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>2+2=4</think>\n\n\\boxed{4}"},
    ]
    assert extract_prompt_and_gold(msgs) == ("What is 2+2?", "4")


def test_extract_prompt_and_gold_missing_box_returns_none():
    """Stage 1 should already have filtered these out, but defend in depth."""
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "the answer is 4"},
    ]
    assert extract_prompt_and_gold(msgs) is None


def test_extract_prompt_and_gold_swapped_roles_returns_none():
    msgs = [
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "..."},
    ]
    assert extract_prompt_and_gold(msgs) is None


def test_extract_prompt_and_gold_missing_assistant_returns_none():
    msgs = [{"role": "user", "content": "..."}]
    assert extract_prompt_and_gold(msgs) is None


def test_extract_prompt_and_gold_picks_last_box():
    """Same semantics as ``extract_boxed_answer`` — last ``\\boxed{}`` wins.
    Mid-think wrong boxes get superseded by the final answer."""
    msgs = [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "<think>\\boxed{3}, no wait</think>\n\\boxed{4}"},
    ]
    assert extract_prompt_and_gold(msgs) == ("...", "4")


# =============================================================================
# validate_pool_row — rejects malformed rows with WARNING, not exceptions.
# =============================================================================

def test_validate_pool_row_accepts_valid_row():
    row = {
        "messages": [
            {"role": "user", "content": "P"},
            {"role": "assistant", "content": "<think>r</think>\n\\boxed{A}"},
        ],
    }
    out = validate_pool_row(row, line_no=1)
    assert out == [{"prompt": "P", "answer": "A"}]


def test_validate_pool_row_rejects_non_dict():
    assert validate_pool_row("not a dict", line_no=1) == []
    assert validate_pool_row([1, 2, 3], line_no=1) == []


def test_validate_pool_row_rejects_missing_messages():
    assert validate_pool_row({}, line_no=1) == []
    assert validate_pool_row({"messages": "not a list"}, line_no=1) == []


def test_validate_pool_row_rejects_unboxed_assistant():
    row = {
        "messages": [
            {"role": "user", "content": "P"},
            {"role": "assistant", "content": "no box here"},
        ],
    }
    assert validate_pool_row(row, line_no=1) == []


# =============================================================================
# load_pool_jsonl — end-to-end JSONL parsing with the malformed rows
# detected and skipped (rather than raising).
# =============================================================================

def test_load_pool_jsonl_happy_path(tmp_path: Path):
    p = tmp_path / "pool.jsonl"
    rows = [
        {"messages": [
            {"role": "user", "content": f"Q{i}"},
            {"role": "assistant", "content": f"<think>r</think>\n\\boxed{{{i}}}"},
        ]}
        for i in range(5)
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = load_pool_jsonl(p)
    assert len(out) == 5
    assert out[0] == {"prompt": "Q0", "answer": "0"}
    assert out[4] == {"prompt": "Q4", "answer": "4"}


def test_load_pool_jsonl_skips_invalid_lines(tmp_path: Path):
    """A malformed JSON line and an unboxed-assistant row both get
    skipped without aborting the load. Caller sees only the valid rows.

    Concrete failure mode this guards: a single corrupted line in a
    ~280k-row Stage 1 file should NOT take the whole curation pass
    down.
    """
    p = tmp_path / "pool.jsonl"
    p.write_text(
        json.dumps({"messages": [
            {"role": "user", "content": "good"},
            {"role": "assistant", "content": "<think>r</think>\n\\boxed{1}"},
        ]}) + "\n"
        + "{not valid json\n"
        + json.dumps({"messages": [
            {"role": "user", "content": "no_box"},
            {"role": "assistant", "content": "the answer is 2"},
        ]}) + "\n"
        + json.dumps({"messages": [
            {"role": "user", "content": "good2"},
            {"role": "assistant", "content": "<think>r</think>\n\\boxed{3}"},
        ]}) + "\n",
        encoding="utf-8",
    )
    out = load_pool_jsonl(p)
    assert len(out) == 2
    assert out[0]["prompt"] == "good"
    assert out[1]["prompt"] == "good2"


def test_load_pool_jsonl_max_rows_caps_valid_rows(tmp_path: Path):
    """``max_rows`` caps the count of *valid* rows returned, not lines
    consumed — important when the input has a high reject rate (Stage 1
    DART had a ~52% drop rate)."""
    p = tmp_path / "pool.jsonl"
    rows = [
        {"messages": [
            {"role": "user", "content": f"Q{i}"},
            {"role": "assistant", "content": f"<think>r</think>\n\\boxed{{{i}}}"},
        ]}
        for i in range(10)
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = load_pool_jsonl(p, max_rows=3)
    assert len(out) == 3


# =============================================================================
# write_jsonl — round-trip; one JSON object per line; no array wrapper.
# =============================================================================

def test_write_jsonl_round_trip(tmp_path: Path):
    rows = [
        {"prompt": "p1", "answer": "a1", "solve_rate": 0.5},
        {"prompt": "p2", "answer": "a2", "solve_rate": 0.25},
    ]
    out = tmp_path / "out.jsonl"
    n = write_jsonl(rows, out)
    assert n == 2
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == rows[0]
    assert json.loads(lines[1]) == rows[1]


def test_write_jsonl_creates_parent_dir(tmp_path: Path):
    out = tmp_path / "nested" / "subdir" / "out.jsonl"
    n = write_jsonl([{"x": 1}], out)
    assert n == 1
    assert out.is_file()


# =============================================================================
# build_scored_row — chat-template invariant (2026-05-12 incident).
#
# The retry2 RLVR run on 2026-05-12 trained for hours with reward_std=0,
# 100% of rollouts hitting the 4096-token cap, because the output rows
# had RAW prompts instead of chat-templated ones. GRPO fed the raw
# strings to the model, which never emitted <|im_end|>. This helper +
# its tests are the bug-class guard: any future code path that builds
# an output row must go through build_scored_row, which refuses to
# accept a prompt that's missing the chat-open marker.
# =============================================================================

# Chat-template wrap + THINK_PREFIX. Mirrors what the locked
# chat_template/chat_template.jinja produces with add_generation_prompt=True,
# plus the "<think>\n" appended by prepare_rlvr.main().
_TEMPLATED_PROMPT = (
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
)
# Bare chat-template output — what apply_chat_template() alone produces.
# Has the chat wrap, but NO <think>\n suffix. This is the 2026-05-12
# retry3 bug shape.
_TEMPLATED_NO_THINK = (
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def test_build_scored_row_writes_templated_prompt():
    """Happy path — output row contains the templated + <think> prompt."""
    out = build_scored_row(
        rendered_prompt=_TEMPLATED_PROMPT,
        gold="4",
        n_correct=3,
        n_rollouts=8,
    )
    assert out["prompt"] == _TEMPLATED_PROMPT
    assert CHAT_TEMPLATE_OPEN_MARKER in out["prompt"]
    assert out["prompt"].endswith(THINK_PREFIX)
    assert out["answer"] == "4"
    assert out["solve_rate"] == 3 / 8


def test_build_scored_row_rejects_raw_prompt():
    """Defensive guard — raw (unwrapped) prompts must raise (retry2)."""
    raw_problem = r"Let $\mathbf{a} = \langle x, y\rangle$. Find ..."
    assert CHAT_TEMPLATE_OPEN_MARKER not in raw_problem
    with pytest.raises(ValueError, match="chat-template marker"):
        build_scored_row(
            rendered_prompt=raw_problem,
            gold="42",
            n_correct=2,
            n_rollouts=8,
        )


def test_build_scored_row_rejects_templated_but_no_think_prefix():
    """Defensive guard — bare chat-template output must raise (retry3).

    The 2026-05-12 retry3 incident: prompts had ``<|im_start|>`` but
    ended at ``<|im_start|>assistant\\n`` without ``<think>\\n``. The
    v3 SFT model was trained on assistant turns that begin with
    <think>, so missing this suffix causes rollouts to unreliably
    drop the think token, run off-distribution, and never terminate."""
    assert CHAT_TEMPLATE_OPEN_MARKER in _TEMPLATED_NO_THINK
    assert not _TEMPLATED_NO_THINK.endswith(THINK_PREFIX)
    with pytest.raises(ValueError, match=r"<think>"):
        build_scored_row(
            rendered_prompt=_TEMPLATED_NO_THINK,
            gold="4",
            n_correct=3,
            n_rollouts=8,
        )


def test_build_scored_row_round_trips_through_write_jsonl(tmp_path: Path):
    """End-to-end: a row built via build_scored_row → written → re-read
    must still carry BOTH the chat-template marker AND the THINK_PREFIX.
    Pins the on-disk schema that scripts/train_rlvr.py P0 check relies on."""
    row = build_scored_row(
        rendered_prompt=_TEMPLATED_PROMPT,
        gold="4",
        n_correct=4,
        n_rollouts=8,
    )
    out = tmp_path / "rlvr_prompts.jsonl"
    write_jsonl([row], out)
    reread = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert CHAT_TEMPLATE_OPEN_MARKER in reread["prompt"]
    assert reread["prompt"].endswith(THINK_PREFIX)
    assert reread["solve_rate"] == 0.5


# =============================================================================
# CLI argument parsing — required defaults present, conflicting bands rejected
# at runtime (filter raises). This pins the surface that submit_rlvr.sh and
# Stage 7 docs reference.
# =============================================================================

def test_parse_args_defaults():
    args = _parse_args([])
    assert args.pool_size == 10000
    assert args.target_size == 5000
    assert args.num_generations == 8
    assert args.max_new_tokens == 4096
    assert args.temperature == 0.8
    assert args.seed == 42
    assert args.difficulty_lo == DIFFICULTY_LO
    assert args.difficulty_hi == DIFFICULTY_HI
    assert args.dry_run is False


def test_parse_args_dry_run_flag():
    args = _parse_args(["--dry-run"])
    assert args.dry_run is True


def test_parse_args_custom_band():
    args = _parse_args(["--difficulty-lo", "0.1", "--difficulty-hi", "0.9"])
    assert args.difficulty_lo == 0.1
    assert args.difficulty_hi == 0.9
