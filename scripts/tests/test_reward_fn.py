"""Tests for ``scripts/reward_fn.py`` — the GRPO reward callback.

Pure Python; no torch/trl imports. The vendored ``evaluate`` package is
all stdlib + regex, so these tests run in <1s on the user's laptop.

Reward shape under test (decision D4, 2026-05-09):

    reward = 1.0 * correct + 0.05 * has_box

so the truth table is:

    correct + boxed       → 1.05
    wrong   + boxed       → 0.05
    wrong   + no box      → 0.00
    correct + no box      → 0.00 (impossible: correct *requires* a box)
"""
from __future__ import annotations

from scripts.reward_fn import (
    CORRECTNESS_WEIGHT,
    FORMAT_WEIGHT,
    batch_rewards,
    compute_reward,
)


# =============================================================================
# Truth table — the four input combinations the user asked the tests to cover.
# =============================================================================

def test_correct_with_box_returns_full_reward():
    gen = "<think>2+2=4</think>\n\n\\boxed{4}"
    assert compute_reward(gen, "4") == CORRECTNESS_WEIGHT + FORMAT_WEIGHT


def test_wrong_with_box_returns_format_reward_only():
    gen = "<think>I think 2+2=5</think>\n\n\\boxed{5}"
    assert compute_reward(gen, "4") == FORMAT_WEIGHT


def test_wrong_no_box_returns_zero():
    gen = "the answer is 5"
    assert compute_reward(gen, "4") == 0.0


def test_no_box_even_with_correct_substring_returns_zero():
    """A bare '4' in the prose without ``\\boxed{}`` MUST score 0.

    This matches the CI's behavior: ``evaluate.score.score_generations``
    counts unboxed answers as wrong regardless of substring content.
    Drifting away from that here would teach the model to skip the box.
    """
    gen = "<think>2+2=4 obviously</think>\n\nthe answer is 4"
    assert compute_reward(gen, "4") == 0.0


# =============================================================================
# is_equiv passthrough — corner cases CI handles via aggressive normalization.
# These prove the reward delegates to evaluate.is_equiv rather than a
# re-implementation; if any of them break, evaluate/extract_answer.py changed
# and our reward will silently drift from the CI grader.
# =============================================================================

def test_fraction_equivalent_to_decimal_scores_correct():
    """``\\frac{1}{2}`` and ``0.5`` are equivalent under OpenCompass
    normalization (the CI's grader). Reward must agree."""
    gen = "<think>half</think>\n\n\\boxed{0.5}"
    assert compute_reward(gen, r"\frac{1}{2}") == CORRECTNESS_WEIGHT + FORMAT_WEIGHT


def test_unit_strip_equivalence_scores_correct():
    """``42`` and ``42 \\text{ km}`` should be equivalent: ``is_equiv``
    strips units and ``\\text{}`` wrappers as part of normalization."""
    gen = "<think>distance</think>\n\n\\boxed{42 \\text{ km}}"
    assert compute_reward(gen, "42") == CORRECTNESS_WEIGHT + FORMAT_WEIGHT


def test_picks_last_boxed_when_multiple():
    """``extract_boxed_answer`` uses ``last_boxed_only_string``; if the
    model writes a wrong ``\\boxed{}`` mid-think and the right one at the
    end, the right one wins. The reward MUST inherit this."""
    gen = (
        "<think>maybe \\boxed{3}, no wait — let me redo this.\n"
        "answer is 4.</think>\n\n\\boxed{4}"
    )
    assert compute_reward(gen, "4") == CORRECTNESS_WEIGHT + FORMAT_WEIGHT


# =============================================================================
# Batch helper — same per-prompt loop the GRPO inner step runs.
# =============================================================================

def test_batch_rewards_returns_list_in_order():
    gens = [
        "<think>...</think>\n\n\\boxed{4}",      # correct + boxed
        "<think>...</think>\n\n\\boxed{5}",      # wrong + boxed
        "no box anywhere",                         # wrong + no box
        "<think>...</think>\n\n\\boxed{4.0}",    # correct (4.0 == 4)
    ]
    out = batch_rewards(gens, "4")
    assert len(out) == 4
    assert out[0] == CORRECTNESS_WEIGHT + FORMAT_WEIGHT
    assert out[1] == FORMAT_WEIGHT
    assert out[2] == 0.0
    assert out[3] == CORRECTNESS_WEIGHT + FORMAT_WEIGHT


def test_batch_rewards_empty_input():
    """Edge case: an empty rollout list yields an empty reward list.

    GRPOTrainer should never call us with this, but we shouldn't blow up.
    """
    assert batch_rewards([], "4") == []


def test_format_weight_is_small_relative_to_correctness():
    """Sanity-check the shape of the reward. If FORMAT_WEIGHT ever creeps
    above ~0.5, the model can equilibrium at "always box, even nonsense"
    — the team agreed the format reward stays tiny. This test pins the
    invariant rather than the exact value."""
    assert FORMAT_WEIGHT < 0.2
    assert FORMAT_WEIGHT < CORRECTNESS_WEIGHT / 4
