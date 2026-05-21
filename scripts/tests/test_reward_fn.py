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

import warnings

import pytest

from scripts.reward_fn import (
    CORRECTNESS_WEIGHT,
    FORMAT_WEIGHT,
    LENGTH_BONUS_WEIGHT,
    TARGET_LENGTH_TOKENS,
    batch_rewards,
    compute_reward,
)


class _FakeTokenizer:
    """Stand-in for an HF tokenizer used only by the length-bonus tests.

    Returns ``n`` integer "token ids" so ``len(encode(...))`` equals ``n``
    by construction. No real tokenization happens — the actual GRPO loop
    will pass a real tokenizer; the reward function only cares about the
    length of the returned sequence.
    """

    def __init__(self, n: int) -> None:
        self.n = n

    def encode(self, _text: str) -> list[int]:
        return [0] * self.n


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


def test_empty_box_returns_zero_not_format_reward():
    """``\\boxed{}`` MUST score 0.0, not 0.05.

    ``extract_boxed_answer`` returns the empty string (not None) when
    the model emits an empty box, which would naively trip
    ``has_box=True`` and harvest the 0.05 format reward without any
    answer content. Under GRPO that is a free 0.05/rollout attractor:
    a policy that learns to give up by emitting ``\\boxed{}`` gets
    rewarded for it. The reward function must explicitly require
    non-empty stripped payload. See scripts/tests/reward_fn_audit.md
    → "Empty-box gaming risk"."""
    gen = "<think>I'm not sure</think>\n\n\\boxed{}"
    assert compute_reward(gen, "42") == 0.0


def test_whitespace_only_box_returns_zero_not_format_reward():
    """``\\boxed{ }`` MUST score 0.0 — same gaming-risk as the empty
    box, just with whitespace inside. ``extract_boxed_answer`` returns
    ``" "`` (the literal space) which would naively trip ``has_box``.

    Stripped-payload check must catch this. Also defends against
    ``\\boxed{\\n}`` and ``\\boxed{\\t}``, which are the same semantic
    case in different lexical clothing."""
    gen = "<think>...</think>\n\n\\boxed{ }"
    assert compute_reward(gen, "42") == 0.0


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


# =============================================================================
# Optional length-shaping bonus (LENGTH_BONUS_WEIGHT > 0).
#
# Default LENGTH_BONUS_WEIGHT is 0.0 (OFF). When enabled, correct rollouts
# get an additional conciseness bonus in [0, LENGTH_BONUS_WEIGHT * 1.0];
# wrong rollouts get NOTHING from the length term regardless of length.
# The tests below pin the safety properties first, the gradient second.
# =============================================================================

def test_default_length_bonus_weight_is_zero():
    """Backward-compat contract: the imported constant ships at 0.0.

    Any future change must flip this deliberately, with a code review;
    we don't want a silent default flip to add length shaping to
    in-flight RLVR runs.
    """
    assert LENGTH_BONUS_WEIGHT == 0.0


def test_backward_compat_default_weight_byte_identical():
    """With the default LENGTH_BONUS_WEIGHT=0.0, results are byte-identical
    to the pre-length-bonus reward — even when a tokenizer is passed.

    This is the explicit backward-compat assertion: legacy RLVR runs that
    happen to pass a tokenizer through (or don't) must score exactly the
    same as before the length-bonus feature existed.
    """
    gen_correct = "<think>...</think>\n\n\\boxed{4}"
    gen_wrong = "<think>...</think>\n\n\\boxed{5}"
    # No tokenizer (legacy call shape):
    assert compute_reward(gen_correct, "4") == CORRECTNESS_WEIGHT + FORMAT_WEIGHT
    assert compute_reward(gen_wrong, "4") == FORMAT_WEIGHT
    # With tokenizer (still no bonus at weight 0):
    tok = _FakeTokenizer(10)
    assert compute_reward(gen_correct, "4", tokenizer=tok) == CORRECTNESS_WEIGHT + FORMAT_WEIGHT
    assert compute_reward(gen_wrong, "4", tokenizer=tok) == FORMAT_WEIGHT


def test_length_bonus_only_applies_to_correct_answers(monkeypatch):
    """SAFETY: a short WRONG answer must never beat a long RIGHT answer.

    The conciseness gate is on ``correct`` (is_equiv agreement), not on
    ``has_box``. Otherwise GRPO would learn the degenerate strategy
    "emit short wrong \\boxed{} to harvest the length bonus". The test
    pins both the per-rollout value (short wrong → format only, no
    length term) and the cross-rollout ordering GRPO uses for advantage.
    """
    import scripts.reward_fn as rf

    monkeypatch.setattr(rf, "LENGTH_BONUS_WEIGHT", 0.2)
    monkeypatch.setattr(rf, "TARGET_LENGTH_TOKENS", 1024)

    short_wrong = "<think>...</think>\n\n\\boxed{5}"
    long_correct = "<think>...</think>\n\n\\boxed{4}"

    short_wrong_score = rf.compute_reward(
        short_wrong, "4", tokenizer=_FakeTokenizer(10)
    )
    long_correct_score = rf.compute_reward(
        long_correct, "4", tokenizer=_FakeTokenizer(1900)
    )

    # Short wrong gets only the format reward — NO length bonus.
    assert short_wrong_score == FORMAT_WEIGHT
    # Long correct still beats it by at least the correctness mass.
    assert long_correct_score > short_wrong_score
    assert long_correct_score >= CORRECTNESS_WEIGHT + FORMAT_WEIGHT


def test_shorter_correct_scores_higher(monkeypatch):
    """When both rollouts are CORRECT, the shorter one scores strictly higher.

    This is the conciseness gradient the length term introduces. At the
    spec'd shape, a 100-token correct answer should score more than a
    1500-token correct answer, and both should score at least
    CORRECTNESS + FORMAT (the floor for any correct rollout).
    """
    import scripts.reward_fn as rf

    monkeypatch.setattr(rf, "LENGTH_BONUS_WEIGHT", 0.2)
    monkeypatch.setattr(rf, "TARGET_LENGTH_TOKENS", 1024)

    gen = "<think>...</think>\n\n\\boxed{4}"
    short_score = rf.compute_reward(gen, "4", tokenizer=_FakeTokenizer(100))
    long_score = rf.compute_reward(gen, "4", tokenizer=_FakeTokenizer(1500))

    assert short_score > long_score
    # Both correct → floor is CORRECTNESS + FORMAT for both.
    assert long_score >= CORRECTNESS_WEIGHT + FORMAT_WEIGHT
    # And the short one beats that floor (bonus > 0 at length 100).
    assert short_score > CORRECTNESS_WEIGHT + FORMAT_WEIGHT


def test_length_bonus_skipped_without_tokenizer(monkeypatch):
    """Misconfiguration is safe: LENGTH_BONUS_WEIGHT > 0 with no tokenizer
    falls back to correctness + format only, emitting a one-shot warning.

    Defends existing call sites in train_rlvr.py that may not yet thread
    a tokenizer through. Better to lose the bonus and warn than to crash
    the GRPO inner loop on the first batch.
    """
    import scripts.reward_fn as rf

    monkeypatch.setattr(rf, "LENGTH_BONUS_WEIGHT", 0.2)
    # Reset the latch so this test sees the warning regardless of order.
    monkeypatch.setattr(rf, "_warned_missing_tokenizer", False)

    gen = "<think>...</think>\n\n\\boxed{4}"

    with pytest.warns(RuntimeWarning, match="no tokenizer"):
        score = rf.compute_reward(gen, "4")  # no tokenizer
    # Reward collapses to the legacy correctness + format value.
    assert score == CORRECTNESS_WEIGHT + FORMAT_WEIGHT

    # Second call: warning should NOT fire again (latch held).
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # If the latch were broken this would raise.
        assert rf.compute_reward(gen, "4") == CORRECTNESS_WEIGHT + FORMAT_WEIGHT
