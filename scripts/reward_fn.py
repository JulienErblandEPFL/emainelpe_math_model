"""Reward function for RLVR (Stage 7 / Phase 2).

Used by ``scripts/train_rlvr.py`` as the reward callback for TRL's
``GRPOTrainer``. Kept in its own module so tests run on the user's
laptop without importing TRL/torch.

Default reward shape (decision D4, 2026-05-09):

    reward = 1.0 * correct + 0.05 * has_box

where ``correct`` is OpenCompass ``is_equiv`` agreement between the
last ``\\boxed{...}`` payload in the generation and the gold answer,
and ``has_box`` is the boolean "the generation produced a parseable
``\\boxed{}``". The small format reward keeps gradient alive when the
SFT model temporarily regresses on boxing during early exploration; it
is small enough that boxing garbage is dominated by the correctness
signal.

Optional conciseness shaping (``LENGTH_BONUS_WEIGHT > 0``):

    reward = 1.0 * correct + 0.05 * has_box + LENGTH_BONUS_WEIGHT * conciseness

where ``conciseness`` is 0 for any incorrect answer (the critical
safety gate — short wrong answers must never beat long right ones)
and otherwise decays linearly from 1.0 at 0 tokens to 0.5 at
``TARGET_LENGTH_TOKENS`` to 0.0 at twice that, measured by the
optional ``tokenizer`` parameter. Default ``LENGTH_BONUS_WEIGHT=0.0``
keeps the legacy reward byte-identical for backward compatibility.

Equivalence is delegated byte-for-byte to ``evaluate.is_equiv`` (the
vendored OpenCompass copy used by the nightly CI). Re-implementing
would silently drift from CI scoring.
"""
from __future__ import annotations

from pathlib import Path
import sys
import warnings

# Allow ``import evaluate`` when this module is imported as
# ``scripts.reward_fn`` from inside the repo (tests do this) or as
# ``reward_fn`` from a script (TRL's reward-callback registration may).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluate.extract_answer import extract_boxed_answer, is_equiv  # noqa: E402

CORRECTNESS_WEIGHT = 1.0
FORMAT_WEIGHT = 0.05
# Default OFF. When > 0 and a tokenizer is supplied, adds a conciseness
# bonus to CORRECT rollouts. Suggested operational range when enabling:
# 0.1-0.2. Even at LENGTH_BONUS_WEIGHT=1.0 the max bonus equals the
# correctness mass, so any value < 1.0 keeps correctness dominant.
LENGTH_BONUS_WEIGHT = 0.0
# "Good" target length — a correct answer of this length gets half the
# max bonus; the bonus is zero past 2 * TARGET_LENGTH_TOKENS.
TARGET_LENGTH_TOKENS = 1024

# Module-level latch so the "tokenizer missing" warning fires exactly once
# even though the reward is called per rollout in the GRPO hot loop. Reset
# in tests via monkeypatch (`monkeypatch.setattr(rf, "_warned_missing_tokenizer", False)`).
_warned_missing_tokenizer = False


def _conciseness_bonus(completion_length: int) -> float:
    """Linear decay: 1.0 at 0 tokens → 0.5 at TARGET → 0.0 at 2*TARGET.

    Clamped at 0.0 for any length past the decay tail so the policy
    can't be penalized in absolute terms for very long outputs (it
    just stops being rewarded for brevity).
    """
    return max(0.0, 1.0 - completion_length / (2.0 * TARGET_LENGTH_TOKENS))


def compute_reward(generation: str, gold: str, tokenizer=None) -> float:
    """Score a single rollout against the gold answer.

    Uses ``extract_boxed_answer(..., strip_double_curly_brace=True)``
    to peel one extra ``{...}`` layer when the model double-wraps —
    same call site as ``evaluate.score.score_generations``.

    With default ``LENGTH_BONUS_WEIGHT == 0.0`` the result is byte-
    identical to the pre-length-bonus reward. With ``LENGTH_BONUS_WEIGHT
    > 0`` and a ``tokenizer`` argument, correct rollouts additionally
    receive a conciseness bonus; the gate on ``correct`` (not
    ``has_box``) is what stops the policy from gaming this by emitting
    short empty/wrong boxes.
    """
    extracted = extract_boxed_answer(generation, strip_double_curly_brace=True)
    # Require non-empty stripped payload so empty \boxed{} and \boxed{ }
    # don't harvest the format reward without contributing an answer —
    # would otherwise give GRPO a "give up, emit empty box" attractor
    # at 0.05 per rollout. See scripts/tests/reward_fn_audit.md →
    # "Empty-box gaming risk".
    has_box = bool(extracted is not None and extracted.strip())
    correct = bool(has_box and is_equiv(extracted, gold))
    reward = CORRECTNESS_WEIGHT * float(correct) + FORMAT_WEIGHT * float(has_box)

    if LENGTH_BONUS_WEIGHT > 0.0:
        if tokenizer is None:
            global _warned_missing_tokenizer
            if not _warned_missing_tokenizer:
                warnings.warn(
                    "LENGTH_BONUS_WEIGHT > 0 but no tokenizer was passed to "
                    "compute_reward; conciseness bonus is being skipped. "
                    "Pass tokenizer=... to enable length shaping.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _warned_missing_tokenizer = True
        elif correct:
            length = len(tokenizer.encode(generation))
            reward += LENGTH_BONUS_WEIGHT * _conciseness_bonus(length)

    return reward


def batch_rewards(
    generations: list[str], gold: str, tokenizer=None
) -> list[float]:
    """Score a batch of rollouts that share the same gold answer.

    Convenience for the GRPO inner loop, which generates ``num_generations``
    completions per prompt and scores each independently. The function is
    stateless so this is just a list comprehension; kept as a named
    helper so the per-prompt batch is profileable.
    """
    return [compute_reward(g, gold, tokenizer=tokenizer) for g in generations]
