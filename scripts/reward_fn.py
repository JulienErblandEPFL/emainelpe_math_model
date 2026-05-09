"""Reward function for RLVR (Stage 7 / Phase 2).

Used by ``scripts/train_rlvr.py`` as the reward callback for TRL's
``GRPOTrainer``. Kept in its own module so tests run on the user's
laptop without importing TRL/torch.

Reward shape (decision D4, 2026-05-09):

    reward = 1.0 * correct + 0.05 * has_box

where ``correct`` is OpenCompass ``is_equiv`` agreement between the
last ``\\boxed{...}`` payload in the generation and the gold answer,
and ``has_box`` is the boolean "the generation produced a parseable
``\\boxed{}``". The small format reward keeps gradient alive when the
SFT model temporarily regresses on boxing during early exploration; it
is small enough that boxing garbage is dominated by the correctness
signal.

Equivalence is delegated byte-for-byte to ``evaluate.is_equiv`` (the
vendored OpenCompass copy used by the nightly CI). Re-implementing
would silently drift from CI scoring.
"""
from __future__ import annotations

from pathlib import Path
import sys

# Allow ``import evaluate`` when this module is imported as
# ``scripts.reward_fn`` from inside the repo (tests do this) or as
# ``reward_fn`` from a script (TRL's reward-callback registration may).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluate.extract_answer import extract_boxed_answer, is_equiv  # noqa: E402

CORRECTNESS_WEIGHT = 1.0
FORMAT_WEIGHT = 0.05


def compute_reward(generation: str, gold: str) -> float:
    """Score a single rollout against the gold answer.

    Uses ``extract_boxed_answer(..., strip_double_curly_brace=True)``
    to peel one extra ``{...}`` layer when the model double-wraps —
    same call site as ``evaluate.score.score_generations``.
    """
    extracted = extract_boxed_answer(generation, strip_double_curly_brace=True)
    has_box = extracted is not None
    correct = bool(has_box and is_equiv(extracted, gold))
    return CORRECTNESS_WEIGHT * float(correct) + FORMAT_WEIGHT * float(has_box)


def batch_rewards(generations: list[str], gold: str) -> list[float]:
    """Score a batch of rollouts that share the same gold answer.

    Convenience for the GRPO inner loop, which generates ``num_generations``
    completions per prompt and scores each independently. The function is
    stateless so this is just a list comprehension; kept as a named
    helper so the per-prompt batch is profileable.
    """
    return [compute_reward(g, gold) for g in generations]
