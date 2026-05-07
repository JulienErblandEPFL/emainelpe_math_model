"""Unbiased pass@k estimator.

From Chen et al. (2021) "Evaluating Large Language Models Trained on Code".
"""

from __future__ import annotations

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute the unbiased estimator of pass@k.

    pass@k = 1 - C(n-c, k) / C(n, k)

    Computed numerically as:
        1 - prod_{i=n-c+1}^{n} (1 - k/i)
    to avoid overflow from large binomial coefficients.

    Args:
        n: total number of samples generated per problem.
        c: number of correct samples for this problem.
        k: the k in pass@k.

    Returns:
        Estimated probability that at least one of k random samples is correct.
    """
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def compute_pass_at_k_for_dataset(
    per_problem_correct: list[int],
    n: int,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute average pass@k across all problems in a dataset.

    Args:
        per_problem_correct: list of correct counts (c) for each problem.
        n: number of samples per problem (same for all).
        k_values: list of k values to compute. Defaults to [1, 8].

    Returns:
        Dict like {"pass@1": 0.45, "pass@8": 0.72}.
    """
    if k_values is None:
        k_values = [1, 8]

    results = {}
    for k in k_values:
        if k > n:
            raise ValueError(f"k={k} > n={n}: cannot compute pass@{k} with only {n} samples")
        scores = [pass_at_k(n, c, k) for c in per_problem_correct]
        results[f"pass@{k}"] = float(np.mean(scores))
    return results


def mean_at_k_for_dataset(per_problem_correct: list[int], n: int, k: int) -> float:
    """Compute mean correctness over k sampled completions per problem.

    This is equivalent to running pass@1 on the same dataset k times, once per
    completion slot, then averaging those k accuracies.
    """
    if k != n:
        raise ValueError(f"mean@{k} requires exactly {k} samples per problem; got n={n}")
    return float(np.mean([c / k for c in per_problem_correct]))
