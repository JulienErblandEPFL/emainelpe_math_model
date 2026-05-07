"""CLI: score a generations JSONL with the same logic the nightly CI uses.

Usage:
    python -m evaluate.score \\
        --generations my_math_gens.jsonl \\
        --benchmark math \\
        [--output scored.json]

Input JSONL schema (one object per problem):
    {"prompt": "...", "answer": "<gold>", "completions": ["<gen 1>", "<gen 2>", ...]}

`reference` is accepted as a synonym for `answer`. All rows must have the same
number of completions; that count is used as `n` for pass@k.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmarks import VALID_METHODS, extract_benchmark_answer, is_correct_benchmark_answer
from .pass_at_k import compute_pass_at_k_for_dataset


# Mirrors config/benchmarks.yaml in the CI repo.
BENCHMARK_TO_METHOD = {
    "math": "boxed",
    "knowledge": "knowledge",
    "multilingual": "boxed",
    "safety": "boxed",
}


def _read_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {e}")
    return items


def _gold(item: dict) -> str:
    if "reference" in item:
        return str(item["reference"])
    if "answer" in item:
        return str(item["answer"])
    raise SystemExit(
        "Each row must contain either 'answer' or 'reference' as the gold field."
    )


def score_generations(
    items: list[dict],
    method: str,
) -> dict:
    if not items:
        raise SystemExit("Generations file is empty.")

    n_completions = None
    per_problem_correct: list[int] = []
    detailed: list[dict] = []

    for i, item in enumerate(items):
        completions = item.get("completions")
        if not isinstance(completions, list) or not completions:
            raise SystemExit(
                f"Row {i}: 'completions' must be a non-empty list of strings."
            )
        if n_completions is None:
            n_completions = len(completions)
        elif len(completions) != n_completions:
            raise SystemExit(
                f"Row {i}: has {len(completions)} completions, expected {n_completions}. "
                "All rows must have the same number of completions."
            )

        reference = _gold(item)
        c = 0
        comp_details = []
        for comp in completions:
            comp_text = str(comp)
            extracted = extract_benchmark_answer(comp_text, method, reference)
            correct = is_correct_benchmark_answer(extracted, reference, method)
            c += int(correct)
            comp_details.append({"extracted": extracted, "correct": correct})

        per_problem_correct.append(c)
        detailed.append({
            "index": i,
            "prompt": item.get("prompt"),
            "reference": reference,
            "n": n_completions,
            "c": c,
            "completions": comp_details,
        })

    n = n_completions or 0
    k_values = [k for k in (1, 8) if k <= n]
    if not k_values:
        raise SystemExit(f"Need at least n=1 completions per row; got n={n}.")
    metrics = compute_pass_at_k_for_dataset(per_problem_correct, n, k_values)

    return {
        "benchmark_method": method,
        "n_problems": len(items),
        "n_completions": n,
        "metrics": metrics,
        "detailed_results": detailed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evaluate.score",
        description="Score generations with the same logic the MNLP nightly CI uses.",
    )
    parser.add_argument(
        "--generations",
        required=True,
        type=Path,
        help="Path to a JSONL file with one row per problem (see module docstring for schema).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--benchmark",
        choices=sorted(BENCHMARK_TO_METHOD),
        help="Benchmark name; selects the extraction method used by the CI.",
    )
    group.add_argument(
        "--method",
        choices=VALID_METHODS,
        help="Extraction method override (advanced; usually pass --benchmark instead).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write detailed per-problem results as JSON.",
    )
    args = parser.parse_args(argv)

    method = args.method or BENCHMARK_TO_METHOD[args.benchmark]
    items = _read_jsonl(args.generations)
    result = score_generations(items, method)

    parts = [f"{k}={v:.4f}" for k, v in result["metrics"].items()]
    summary = (
        f"{', '.join(parts)} "
        f"(n_problems={result['n_problems']}, n_completions={result['n_completions']}, "
        f"method={method})"
    )
    print(summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Wrote detailed results to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
