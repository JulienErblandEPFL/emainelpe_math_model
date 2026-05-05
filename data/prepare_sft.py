"""Prepare hkust-nlp/dart-math-uniform for SFT.

Loads the dataset, normalizes each response to
``<think>{reasoning}</think>\\n\\n\\boxed{{answer}}``, applies a per-question
solution cap, subsamples, splits into train/eval, and writes JSONL files
ready for ``trl.SFTTrainer`` (chat format).

The pure helpers (extraction, formatting, capping, writing) are tested on
synthetic data on a CPU laptop. The dataset download happens only when
``main()`` runs, typically on RCP. The ``HF_HOME`` environment variable
controls the cache location (set by ``rcp/submit_train.sh`` to
``/scratch/hf_cache``); the ``datasets`` library reads it automatically.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_BOXED_OPEN = re.compile(r"\\boxed\s*\{")


def extract_last_boxed(text: str) -> tuple[str, str] | None:
    """Return ``(text_before_boxed, content_inside_boxed)`` for the LAST
    ``\\boxed{...}`` in *text*, or ``None`` if none is found or the last
    one is unbalanced.

    Brace-depth counted manually so nested braces (``\\frac{a}{b}``) and
    escaped braces (``\\{``, ``\\}`` for set notation) are handled correctly.
    """
    matches = list(_BOXED_OPEN.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[: m.start()], text[start : i - 1]


def format_response(reasoning: str, answer: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>\n\n\\boxed{{{answer}}}"


def make_example(query: str, reasoning: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": format_response(reasoning, answer)},
        ]
    }


def apply_per_question_cap(
    rows: list[dict], cap: int, rng: random.Random
) -> list[dict]:
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")
    by_query: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_query[row["query"]].append(row)
    out: list[dict] = []
    for group in by_query.values():
        if len(group) > cap:
            rng.shuffle(group)
            group = group[:cap]
        out.extend(group)
    return out


def write_jsonl(examples: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def build_pipeline(
    raw_rows: Iterable[dict],
    *,
    n_samples: int,
    per_question_cap: int,
    eval_size: int,
    max_response_chars: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Filter → cap → subsample → shuffle → split.

    Returns ``(train_examples, eval_examples)`` as TRL chat dicts. Pure over
    the input iterable — no I/O, no dataset library needed.
    """
    rng = random.Random(seed)

    kept: list[dict] = []
    n_no_box = 0
    n_too_long = 0
    for row in raw_rows:
        response = row["response"]
        if len(response) > max_response_chars:
            n_too_long += 1
            continue
        result = extract_last_boxed(response)
        if result is None:
            n_no_box += 1
            continue
        reasoning, answer = result
        kept.append(
            {"query": row["query"], "reasoning": reasoning, "answer": answer}
        )
    logger.info(
        "filter: kept=%d dropped_no_box=%d dropped_too_long=%d",
        len(kept), n_no_box, n_too_long,
    )

    capped = apply_per_question_cap(kept, cap=per_question_cap, rng=rng)
    logger.info("per-question cap (cap=%d): kept=%d", per_question_cap, len(capped))

    rng.shuffle(capped)
    if n_samples < len(capped):
        capped = capped[:n_samples]
    logger.info("subsample (n=%d): kept=%d", n_samples, len(capped))

    if eval_size > len(capped):
        raise ValueError(
            f"eval_size ({eval_size}) exceeds rows after pipeline ({len(capped)})"
        )
    eval_rows = capped[:eval_size]
    train_rows = capped[eval_size:]

    train = [make_example(r["query"], r["reasoning"], r["answer"]) for r in train_rows]
    eval_ = [make_example(r["query"], r["reasoning"], r["answer"]) for r in eval_rows]
    return train, eval_


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data_out"))
    parser.add_argument("--n-samples", type=int, default=50_000)
    parser.add_argument("--per-question-cap", type=int, default=4)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-response-chars", type=int, default=8000)
    parser.add_argument("--dataset-name", default="hkust-nlp/dart-math-uniform")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Lazy import: keeps the unit tests free of the `datasets` dependency.
    from datasets import load_dataset

    logger.info("loading %s split=%s", args.dataset_name, args.split)
    ds = load_dataset(args.dataset_name, split=args.split)
    logger.info("loaded %d raw rows", len(ds))

    raw_rows = ({"query": r["query"], "response": r["response"]} for r in ds)
    train, eval_ = build_pipeline(
        raw_rows,
        n_samples=args.n_samples,
        per_question_cap=args.per_question_cap,
        eval_size=args.eval_size,
        max_response_chars=args.max_response_chars,
        seed=args.seed,
    )

    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "eval.jsonl"
    n_train = write_jsonl(train, train_path)
    n_eval = write_jsonl(eval_, eval_path)
    logger.info("wrote %s (n=%d) and %s (n=%d)", train_path, n_train, eval_path, n_eval)


if __name__ == "__main__":
    main()
