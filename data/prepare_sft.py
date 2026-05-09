"""Prepare math SFT data for ``trl.SFTTrainer`` (chat format).

v1: ``hkust-nlp/dart-math-uniform`` only — Stage 1 of the implementation
plan. Loads the dataset, normalizes each response to
``<think>{reasoning}</think>\\n\\n\\boxed{{answer}}``, applies a per-question
solution cap, subsamples, splits into train/eval, writes JSONL.

v2 (2026-05-09): adds ``nvidia/OpenMathInstruct-2`` as a second source.
The two are mixed (default 50/50) into a single ~50k-example output.

Why mix: OMI2's solutions come from Llama3.1-405B-Instruct, a
substantially stronger teacher than DART-Math's DeepSeekMath-7B-RL.
The proposal-anchored DART subset stays as the diversity / per-problem
multi-solution backbone; OMI2 brings stronger teacher CoT. Locked
decisions live in the ``--source mixed`` block in ``main()`` and in
``IMPLEMENTATION_PLAN.md`` Stage 1 v2.

Architecture notes:
- The OMI2 path normalizes its rows into the DART ``{query, response}``
  shape by appending ``\\boxed{expected_answer}`` to the cleaned
  ``generated_solution``. Reuses 100% of the existing extract/strip/
  cap/format pipeline (no re-implementation, no parallel branch).
- The new token-length filter (D4) takes a ``tokenize_fn`` callable so
  the heavy ``transformers`` import stays inside ``main()`` and the
  unit tests inject a fake counter. Existing DART tests pass an
  unchanged ``build_pipeline`` (the new kwargs default to ``None`` and
  are no-ops).

The pure helpers (extraction, formatting, capping, writing,
normalization) are tested on synthetic data on a CPU laptop. The
dataset download happens only when ``main()`` runs, typically on RCP.
The ``HF_HOME`` environment variable controls the cache location.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHAT_TEMPLATE = REPO_ROOT / "chat_template" / "chat_template.jinja"

# v2 default split (per the locked design): nvidia/OpenMathInstruct-2 ships
# pre-downsampled splits at 1M / 2M / 5M / 14M. ``train_1M`` is the smallest
# fair-downsampled slice; sufficient for a 25k subsample and downloads in
# minutes rather than hours.
OMI2_DEFAULT_SPLIT = "train_1M"

_BOXED_OPEN = re.compile(r"\\boxed\s*\{")

# Literal trailing fragments that show up right before \boxed{...} and
# become orphans once the boxed payload is extracted. Checked in this
# order; longer LaTeX openers must come before their shorter prefixes
# (e.g., '$$' before '$') so endswith picks the longer match.
_TRAILING_DELIMITERS: tuple[str, ...] = ("$$", "$", r"\[", r"\(")

# Answer-preamble phrases. Stored lowercase and matched
# case-insensitively against the suffix of the (rstripped) text. Order
# matters: the longer 'the answer is:' must be tried before the bare
# 'the answer is' so the colon-terminated form is consumed in one pass.
_TRAILING_PHRASES: tuple[str, ...] = (
    "the answer is:",
    "the answer is",
    "final answer:",
    "answer:",
)


def strip_trailing_preamble(text: str, max_iterations: int = 10) -> str:
    """Iteratively strip orphan math-mode delimiters and answer-preamble
    phrases from the END of *text*.

    Conservative by design: only literal patterns anchored to the suffix
    are removed. Patterns that appear mid-text are untouched. Phrase
    matches require the prior character to be whitespace (or the phrase
    to be the entire string) so we don't slice into a word.

    The cascade is iterative because patterns nest:
    ``"reasoning. The answer is: $"`` requires stripping ``$`` first,
    then ``The answer is:``. The ``max_iterations`` cap is defensive
    only; convergence is reached in <=4 passes for any plausible input.
    """
    for _ in range(max_iterations):
        new = _strip_one_pass(text)
        if new == text:
            break
        text = new
    return text.rstrip()


def _strip_one_pass(text: str) -> str:
    text = text.rstrip()
    if not text:
        return text
    for delim in _TRAILING_DELIMITERS:
        if text.endswith(delim):
            return text[: -len(delim)]
    lower = text.lower()
    for phrase in _TRAILING_PHRASES:
        if lower.endswith(phrase):
            start = len(text) - len(phrase)
            if start == 0 or text[start - 1].isspace():
                return text[:start]
    return text


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


def normalize_openmathinstruct_row(row: dict) -> dict:
    """Convert an ``nvidia/OpenMathInstruct-2`` row to DART ``{query, response}``.

    Schema in: ``{problem, generated_solution, expected_answer, problem_source}``.
    Schema out: ``{query, response}`` matching what DART's ``query`` /
    ``response`` columns produce, so the row can be fed through the same
    ``build_pipeline`` extract→strip→cap→format flow.

    Boxing strategy (decision D2, 2026-05-09): we always APPEND
    ``\\boxed{expected_answer}`` to the cleaned ``generated_solution``,
    regardless of whether the solution already contains a ``\\boxed{}``.
    The ``extract_last_boxed`` helper takes the LAST ``\\boxed{}``, so
    our appended one always wins; any mid-text ``\\boxed{}`` left in the
    solution becomes part of the reasoning, which is semantically fine.
    The result: every OMI2 row reliably extracts the gold answer, and
    the format matches the canonical training shape.
    """
    problem = row["problem"]
    solution = str(row["generated_solution"]).rstrip()
    answer = str(row["expected_answer"])
    response = f"{solution}\n\\boxed{{{answer}}}"
    return {"query": problem, "response": response}


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
    min_reasoning_chars: int,
    max_answer_chars: int,
    seed: int,
    max_formatted_tokens: int | None = None,
    tokenize_fn: Callable[[list[dict]], int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Filter → token-cap → cap → subsample → shuffle → split.

    Returns ``(train_examples, eval_examples)`` as TRL chat dicts. Pure over
    the input iterable — no I/O, no dataset library needed.

    Per-row pipeline:
      1. response length cap (pre-extract early-out, character-based)
      2. extract_last_boxed → (text_before_boxed, content_inside_boxed)
      3. answer length cap (max_answer_chars)
      4. strip_trailing_preamble on text_before_boxed
      5. reasoning length floor (min_reasoning_chars) — applied to the
         CLEANED text so e.g. "reasoning $" becomes "reasoning" before
         being measured
      6. (v2) token-length cap on the formatted message sequence.
         Only runs if BOTH ``max_formatted_tokens`` is set AND
         ``tokenize_fn`` is provided. Designed for OMI2's verbose
         Llama3.1-405B solutions but applies to any source. The
         ``tokenize_fn`` callable lets the heavy ``transformers`` import
         stay inside ``main()`` so unit tests inject a fake counter.

    Backward compatibility: when ``max_formatted_tokens`` and
    ``tokenize_fn`` are both ``None`` (the defaults), the v2 step is a
    no-op and the function behaves identically to the v1 pipeline.
    """
    rng = random.Random(seed)

    kept: list[dict] = []
    n_no_box = 0
    n_too_long = 0
    n_too_short_reasoning = 0
    n_too_long_answer = 0
    n_strip_called = 0
    n_strip_fired = 0
    for row in raw_rows:
        response = row["response"]
        if len(response) > max_response_chars:
            n_too_long += 1
            continue
        result = extract_last_boxed(response)
        if result is None:
            n_no_box += 1
            continue
        raw_reasoning, answer = result
        if len(answer) > max_answer_chars:
            n_too_long_answer += 1
            continue
        n_strip_called += 1
        reasoning = strip_trailing_preamble(raw_reasoning)
        if reasoning != raw_reasoning.rstrip():
            n_strip_fired += 1
        if len(reasoning) < min_reasoning_chars:
            n_too_short_reasoning += 1
            continue
        kept.append(
            {"query": row["query"], "reasoning": reasoning, "answer": answer}
        )
    logger.info(
        "filter: kept=%d dropped_no_box=%d dropped_too_long=%d "
        "dropped_too_short_reasoning=%d dropped_too_long_answer=%d",
        len(kept), n_no_box, n_too_long,
        n_too_short_reasoning, n_too_long_answer,
    )
    strip_pct = (n_strip_fired / n_strip_called * 100) if n_strip_called else 0.0
    logger.info(
        "[prepare_sft] strip_trailing_preamble fired on %d/%d rows (%.1f%%)",
        n_strip_fired, n_strip_called, strip_pct,
    )

    # v2: token-length filter. The character cap above (max_response_chars)
    # is a cheap pre-filter; this is the precise check that mirrors the
    # train-time filter in ``scripts/train_sft.filter_long_rows``. Running
    # it at data-prep time means oversized rows never touch the JSONL.
    if max_formatted_tokens is not None and tokenize_fn is not None:
        before = len(kept)
        n_dropped_too_many_tokens = 0
        kept_after_tokens = []
        for r in kept:
            messages = [
                {"role": "user", "content": r["query"]},
                {
                    "role": "assistant",
                    "content": format_response(r["reasoning"], r["answer"]),
                },
            ]
            n_tokens = tokenize_fn(messages)
            if n_tokens <= max_formatted_tokens:
                kept_after_tokens.append(r)
            else:
                n_dropped_too_many_tokens += 1
        kept = kept_after_tokens
        pct = (n_dropped_too_many_tokens / before * 100) if before else 0.0
        logger.info(
            "token-length filter (max=%d): kept=%d dropped=%d (%.1f%%)",
            max_formatted_tokens, len(kept), n_dropped_too_many_tokens, pct,
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


def resolve_n_samples(
    *,
    n_samples_arg: int | None,
    train_size_arg: int | None,
    eval_size: int,
    default_n_samples: int = 50_000,
) -> int:
    """Resolve the conflicting ``--n-samples`` / ``--train-size`` flags.

    - Both unset → fall back to ``default_n_samples`` (matches v1 behavior).
    - Only ``--n-samples`` → use it as-is (v1 semantics).
    - Only ``--train-size`` → ``n_samples = train_size + eval_size``
      (v2 semantics: train_size is the desired train-file row count).
    - Both set → raise ``ValueError`` (mutually exclusive).
    """
    if n_samples_arg is not None and train_size_arg is not None:
        raise ValueError(
            "--train-size and --n-samples are mutually exclusive; pick one."
        )
    if train_size_arg is not None:
        return train_size_arg + eval_size
    if n_samples_arg is not None:
        return n_samples_arg
    return default_n_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data_out"))
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Total rows after filtering, BEFORE the train/eval split. "
             "v1 semantics. Mutually exclusive with --train-size. "
             "Default 50000 when neither is set.",
    )
    parser.add_argument(
        "--train-size", type=int, default=None,
        help="Rows to write to train.jsonl (v2 semantics). Internally "
             "translates to --n-samples = --train-size + --eval-size. "
             "Mutually exclusive with --n-samples.",
    )
    parser.add_argument("--per-question-cap", type=int, default=4)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-response-chars", type=int, default=8000)
    parser.add_argument("--min-reasoning-chars", type=int, default=150)
    parser.add_argument("--max-answer-chars", type=int, default=200)
    parser.add_argument(
        "--max-formatted-tokens", type=int, default=None,
        help="v2 token-length cap (drops rows whose formatted chat exceeds "
             "this many Qwen3 tokens). Only enforced for --source mixed and "
             "--source openmathinstruct (default 3500). For --source dart "
             "the default stays None to keep v1 behavior byte-stable.",
    )
    parser.add_argument(
        "--source", choices=["dart", "openmathinstruct", "mixed"],
        default="dart",
        help="Data source. 'dart' = v1 default (unchanged). "
             "'openmathinstruct' = nvidia/OpenMathInstruct-2 only. "
             "'mixed' = ~50/50 v2 mix (controlled by --dart-fraction).",
    )
    parser.add_argument(
        "--dart-fraction", type=float, default=0.5,
        help="For --source mixed: fraction of the output coming from DART. "
             "Remainder comes from OpenMathInstruct-2.",
    )
    parser.add_argument("--dataset-name", default="hkust-nlp/dart-math-uniform")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--openmathinstruct-name", default="nvidia/OpenMathInstruct-2",
    )
    parser.add_argument(
        "--openmathinstruct-split", default=OMI2_DEFAULT_SPLIT,
    )
    parser.add_argument(
        "--chat-template", type=Path, default=DEFAULT_CHAT_TEMPLATE,
        help="Locked Qwen3 chat template (used to build the token "
             "counter when --max-formatted-tokens is set).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Resolve --n-samples vs --train-size before any data is loaded so a
    # CLI conflict surfaces as a clean error, not a silent split mismatch.
    try:
        n_samples = resolve_n_samples(
            n_samples_arg=args.n_samples,
            train_size_arg=args.train_size,
            eval_size=args.eval_size,
        )
    except ValueError as e:
        parser.error(str(e))

    # Auto-default the token cap for paths that include OMI2. Keeping the
    # DART-only path's default at None preserves v1 byte-stability — the
    # existing tests don't pass --max-formatted-tokens and don't expect
    # a token filter to fire.
    max_formatted_tokens = args.max_formatted_tokens
    if (
        max_formatted_tokens is None
        and args.source in ("openmathinstruct", "mixed")
    ):
        max_formatted_tokens = 3500

    # Build the tokenize_fn (heavy import) only if the filter will fire.
    tokenize_fn: Callable[[list[dict]], int] | None = None
    if max_formatted_tokens is not None:
        from transformers import AutoTokenizer
        chat_template = args.chat_template.read_text(encoding="utf-8")
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        tok.chat_template = chat_template
        if tok.chat_template != chat_template:
            raise RuntimeError(
                "tokenizer.chat_template differs from the assigned string after "
                "assignment; refusing to filter against a stale template."
            )

        def tokenize_fn(messages: list[dict]) -> int:
            return len(tok.apply_chat_template(messages, tokenize=True))

    # Lazy import: keeps the unit tests free of the `datasets` dependency.
    from datasets import load_dataset

    common_kwargs = dict(
        per_question_cap=args.per_question_cap,
        max_response_chars=args.max_response_chars,
        min_reasoning_chars=args.min_reasoning_chars,
        max_answer_chars=args.max_answer_chars,
        seed=args.seed,
        max_formatted_tokens=max_formatted_tokens,
        tokenize_fn=tokenize_fn,
    )

    if args.source == "dart":
        logger.info("loading %s split=%s", args.dataset_name, args.split)
        ds = load_dataset(args.dataset_name, split=args.split)
        logger.info("loaded %d raw DART rows", len(ds))
        raw_rows = ({"query": r["query"], "response": r["response"]} for r in ds)
        train, eval_ = build_pipeline(
            raw_rows,
            n_samples=n_samples,
            eval_size=args.eval_size,
            **common_kwargs,
        )

    elif args.source == "openmathinstruct":
        logger.info(
            "loading %s split=%s",
            args.openmathinstruct_name, args.openmathinstruct_split,
        )
        ds = load_dataset(
            args.openmathinstruct_name, split=args.openmathinstruct_split,
        )
        logger.info("loaded %d raw OpenMathInstruct-2 rows", len(ds))
        raw_rows = (normalize_openmathinstruct_row(r) for r in ds)
        train, eval_ = build_pipeline(
            raw_rows,
            n_samples=n_samples,
            eval_size=args.eval_size,
            **common_kwargs,
        )

    elif args.source == "mixed":
        # Per-source budgets — round to nearest int and let the leftover
        # land on OMI2. Both sources go through their own build_pipeline
        # so the per-question cap operates per-source (D3) and a problem
        # appearing in BOTH sources gets capped twice (once per source).
        if not 0.0 < args.dart_fraction < 1.0:
            parser.error(
                f"--dart-fraction must be strictly between 0 and 1, "
                f"got {args.dart_fraction}"
            )
        dart_n = int(round(n_samples * args.dart_fraction))
        omi_n = n_samples - dart_n
        dart_eval = int(round(args.eval_size * args.dart_fraction))
        omi_eval = args.eval_size - dart_eval
        logger.info(
            "mixed source budgets: DART(n=%d, eval=%d) + "
            "OpenMathInstruct-2(n=%d, eval=%d)",
            dart_n, dart_eval, omi_n, omi_eval,
        )

        logger.info("loading %s split=%s", args.dataset_name, args.split)
        dart_ds = load_dataset(args.dataset_name, split=args.split)
        logger.info("loaded %d raw DART rows", len(dart_ds))
        dart_rows = (
            {"query": r["query"], "response": r["response"]} for r in dart_ds
        )
        dart_train, dart_eval_ = build_pipeline(
            dart_rows,
            n_samples=dart_n + dart_eval,
            eval_size=dart_eval,
            **common_kwargs,
        )

        logger.info(
            "loading %s split=%s",
            args.openmathinstruct_name, args.openmathinstruct_split,
        )
        omi_ds = load_dataset(
            args.openmathinstruct_name, split=args.openmathinstruct_split,
        )
        logger.info("loaded %d raw OpenMathInstruct-2 rows", len(omi_ds))
        omi_rows = (normalize_openmathinstruct_row(r) for r in omi_ds)
        omi_train, omi_eval_ = build_pipeline(
            omi_rows,
            n_samples=omi_n + omi_eval,
            eval_size=omi_eval,
            **common_kwargs,
        )

        # Concatenate then shuffle so the train and eval JSONLs are not
        # source-grouped (TRL doesn't reshuffle between epochs by default).
        train = list(dart_train) + list(omi_train)
        eval_ = list(dart_eval_) + list(omi_eval_)
        rng = random.Random(args.seed)
        rng.shuffle(train)
        rng.shuffle(eval_)
        logger.info(
            "mixed merged: train=%d (DART %d + OMI2 %d), eval=%d (DART %d + OMI2 %d)",
            len(train), len(dart_train), len(omi_train),
            len(eval_), len(dart_eval_), len(omi_eval_),
        )

    else:
        parser.error(f"unknown --source {args.source!r}")

    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "eval.jsonl"
    n_train = write_jsonl(train, train_path)
    n_eval = write_jsonl(eval_, eval_path)
    logger.info("wrote %s (n=%d) and %s (n=%d)", train_path, n_train, eval_path, n_eval)


if __name__ == "__main__":
    main()
