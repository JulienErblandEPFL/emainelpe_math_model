"""Curate the RLVR prompt set (Stage 7 / Phase 2).

Implements decision D3b (2026-05-09): score candidate prompts with the
SFT checkpoint, keep those whose empirical solve rate falls in the
[0.2, 0.8] difficulty band — the proposal's "20–80%" sweet spot where
GRPO has both a learning signal (positive rewards exist) and headroom
(the SFT model can't already solve it).

Input  : Stage 1's ``train.jsonl`` (``{"messages": [user, assistant]}``).
Output : RLVR prompt JSONL (``{"prompt", "answer", "solve_rate"}``).

Pure helpers (``difficulty_filter``, ``extract_prompt_and_gold``,
``validate_pool_row``, ``solve_rate``) are CPU-testable and live at
module scope. The heavy ML imports (``torch``, ``vllm``, ``transformers``)
are deferred into ``main()`` so the unit tests run on a laptop.

Why empirical ``solve_rate = c/n`` and not ``pass@8``: with n=k=8 the
unbiased Chen-2021 estimator is binary (1 if any correct, 0 if none),
which collapses the [0.2, 0.8] difficulty band to nothing. The empirical
solve rate keeps the granular signal we need for filtering.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

# evaluate is at the repo root; this script lives in data/ so add the
# repo root to sys.path before importing reward primitives.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluate.extract_answer import extract_boxed_answer  # noqa: E402

logger = logging.getLogger("prepare_rlvr")

DEFAULT_INPUT = Path("/scratch/Julien/data_out_v3/train.jsonl")
DEFAULT_OUTPUT = Path("/scratch/Julien/data_out_v3/rlvr_prompts.jsonl")
DEFAULT_SFT_MODEL = Path("/scratch/Julien/merged/math_model_v3")

DIFFICULTY_LO = 0.2
DIFFICULTY_HI = 0.8

# Sampling config for the difficulty-scoring inference pass. Matches the
# Stage 4 eval contract (n=8, max_tokens=4096) so the difficulty signal
# we measure here is comparable to what the CI sees.
SCORING_NUM_GENERATIONS = 8
SCORING_MAX_TOKENS = 4096
SCORING_TEMPERATURE = 0.8        # higher than CI's 0.3 — we want diverse
                                 # rollouts so c/n is informative
SCORING_SEED = 42


# =============================================================================
# Pure helpers — CPU-testable, no torch/vllm imports.
# =============================================================================

def solve_rate(num_correct: int, num_rollouts: int) -> float:
    """Empirical fraction-correct over rollouts.

    Distinct from ``evaluate.pass_at_k`` (which is the unbiased Chen-2021
    estimator). For difficulty filtering at n=k=8, c/n is what we want:
    pass@8 in that regime is binary and collapses the [0.2, 0.8] band.
    """
    if num_rollouts <= 0:
        raise ValueError(f"num_rollouts must be positive, got {num_rollouts}")
    return num_correct / num_rollouts


def difficulty_filter(
    rows: list[dict],
    *,
    lo: float = DIFFICULTY_LO,
    hi: float = DIFFICULTY_HI,
) -> list[dict]:
    """Keep rows whose ``solve_rate`` falls in ``[lo, hi]``.

    Inclusive on both ends — a problem the SFT model solves *exactly*
    20% of the time is still in the band.
    """
    if not 0.0 <= lo <= hi <= 1.0:
        raise ValueError(
            f"difficulty band must satisfy 0 <= lo <= hi <= 1, got [{lo}, {hi}]"
        )
    return [r for r in rows if lo <= r["solve_rate"] <= hi]


def extract_prompt_and_gold(messages: list[dict]) -> tuple[str, str] | None:
    """Pull the user prompt + gold answer out of a Stage 1 ``messages``
    pair.

    Returns ``None`` when the row's assistant turn has no parseable
    ``\\boxed{...}`` — those rows can't be used as RLVR prompts because
    we have no gold answer to score rollouts against.
    """
    if len(messages) < 2:
        return None
    user_msg = messages[0]
    asst_msg = messages[1]
    if user_msg.get("role") != "user" or asst_msg.get("role") != "assistant":
        return None
    prompt = user_msg.get("content")
    asst_content = asst_msg.get("content")
    if not isinstance(prompt, str) or not isinstance(asst_content, str):
        return None
    gold = extract_boxed_answer(asst_content, strip_double_curly_brace=True)
    if gold is None:
        return None
    return prompt, gold


def validate_pool_row(row: dict, line_no: int) -> list[dict]:
    """Validate a single Stage 1 input row, return [{prompt, answer}] or [].

    Rejects malformed rows with a logged WARNING (so the user can see how
    many rows are dropped at curation time) but does NOT raise — the
    pipeline keeps going. A whole-file rejection is surfaced by main()
    when zero rows survive.
    """
    if not isinstance(row, dict):
        logger.warning("line %d: not a JSON object, skipping", line_no)
        return []
    messages = row.get("messages")
    if not isinstance(messages, list):
        logger.warning(
            "line %d: missing/invalid 'messages' field, skipping", line_no,
        )
        return []
    pair = extract_prompt_and_gold(messages)
    if pair is None:
        logger.debug("line %d: cannot extract prompt+gold, skipping", line_no)
        return []
    prompt, gold = pair
    return [{"prompt": prompt, "answer": gold}]


def load_pool_jsonl(path: Path, *, max_rows: int | None = None) -> list[dict]:
    """Read Stage 1 ``train.jsonl`` and project to ``{prompt, answer}``.

    Drops malformed/un-parseable rows with a WARNING log. ``max_rows``
    caps the number of *valid* rows returned (not the lines read), so a
    pool with high reject rates still yields the requested count.
    """
    out: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning("line %d: not valid JSON (%s), skipping", line_no, e)
                continue
            for parsed in validate_pool_row(row, line_no):
                out.append(parsed)
                if max_rows is not None and len(out) >= max_rows:
                    return out
    return out


def write_jsonl(rows: Iterable[dict], path: Path) -> int:
    """Write ``rows`` as JSONL, return the count written.

    Creates the parent dir if missing. One JSON object per line, no
    trailing comma, no array wrapper — the format the rest of the repo
    (Stage 1 train, Stage 4 generations) reads.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


# =============================================================================
# CLI / main — heavy imports deferred into the body.
# =============================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-jsonl", type=Path, default=DEFAULT_INPUT,
        help=f"Stage 1 train.jsonl. Default is the v3 SFT pool — pure OMI2, "
             f"the post-2026-05-11 temperature-sweep choice "
             f"(default: {DEFAULT_INPUT}).",
    )
    p.add_argument(
        "--sft-model-path", type=Path, default=DEFAULT_SFT_MODEL,
        help="Path to the MERGED SFT checkpoint used to score difficulty. "
             "vLLM loads this — must be a complete model dir, not an adapter "
             "(different from train_rlvr.py's --adapter-dir, which takes the "
             "unmerged adapter). Default matches the v3 SFT winner.",
    )
    p.add_argument(
        "--output-jsonl", type=Path, default=DEFAULT_OUTPUT,
        help="Where to write the curated RLVR prompts. Default colocates "
             "with the v3 SFT data so a single --output-dir override on "
             "submit_rlvr.sh covers both.",
    )
    p.add_argument(
        "--pool-size", type=int, default=10000,
        help="Number of valid prompts to score (input pool size).",
    )
    p.add_argument(
        "--target-size", type=int, default=5000,
        help="Cap on the curated output size after difficulty filtering.",
    )
    p.add_argument(
        "--num-generations", type=int, default=SCORING_NUM_GENERATIONS,
        help="Rollouts per prompt for difficulty scoring (matches CI n=8).",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=SCORING_MAX_TOKENS,
        help="Per-rollout token budget (matches CI max_tokens=4096).",
    )
    p.add_argument(
        "--temperature", type=float, default=SCORING_TEMPERATURE,
        help="Rollout temperature for scoring; higher than CI's 0.3 so the "
             "solve_rate signal is informative rather than collapsed.",
    )
    p.add_argument("--seed", type=int, default=SCORING_SEED)
    p.add_argument(
        "--difficulty-lo", type=float, default=DIFFICULTY_LO,
        help="Inclusive lower bound for solve_rate (default 0.2).",
    )
    p.add_argument(
        "--difficulty-hi", type=float, default=DIFFICULTY_HI,
        help="Inclusive upper bound for solve_rate (default 0.8).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate config + load the input pool, but skip the GPU "
             "scoring pass. Writes a no-op summary to stdout.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="[%(name)s] %(levelname)s: %(message)s",
        stream=sys.stdout,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.log_level)

    if not args.input_jsonl.is_file():
        logger.error("Input pool not found: %s", args.input_jsonl)
        return 2

    logger.info("Loading pool from %s (max %d valid rows)",
                args.input_jsonl, args.pool_size)
    pool = load_pool_jsonl(args.input_jsonl, max_rows=args.pool_size)
    if not pool:
        logger.error(
            "Input pool yielded zero valid rows; check the JSONL schema."
        )
        return 3
    logger.info("Pool size after schema validation: %d", len(pool))

    if args.dry_run:
        print("=" * 60)
        print("prepare_rlvr.py --dry-run summary")
        print("=" * 60)
        print(f"  input_jsonl       : {args.input_jsonl}")
        print(f"  pool_size (valid) : {len(pool)}")
        print(f"  sft_model_path    : {args.sft_model_path}")
        print(f"  output_jsonl      : {args.output_jsonl} (NOT written)")
        print(f"  num_generations   : {args.num_generations}")
        print(f"  max_new_tokens    : {args.max_new_tokens}")
        print(f"  temperature       : {args.temperature}")
        print(f"  difficulty band   : [{args.difficulty_lo}, {args.difficulty_hi}]")
        print("Dry-run: skipping GPU scoring pass.")
        return 0

    if not args.sft_model_path.is_dir():
        logger.error("SFT model path not found: %s", args.sft_model_path)
        return 2

    # ---- GPU scoring pass --------------------------------------------------
    # Heavy imports deferred so the CPU helpers above remain testable on a
    # laptop without vLLM/torch wheels.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from scripts.reward_fn import compute_reward

    logger.info("Loading tokenizer + chat template from %s",
                args.sft_model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(args.sft_model_path))
    if not tokenizer.chat_template:
        logger.error(
            "Tokenizer at %s has no chat_template; refusing to score "
            "without one (would silently render no <think> markers).",
            args.sft_model_path,
        )
        return 4

    logger.info("Loading vLLM engine from %s", args.sft_model_path)
    llm = LLM(
        model=str(args.sft_model_path),
        dtype="bfloat16",
        max_model_len=args.max_new_tokens,
        gpu_memory_utilization=0.85,
    )
    sampling = SamplingParams(
        n=args.num_generations,
        temperature=args.temperature,
        top_p=0.95,
        top_k=20,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    rendered_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in pool
    ]

    logger.info("Generating %d rollouts/prompt × %d prompts",
                args.num_generations, len(pool))
    outputs = llm.generate(rendered_prompts, sampling)

    # vLLM may not return outputs in input order in some configs; we
    # built rendered_prompts in pool order and outputs[i].request_id
    # would tell us, but the standard behavior is index-aligned. Be
    # explicit about the assumption with a length check.
    if len(outputs) != len(pool):
        logger.error(
            "vLLM returned %d outputs for %d inputs; index alignment broken.",
            len(outputs), len(pool),
        )
        return 5

    scored: list[dict] = []
    for row, out in zip(pool, outputs):
        completions = [c.text for c in out.outputs]
        rewards = [compute_reward(c, row["answer"]) for c in completions]
        # solve_rate uses correctness only — format-only rewards (0.05) do
        # NOT count as solves. Otherwise the band would skew toward boxing
        # garbage rather than actually solving.
        n_correct = sum(1 for r in rewards if r >= 1.0)
        rate = solve_rate(n_correct, args.num_generations)
        scored.append({
            "prompt": row["prompt"],
            "answer": row["answer"],
            "solve_rate": rate,
        })

    logger.info("Scoring done; applying difficulty filter [%.2f, %.2f]",
                args.difficulty_lo, args.difficulty_hi)
    kept = difficulty_filter(scored, lo=args.difficulty_lo, hi=args.difficulty_hi)
    logger.info(
        "Difficulty filter: %d/%d kept (%.1f%%)",
        len(kept), len(scored), 100.0 * len(kept) / max(len(scored), 1),
    )

    if len(kept) > args.target_size:
        # Deterministic truncation — first N after the input order, which is
        # already shuffled by Stage 1's seed=42 subsample. Keeps the band's
        # difficulty distribution roughly uniform.
        kept = kept[: args.target_size]
        logger.info("Truncated to target_size=%d", args.target_size)

    n_written = write_jsonl(kept, args.output_jsonl)
    logger.info("Wrote %d rows to %s", n_written, args.output_jsonl)

    print("=" * 60)
    print("prepare_rlvr.py SUMMARY")
    print("=" * 60)
    print(f"  pool_size (valid) : {len(pool)}")
    print(f"  scored            : {len(scored)}")
    print(f"  kept (in band)    : {len(kept)}")
    print(f"  output_jsonl      : {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
