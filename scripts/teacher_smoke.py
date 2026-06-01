"""Smoke-test a quantized teacher math model on a small problem set.

Loads a large math-tuned teacher (default ``Qwen/Qwen3-32B-AWQ``) via
vLLM with AWQ quantization and probes whether it produces
well-formatted ``<think>...</think>`` + ``\\boxed{...}`` solutions on
the local validation problems. Scoring goes through the same
``evaluate.score.score_generations`` path as ``scripts/run_eval.py``
so the teacher's pass rate is directly comparable to the student SFT
checkpoint's headline number.

Key design choices:

  - **Qwen3-32B-AWQ in thinking mode.** Qwen3's tokenizer chat template
    enables thinking mode by default, so the model emits
    ``<think>...</think>`` natively without any system-prompt
    engineering. The simplified system prompt asks only for
    ``\\boxed{}`` at the end. Defaults follow Qwen3's recommended
    sampling: ``temperature=0.6``, ``top_p=0.95``, ``top_k=20``,
    ``presence_penalty=1.5``, ``min_p=0.0``.
  - **No locked chat template.** The teacher uses its own bundled
    template via ``llm.chat(...)``; overriding it with our Qwen3-1.7B
    Jinja would break thinking mode.
  - **Generation cap.** ``max_tokens=4096`` lets Qwen3 emit long
    reasoning chains without running forever.

Pure helpers (``build_chat_messages``, ``is_well_formatted``,
``build_dump_row``, ``write_jsonl``, ``summarize``) are CPU-testable.
``torch`` / ``vllm`` imports are deferred into ``main()`` so unit
tests run on a laptop without those wheels.

CLI::

    python scripts/teacher_smoke.py \\
        --teacher Qwen/Qwen3-32B-AWQ \\
        --test-file validation_samples/math.jsonl \\
        --output-file /scratch/Julien/teacher_smoke_qwen3_32b/generations.jsonl \\
        --n 2 --temperature 0.6 --top-p 0.95 --top-k 20 \\
        --presence-penalty 1.5 --min-p 0.0 \\
        --quantization awq --gpu-memory-utilization 0.85
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse run_eval's stdlib-only JSONL helpers (no torch).
from scripts.run_eval import load_eval_jsonl, normalize_input_row

logger = logging.getLogger("teacher_smoke")

DEFAULT_TEACHER = "Qwen/Qwen3-32B-AWQ"
DEFAULT_TEST_FILE = REPO_ROOT / "validation_samples" / "math.jsonl"
DEFAULT_N = 2
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
DEFAULT_PRESENCE_PENALTY = 1.5
DEFAULT_MIN_P = 0.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85
DEFAULT_QUANTIZATION = "awq"
DEFAULT_SEED = 42

# Qwen3 emits <think>...</think> natively via its tokenizer chat
# template (thinking mode is on by default). The system prompt only
# needs to enforce the \boxed{} terminator the CI scorer extracts.
SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "Provide the final answer enclosed in \\boxed{}."
)

# Pre-compiled regexes for well-formed detection. DOTALL so a multiline
# reasoning span matches; non-greedy on <think>...</think> so chained
# segments don't gobble each other.
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{")


# =============================================================================
# Pure helpers — no torch / vllm. CPU-testable.
# =============================================================================

def build_chat_messages(problem: str) -> list[dict]:
    """Build the [system, user] message pair for the teacher.

    Returned shape is the OpenAI / Qwen chat schema that
    ``vllm.LLM.chat`` expects; vLLM will apply the teacher's own
    tokenizer chat template at generation time.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def is_well_formatted(text: str) -> bool:
    """True iff ``text`` contains BOTH a matched ``<think>...</think>``
    pair AND a ``\\boxed{`` marker.

    Note: an empty ``<think></think>`` still satisfies the regex —
    the smoke test cares whether the teacher *uses the format*, not
    whether it filled the reasoning span with useful tokens.
    """
    return bool(THINK_RE.search(text)) and bool(BOXED_RE.search(text))


def build_dump_row(
    item: dict,
    completions: list[str],
    n_correct: int,
) -> dict:
    """Assemble one output JSONL row.

    ``n_well_formatted`` is derived deterministically from the
    completion strings; ``n_correct`` is the per-problem ``c`` value
    from the CI scorer.
    """
    n_well_formatted = sum(
        1 for comp in completions if is_well_formatted(comp)
    )
    return {
        "prompt": item["prompt"],
        "answer": item["answer"],
        "teacher_completions": list(completions),
        "n_correct": int(n_correct),
        "n_well_formatted": int(n_well_formatted),
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict], n_samples: int) -> str:
    """One-line summary: total problems / generations / pass / format rate."""
    n_problems = len(rows)
    n_generations = n_problems * n_samples
    total_correct = sum(r["n_correct"] for r in rows)
    total_well_formatted = sum(r["n_well_formatted"] for r in rows)
    pass_rate = total_correct / n_generations if n_generations else 0.0
    fmt_rate = total_well_formatted / n_generations if n_generations else 0.0
    return (
        f"teacher_smoke: n_problems={n_problems}, "
        f"n_generations={n_generations}, "
        f"pass_rate={pass_rate:.3f}, format_rate={fmt_rate:.3f}"
    )


# =============================================================================
# CLI / main — heavy imports deferred.
# =============================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke-test a quantized teacher math model via vLLM.",
    )
    p.add_argument(
        "--teacher", default=DEFAULT_TEACHER,
        help=f"Teacher model HF id. Default: {DEFAULT_TEACHER}.",
    )
    p.add_argument(
        "--test-file", type=Path, default=DEFAULT_TEST_FILE,
        help=(
            f"JSONL with {{prompt,answer}} per line. Default: "
            f"{DEFAULT_TEST_FILE.relative_to(REPO_ROOT)}."
        ),
    )
    p.add_argument(
        "--output-file", type=Path, required=True,
        help="Path to write the generations JSONL.",
    )
    p.add_argument(
        "--n", type=int, default=DEFAULT_N,
        help=f"Samples per problem. Default: {DEFAULT_N}.",
    )
    p.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=(
            f"Temperature. Default: {DEFAULT_TEMPERATURE} (Qwen3 "
            "thinking-mode recommended)."
        ),
    )
    p.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, dest="top_p",
        help=f"vLLM SamplingParams.top_p. Default: {DEFAULT_TOP_P}.",
    )
    p.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, dest="top_k",
        help=f"vLLM SamplingParams.top_k. Default: {DEFAULT_TOP_K}.",
    )
    p.add_argument(
        "--presence-penalty", type=float, default=DEFAULT_PRESENCE_PENALTY,
        dest="presence_penalty",
        help=(
            f"vLLM SamplingParams.presence_penalty. Default: "
            f"{DEFAULT_PRESENCE_PENALTY} (Qwen3 thinking-mode recommended; "
            "discourages literal repetition in long reasoning chains)."
        ),
    )
    p.add_argument(
        "--min-p", type=float, default=DEFAULT_MIN_P, dest="min_p",
        help=f"vLLM SamplingParams.min_p. Default: {DEFAULT_MIN_P}.",
    )
    p.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=(
            f"vLLM SamplingParams.max_tokens. Default: {DEFAULT_MAX_TOKENS}. "
            "Qwen3 thinking-mode traces can be long; this cap stops runaway "
            "generations without truncating well-bounded reasoning."
        ),
    )
    p.add_argument(
        "--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN,
        help=f"vLLM context window. Default: {DEFAULT_MAX_MODEL_LEN}.",
    )
    p.add_argument(
        "--gpu-memory-utilization", type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help=(
            f"vLLM gpu_memory_utilization. Default: "
            f"{DEFAULT_GPU_MEMORY_UTILIZATION}."
        ),
    )
    p.add_argument(
        "--quantization", default=DEFAULT_QUANTIZATION,
        help=(
            f"vLLM quantization arg (passed straight to LLM(...)). "
            f"Default: {DEFAULT_QUANTIZATION}. The default teacher is an "
            "AWQ checkpoint and won't load without it."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"vLLM SamplingParams.seed. Default: {DEFAULT_SEED}.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # 1. Load + normalize problems.
    raw_rows = load_eval_jsonl(args.test_file)
    if not raw_rows:
        logger.error("test file is empty: %s", args.test_file)
        return 2
    items = [normalize_input_row(r) for r in raw_rows]
    logger.info("loaded %d problems from %s", len(items), args.test_file)

    # 2. Build per-problem chat messages (teacher's own template applied
    #    later by vLLM).
    messages_per_item = [build_chat_messages(it["prompt"]) for it in items]

    # 3. Load vLLM with AWQ quantization. Fail loudly on AWQ kernel
    #    errors, OOM, or HF download issues; the script's only output
    #    that matters is the generations JSONL, so a half-loaded teacher
    #    is worse than an early exit.
    try:
        from vllm import LLM, SamplingParams
        logger.info(
            "loading teacher=%s quantization=%s max_model_len=%d gpu_mem=%.2f",
            args.teacher, args.quantization, args.max_model_len,
            args.gpu_memory_utilization,
        )
        llm = LLM(
            model=args.teacher,
            quantization=args.quantization,
            dtype="auto",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error(
            "Failed to load teacher %s with quantization=%s: %s",
            args.teacher, args.quantization, e,
        )
        return 3

    sampling = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # 4. Generate via llm.chat — applies the teacher's own chat template
    #    (NOT our locked Qwen3-1.7B Jinja).
    logger.info("generating n=%d completions per problem", args.n)
    try:
        outputs = llm.chat(messages_per_item, sampling)
    except Exception as e:
        logger.error("Teacher generation failed: %s", e)
        return 4

    completions_per_item = [
        [co.text for co in out.outputs] for out in outputs
    ]

    # 5. Score via the vendored CI scorer (same path as run_eval).
    from evaluate.score import score_generations
    score_input = [
        {"prompt": it["prompt"], "answer": it["answer"], "completions": comps}
        for it, comps in zip(items, completions_per_item)
    ]
    try:
        result = score_generations(score_input, method="boxed")
    except SystemExit as e:
        # score_generations raises SystemExit on schema violations;
        # convert to a clear non-zero return.
        logger.error("Scoring failed: %s", e)
        return 5

    # 6. Build output rows + per-problem logging.
    rows = []
    for it, comps, detail in zip(
        items, completions_per_item, result["detailed_results"],
    ):
        row = build_dump_row(it, comps, n_correct=int(detail["c"]))
        rows.append(row)
        logger.info(
            "problem: prompt=%r n=%d n_correct=%d n_well_formatted=%d",
            it["prompt"][:60], args.n, row["n_correct"],
            row["n_well_formatted"],
        )

    # 7. Dump + summarize.
    write_jsonl(rows, args.output_file)
    logger.info("wrote %d rows to %s", len(rows), args.output_file)
    print(summarize(rows, n_samples=args.n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
