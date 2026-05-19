"""Distill a strong math teacher's CoT solutions into SFT training data.

Production-scale counterpart to ``scripts/teacher_smoke.py``: same
teacher (default ``Qwen/Qwen3-32B-AWQ``), same thinking-mode sampling
defaults, same well-formed regex check, but the output is SFT training
JSONL — one row per kept (correct + well-formatted) teacher sample,
ready to feed ``data/prepare_sft.py`` or ``scripts/train_sft.py``.

The output row schema is byte-compatible with what ``prepare_sft.py``'s
``make_example`` emits, so the distilled set can be concatenated with
OMI2- or DART-derived rows without further plumbing::

    {
      "messages": [
        {"role": "user", "content": <problem>},
        {"role": "assistant", "content": <teacher's raw completion>},
      ],
      "source": "teacher_qwen3_32b_awq",   # or whatever --source-tag
      "problem_idx": int,
      "sample_idx": int,
      "teacher_pass_at_n": float,
    }

Note on the assistant content: we keep the teacher's RAW completion
(not a reformatted ``<think>\\n{reasoning}\\n</think>\\n\\n\\boxed{...}``
canonicalization), because:

  1. Distillation semantically preserves the teacher's exact output.
  2. The well-formed filter guarantees the completion contains a
     ``<think>...</think>`` block AND a ``\\boxed{`` marker, so the
     trained student still learns the format.
  3. Whitespace / quirks are wrapped by our locked chat template at
     training time; they do not affect the SFT loss target.

Heavy ``vllm`` import is deferred into ``main()``; pure helpers
(``should_keep_problem``, ``build_distill_row``,
``format_progress_summary``, ``derive_source_tag``) live at module
scope and are CPU-testable.

CLI::

    python scripts/teacher_distill.py \\
        --teacher Qwen/Qwen3-32B-AWQ \\
        --problem-set <jsonl with {prompt,answer} per line> \\
        --output-file /scratch/Julien/teacher_distill/v7_solutions.jsonl \\
        --n 2 --min-correct 1 --quantization awq
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Pure helpers shared with teacher_smoke; both reuse eval_local's
# stdlib-only loaders. None of these pulls torch at import time.
from scripts.eval_local import load_eval_jsonl, normalize_input_row
from scripts.teacher_smoke import (
    SYSTEM_PROMPT,
    build_chat_messages,
    is_well_formatted,
)

logger = logging.getLogger("teacher_distill")

# Defaults aligned with teacher_smoke (Qwen3 thinking-mode contract),
# except max_model_len is doubled to give long thinking traces room.
DEFAULT_TEACHER = "Qwen/Qwen3-32B-AWQ"
DEFAULT_N = 2
DEFAULT_MIN_CORRECT = 1
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
DEFAULT_PRESENCE_PENALTY = 1.5
DEFAULT_MIN_P = 0.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85
DEFAULT_QUANTIZATION = "awq"
DEFAULT_SEED = 42
DEFAULT_CHUNK_SIZE = 50  # also the progress-logging cadence
DEFAULT_LOG_EVERY = 50


# =============================================================================
# Pure helpers — no torch / no vllm. CPU-testable.
# =============================================================================

def derive_source_tag(teacher: str) -> str:
    """Turn an HF model id into a stable snake_case source tag.

    ``Qwen/Qwen3-32B-AWQ`` → ``teacher_qwen3_32b_awq``. Drops the org
    prefix; lowercases; replaces dashes and dots with underscores.
    """
    base = teacher.rsplit("/", 1)[-1]
    slug = base.lower().replace("-", "_").replace(".", "_")
    return f"teacher_{slug}"


def should_keep_problem(c: int, min_correct: int) -> bool:
    """Per-problem filter: keep iff the teacher got ``c >= min_correct``.

    ``min_correct=1`` (default) keeps any problem where the teacher
    landed at least one correct sample — strong enough to be a useful
    distillation signal.
    """
    return c >= int(min_correct)


def build_distill_row(
    *,
    problem_idx: int,
    sample_idx: int,
    item: dict,
    teacher_solution: str,
    pass_at_n: float,
    source: str,
) -> dict:
    """Assemble one SFT-compatible row.

    Schema matches ``data/prepare_sft.make_example`` for the
    ``messages`` field (so distillation rows blend with OMI2/DART
    rows), plus minimal trace fields (``source``, ``problem_idx``,
    ``sample_idx``, ``teacher_pass_at_n``) for downstream auditing.
    """
    return {
        "messages": [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": teacher_solution.strip()},
        ],
        "source": source,
        "problem_idx": int(problem_idx),
        "sample_idx": int(sample_idx),
        "teacher_pass_at_n": float(pass_at_n),
    }


def format_progress_summary(
    *,
    n_processed: int,
    n_total: int,
    total_attempts: int,
    total_correct: int,
    total_keepers: int,
) -> str:
    pass_rate = total_correct / total_attempts if total_attempts else 0.0
    return (
        f"progress {n_processed}/{n_total} "
        f"({n_processed / n_total * 100:.1f}%) | "
        f"teacher_pass_rate={pass_rate:.3f} "
        f"({total_correct}/{total_attempts}) | "
        f"keepers={total_keepers}"
    )


# =============================================================================
# CLI / main — heavy imports deferred.
# =============================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Distill teacher CoT solutions into SFT training data.",
    )
    p.add_argument(
        "--teacher", default=DEFAULT_TEACHER,
        help=f"Teacher model HF id. Default: {DEFAULT_TEACHER}.",
    )
    p.add_argument(
        "--problem-set", type=Path, required=True,
        help="JSONL with {'prompt','answer'} per line.",
    )
    p.add_argument(
        "--output-file", type=Path, required=True,
        help="Output JSONL path; one row per kept teacher sample.",
    )
    p.add_argument(
        "--n", type=int, default=DEFAULT_N,
        help=f"Teacher samples per problem. Default: {DEFAULT_N}.",
    )
    p.add_argument(
        "--min-correct", type=int, default=DEFAULT_MIN_CORRECT,
        help=(
            f"Per-problem filter: keep only problems where teacher got "
            f">= --min-correct samples right. Default: {DEFAULT_MIN_CORRECT}."
        ),
    )
    p.add_argument(
        "--source-tag", default=None,
        help=(
            "Slug stored in each output row's 'source' field. Default: "
            "auto-derived from --teacher (e.g. Qwen/Qwen3-32B-AWQ → "
            "teacher_qwen3_32b_awq)."
        ),
    )
    p.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature. Default: {DEFAULT_TEMPERATURE}.",
    )
    p.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, dest="top_p",
    )
    p.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, dest="top_k",
    )
    p.add_argument(
        "--presence-penalty", type=float, default=DEFAULT_PRESENCE_PENALTY,
        dest="presence_penalty",
    )
    p.add_argument(
        "--min-p", type=float, default=DEFAULT_MIN_P, dest="min_p",
    )
    p.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"vLLM SamplingParams.max_tokens. Default: {DEFAULT_MAX_TOKENS}.",
    )
    p.add_argument(
        "--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN,
        help=(
            f"vLLM context window. Default: {DEFAULT_MAX_MODEL_LEN} "
            "(doubled from teacher_smoke's 4096 to give long thinking "
            "traces room)."
        ),
    )
    p.add_argument(
        "--gpu-memory-utilization", type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    p.add_argument(
        "--quantization", default=DEFAULT_QUANTIZATION,
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
    )
    p.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, dest="chunk_size",
        help=(
            f"Problems per vLLM batch (also progress-logging cadence). "
            f"Default: {DEFAULT_CHUNK_SIZE}. Larger = better throughput, "
            "longer between progress lines."
        ),
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

    source_tag = args.source_tag or derive_source_tag(args.teacher)
    logger.info("source tag: %s", source_tag)

    # 1. Load + normalize problems.
    raw_rows = load_eval_jsonl(args.problem_set)
    if not raw_rows:
        logger.error("problem set is empty: %s", args.problem_set)
        return 2
    items = [normalize_input_row(r) for r in raw_rows]
    logger.info("loaded %d problems from %s", len(items), args.problem_set)

    # 2. Load vLLM with AWQ quantization. Fail loudly on AWQ/OOM/HF.
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

    # 3. Chunked generation + scoring + keeper write. Output file is
    # opened once and rows are flushed per chunk so a partial run still
    # leaves usable data on disk.
    from evaluate.score import score_generations

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    total_attempts = 0
    total_correct = 0
    total_keepers = 0

    with open(args.output_file, "w", encoding="utf-8") as out_fp:
        for chunk_start in range(0, len(items), args.chunk_size):
            chunk = items[chunk_start:chunk_start + args.chunk_size]
            messages_per_item = [build_chat_messages(it["prompt"]) for it in chunk]

            try:
                outputs = llm.chat(messages_per_item, sampling)
            except Exception as e:
                logger.error(
                    "Generation failed on chunk starting at %d: %s",
                    chunk_start, e,
                )
                return 4

            completions_per_item = [
                [co.text for co in out.outputs] for out in outputs
            ]

            score_input = [
                {
                    "prompt": it["prompt"],
                    "answer": it["answer"],
                    "completions": comps,
                }
                for it, comps in zip(chunk, completions_per_item)
            ]
            try:
                result = score_generations(score_input, method="boxed")
            except SystemExit as e:
                logger.error("Scoring failed on chunk at %d: %s", chunk_start, e)
                return 5

            for chunk_idx, (it, comps, detail) in enumerate(zip(
                chunk, completions_per_item, result["detailed_results"],
            )):
                problem_idx = chunk_start + chunk_idx
                c = int(detail["c"])
                n = int(detail["n"])
                total_attempts += n
                total_correct += c

                if not should_keep_problem(c, min_correct=args.min_correct):
                    continue

                pass_at_n = c / n if n else 0.0
                for sample_idx, (comp, comp_detail) in enumerate(zip(
                    comps, detail["completions"],
                )):
                    if not comp_detail["correct"]:
                        continue
                    if not is_well_formatted(comp):
                        continue
                    row = build_distill_row(
                        problem_idx=problem_idx,
                        sample_idx=sample_idx,
                        item=it,
                        teacher_solution=comp,
                        pass_at_n=pass_at_n,
                        source=source_tag,
                    )
                    out_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total_keepers += 1

            out_fp.flush()
            n_processed = chunk_start + len(chunk)
            logger.info(format_progress_summary(
                n_processed=n_processed, n_total=len(items),
                total_attempts=total_attempts, total_correct=total_correct,
                total_keepers=total_keepers,
            ))

    # 4. Final summary.
    final_pass_rate = total_correct / total_attempts if total_attempts else 0.0
    print(
        f"teacher_distill: n_problems={len(items)}, "
        f"teacher_attempts={total_attempts}, "
        f"correct={total_correct} (pass_rate={final_pass_rate:.3f}), "
        f"kept_solutions={total_keepers}"
    )
    logger.info("wrote keepers to %s", args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
