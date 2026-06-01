"""Sample a math model on a problem set and dump the failures.

Companion to ``scripts/run_eval.py``: same vLLM front-end, same
locked chat template, same ``evaluate.score.score_generations`` scoring
path, but the output is the *per-problem* failure list rather than the
pass@k summary.

A failure is a problem where ``c / n <= --failure-threshold``, where
``c`` is the number of correct completions (via the vendored CI scorer)
and ``n`` is the sample count per problem. Default ``threshold=0`` is
strict — a problem is a failure only if zero of n samples were
correct (pass@n = 0).

Pure helpers (``is_failure``, ``build_failure_rows``,
``format_failure_summary``, ``resolve_sampling_params``,
``write_failures_jsonl``) live at module scope and are CPU-testable.
Heavy runtime helpers are imported from ``scripts.run_eval`` and
defer ``torch`` / ``transformers`` / ``vllm`` into their bodies, so
laptop unit tests run without those wheels.

CLI::

    python scripts/sample_failures.py \\
        --model /scratch/Julien/merged/math_model_v5_omi2_100k \\
        --prompt-set <path to JSONL with prompts> \\
        --output-dir /scratch/Julien/failures/v5_math_train_failures \\
        --n 4 --temperature 0.4 --seed 42

Use cases:
  - Mine v5 failure modes on MATH-train to seed a v7 SFT mix.
  - Diff v5 failures vs v3 failures (run twice with --model-tag v5/v3
    and compare output JSONLs).
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

# Reuse run_eval's pure helpers (cheap stdlib only). Runtime helpers
# that need torch/transformers/vllm are imported inside main().
from scripts.run_eval import (
    DEFAULT_CHAT_TEMPLATE,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_SEED,
    FALLBACK_TOP_K,
    FALLBACK_TOP_P,
    build_generations_dump,
    load_eval_jsonl,
    load_generation_config_from_model_dir,
    normalize_input_row,
    resolve_context_caps,
    write_generations_jsonl,
)

logger = logging.getLogger("sample_failures")

DEFAULT_N = 4
DEFAULT_TEMPERATURE = 0.4
DEFAULT_FAILURE_THRESHOLD = 0.0
DEFAULT_MODEL_TAG = "model"


# =============================================================================
# Pure helpers — no torch / transformers / vllm. CPU-testable.
# =============================================================================

def is_failure(c: int, n: int, threshold: float) -> bool:
    """Return True if a problem is a failure under the threshold.

    Per-problem pass@n = ``c / n``. Failure iff that fraction is
    ``<= threshold``. Default threshold=0.0 → failure iff ``c == 0``
    (no correct completions across n samples).
    """
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1]; got {threshold}")
    return (c / n) <= threshold


def build_failure_rows(
    detailed_results: list[dict],
    items: list[dict],
    completions_per_item: list[list[str]],
    threshold: float,
    model_tag: str,
) -> list[dict]:
    """Filter detailed_results to failures and build the dump rows.

    Output schema (per failed problem)::

        {
          "prompt": str,
          "answer": str,
          "<tag>_pass_at_n": float,        # c / n
          "<tag>_completions": list[str],  # all n completions
        }

    All three input lists must be positionally aligned:
    ``detailed_results[i]`` corresponds to ``items[i]`` and
    ``completions_per_item[i]``.
    """
    if not (len(detailed_results) == len(items) == len(completions_per_item)):
        raise ValueError(
            f"length mismatch: detailed_results={len(detailed_results)}, "
            f"items={len(items)}, completions={len(completions_per_item)}"
        )

    pass_key = f"{model_tag}_pass_at_n"
    comps_key = f"{model_tag}_completions"

    failures: list[dict] = []
    for detail, item, comps in zip(detailed_results, items, completions_per_item):
        c = int(detail["c"])
        n = int(detail["n"])
        if is_failure(c, n, threshold):
            failures.append({
                "prompt": item["prompt"],
                "answer": item["answer"],
                pass_key: c / n,
                comps_key: list(comps),
            })
    return failures


def format_failure_summary(
    n_total: int, n_failures: int, threshold: float, n_samples: int,
) -> str:
    rate = n_failures / n_total if n_total else 0.0
    return (
        f"sample_failures: n_problems={n_total}, n_failures={n_failures} "
        f"(rate={rate:.3f}), threshold={threshold:g}, n={n_samples}"
    )


def write_failures_jsonl(rows: list[dict], path: Path) -> None:
    """Serialize failure rows as JSONL (one object per line, UTF-8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_sampling_params(
    args: argparse.Namespace, gen_config_dict: dict | None,
) -> dict:
    """Two-tier resolution for vLLM ``SamplingParams`` kwargs.

    Priority (highest first):

    1. CLI flags — ``--temperature`` / ``--top-p`` / ``--top-k`` if not
       at their default sentinel. ``--temperature`` defaults to
       ``DEFAULT_TEMPERATURE`` (0.4, matching v5's pushed
       ``generation_config.json``); ``--top-p`` / ``--top-k`` default
       to ``None`` so they fall through to the model's config.
    2. ``generation_config.json`` from the model dir (if present), then
       run_eval's ``FALLBACK_TOP_P`` / ``FALLBACK_TOP_K``.

    Unlike ``run_eval.resolve_sampling_params`` this does *not*
    WARNING-log on CLI presence — the failure-mining script's defaults
    are deliberately aligned with the v5 pushed sampling contract, so a
    matching CLI value is not "drift" worth flagging.
    """
    cfg = gen_config_dict or {}
    final = {
        "temperature": args.temperature,
        "top_p": (
            args.top_p if args.top_p is not None
            else cfg.get("top_p", FALLBACK_TOP_P)
        ),
        "top_k": (
            args.top_k if args.top_k is not None
            else cfg.get("top_k", FALLBACK_TOP_K)
        ),
        "n": args.n,
        "max_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    return final


# =============================================================================
# CLI / main — heavy imports deferred.
# =============================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample a math model and dump per-problem failures.",
    )
    p.add_argument(
        "--model", required=True,
        help="HF hub id or path to a local merged checkpoint.",
    )
    p.add_argument(
        "--prompt-set", type=Path, required=True,
        help=(
            "JSONL with {'prompt','answer'} per line (also accepts the "
            "{'messages': [...]} schema emitted by data/prepare_sft.py)."
        ),
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help=(
            "Directory for failures.jsonl, generations.jsonl, and scored.json. "
            "The latter two are kept so a downstream rerun of the CI scorer "
            "can reproduce the failure list without re-sampling the model."
        ),
    )
    p.add_argument(
        "--n", type=int, default=DEFAULT_N,
        help=f"Completions per problem (default: {DEFAULT_N}).",
    )
    p.add_argument(
        "--failure-threshold", type=float,
        default=DEFAULT_FAILURE_THRESHOLD,
        help=(
            "Per-problem pass@n cutoff. A problem is a failure iff "
            "(c / n) <= threshold. Default 0.0 (strict, c==0). Pass e.g. "
            "0.25 to also flag problems where 1-of-4 got it right."
        ),
    )
    p.add_argument(
        "--model-tag", default=DEFAULT_MODEL_TAG,
        help=(
            "Prefix for the per-problem output keys "
            "('<tag>_pass_at_n', '<tag>_completions'). Default: 'model'. "
            "Pass --model-tag v5 → 'v5_pass_at_n' / 'v5_completions'."
        ),
    )
    p.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=(
            f"Sampling temperature. Default {DEFAULT_TEMPERATURE} — matches "
            "the v5 / v6 pushed generation_config.json (see CLAUDE.md "
            "'Inference temperature')."
        ),
    )
    p.add_argument(
        "--top-p", type=float, default=None, dest="top_p",
        help=f"Override top_p. Default: from <model>/generation_config.json "
             f"if present, else {FALLBACK_TOP_P}.",
    )
    p.add_argument(
        "--top-k", type=int, default=None, dest="top_k",
        help=f"Override top_k. Default: from <model>/generation_config.json "
             f"if present, else {FALLBACK_TOP_K}.",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"vLLM SamplingParams.seed. Default {DEFAULT_SEED}.",
    )
    p.add_argument(
        "--chat-template", type=Path, default=DEFAULT_CHAT_TEMPLATE,
        help="Locked chat-template Jinja.",
    )
    p.add_argument(
        "--no-ci-mode", action="store_true", dest="no_ci_mode",
        help=(
            "Drop CI-faithful context caps (use legacy 20480/16384 "
            "instead). Off by default — CI-faithful is the predictive "
            "setting."
        ),
    )
    p.add_argument(
        "--max-model-len", type=int, default=None,
        help="vLLM context window override.",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=None,
        help="vLLM max_tokens override.",
    )
    p.add_argument(
        "--gpu-memory-utilization", type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 0. Resolve context caps (mode-derived; explicit overrides honored).
    args.max_model_len, args.max_new_tokens = resolve_context_caps(
        legacy_mode=args.no_ci_mode,
        max_model_len_arg=args.max_model_len,
        max_new_tokens_arg=args.max_new_tokens,
    )
    mode_label = "legacy" if args.no_ci_mode else "ci-faithful"
    logger.info(
        "context caps: max_model_len=%d, max_new_tokens=%d (mode=%s)",
        args.max_model_len, args.max_new_tokens, mode_label,
    )

    # 1. Load + normalize the prompt set.
    raw_rows = load_eval_jsonl(args.prompt_set)
    if not raw_rows:
        raise SystemExit(f"prompt set is empty: {args.prompt_set}")
    items = [normalize_input_row(r) for r in raw_rows]
    logger.info("loaded %d problems from %s", len(items), args.prompt_set)

    # 2. Runtime imports happen here (run_eval helpers).
    from scripts.run_eval import (
        assert_model_supports_max_len,
        load_tokenizer_with_locked_template,
        render_prompts,
        run_vllm,
    )

    assert_model_supports_max_len(args.model, args.max_model_len)
    tokenizer = load_tokenizer_with_locked_template(args.model, args.chat_template)
    prompts = render_prompts(tokenizer, items)

    # 3. Sampling params.
    gen_config = load_generation_config_from_model_dir(args.model)
    sampling_params = resolve_sampling_params(args, gen_config)
    logger.info("sampling params: %s", sampling_params)

    # 4. Generate.
    completions_per_item = run_vllm(
        model=args.model,
        prompts=prompts,
        sampling_params=sampling_params,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # 5. Score via the vendored CI scorer.
    dump = build_generations_dump(items, completions_per_item)
    gen_path = args.output_dir / "generations.jsonl"
    write_generations_jsonl(dump, gen_path)
    logger.info("wrote generations to %s", gen_path)

    from evaluate.score import score_generations
    result = score_generations(dump, method="boxed")
    scored_path = args.output_dir / "scored.json"
    scored_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("wrote scored results to %s", scored_path)

    # 6. Filter to failures + dump.
    failures = build_failure_rows(
        detailed_results=result["detailed_results"],
        items=items,
        completions_per_item=completions_per_item,
        threshold=args.failure_threshold,
        model_tag=args.model_tag,
    )
    failures_path = args.output_dir / "failures.jsonl"
    write_failures_jsonl(failures, failures_path)
    logger.info("wrote failures to %s", failures_path)

    print(format_failure_summary(
        n_total=len(items),
        n_failures=len(failures),
        threshold=args.failure_threshold,
        n_samples=args.n,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
