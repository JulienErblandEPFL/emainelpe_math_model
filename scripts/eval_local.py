"""Local eval mirroring the CS-552 nightly CI for the math expert.

vLLM front-end + thin glue around the vendored CI scorer at ``evaluate/``.
Loads a merged checkpoint (or bare HF id), generates n=8 completions per
problem on the validation snapshot, dumps a generations JSONL, and pipes
that through ``evaluate.score.score_generations`` to report pass@1 / pass@8.

Pure helpers (``load_eval_jsonl``, ``normalize_input_row``,
``build_generations_dump``, ``resolve_sampling_params``,
``_check_max_model_len``, ``format_summary``, etc.) live at module scope and
are CPU-testable. Heavy imports (``torch`` / ``transformers`` / ``vllm``)
are deferred into ``main()`` and the runtime helpers so unit tests don't
pull those wheels.

CI contract values (n=8, seed=42) are pinned. Context caps are
two-mode:

  - **Default (CI-faithful)**: max_model_len=4096, max_tokens=4096.
    Matches the team project README's "Max model length 4096"
    (combined prompt + generation). This is the binding ceiling for
    any CI prediction; local pass@1 / pass@8 measured here are
    calibrated against what CI will see.
  - **--no-ci-mode (legacy escape hatch)**: max_model_len=20480,
    max_tokens=16384. Mirrors ``docs/project_description.pdf`` page 3
    (``Max new tokens: 16384``), the older course doc. More permissive
    than the CI; numbers measured under it *overstate* CI scores. Use
    only for ablations where the longer generation budget matters
    (e.g., probing whether the model would have boxed an answer given
    more tokens).

Explicit ``--max-model-len`` / ``--max-new-tokens`` always override the
mode-derived defaults. See CLAUDE.md "Eval contract" for the
README-vs-project_description.pdf conflict write-up.

Sampling defaults are three-tiered (highest priority first):

  1. CLI flags — ``--temperature`` / ``--top-p`` / ``--top-k`` if set.
     Each override is logged at WARNING.
  2. ``<model>/generation_config.json`` if present and contains any of
     the three sampling fields. (Lets us track whatever Stage 5 pushed.)
  3. Stage-4 hardcoded fallback: temp=0.3, top_p=0.95, top_k=20.

See CLAUDE.md "Eval contract" and IMPLEMENTATION_PLAN.md Stage 4 for the
underlying decisions.

Smoke run on RCP (Stage 4 "Done when" criterion):

    python scripts/eval_local.py \\
        --model Qwen/Qwen3-1.7B \\
        --output-dir runs/eval_baseline
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Put repo root on sys.path so ``from evaluate.score import ...`` works
# whether the script is invoked via ``python scripts/eval_local.py``
# (which prepends scripts/ to sys.path, hiding the evaluate package) or
# ``python -m scripts.eval_local``. Same idiom as scripts/tests/conftest.py.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# evaluate/ is pure Python (numpy + stdlib); cheap to import at module scope.
# Heavier imports (torch / transformers / vllm) stay deferred inside main().
from evaluate.extract_answer import extract_boxed_answer

logger = logging.getLogger(__name__)

DEFAULT_CHAT_TEMPLATE = REPO_ROOT / "chat_template" / "chat_template.jinja"
DEFAULT_EVAL_FILE = REPO_ROOT / "validation_samples" / "math.jsonl"

# Stage-4 fallback sampling values (math-tuned defaults from Stage 5 plan).
FALLBACK_TEMPERATURE = 0.3
FALLBACK_TOP_P = 0.95
FALLBACK_TOP_K = 20

# vLLM context-window defaults. Two modes; see module docstring.
#  - Default (CI-faithful): tracks the team project README's
#    max_model_len=4096 cap. max_tokens budget shrinks to whatever fits
#    within 4096 minus the rendered prompt; we set max_tokens=4096 and
#    let vLLM clamp.
#  - --no-ci-mode (legacy escape hatch): tracks
#    docs/project_description.pdf page 3 (max_new_tokens=16384).
#    max_model_len sized to fit prompt(≤4096) + max_new_tokens(16384).
CI_MAX_MODEL_LEN = 4096
CI_MAX_NEW_TOKENS = 4096
LEGACY_MAX_MODEL_LEN = 20480
LEGACY_MAX_NEW_TOKENS = 16384
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85

DEFAULT_N = 8
DEFAULT_SEED = 42


# =============================================================================
# Pure helpers — CPU-testable, no torch / transformers / vllm imports.
# =============================================================================

def load_eval_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file; skip blank lines; raise on malformed JSON."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}")
    return rows


def normalize_input_row(row: dict) -> dict:
    """Convert any supported input row to ``{"prompt": str, "answer": str}``.

    Two recognized schemas:

    - ``validation_samples/*.jsonl`` — ``{"prompt": str, "answer": str}``,
      passthrough.
    - ``data_out/eval.jsonl`` (output of ``data/prepare_sft.py``) —
      ``{"messages": [user_msg, assistant_msg]}``. Extract user content
      as the prompt; pull the last ``\\boxed{...}`` from assistant content
      as the gold answer using the same extractor the CI scorer uses, so
      input-side and scoring-side agree byte-for-byte.
    """
    if "prompt" in row and "answer" in row:
        return {"prompt": str(row["prompt"]), "answer": str(row["answer"])}
    if "messages" in row:
        return _normalize_from_messages(row["messages"])
    raise ValueError(
        f"unrecognized row schema; expected 'prompt'+'answer' or 'messages', "
        f"got keys={sorted(row.keys())}"
    )


def _normalize_from_messages(messages: list[dict]) -> dict:
    if len(messages) < 2:
        raise ValueError(
            f"messages list too short (need user+assistant): {messages}"
        )
    user_msg, asst_msg = messages[0], messages[1]
    if user_msg.get("role") != "user":
        raise ValueError(
            f"first message is not 'user' role: {user_msg.get('role')!r}"
        )
    if asst_msg.get("role") != "assistant":
        raise ValueError(
            f"second message is not 'assistant' role: {asst_msg.get('role')!r}"
        )

    prompt = str(user_msg["content"])
    answer = extract_boxed_answer(
        str(asst_msg["content"]), strip_double_curly_brace=True
    )
    if answer is None:
        raise ValueError(
            f"no \\boxed{{}} in assistant content: "
            f"{str(asst_msg['content'])[:120]!r}"
        )
    return {"prompt": prompt, "answer": answer}


def build_generations_dump(
    items: list[dict],
    completions_per_item: list[list[str]],
) -> list[dict]:
    """Combine input rows with their per-row completions into the JSONL shape
    ``evaluate.score.score_generations`` expects.

    Each output row carries the input's ``prompt`` and ``answer`` plus a
    ``completions`` list. All rows must have the same number of completions
    (n=8 in production). Validated here so a count mismatch surfaces before
    we try to score, not several minutes after vLLM finished generating.
    """
    if len(items) != len(completions_per_item):
        raise ValueError(
            f"items count ({len(items)}) != completions count "
            f"({len(completions_per_item)})"
        )
    if not items:
        return []

    n = len(completions_per_item[0])
    rows: list[dict] = []
    for i, (item, comps) in enumerate(zip(items, completions_per_item)):
        if len(comps) != n:
            raise ValueError(
                f"row {i}: {len(comps)} completions, expected {n} "
                f"(uniform n required by evaluate.score)"
            )
        rows.append({
            "prompt": item["prompt"],
            "answer": item["answer"],
            "completions": list(comps),
        })
    return rows


def write_generations_jsonl(rows: list[dict], path: Path) -> None:
    """Serialize rows as JSONL (one JSON object per line, UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_generation_config_from_model_dir(model_path: str) -> dict | None:
    """Try to read ``<model_path>/generation_config.json``.

    Returns the parsed dict if present, ``None`` otherwise. Returns ``None``
    when:

    - ``model_path`` is not a local directory (e.g., HF hub id like
      ``Qwen/Qwen3-1.7B``)
    - the directory exists but has no ``generation_config.json``
    - the file exists but is malformed JSON (logs a WARNING)
    """
    p = Path(model_path)
    if not p.is_dir():
        return None
    gc_path = p / "generation_config.json"
    if not gc_path.exists():
        return None
    try:
        return json.loads(gc_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.warning(
            "malformed generation_config.json at %s: %s; ignoring",
            gc_path, e,
        )
        return None


def resolve_sampling_params(
    args: argparse.Namespace,
    gen_config_dict: dict | None,
) -> dict:
    """Three-tier resolution of vLLM ``SamplingParams`` kwargs.

    Priority (highest first):

    1. CLI flags — ``--temperature`` / ``--top-p`` / ``--top-k`` if set.
       Each override is logged at WARNING so the operator notices the
       drift from the pushed-checkpoint contract.
    2. ``generation_config.json`` from the model dir, when ``gen_config_dict``
       contains any of the three sampling fields. (Lets local eval track
       whatever Stage 5 pushed.)
    3. Stage-4 hardcoded fallback (``FALLBACK_TEMPERATURE`` etc.).

    The CI-contract values ``n``, ``max_tokens``, ``seed`` are taken
    straight from ``args`` and not subject to fallback resolution.
    """
    if gen_config_dict and any(
        k in gen_config_dict for k in ("temperature", "top_p", "top_k")
    ):
        source = "generation_config.json"
        defaults = {
            "temperature": gen_config_dict.get("temperature", FALLBACK_TEMPERATURE),
            "top_p": gen_config_dict.get("top_p", FALLBACK_TOP_P),
            "top_k": gen_config_dict.get("top_k", FALLBACK_TOP_K),
        }
    else:
        source = "Stage-4 fallback (no generation_config.json with sampling fields found)"
        defaults = {
            "temperature": FALLBACK_TEMPERATURE,
            "top_p": FALLBACK_TOP_P,
            "top_k": FALLBACK_TOP_K,
        }
    logger.info("sampling defaults source: %s — %s", source, defaults)

    final = dict(defaults)
    for name in ("temperature", "top_p", "top_k"):
        cli_val = getattr(args, name)
        if cli_val is not None:
            logger.warning(
                "%s overridden via CLI: %s → %s "
                "(drift from pushed-checkpoint contract)",
                name, defaults[name], cli_val,
            )
            final[name] = cli_val

    final["n"] = args.n
    final["max_tokens"] = args.max_new_tokens
    final["seed"] = args.seed
    return final


def resolve_context_caps(
    *,
    legacy_mode: bool = False,
    max_model_len_arg: int | None = None,
    max_new_tokens_arg: int | None = None,
) -> tuple[int, int]:
    """Resolve ``(max_model_len, max_new_tokens)`` from CLI args.

    Two-source resolution (highest priority first):

    1. Explicit ``--max-model-len`` / ``--max-new-tokens`` if the user
       passed them.
    2. Mode-derived defaults: 20480/16384 under ``legacy_mode=True``
       (i.e., when ``--no-ci-mode`` is passed at the CLI), 4096/4096
       otherwise. The CI-faithful 4096/4096 is the *default* —
       calling this helper with no flags returns CI caps.

    The explicit overrides exist so an operator can run an ablation
    (e.g., ``--max-new-tokens 8192`` while staying in CI mode) without
    juggling two flags. Each override is independent — passing one does
    not affect the resolution of the other.
    """
    if legacy_mode:
        default_mml = LEGACY_MAX_MODEL_LEN
        default_mnt = LEGACY_MAX_NEW_TOKENS
    else:
        default_mml = CI_MAX_MODEL_LEN
        default_mnt = CI_MAX_NEW_TOKENS
    final_mml = max_model_len_arg if max_model_len_arg is not None else default_mml
    final_mnt = max_new_tokens_arg if max_new_tokens_arg is not None else default_mnt
    return final_mml, final_mnt


def _check_max_model_len(positional_ceiling: int, max_model_len: int) -> None:
    """Refuse to launch if the model can't support the requested context.

    Qwen3-1.7B has ``max_position_embeddings=40960`` (Qwen3 default), so
    ``DEFAULT_MAX_MODEL_LEN=20480`` fits easily. A misconfiguration that
    picks a smaller-context model would otherwise crash deep inside vLLM
    with a confusing error.
    """
    if positional_ceiling < max_model_len:
        raise RuntimeError(
            f"requested max_model_len={max_model_len} exceeds the model's "
            f"max_position_embeddings={positional_ceiling}. Either lower "
            f"--max-model-len or pick a model that supports the requested context."
        )


def format_summary(score_result: dict) -> str:
    """Render ``score_generations``'s output as a one-line summary.

    Format intentionally mirrors ``evaluate/score.py``'s own stdout line
    so re-running ``python -m evaluate.score`` on the dumped generations
    produces an identical summary.
    """
    metrics = score_result["metrics"]
    parts = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    return (
        f"{parts} "
        f"(n_problems={score_result['n_problems']}, "
        f"n_completions={score_result['n_completions']}, "
        f"method={score_result['benchmark_method']})"
    )


# =============================================================================
# Runtime helpers — heavy imports inside.
# =============================================================================

def load_tokenizer_with_locked_template(model: str, template_path: Path):
    """Load the tokenizer for ``model`` and overwrite ``chat_template`` with
    the locked Jinja. Same idiom as ``scripts/verify_chat_template.py`` and
    ``scripts/train_sft.py``."""
    from transformers import AutoTokenizer

    template = template_path.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.chat_template = template
    if tokenizer.chat_template != template:
        raise RuntimeError(
            "tokenizer.chat_template differs from the assigned string after "
            "assignment. Investigate before running eval."
        )
    return tokenizer


def assert_model_supports_max_len(model: str, max_model_len: int) -> None:
    """Pull ``max_position_embeddings`` from the model's HF config and call
    ``_check_max_model_len``."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model)
    ceiling = int(config.max_position_embeddings)
    logger.info("model %s max_position_embeddings=%d", model, ceiling)
    _check_max_model_len(ceiling, max_model_len)


def render_prompts(tokenizer, items: list[dict]) -> list[str]:
    """Wrap each item's ``prompt`` in a single user turn, apply the chat
    template with ``add_generation_prompt=True``, return the rendered
    strings. vLLM receives these pre-rendered, sidestepping its own
    chat-template handling."""
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for item in items
    ]


def run_vllm(
    model: str,
    prompts: list[str],
    sampling_params: dict,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> list[list[str]]:
    """Generate ``n`` completions per prompt with vLLM in bf16.

    Returns a list-of-lists: outer length matches ``prompts``, inner length
    matches ``sampling_params['n']``. vLLM preserves output order with
    respect to input prompts.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    params = SamplingParams(**sampling_params)
    outputs = llm.generate(prompts, params)
    return [[co.text for co in out.outputs] for out in outputs]


# =============================================================================
# CLI / main
# =============================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", required=True,
        help="HF hub id (e.g. 'Qwen/Qwen3-1.7B') or path to a local merged checkpoint.",
    )
    p.add_argument(
        "--eval-file", type=Path, default=DEFAULT_EVAL_FILE,
        help=(
            f"JSONL with {{'prompt','answer'}} or {{'messages'}} per line. "
            f"Default: {DEFAULT_EVAL_FILE.relative_to(REPO_ROOT)} "
            f"(course-vendored validation snapshot, N=10, OOD competition "
            f"stress test). For lower-variance in-distribution signal, "
            f"pass data_out/eval.jsonl (the DART held-out slice)."
        ),
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for generations.jsonl and scored.json.",
    )
    p.add_argument(
        "--chat-template", type=Path, default=DEFAULT_CHAT_TEMPLATE,
        help="Locked chat-template Jinja. Default: chat_template/chat_template.jinja.",
    )
    p.add_argument(
        "--no-ci-mode", action="store_true", dest="no_ci_mode",
        help=(
            "Drop the CI-faithful default and use the legacy permissive "
            f"caps instead: max_model_len={LEGACY_MAX_MODEL_LEN}, "
            f"max_tokens={LEGACY_MAX_NEW_TOKENS}. The CI-faithful default "
            f"is max_model_len={CI_MAX_MODEL_LEN}, max_tokens={CI_MAX_NEW_TOKENS} "
            "(matches the team project README's combined-context cap). "
            "Use --no-ci-mode only for ablations where the longer "
            "generation budget matters; numbers measured under it "
            "*overstate* CI scores."
        ),
    )
    p.add_argument(
        "--max-model-len", type=int, default=None,
        help=(
            f"vLLM context window. Overrides the mode-derived default "
            f"({CI_MAX_MODEL_LEN} CI-faithful, {LEGACY_MAX_MODEL_LEN} legacy)."
        ),
    )
    p.add_argument(
        "--gpu-memory-utilization", type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help=f"vLLM GPU memory fraction. Default: {DEFAULT_GPU_MEMORY_UTILIZATION}.",
    )
    p.add_argument(
        "--n", type=int, default=DEFAULT_N,
        help=f"Completions per problem (CI contract: {DEFAULT_N}).",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=None,
        help=(
            f"vLLM SamplingParams.max_tokens. Overrides the mode-derived "
            f"default ({CI_MAX_NEW_TOKENS} CI-faithful, "
            f"{LEGACY_MAX_NEW_TOKENS} legacy)."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"vLLM SamplingParams.seed (CI contract: {DEFAULT_SEED}).",
    )
    p.add_argument(
        "--temperature", type=float, default=None,
        help=f"Override sampling temperature; logs WARNING. Default: from "
             f"<model>/generation_config.json if present, else {FALLBACK_TEMPERATURE}.",
    )
    p.add_argument(
        "--top-p", type=float, default=None, dest="top_p",
        help=f"Override sampling top_p; logs WARNING. Default: from "
             f"<model>/generation_config.json if present, else {FALLBACK_TOP_P}.",
    )
    p.add_argument(
        "--top-k", type=int, default=None, dest="top_k",
        help=f"Override sampling top_k; logs WARNING. Default: from "
             f"<model>/generation_config.json if present, else {FALLBACK_TOP_K}.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 0. Resolve context caps from --no-ci-mode / explicit overrides.
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

    # 1. Load + normalize input.
    raw_rows = load_eval_jsonl(args.eval_file)
    if not raw_rows:
        raise SystemExit(f"eval file is empty: {args.eval_file}")
    items = [normalize_input_row(r) for r in raw_rows]
    logger.info("loaded %d eval items from %s", len(items), args.eval_file)

    # 2. Verify model supports the requested context window.
    assert_model_supports_max_len(args.model, args.max_model_len)

    # 3. Tokenizer + locked chat template; render prompts.
    tokenizer = load_tokenizer_with_locked_template(args.model, args.chat_template)
    prompts = render_prompts(tokenizer, items)

    # 4. Resolve sampling params (info + warnings logged inside).
    gen_config = load_generation_config_from_model_dir(args.model)
    sampling_params = resolve_sampling_params(args, gen_config)
    logger.info("sampling params: %s", sampling_params)

    # 5. Inference.
    completions_per_item = run_vllm(
        model=args.model,
        prompts=prompts,
        sampling_params=sampling_params,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # 6. Build + write generations dump.
    dump = build_generations_dump(items, completions_per_item)
    gen_path = args.output_dir / "generations.jsonl"
    write_generations_jsonl(dump, gen_path)
    logger.info("wrote generations to %s", gen_path)

    # 7. Score using the vendored CI scorer (byte-identical to nightly).
    from evaluate.score import score_generations
    result = score_generations(dump, method="boxed")

    # 8. Persist + print.
    scored_path = args.output_dir / "scored.json"
    scored_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("wrote scored results to %s", scored_path)
    print(format_summary(result))


if __name__ == "__main__":
    main()
