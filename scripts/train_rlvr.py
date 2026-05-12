"""RLVR training (Stage 7 / Phase 2) for the math expert.

**v0 — IMPLEMENTED 2026-05-09, NOT YET TRAINED.** The hyperparameters
below are conservative pre-fills from Tülu 3 + Dang & Ngo 2025; expect
to iterate. RLVR on small models is fragile (Dang & Ngo 2025) — this
script is a starting point, not a recipe for guaranteed gains.

Decisions encoded (2026-05-09):

  D1. TRL GRPOTrainer (consistency with the SFT pipeline).
  D2. Continue training the SFT LoRA adapter on top of Qwen3-1.7B base
      (Phase 3-merge-compatible).
  D3. Prompt set is the difficulty-band [0.2, 0.8] curation from
      ``data/prepare_rlvr.py``.
  D4. reward = 1.0 * correct + 0.05 * has_box, via ``scripts/reward_fn``.
  D5. lr=3e-6, beta(KL)=0.04, num_generations=8, rollout_temp=0.8,
      max_prompts=5000, max_new_tokens=4096, per-device-batch=1,
      grad_accum=8, epochs=1.

Critical preflights (P1/P2/P3) — see preflight checks in main():

  P1. SFT adapter loads + emits well-formed output (<think>, \\boxed{}).
  P2. Reward variance > threshold on a 10-prompt × 8-rollout sample.
  P3. KL divergence does not spike above ``KL_SPIKE_THRESHOLD`` in the
      first ``KL_SPIKE_WINDOW_STEPS`` steps (logged via callback).

Pure helpers (``grpo_config_kwargs``, ``check_reward_variance``,
``KLSpikeCallback._is_kl_spike``, ``default_run_name``,
``load_prompt_set_jsonl``, ``validate_max_new_tokens``) live at module
scope and are CPU-testable. The heavy ML imports (``torch``, ``peft``,
``trl``, ``transformers``) are deferred into ``main()``.

Saves the trained adapter to ``<output-dir>/final/`` with the SAME on-
disk shape as ``scripts/train_sft.py``, so ``scripts/merge_and_push.py
--adapter-dir <output-dir>/final`` will fold the RLVR-tuned adapter
into a deployable checkpoint without code changes.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path

import yaml

logger = logging.getLogger("train_rlvr")

REPO_ROOT = Path(__file__).resolve().parents[1]

# Put repo root on sys.path so ``from scripts.reward_fn import ...`` and
# ``from evaluate.X import ...`` work whether the script is invoked via
# ``python scripts/train_rlvr.py`` (which prepends scripts/ to sys.path,
# hiding the scripts package) or ``python -m scripts.train_rlvr``. Same
# idiom as scripts/eval_local.py and data/prepare_rlvr.py.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_LORA_YAML = REPO_ROOT / "configs" / "lora.yaml"
DEFAULT_CHAT_TEMPLATE = REPO_ROOT / "chat_template" / "chat_template.jinja"

# Defaults reflect the v3 SFT pipeline outputs as of 2026-05-11
# (post-temperature-sweep — v3 OMI2-only is the SFT winner).
DEFAULT_ADAPTER_DIR = Path(
    "/scratch/Julien/runs/cs552-erbland-g65-v3-omi2-fix2-20260511-152150/final"
)
DEFAULT_PROMPT_SET = Path("/scratch/Julien/data_out/rlvr_prompts.jsonl")

WANDB_PROJECT_DEFAULT = "emainelpe-math"

# Preflight thresholds. Tuned conservatively; flip via CLI if a
# legitimate run is being false-positive-rejected.
REWARD_VARIANCE_THRESHOLD = 0.01     # std² across 10 prompts × 8 rollouts
KL_SPIKE_THRESHOLD = 0.5             # Dang & Ngo 2025 small-model warn line
KL_SPIKE_WINDOW_STEPS = 100          # first N optimizer steps watched


# =============================================================================
# Pure helpers — CPU-testable, no torch/peft/trl imports.
# =============================================================================

def load_lora_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_chat_template(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def default_run_name(*, now: _dt.datetime | None = None) -> str:
    when = now or _dt.datetime.now()
    return f"rlvr-{when.strftime('%Y%m%d-%H%M')}"


def validate_max_new_tokens(
    max_new_tokens: int, *, training_seq_length: int, ci_eval_cap: int
) -> list[str]:
    """Return a list of warning strings for ``--max-new-tokens`` choices.

    No exception raised — these are advisories; the operator may know
    something the script doesn't (e.g. an ablation deliberately exceeding
    the CI cap to measure what's left on the table).
    """
    warns: list[str] = []
    if max_new_tokens > ci_eval_cap:
        warns.append(
            f"--max-new-tokens={max_new_tokens} exceeds the CI eval cap "
            f"({ci_eval_cap}). The model will sample tokens the CI will "
            "never see. Use only for ablations."
        )
    if max_new_tokens > training_seq_length:
        warns.append(
            f"--max-new-tokens={max_new_tokens} exceeds the training "
            f"max_seq_length ({training_seq_length}); ROPE behavior past "
            "the trained range is undefined."
        )
    return warns


def check_reward_variance(
    rewards: list[list[float]],
    *,
    threshold: float = REWARD_VARIANCE_THRESHOLD,
) -> tuple[bool, float]:
    """Verify P2: per-prompt reward variance is non-trivially non-zero.

    Computes mean per-prompt variance across the rollouts (each inner
    list = the n_generations rollouts for one prompt). Returns
    ``(passed, mean_variance)``.

    GRPO computes advantages as ``(r - mean(r)) / std(r)`` per prompt,
    so per-prompt variance is what matters; cross-prompt variance is a
    red herring (you can have prompts that all-pass and prompts that
    all-fail and *zero* useful signal).
    """
    if not rewards:
        raise ValueError("rewards must contain at least one prompt's rollouts")
    per_prompt_var = []
    for r_list in rewards:
        if len(r_list) < 2:
            raise ValueError(
                f"each prompt needs ≥2 rollouts to compute variance, got {len(r_list)}"
            )
        m = sum(r_list) / len(r_list)
        v = sum((r - m) ** 2 for r in r_list) / len(r_list)
        per_prompt_var.append(v)
    mean_var = sum(per_prompt_var) / len(per_prompt_var)
    return mean_var >= threshold, mean_var


def grpo_config_kwargs(
    *,
    args,
    yaml_dict: dict,
    precision: str,
    run_name: str,
    use_wandb: bool,
) -> dict:
    """Map CLI args + locked yaml + dtype → ``trl.GRPOConfig`` kwargs.

    Mirrors the structure of ``scripts.train_sft.sft_config_kwargs``.
    GRPO-specific knobs:

      - ``beta``: KL coefficient (Tülu 3 default 0.04).
      - ``num_generations``: rollouts per prompt (matches CI n=8).
      - ``temperature``: rollout sampling temp; intentionally separate
        from the ``generation_config.json`` written at Stage 5 (which
        is the eval-time temperature for CI inference).
      - ``max_completion_length``: per-rollout token budget.

    Prompt vs. completion budget. ``yaml_dict["max_seq_length"]=4096``
    is the SFT *training* sequence cap, not Qwen3-1.7B's hard context
    (which is ~32k). For RLVR rollouts we keep ``max_completion_length``
    pinned at the CI ``max_tokens=4096`` to mirror eval. The previously-
    set ``max_prompt_length`` knob was dropped on 2026-05-12 because the
    course-image TRL 0.19.1 ``GRPOConfig.__init__`` rejects it as an
    unexpected keyword argument (verified via ``inspect.signature``).
    Prompt-length truncation now defers to the tokenizer's own limits,
    which is fine — the curated math prompts are comfortably short (the
    ``--max-prompt-length`` v0 default of 1024 was never tight).

    ``per_device_train_batch_size`` here is *prompts per step*, not
    rollouts; total trajectories per gradient update is
    ``per_device_batch * gradient_accumulation_steps * num_generations``.
    With the v0 defaults: 1 × 8 × 8 = 64 trajectories/update.
    """
    return {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        # NOTE: max_prompt_length intentionally omitted — TRL 0.19.1's
        # GRPOConfig rejects it. See docstring above.
        "max_completion_length": args.max_new_tokens,
        "num_generations": args.num_generations,
        "temperature": args.rollout_temp,
        "top_p": 0.95,
        "top_k": 20,
        "beta": args.kl_coef,
        "bf16": precision == "bf16",
        "fp16": precision == "fp16",
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": 1,             # log every step — RL runs are short
        "save_strategy": "steps",
        "save_steps": 50,
        "save_total_limit": 2,
        "report_to": "wandb" if use_wandb else "none",
        "run_name": run_name,
        "seed": args.seed,
    }


# Same markers as data/prepare_rlvr (intentionally re-declared so the
# runtime sanity check doesn't pull data/ into the import graph). Keep
# in sync with data/prepare_rlvr.CHAT_TEMPLATE_OPEN_MARKER / THINK_PREFIX.
CHAT_TEMPLATE_OPEN_MARKER = "<|im_start|>"
THINK_PREFIX = "<think>\n"


def assert_prompts_are_chat_templated(
    prompts: list[dict], *, sample_size: int = 5
) -> None:
    """Defend against the 2026-05-12 retry2 and retry3 incidents.

    Curated prompts MUST satisfy two invariants:

      1. ``CHAT_TEMPLATE_OPEN_MARKER`` present — the prompt is the
         output of ``tokenizer.apply_chat_template(...)``, not raw
         text. Raw text → no ``<|im_end|>`` → 100% token-cap clipping
         → ``reward_std`` collapses to 0 (retry2, 2026-05-12).
      2. Prompt ends with ``THINK_PREFIX`` — the v3 SFT model was
         trained on assistant turns starting with ``<think>\\n``, but
         the locked chat template's generation prompt omits it.
         Without this prefix, rollouts at temp=0.8 sometimes skip
         ``<think>``, drop out of the trained regime, and never
         terminate (retry3, 2026-05-12).

    We check the first ``sample_size`` prompts and raise with a
    precise pointer to ``data/prepare_rlvr.build_scored_row`` if any
    fail either invariant.
    """
    if not prompts:
        return
    for idx, row in enumerate(prompts[:sample_size]):
        prompt = row.get("prompt", "")
        if CHAT_TEMPLATE_OPEN_MARKER not in prompt:
            preview = prompt[:120].replace("\n", "\\n")
            raise RuntimeError(
                f"Prompt set is NOT chat-templated. Row {idx} starts "
                f"with: {preview!r}. Expected the output of "
                f"tokenizer.apply_chat_template(...). This causes "
                f"100% token-cap clipping and reward_std=0 during "
                f"GRPO (see 2026-05-12 retry2 incident). Re-run "
                f"data/prepare_rlvr.py with the post-2026-05-12 fix "
                f"to regenerate rlvr_prompts.jsonl."
            )
        if not prompt.endswith(THINK_PREFIX):
            preview_tail = prompt[-120:].replace("\n", "\\n")
            raise RuntimeError(
                f"Prompt set is missing the {THINK_PREFIX!r} suffix. "
                f"Row {idx} ends with: {preview_tail!r}. The v3 SFT "
                f"model was trained on assistant turns starting with "
                f"'<think>\\n', but the chat template's generation "
                f"prompt does NOT emit it. Without this suffix, "
                f"rollouts at temp=0.8 unreliably skip <think> and "
                f"never terminate (see 2026-05-12 retry3 incident). "
                f"Re-run data/prepare_rlvr.py — build_scored_row "
                f"enforces the suffix at curation time."
            )


def load_prompt_set_jsonl(
    path: Path, *, max_prompts: int | None = None
) -> list[dict]:
    """Read the curated RLVR prompt JSONL produced by ``data/prepare_rlvr``.

    Schema enforced: each line must have ``prompt: str`` and ``answer: str``.
    Raises ``ValueError`` (not WARN) on malformed rows because the curation
    script already log-and-skipped the dirty input — anything that reaches
    this layer is expected to be clean.
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
                raise ValueError(
                    f"{path}:{line_no} not valid JSON: {e}"
                ) from e
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} not a JSON object")
            prompt = row.get("prompt")
            answer = row.get("answer")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(
                    f"{path}:{line_no} missing/empty 'prompt' (str required)"
                )
            if not isinstance(answer, str) or not answer:
                raise ValueError(
                    f"{path}:{line_no} missing/empty 'answer' (str required)"
                )
            out.append({"prompt": prompt, "answer": answer})
            if max_prompts is not None and len(out) >= max_prompts:
                break
    return out


def choose_precision(
    requested: str, *, cuda_available: bool, bf16_supported: bool
) -> str:
    """Same shape as ``scripts.train_sft.choose_precision``."""
    if not cuda_available:
        raise RuntimeError(
            "CUDA is not available. RLVR training requires a GPU; refusing "
            "to fall back to CPU silently."
        )
    if requested == "auto":
        return "bf16" if bf16_supported else "fp16"
    if requested == "bf16":
        if not bf16_supported:
            raise RuntimeError(
                "Requested --precision bf16 but the GPU does not advertise "
                "bf16 support."
            )
        return "bf16"
    if requested == "fp16":
        return "fp16"
    raise ValueError(f"unknown precision: {requested!r}")


# =============================================================================
# CLI / main
# =============================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR,
        help="Trained SFT adapter dir (PeftModel.from_pretrained input). "
             "Default is the v3 SFT winner — pure OMI2, the post-2026-05-11 "
             "temperature-sweep choice. NOT the merged checkpoint dir; this "
             "must contain adapter_config.json + adapter_model.safetensors. "
             f"Default: {DEFAULT_ADAPTER_DIR}",
    )
    p.add_argument(
        "--prompt-set", type=Path, default=DEFAULT_PROMPT_SET,
        help="Curated RLVR prompts JSONL (from data/prepare_rlvr.py).",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--learning-rate", type=float, default=3e-6)
    p.add_argument("--kl-coef", type=float, default=0.04,
                   help="GRPO beta — KL coefficient. Tülu 3 default 0.04.")
    p.add_argument("--rollout-temp", type=float, default=0.8,
                   help="Rollout sampling temperature; separate from the "
                        "eval-time temp in generation_config.json.")
    p.add_argument("--num-generations", type=int, default=8,
                   help="Rollouts per prompt; matches CI n=8.")
    p.add_argument("--max-prompts", type=int, default=5000,
                   help="Cap on prompt-set rows used (one epoch).")
    p.add_argument("--max-new-tokens", type=int, default=4096,
                   help="Per-rollout token budget; matches CI max_tokens=4096.")
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--precision", choices=["auto", "bf16", "fp16"], default="auto",
    )
    p.add_argument("--run-name", default=None)
    p.add_argument("--lora-yaml", type=Path, default=DEFAULT_LORA_YAML)
    p.add_argument("--chat-template", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    p.add_argument(
        "--variance-preflight-prompts", type=int, default=10,
        help="P2 preflight: number of prompts to sample for the reward-"
             "variance check before training begins.",
    )
    p.add_argument(
        "--skip-preflights", action="store_true",
        help="Skip P1/P2 preflights. ONLY for debugging the trainer wiring; "
             "running real RLVR with this set is unsupported.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate config + load prompts + print the resolved GRPOConfig "
             "values, then exit. No model loading, no GPU work.",
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


# =============================================================================
# Runtime helpers — used by main(); import torch/transformers internally.
# =============================================================================

def smoke_inference_p1(model, tokenizer, *, max_new_tokens: int = 2048) -> str:
    """P1: sanity-check the loaded adapter emits well-formed output.

    Same shape as ``train_sft.smoke_inference``. Aborts via RuntimeError
    if the output is missing ``<think>`` or ``\\boxed{``. Greedy decode
    so the smoke result is reproducible across runs.
    """
    import torch

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=False)
    logger.info("=== P1 smoke ===\n%s\n=== end P1 ===", decoded)
    if "<think>" not in decoded:
        raise RuntimeError(
            "P1 preflight FAILED: smoke output missing <think>. "
            "The starting adapter is broken; refusing to start training."
        )
    if r"\boxed{" not in decoded:
        raise RuntimeError(
            "P1 preflight FAILED: smoke output missing \\boxed{. "
            "The starting adapter is not emitting reward-scoreable answers; "
            "refusing to start training."
        )
    return decoded


def reward_variance_preflight_p2(
    model, tokenizer, prompts: list[dict], *,
    num_generations: int, rollout_temp: float, max_new_tokens: int, seed: int,
) -> tuple[bool, float]:
    """P2: rollouts on a small sample must have non-trivial reward variance.

    Without this, GRPO's advantage = (r - mean(r)) / std(r) is 0/0 and
    training is silent garbage. BASELINE.md explicitly flagged this as
    a real risk for our checkpoint at low sampling temperatures.
    """
    import torch

    from scripts.reward_fn import compute_reward

    rewards: list[list[float]] = []
    for row in prompts:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=rollout_temp,
                top_p=0.95,
                top_k=20,
                num_return_sequences=num_generations,
                # PyTorch's seed is set by the trainer; passing it here
                # would force identical rollouts across prompts.
            )
        # Decode only the new tokens (everything after the prompt length).
        prompt_len = inputs["input_ids"].shape[1]
        completions = [
            tokenizer.decode(seq[prompt_len:], skip_special_tokens=False)
            for seq in out
        ]
        per_prompt = [compute_reward(c, row["answer"]) for c in completions]
        rewards.append(per_prompt)
        logger.info(
            "P2 prompt rewards (mean=%.3f, std=%.3f): %s",
            sum(per_prompt) / len(per_prompt),
            (sum((r - sum(per_prompt)/len(per_prompt))**2 for r in per_prompt)
             / len(per_prompt)) ** 0.5,
            [f"{r:.2f}" for r in per_prompt],
        )

    passed, mean_var = check_reward_variance(rewards)
    return passed, mean_var


# =============================================================================
# KL spike monitor (P3) — wired into the trainer as a callback.
# =============================================================================

def _is_kl_spike(
    kl_value: float, step: int,
    *,
    threshold: float = KL_SPIKE_THRESHOLD,
    window: int = KL_SPIKE_WINDOW_STEPS,
) -> bool:
    """Pure helper: is this step's KL above the spike threshold within
    the early-training watch window?

    Extracted as a function so the test suite can pin the boundary
    behavior (off-by-one, exact-threshold equality) without spinning up
    a TRL trainer.
    """
    if step >= window:
        return False
    return kl_value > threshold


def _build_kl_spike_callback():
    """Construct the TrainerCallback subclass.

    Done inside a function (not at module scope) so the
    ``transformers.TrainerCallback`` import stays out of CPU-only test
    environments.
    """
    from transformers import TrainerCallback

    class KLSpikeCallback(TrainerCallback):
        """P3 monitor: WARN if KL divergence spikes early in training.

        Doesn't abort the run — Dang & Ngo 2025 reports cases where the
        KL recovers after a brief spike — but it does flag prominently
        in stdout (and stderr-tagged in W&B if available). Operator
        decides whether to early-stop.
        """

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            kl = logs.get("kl") or logs.get("objective/kl")
            if kl is None:
                return
            if _is_kl_spike(kl, state.global_step):
                logger.warning(
                    "P3 ALERT: KL=%.3f at step %d exceeds %.2f within the "
                    "first %d steps. Dang & Ngo 2025 small-model "
                    "instability signal — consider lowering --learning-rate "
                    "or raising --kl-coef.",
                    kl, state.global_step, KL_SPIKE_THRESHOLD,
                    KL_SPIKE_WINDOW_STEPS,
                )

    return KLSpikeCallback


# =============================================================================
# main()
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.log_level)

    # Locked configs first — pure-Python so any drift surfaces immediately.
    lora_yaml = load_lora_yaml(args.lora_yaml)
    chat_template = load_chat_template(args.chat_template)

    # Sanity-check max-new-tokens vs training/CI caps.
    for w in validate_max_new_tokens(
        args.max_new_tokens,
        training_seq_length=lora_yaml["max_seq_length"],
        ci_eval_cap=4096,
    ):
        logger.warning(w)

    # Load the curated prompt set up-front so dry-run can validate it.
    if not args.prompt_set.is_file():
        logger.error("Prompt set not found: %s", args.prompt_set)
        return 2
    try:
        prompts = load_prompt_set_jsonl(args.prompt_set, max_prompts=args.max_prompts)
    except ValueError as e:
        logger.error("Prompt set malformed: %s", e)
        return 3
    if not prompts:
        logger.error("Prompt set is empty after load: %s", args.prompt_set)
        return 3
    logger.info("Loaded %d prompts (capped at --max-prompts=%d)",
                len(prompts), args.max_prompts)

    # P0 (cheap, pre-GPU) — prompts must already be chat-templated. Raw
    # prompts cause the 2026-05-12 retry2 degenerate-rollout failure:
    # GRPO sends each prompt verbatim to the model, and an unwrapped
    # prompt never produces <|im_end|>, so every rollout hits the token
    # cap and reward_std collapses to 0. We block here rather than 10h
    # later when the run finishes with no learning signal.
    try:
        assert_prompts_are_chat_templated(prompts)
    except RuntimeError as e:
        logger.error("%s", e)
        return 4

    run_name = args.run_name or default_run_name()
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT_DEFAULT)
    else:
        logger.warning(
            "WANDB_API_KEY is not set; running with report_to='none'. "
            "Reward variance and KL trajectory will appear in stdout only."
        )

    if args.dry_run:
        # Build the GRPO config dict to print it, but skip --bf16/--fp16
        # resolution (no GPU available here in dry mode).
        cfg_kwargs = grpo_config_kwargs(
            args=args, yaml_dict=lora_yaml, precision="bf16",
            run_name=run_name, use_wandb=use_wandb,
        )
        print("=" * 60)
        print("train_rlvr.py --dry-run summary")
        print("=" * 60)
        print(f"  adapter_dir       : {args.adapter_dir}")
        print(f"  prompt_set        : {args.prompt_set}")
        print(f"  prompts loaded    : {len(prompts)}")
        print(f"  output_dir        : {args.output_dir}")
        print(f"  run_name          : {run_name}")
        print(f"  use_wandb         : {use_wandb}")
        print()
        print("Resolved GRPOConfig kwargs:")
        for k, v in cfg_kwargs.items():
            print(f"  {k:32s} = {v!r}")
        print()
        print("Dry-run: no model loading, no GPU work.")
        return 0

    # Heavy ML imports deferred so unit tests don't need these wheels.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    bf16_supported = (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    precision = choose_precision(
        args.precision,
        cuda_available=torch.cuda.is_available(),
        bf16_supported=bf16_supported,
    )
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    logger.info("precision=%s dtype=%s", precision, dtype)

    # Tokenizer + locked chat template.
    base_model_id = lora_yaml["base_model"]
    logger.info("loading tokenizer from %s", base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.chat_template = chat_template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base + SFT adapter (D2: continue the SFT LoRA).
    logger.info("loading base model %s (dtype=%s)", base_model_id, dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, dtype=dtype, device_map="auto",
    )
    logger.info("attaching SFT adapter from %s (is_trainable=True)", args.adapter_dir)
    peft_model = PeftModel.from_pretrained(
        base_model, str(args.adapter_dir), is_trainable=True,
    )
    peft_model.gradient_checkpointing_enable()
    peft_model.config.use_cache = False

    # ---- P1 preflight: well-formed output from the starting adapter -------
    if not args.skip_preflights:
        smoke_inference_p1(peft_model, tokenizer)

    # ---- P2 preflight: reward variance is non-trivial ---------------------
    if not args.skip_preflights:
        sample_prompts = prompts[: args.variance_preflight_prompts]
        passed, mean_var = reward_variance_preflight_p2(
            peft_model, tokenizer, sample_prompts,
            num_generations=args.num_generations,
            rollout_temp=args.rollout_temp,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        logger.info(
            "P2 preflight: mean per-prompt reward variance = %.4f "
            "(threshold %.4f)",
            mean_var, REWARD_VARIANCE_THRESHOLD,
        )
        if not passed:
            raise RuntimeError(
                f"P2 preflight FAILED: per-prompt reward variance "
                f"({mean_var:.4f}) below threshold ({REWARD_VARIANCE_THRESHOLD}). "
                "GRPO has no signal — every rollout is scoring identically. "
                "Likely fixes: raise --rollout-temp, drop the difficulty band, "
                "or check that the prompt set actually contains in-band rows."
            )

    # ---- Build trainer ----------------------------------------------------
    grpo_config = GRPOConfig(**grpo_config_kwargs(
        args=args, yaml_dict=lora_yaml, precision=precision,
        run_name=run_name, use_wandb=use_wandb,
    ))

    # TRL expects a HF datasets.Dataset (or compatible). Build from list-of-
    # dicts; the column names ('prompt', 'answer') are referenced by the
    # reward function below.
    from datasets import Dataset
    train_ds = Dataset.from_list(prompts)

    # TRL reward callback signature: (completions, **kwargs) -> list[float].
    # The dataset's extra columns arrive as parallel lists; we destructure
    # 'answer' and call the per-row reward.
    from scripts.reward_fn import compute_reward

    def _reward_callback(completions, **kwargs):
        gold_list = kwargs.get("answer")
        if gold_list is None:
            raise RuntimeError(
                "Reward callback received no 'answer' column. The dataset "
                "must have prompt+answer; check the prompt-set schema."
            )
        return [compute_reward(c, g) for c, g in zip(completions, gold_list)]

    trainer = GRPOTrainer(
        model=peft_model,
        args=grpo_config,
        reward_funcs=[_reward_callback],
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    # P3: KL spike monitor (warn-only).
    if not args.skip_preflights:
        trainer.add_callback(_build_kl_spike_callback()())

    # ---- Train ------------------------------------------------------------
    trainer.train()

    # Save adapter + tokenizer (with chat template). Same on-disk shape as
    # train_sft.py so merge_and_push.py works without changes.
    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("saved RLVR adapter + tokenizer to %s", final_dir)

    # Re-load and assert chat template survives save (same guard as Stage 3).
    reloaded = AutoTokenizer.from_pretrained(str(final_dir))
    if reloaded.chat_template != chat_template:
        raise RuntimeError(
            "after save_pretrained, the reloaded tokenizer.chat_template "
            "does NOT match the locked Jinja byte-for-byte. The Phase 3 "
            "merge will be broken until this is fixed."
        )
    logger.info("chat_template round-trip OK after save")

    # Final smoke (reuse P1 shape) — confirms the post-RLVR adapter still
    # emits well-formed output. If this regresses, the run probably
    # collapsed; eval before pushing.
    smoke_inference_p1(trainer.model, tokenizer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
