"""Train a LoRA adapter on the JSONL produced by ``data/prepare_sft.py``.

Anchored to the locked decisions in ``CLAUDE.md`` and the locked LoRA shape
in ``configs/lora.yaml``. The shared chat template at
``chat_template/chat_template.jinja`` is loaded onto the tokenizer before
``trl.SFTTrainer`` sees it; the same Jinja is then re-asserted byte-identical
after ``save_pretrained`` so the merge in Phase 3 cannot silently drift.

Pure helpers (``load_lora_yaml``, ``load_chat_template``, ``lora_config_kwargs``,
``sft_config_kwargs``, ``choose_precision``, ``default_run_name``,
``validate_init_adapter_config``) are CPU-testable and live at module
scope. The heavy ML imports (``torch``, ``peft``, ``trl``, ``transformers``,
``datasets``) are deferred into ``main()`` so the unit tests can run on a
laptop without those wheels installed.

Two ways to start training from a prior checkpoint:

  - ``--resume``: continue a previously interrupted training run.
    Reloads the HF Trainer state (optimizer momenta, LR scheduler
    position, RNG state, dataset position). The training data must
    match the original run; this is just a "pick up where we left off"
    operation.

  - ``--init-from-adapter PATH``: start a FRESH training run on
    ``--train-file`` (typically a new dataset), but initialize the
    LoRA weights from the adapter at PATH instead of from a random
    LoRA init. Fresh optimizer + LR scheduler + RNG. Used for v4
    training where we want to build on v3's learned weights without
    inheriting v3's optimizer state (which was tuned for the v3 OMI2
    dataset, not the v4-mix). Validates the adapter's r / alpha /
    target_modules against the team-locked ``configs/lora.yaml``
    before training starts — refuses to launch on a mismatched
    adapter to keep the Phase 3 merge safe.

The two flags are mutually exclusive (enforced by argparse).

Smoke run on RCP (Stage 3 "Done when" criterion):

    python scripts/train_sft.py \\
        --train-file data_out/train.jsonl \\
        --eval-file  data_out/eval.jsonl \\
        --output-dir runs/smoke \\
        --epochs 1 \\
        --max-train-samples 200
"""
from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LORA_YAML = REPO_ROOT / "configs" / "lora.yaml"
DEFAULT_CHAT_TEMPLATE = REPO_ROOT / "chat_template" / "chat_template.jinja"

WANDB_PROJECT_DEFAULT = "emainelpe-math"


# =============================================================================
# Pure helpers — CPU-testable, no torch/peft/trl imports.
# =============================================================================

def load_lora_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_chat_template(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def default_run_name(
    *, epochs: int, rank: int, now: _dt.datetime | None = None
) -> str:
    when = now or _dt.datetime.now()
    return f"sft-{when.strftime('%Y%m%d-%H%M')}-{epochs}ep-r{rank}"


def compute_warmup_steps(
    *,
    n_train_examples: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    warmup_ratio: float = 0.03,
) -> int:
    """Total training steps × ``warmup_ratio``, rounded, with a floor of 1.

    Replaces ``SFTConfig.warmup_ratio`` (deprecated in TRL 0.21+). Computed
    once up-front from the post-filter train size; the smoke run on 200
    examples produces ``total_steps=7`` so the ``max(1, ...)`` floor matters.
    """
    import math
    effective_batch = per_device_batch_size * gradient_accumulation_steps
    total_steps = math.ceil(n_train_examples / effective_batch) * epochs
    return max(1, round(warmup_ratio * total_steps))


def lora_config_kwargs(yaml_dict: dict) -> dict:
    """Map locked ``lora.yaml`` keys → ``peft.LoraConfig`` kwargs.

    PEFT names them ``lora_alpha`` / ``lora_dropout``; ``lora.yaml`` stores
    them as ``alpha`` / ``dropout``. Drift on this rename silently default-
    initializes alpha=8 and breaks the Phase 3 merge.
    """
    lora = yaml_dict["lora"]
    return {
        "r": lora["r"],
        "lora_alpha": lora["alpha"],
        "lora_dropout": lora["dropout"],
        "bias": lora["bias"],
        "task_type": lora["task_type"],
        "target_modules": list(lora["target_modules"]),
    }


def validate_init_adapter_config(
    adapter_cfg: dict, lora_yaml: dict,
) -> None:
    """Assert a loaded adapter's LoRA shape matches the team-locked spec.

    Used by ``--init-from-adapter``: when training v4 from v3's adapter,
    we MUST verify v3's adapter was trained with the team-locked r,
    alpha, and target_modules. Mismatch here would silently produce a v4
    adapter whose shape diverges from the locked spec, breaking the
    Phase 3 merge.

    ``adapter_cfg`` is the parsed ``adapter_config.json`` dict (PEFT's
    canonical on-disk format for LoRA adapters). ``lora_yaml`` is the
    parsed ``configs/lora.yaml``. Raises ``RuntimeError`` with a precise
    field-by-field diff when any of {r, alpha, target_modules}
    mismatches; returns ``None`` on a clean match.
    """
    expected = lora_yaml["lora"]
    actual_r = adapter_cfg.get("r")
    actual_alpha = adapter_cfg.get("lora_alpha")
    actual_modules = adapter_cfg.get("target_modules") or []

    if actual_r != expected["r"]:
        raise RuntimeError(
            f"--init-from-adapter: adapter r={actual_r} does not match "
            f"locked configs/lora.yaml r={expected['r']}. The Phase 3 "
            f"merge requires identical rank across all four experts; "
            f"refusing to launch on a mismatched adapter."
        )
    if actual_alpha != expected["alpha"]:
        raise RuntimeError(
            f"--init-from-adapter: adapter lora_alpha={actual_alpha} does "
            f"not match locked configs/lora.yaml alpha={expected['alpha']}. "
            f"The Phase 3 merge requires identical alpha across all four "
            f"experts; refusing to launch on a mismatched adapter."
        )
    expected_modules = set(expected["target_modules"])
    if set(actual_modules) != expected_modules:
        missing = sorted(expected_modules - set(actual_modules))
        extra = sorted(set(actual_modules) - expected_modules)
        raise RuntimeError(
            f"--init-from-adapter: adapter target_modules "
            f"{sorted(actual_modules)} does not match locked "
            f"configs/lora.yaml target_modules "
            f"{sorted(expected_modules)}. Missing in adapter: {missing}. "
            f"Extra in adapter: {extra}. The Phase 3 merge requires the "
            f"exact 7-module set across all four experts."
        )


def choose_precision(
    requested: str,
    *,
    cuda_available: bool,
    bf16_supported: bool,
) -> str:
    """Resolve ``--precision`` to a concrete ``"bf16"`` or ``"fp16"``.

    Refuses to fall back to CPU; raises ``RuntimeError`` if no GPU is
    available, regardless of the requested mode.
    """
    if not cuda_available:
        raise RuntimeError(
            "CUDA is not available. SFT training requires a GPU; refusing "
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


def sft_config_kwargs(
    *,
    args,
    yaml_dict: dict,
    precision: str,
    run_name: str,
    use_wandb: bool,
    n_train_examples: int,
) -> dict:
    """Map CLI args + locked yaml + dtype + train-set size → ``trl.SFTConfig``
    kwargs.

    Loss-masking note: ``assistant_only_loss=False`` because the locked Jinja
    in ``chat_template/chat_template.jinja`` lacks ``{% generation %}``
    markers and TRL 0.21+ refuses to auto-patch the Qwen3 template without
    them. We compute loss over the full sequence (user + assistant tokens).
    Adding the markers is a v2 stretch goal — see IMPLEMENTATION_PLAN.md.

    Eval-strategy note: ``eval_steps`` measures token-level cross-entropy
    on a held-out slice of the SAME training distribution. It does NOT
    measure math accuracy — that lives in Stage 4 (scripts/run_eval.py).
    A flat eval-loss curve does not necessarily mean the model has stopped
    improving on math; a falling eval-loss curve does not guarantee
    pass@1 went up.

    Liger Kernel note: ``use_liger_kernel=True`` (default) swaps the stock
    HuggingFace causal-LM linear-cross-entropy path for the fused Liger
    kernel, which never materializes the full ``B × T × vocab × 4B`` logits
    tensor. This is the structural fix for the OOM class that hit v4 at
    step 1514 (and v4-200k at epoch 0.08) — the logits tensor alone
    consumed 7.5-9.3 GiB on near-max-length batches, more than the model
    weights themselves. Affects loss computation only; the locked LoRA
    shape (r=32, alpha=64, 7 target_modules) is preserved and the saved
    adapter is byte-identical to a non-Liger run.
    """
    warmup_steps = compute_warmup_steps(
        n_train_examples=n_train_examples,
        per_device_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
    )
    return {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lr_scheduler_type": "cosine",
        "warmup_steps": warmup_steps,
        "max_length": yaml_dict["max_seq_length"],
        "packing": False,
        "assistant_only_loss": False,
        "bf16": precision == "bf16",
        "fp16": precision == "fp16",
        "use_liger_kernel": args.use_liger_kernel,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": 10,
        "eval_strategy": "steps",
        "eval_steps": 500,
        # Eval batch must be 1 to avoid OOM on logits allocation with long
        # OMI2 sequences; eval_accumulation_steps=4 is insufficient alone.
        "per_device_eval_batch_size": 1,
        "eval_accumulation_steps": 4,
        "save_strategy": "steps",
        "save_steps": 500,
        "save_total_limit": 2,
        "report_to": "wandb" if use_wandb else "none",
        "run_name": run_name,
        "seed": args.seed,
    }


# =============================================================================
# Runtime helpers — used by main(); import torch/transformers internally.
# =============================================================================

def filter_long_rows(dataset, tokenizer, max_length: int):
    """Drop rows whose chat-templated tokenization exceeds ``max_length``.

    With ``assistant_only_loss=True``, TRL truncates from the right at
    ``max_length`` tokens. Truncation can chop off ``\\boxed{...}``, leaving
    a row with no supervised final-answer span. Pre-filtering is preferable:
    every surviving row keeps its full assistant turn.
    """
    def _measure(ex):
        toks = tokenizer.apply_chat_template(ex["messages"], tokenize=True)
        return {"_n_tokens": len(toks)}

    measured = dataset.map(_measure)
    n_total = len(measured)
    kept = measured.filter(lambda ex: ex["_n_tokens"] <= max_length)
    n_kept = len(kept)
    n_dropped = n_total - n_kept
    pct = (n_dropped / n_total * 100) if n_total else 0.0
    logger.info(
        "token-length filter: max=%d kept=%d dropped=%d (%.1f%%)",
        max_length, n_kept, n_dropped, pct,
    )
    if pct > 5.0:
        logger.warning(
            "token-length filter dropped >5%% of rows (%.1f%%); consider "
            "tightening --max-response-chars in prepare_sft.py",
            pct,
        )
    return kept.remove_columns(["_n_tokens"])


def smoke_inference(model, tokenizer) -> None:
    """Hardcoded post-training smoke: 'What is 2+2?' → decoded output.

    Catches "trained but broken" failure modes (NaN logits, empty
    generation, missing chat template) before Stage 4 evaluation runs.
    Always runs; not configurable on purpose.

    ``max_new_tokens=2048`` is enough to clear the ``<think>`` block and
    emit a closing ``</think>`` + ``\\boxed{...}`` for a trivial-arithmetic
    prompt. The first smoke (2026-05-07) used 128 and truncated mid-think,
    making format verification impossible.
    """
    import torch  # local import; main() already pulled torch in

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=False)
    logger.info("=== smoke inference ===\n%s\n=== end ===", decoded)


# =============================================================================
# CLI / main
# =============================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--eval-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--run-name", default=None)
    resume_or_init = p.add_mutually_exclusive_group()
    resume_or_init.add_argument(
        "--resume", default=None,
        help="'latest' to resume from the most recent checkpoint under "
             "--output-dir, or a path to a specific checkpoint dir. "
             "Reloads optimizer + LR scheduler + RNG state from the "
             "checkpoint. Mutually exclusive with --init-from-adapter.",
    )
    resume_or_init.add_argument(
        "--init-from-adapter", type=Path, default=None,
        dest="init_from_adapter",
        help="Path to a trained LoRA adapter directory (Stage 3 final/). "
             "Loads the base model + this adapter via "
             "PeftModel.from_pretrained, then trains on --train-file with "
             "a FRESH optimizer + LR scheduler. Use this to start v4 from "
             "v3's learned weights without inheriting v3's optimizer state "
             "(which was tuned for v3's data, not the new v4-mix). The "
             "adapter's LoRA shape is validated against configs/lora.yaml "
             "before training launches. Mutually exclusive with --resume.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--precision", choices=["auto", "bf16", "fp16"], default="auto",
    )
    p.add_argument(
        "--max-train-samples", type=int, default=None,
        help="Cap the training set (smoke runs). None = use all.",
    )
    p.add_argument("--lora-yaml", type=Path, default=DEFAULT_LORA_YAML)
    p.add_argument("--chat-template", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    # Liger Kernel: fused linear-cross-entropy that never materializes the
    # full B × T × vocab logits tensor. Default ON because the stock HF
    # causal-LM path OOMs on near-max-length batches with Qwen3-1.7B
    # (vocab=151,643) — the logits tensor alone is 7.5-9.3 GiB. Disable
    # via --no-use-liger-kernel for A/B comparison or if the kernel is
    # unavailable on a future image.
    p.add_argument(
        "--use-liger-kernel", action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Liger Kernel fused cross-entropy in SFTConfig "
             "(eliminates the logits-tensor OOM class). Default: True.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def _resolve_resume(arg: str | None, output_dir: Path):
    """Resolve ``--resume`` to a concrete checkpoint path or ``None``.

    Sorts by the integer step suffix, NOT lexicographically: HF Trainer
    names checkpoints ``checkpoint-{global_step}``, and a lexicographic
    sort would pick ``checkpoint-9500`` over ``checkpoint-16000``.
    """
    if arg is None:
        return None
    if arg == "latest":
        ckpts = list(output_dir.glob("checkpoint-*"))
        if not ckpts:
            raise FileNotFoundError(
                f"--resume latest: no checkpoint-* directories under {output_dir}"
            )
        latest = max(ckpts, key=lambda p: int(p.name.split("-")[1]))
        return str(latest)
    return arg


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Locked configs first — pure-Python so any drift surfaces immediately.
    lora_yaml = load_lora_yaml(args.lora_yaml)
    chat_template = load_chat_template(args.chat_template)

    # Heavy ML imports deferred so unit tests don't need these wheels.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # Resolve precision; refuses to run on CPU.
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

    # Liger Kernel preflight: fail fast at startup if --use-liger-kernel
    # is True but the wheel isn't importable. Stock HF logits-cross-entropy
    # OOMs on Qwen3-1.7B near-max-length batches; running without Liger
    # silently risks the v4 step-1514 OOM class.
    if args.use_liger_kernel:
        try:
            import liger_kernel  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "--use-liger-kernel=True but liger_kernel is not installed. "
                "Install with `pip install liger-kernel>=0.8.0` (already in "
                "requirements.txt) or pass --no-use-liger-kernel to fall "
                "back to the stock HF cross-entropy path (and accept the "
                "logits-tensor OOM risk on long sequences)."
            ) from exc
        logger.info(
            "Liger Kernel enabled: SFTConfig.use_liger_kernel=True. "
            "Fused cross-entropy avoids materializing the B × T × vocab "
            "logits tensor — primary OOM mitigation for Qwen3-1.7B."
        )

    # Tokenizer + locked chat template (same idiom as verify_chat_template.py).
    base_model = lora_yaml["base_model"]
    logger.info("loading tokenizer from %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.chat_template = chat_template
    if tokenizer.chat_template != chat_template:
        raise RuntimeError(
            "tokenizer.chat_template differs from the assigned string after "
            "assignment. Investigate before training."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model.
    logger.info("loading model from %s (dtype=%s)", base_model, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
        device_map="auto",
    )

    # --init-from-adapter: attach a pre-trained LoRA adapter to the base
    # model before SFTTrainer runs. The trainer will then continue training
    # the EXISTING adapter weights with a fresh optimizer + LR scheduler.
    # We validate the adapter's shape against the locked yaml first so a
    # silently-mismatched r/alpha/target_modules cannot start a run that
    # would break the Phase 3 merge.
    init_adapter_in_use = bool(args.init_from_adapter)
    if init_adapter_in_use:
        import json
        from peft import PeftModel

        adapter_dir = args.init_from_adapter
        adapter_cfg_path = adapter_dir / "adapter_config.json"
        if not adapter_cfg_path.is_file():
            raise RuntimeError(
                f"--init-from-adapter: no adapter_config.json at {adapter_cfg_path}. "
                f"The PATH must point at a PEFT adapter directory (e.g., "
                f"Stage 3's final/)."
            )
        with open(adapter_cfg_path, encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        validate_init_adapter_config(adapter_cfg, lora_yaml)
        logger.info(
            "--init-from-adapter: loading adapter from %s (validated against "
            "locked LoRA spec)", adapter_dir,
        )
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)

    model.gradient_checkpointing_enable()
    # use_cache must be off when gradient_checkpointing is on.
    model.config.use_cache = False

    # Data.
    logger.info("loading train=%s eval=%s", args.train_file, args.eval_file)
    raw_train = load_dataset("json", data_files=str(args.train_file), split="train")
    raw_eval = load_dataset("json", data_files=str(args.eval_file), split="train")
    if args.max_train_samples is not None:
        n = min(args.max_train_samples, len(raw_train))
        raw_train = raw_train.select(range(n))
        logger.info("smoke mode: capped train to %d examples", n)

    max_seq_length = lora_yaml["max_seq_length"]
    train_ds = filter_long_rows(raw_train, tokenizer, max_seq_length)
    eval_ds = filter_long_rows(raw_eval, tokenizer, max_seq_length)

    # W&B configuration.
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT_DEFAULT)
    else:
        logger.warning(
            "WANDB_API_KEY is not set; running with report_to='none'. "
            "Loss curves will appear in stdout only."
        )

    run_name = args.run_name or default_run_name(
        epochs=args.epochs, rank=lora_yaml["lora"]["r"],
    )
    logger.info("run_name=%s", run_name)

    # Build configs from the same dicts the unit tests lock against.
    # When --init-from-adapter is set, the model already carries a PEFT
    # adapter; passing peft_config to SFTTrainer would add a SECOND
    # adapter on top of the first one, which is not what we want.
    sft_config = SFTConfig(**sft_config_kwargs(
        args=args,
        yaml_dict=lora_yaml,
        precision=precision,
        run_name=run_name,
        use_wandb=use_wandb,
        n_train_examples=len(train_ds),
    ))
    if init_adapter_in_use:
        trainer_peft_config = None
    else:
        trainer_peft_config = LoraConfig(**lora_config_kwargs(lora_yaml))

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=trainer_peft_config,
    )

    # Train.
    resume_from = _resolve_resume(args.resume, args.output_dir)
    if resume_from is not None:
        logger.info("resuming from %s", resume_from)
    trainer.train(resume_from_checkpoint=resume_from)

    # Save adapter + tokenizer (with chat template).
    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("saved final adapter + tokenizer to %s", final_dir)

    # Re-load and assert the chat template survived save_pretrained.
    reloaded = AutoTokenizer.from_pretrained(str(final_dir))
    if reloaded.chat_template != chat_template:
        raise RuntimeError(
            "after save_pretrained, the reloaded tokenizer.chat_template does "
            "NOT match the locked Jinja byte-for-byte. The Phase 3 merge will "
            "be broken until this is fixed."
        )
    logger.info("chat_template round-trip OK after save")

    # Final smoke inference (always on).
    smoke_inference(trainer.model, tokenizer)


if __name__ == "__main__":
    main()
