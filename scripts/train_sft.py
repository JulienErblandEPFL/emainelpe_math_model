"""Train a LoRA adapter on the JSONL produced by ``data/prepare_sft.py``.

Anchored to the locked decisions in ``CLAUDE.md`` and the locked LoRA shape
in ``configs/lora.yaml``. The shared chat template at
``chat_template/chat_template.jinja`` is loaded onto the tokenizer before
``trl.SFTTrainer`` sees it; the same Jinja is then re-asserted byte-identical
after ``save_pretrained`` so the merge in Phase 3 cannot silently drift.

Pure helpers (``load_lora_yaml``, ``load_chat_template``, ``lora_config_kwargs``,
``sft_config_kwargs``, ``choose_precision``, ``default_run_name``) are CPU-
testable and live at module scope. The heavy ML imports (``torch``, ``peft``,
``trl``, ``transformers``, ``datasets``) are deferred into ``main()`` so the
unit tests can run on a laptop without those wheels installed.

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
    measure math accuracy — that lives in Stage 4 (scripts/eval_local.py).
    A flat eval-loss curve does not necessarily mean the model has stopped
    improving on math; a falling eval-loss curve does not guarantee
    pass@1 went up.
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
    p.add_argument(
        "--resume", default=None,
        help="'latest' to resume from the most recent checkpoint under "
             "--output-dir, or a path to a specific checkpoint dir.",
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
    lora_config = LoraConfig(**lora_config_kwargs(lora_yaml))
    sft_config = SFTConfig(**sft_config_kwargs(
        args=args,
        yaml_dict=lora_yaml,
        precision=precision,
        run_name=run_name,
        use_wandb=use_wandb,
        n_train_examples=len(train_ds),
    ))

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
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
