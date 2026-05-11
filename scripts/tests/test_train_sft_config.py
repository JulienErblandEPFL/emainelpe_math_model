"""Pure-Python tests for the config-kwargs helpers in scripts/train_sft.

These lock the values that the merge in Phase 3 depends on (LoRA shape) and
the values that CLAUDE.md commits us to (SFT schedule, batch sizes, packing
off, max_length=4096). They run without peft/trl because the helpers return
plain dicts; the actual LoraConfig / SFTConfig instantiation is exercised
at runtime on RCP, not in unit tests.
"""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from scripts.train_sft import (
    compute_warmup_steps,
    default_run_name,
    lora_config_kwargs,
    sft_config_kwargs,
)


LOCKED_LORA_YAML = {
    "base_model": "Qwen/Qwen3-1.7B",
    "max_seq_length": 4096,
    "lora": {
        "r": 32,
        "alpha": 64,
        "dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    },
}


def _default_args(**overrides):
    base = {
        "output_dir": "/tmp/run",
        "epochs": 2,
        "learning_rate": 1e-4,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "seed": 42,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _call_sft(
    *,
    args=None,
    yaml_dict=None,
    precision="bf16",
    run_name="sft-test",
    use_wandb=False,
    n_train_examples=50_000,
):
    return sft_config_kwargs(
        args=args or _default_args(),
        yaml_dict=yaml_dict or LOCKED_LORA_YAML,
        precision=precision,
        run_name=run_name,
        use_wandb=use_wandb,
        n_train_examples=n_train_examples,
    )


# ---- lora_config_kwargs ----------------------------------------------------

def test_lora_config_kwargs_locks_rank_alpha_dropout_bias_tasktype():
    kw = lora_config_kwargs(LOCKED_LORA_YAML)
    assert kw["r"] == 32
    assert kw["lora_alpha"] == 64
    assert kw["lora_dropout"] == 0.05
    assert kw["bias"] == "none"
    assert kw["task_type"] == "CAUSAL_LM"


def test_lora_config_kwargs_locks_all_seven_target_modules():
    kw = lora_config_kwargs(LOCKED_LORA_YAML)
    assert set(kw["target_modules"]) == {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    }


def test_lora_config_kwargs_renames_alpha_to_lora_alpha():
    """PEFT uses lora_alpha/lora_dropout, not alpha/dropout. If this rename
    drops, LoraConfig will silently default-initialize alpha=8 and the merge
    in Phase 3 will be broken."""
    kw = lora_config_kwargs(LOCKED_LORA_YAML)
    assert "alpha" not in kw
    assert "dropout" not in kw
    assert "lora_alpha" in kw
    assert "lora_dropout" in kw


# ---- sft_config_kwargs -----------------------------------------------------

def test_sft_config_kwargs_uses_locked_max_seq_length():
    assert _call_sft()["max_length"] == 4096


def test_sft_config_kwargs_disables_packing_and_assistant_only_loss():
    """Both off. assistant_only_loss=False is forced because the locked
    Jinja lacks `{% generation %}` markers (TRL 0.21+ refuses to auto-patch).
    A flip back to True must be coupled with adding generation markers
    in emainelpe-shared and a re-run of the cluster smoke."""
    kw = _call_sft()
    assert kw["packing"] is False
    assert kw["assistant_only_loss"] is False


def test_sft_config_kwargs_uses_cosine_schedule():
    assert _call_sft()["lr_scheduler_type"] == "cosine"


def test_sft_config_kwargs_warmup_steps_is_three_percent_of_total():
    """50k examples, batch 4×8=32, 2 epochs → ceil(50000/32)*2 = 3126 total
    steps → round(0.03*3126) = 94 warmup steps."""
    kw = _call_sft(n_train_examples=50_000)
    assert kw["warmup_steps"] == 94
    assert "warmup_ratio" not in kw


def test_sft_config_kwargs_warmup_steps_floor_is_one_for_smoke():
    """200 examples, 1 epoch → 7 total steps → round(0.21) = 0 → floor to 1.
    Without the floor, SFTConfig refuses warmup_steps=0 in some versions."""
    smoke_args = _default_args(epochs=1)
    kw = _call_sft(args=smoke_args, n_train_examples=200)
    assert kw["warmup_steps"] == 1


def test_sft_config_kwargs_enables_gradient_checkpointing_non_reentrant():
    kw = _call_sft()
    assert kw["gradient_checkpointing"] is True
    assert kw["gradient_checkpointing_kwargs"] == {"use_reentrant": False}


def test_sft_config_kwargs_bf16_vs_fp16_flags():
    bf16 = _call_sft(precision="bf16")
    fp16 = _call_sft(precision="fp16")
    assert bf16["bf16"] is True and bf16["fp16"] is False
    assert fp16["bf16"] is False and fp16["fp16"] is True


def test_sft_config_kwargs_passes_through_cli_overrides():
    kw = _call_sft(
        args=_default_args(
            epochs=1,
            learning_rate=5e-5,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            seed=7,
            output_dir="/tmp/runX",
        ),
        run_name="sft-override",
    )
    assert kw["num_train_epochs"] == 1
    assert kw["learning_rate"] == 5e-5
    assert kw["per_device_train_batch_size"] == 2
    assert kw["gradient_accumulation_steps"] == 16
    assert kw["seed"] == 7
    assert kw["output_dir"] == "/tmp/runX"
    assert kw["run_name"] == "sft-override"


def test_sft_config_kwargs_save_and_eval_cadence():
    kw = _call_sft()
    assert kw["save_strategy"] == "steps"
    assert kw["save_steps"] == 500
    assert kw["eval_strategy"] == "steps"
    assert kw["eval_steps"] == 500
    assert kw["save_total_limit"] == 2


def test_sft_config_kwargs_eval_memory_caps_avoid_oom():
    """Both per_device_eval_batch_size=1 AND eval_accumulation_steps>=4
    must be set to fit eval-time logits on A100 40GB.

    Rationale: with Qwen3-1.7B vocab=151,936 and max_seq=4096, the
    per-batch (B × T × V × 2B) eval-logits tensor — plus its contiguous
    shift_logits copy inside compute_loss — exceeds A100-40GB headroom
    whenever B>1 on pure-OMI2 (v3) data where every eval row is
    token-dense (Llama3.1-405B solutions are long). Observed stack:
        trl/trainer/sft_trainer.py:1349  compute_loss
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        torch.OutOfMemoryError: Tried to allocate 13.77 GiB.
    eval_accumulation_steps alone is INSUFFICIENT — it only controls
    cross-batch accumulator placement, not the per-batch allocation.
    Both knobs target different memory ledgers and both must be set."""
    kw = _call_sft()
    # Per-batch logits-allocation cap — the load-bearing fix.
    assert "per_device_eval_batch_size" in kw, (
        "per_device_eval_batch_size must be set explicitly to 1; the "
        "Trainer default cascades from per_device_train_batch_size (4) "
        "and OOMs on long-sequence v3 eval rows. See the OOM-fix comment "
        "in scripts/train_sft.py:sft_config_kwargs."
    )
    assert kw["per_device_eval_batch_size"] == 1, (
        f"per_device_eval_batch_size={kw['per_device_eval_batch_size']}; "
        "must be 1 — eval_accumulation_steps does not shrink the per-batch "
        "logits tensor, only B=1 does."
    )
    # Cross-batch accumulator placement — complementary, not load-bearing.
    assert "eval_accumulation_steps" in kw
    assert kw["eval_accumulation_steps"] >= 4, (
        f"eval_accumulation_steps={kw['eval_accumulation_steps']}; raise "
        "to >=4 so per-batch predictions are moved to CPU promptly rather "
        "than accumulated on GPU across the full eval set."
    )


def test_sft_config_kwargs_report_to_follows_use_wandb():
    assert _call_sft(use_wandb=True)["report_to"] == "wandb"
    assert _call_sft(use_wandb=False)["report_to"] == "none"


# ---- compute_warmup_steps --------------------------------------------------

def test_compute_warmup_steps_full_run_50k_two_epochs():
    """Anchor case from CLAUDE.md: 50k train, batch 32, 2 epochs."""
    n = compute_warmup_steps(
        n_train_examples=50_000,
        per_device_batch_size=4,
        gradient_accumulation_steps=8,
        epochs=2,
    )
    assert n == 94


def test_compute_warmup_steps_smoke_run_floors_to_one():
    """Smoke: 200 train, 1 epoch → 7 total steps → 0.21 → 0 → floor."""
    n = compute_warmup_steps(
        n_train_examples=200,
        per_device_batch_size=4,
        gradient_accumulation_steps=8,
        epochs=1,
    )
    assert n == 1


def test_compute_warmup_steps_factors_epochs_into_total_steps():
    """Warmup is rounded once from total_steps = ceil(n/batch) * epochs,
    not multiplied per-epoch. ceil(10000/32) = 313 steps_per_epoch.
        1 ep: total=313,  round(0.03*313)  = round(9.39)  = 9
        3 ep: total=939,  round(0.03*939)  = round(28.17) = 28
    (28 != 3*9 because rounding doesn't distribute — that's the point.)"""
    one_ep = compute_warmup_steps(
        n_train_examples=10_000,
        per_device_batch_size=4,
        gradient_accumulation_steps=8,
        epochs=1,
    )
    three_ep = compute_warmup_steps(
        n_train_examples=10_000,
        per_device_batch_size=4,
        gradient_accumulation_steps=8,
        epochs=3,
    )
    assert one_ep == 9
    assert three_ep == 28


def test_compute_warmup_steps_respects_custom_ratio():
    n = compute_warmup_steps(
        n_train_examples=10_000,
        per_device_batch_size=4,
        gradient_accumulation_steps=8,
        epochs=1,
        warmup_ratio=0.10,
    )
    # ceil(10000/32) = 313; round(0.10*313) = 31
    assert n == 31


# ---- default_run_name ------------------------------------------------------

def test_default_run_name_format_is_sft_stamp_epochs_rank():
    when = dt.datetime(2026, 5, 7, 18, 30)
    name = default_run_name(epochs=2, rank=32, now=when)
    assert name == "sft-20260507-1830-2ep-r32"


def test_default_run_name_reflects_epochs_and_rank():
    when = dt.datetime(2026, 6, 1, 9, 5)
    a = default_run_name(epochs=1, rank=32, now=when)
    b = default_run_name(epochs=3, rank=16, now=when)
    assert a == "sft-20260601-0905-1ep-r32"
    assert b == "sft-20260601-0905-3ep-r16"
