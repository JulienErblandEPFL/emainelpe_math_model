"""Pure-Python tests for the config-kwargs helpers in scripts/train_sft.

These lock the values that the merge in Phase 3 depends on (LoRA shape) and
the values that CLAUDE.md commits us to (SFT schedule, batch sizes, packing
off, assistant_only_loss on, max_length=4096). They run without peft/trl
because the helpers return plain dicts; the actual LoraConfig / SFTConfig
instantiation is exercised at runtime on RCP, not in unit tests.
"""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from scripts.train_sft import (
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
    kw = sft_config_kwargs(
        args=_default_args(),
        yaml_dict=LOCKED_LORA_YAML,
        precision="bf16",
        run_name="sft-test",
        use_wandb=False,
    )
    assert kw["max_length"] == 4096


def test_sft_config_kwargs_disables_packing_and_enables_assistant_only_loss():
    kw = sft_config_kwargs(
        args=_default_args(),
        yaml_dict=LOCKED_LORA_YAML,
        precision="bf16",
        run_name="sft-test",
        use_wandb=False,
    )
    assert kw["packing"] is False
    assert kw["assistant_only_loss"] is True


def test_sft_config_kwargs_uses_cosine_schedule_with_warmup():
    kw = sft_config_kwargs(
        args=_default_args(),
        yaml_dict=LOCKED_LORA_YAML,
        precision="bf16",
        run_name="sft-test",
        use_wandb=False,
    )
    assert kw["lr_scheduler_type"] == "cosine"
    assert kw["warmup_ratio"] == 0.03


def test_sft_config_kwargs_enables_gradient_checkpointing_non_reentrant():
    kw = sft_config_kwargs(
        args=_default_args(),
        yaml_dict=LOCKED_LORA_YAML,
        precision="bf16",
        run_name="sft-test",
        use_wandb=False,
    )
    assert kw["gradient_checkpointing"] is True
    assert kw["gradient_checkpointing_kwargs"] == {"use_reentrant": False}


def test_sft_config_kwargs_bf16_vs_fp16_flags():
    bf16 = sft_config_kwargs(
        args=_default_args(),
        yaml_dict=LOCKED_LORA_YAML,
        precision="bf16",
        run_name="sft-test",
        use_wandb=False,
    )
    fp16 = sft_config_kwargs(
        args=_default_args(),
        yaml_dict=LOCKED_LORA_YAML,
        precision="fp16",
        run_name="sft-test",
        use_wandb=False,
    )
    assert bf16["bf16"] is True and bf16["fp16"] is False
    assert fp16["bf16"] is False and fp16["fp16"] is True


def test_sft_config_kwargs_passes_through_cli_overrides():
    kw = sft_config_kwargs(
        args=_default_args(
            epochs=1,
            learning_rate=5e-5,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            seed=7,
            output_dir="/tmp/runX",
        ),
        yaml_dict=LOCKED_LORA_YAML,
        precision="bf16",
        run_name="sft-override",
        use_wandb=False,
    )
    assert kw["num_train_epochs"] == 1
    assert kw["learning_rate"] == 5e-5
    assert kw["per_device_train_batch_size"] == 2
    assert kw["gradient_accumulation_steps"] == 16
    assert kw["seed"] == 7
    assert kw["output_dir"] == "/tmp/runX"
    assert kw["run_name"] == "sft-override"


def test_sft_config_kwargs_save_and_eval_cadence():
    kw = sft_config_kwargs(
        args=_default_args(),
        yaml_dict=LOCKED_LORA_YAML,
        precision="bf16",
        run_name="sft-test",
        use_wandb=False,
    )
    assert kw["save_strategy"] == "steps"
    assert kw["save_steps"] == 500
    assert kw["eval_strategy"] == "steps"
    assert kw["eval_steps"] == 500
    assert kw["save_total_limit"] == 2


def test_sft_config_kwargs_report_to_follows_use_wandb():
    on = sft_config_kwargs(
        args=_default_args(),
        yaml_dict=LOCKED_LORA_YAML,
        precision="bf16",
        run_name="sft-test",
        use_wandb=True,
    )
    off = sft_config_kwargs(
        args=_default_args(),
        yaml_dict=LOCKED_LORA_YAML,
        precision="bf16",
        run_name="sft-test",
        use_wandb=False,
    )
    assert on["report_to"] == "wandb"
    assert off["report_to"] == "none"


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
