"""CPU-only tests for ``scripts/train_rlvr.py`` pure helpers.

The heavy ML imports (``torch``, ``peft``, ``trl``, ``transformers``,
``datasets``) are deferred into ``main()`` so this file never needs to
import them. Target runtime: <2s on the user's laptop.

Coverage:
  - argparse defaults (D5 hyperparameters)
  - argparse rejects required-arg-missing
  - load_prompt_set_jsonl (happy path, schema rejection, max_prompts cap)
  - check_reward_variance (P2: above/below threshold, edge cases)
  - _is_kl_spike (P3: window + threshold edges)
  - validate_max_new_tokens (advisory warnings, no exception)
  - grpo_config_kwargs (locked values match D5)
  - default_run_name (shape only — string includes timestamp + 'rlvr-' prefix)
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest

from scripts.train_rlvr import (
    KL_SPIKE_THRESHOLD,
    KL_SPIKE_WINDOW_STEPS,
    REWARD_VARIANCE_THRESHOLD,
    _is_kl_spike,
    _parse_args,
    check_reward_variance,
    default_run_name,
    grpo_config_kwargs,
    load_prompt_set_jsonl,
    validate_max_new_tokens,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LORA_YAML = REPO_ROOT / "configs" / "lora.yaml"


# =============================================================================
# argparse — D5 hyperparameter defaults pinned here.
# =============================================================================

def test_parse_args_defaults_match_d5():
    args = _parse_args(["--output-dir", "/tmp/x"])
    assert args.learning_rate == 3e-6
    assert args.kl_coef == 0.04
    assert args.rollout_temp == 0.8
    assert args.num_generations == 8
    assert args.max_prompts == 5000
    assert args.max_new_tokens == 4096
    assert args.per_device_train_batch_size == 1
    assert args.gradient_accumulation_steps == 8
    assert args.epochs == 1
    assert args.seed == 42
    assert args.dry_run is False
    assert args.skip_preflights is False


def test_parse_args_requires_output_dir():
    with pytest.raises(SystemExit):
        _parse_args([])


def test_parse_args_dry_run():
    args = _parse_args(["--output-dir", "/tmp/x", "--dry-run"])
    assert args.dry_run is True


# =============================================================================
# load_prompt_set_jsonl — schema enforcement is strict at this layer
# (curation already cleaned input; anything reaching here is expected clean).
# =============================================================================

def test_load_prompt_set_jsonl_happy_path(tmp_path: Path):
    p = tmp_path / "rlvr_prompts.jsonl"
    rows = [
        {"prompt": "What is 2+2?", "answer": "4", "solve_rate": 0.5},
        {"prompt": "What is 5*6?", "answer": "30", "solve_rate": 0.25},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = load_prompt_set_jsonl(p)
    assert len(out) == 2
    # Only prompt+answer projected through; solve_rate dropped at this layer.
    assert out[0] == {"prompt": "What is 2+2?", "answer": "4"}


def test_load_prompt_set_jsonl_max_prompts(tmp_path: Path):
    p = tmp_path / "rlvr_prompts.jsonl"
    rows = [{"prompt": f"Q{i}", "answer": str(i)} for i in range(10)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = load_prompt_set_jsonl(p, max_prompts=3)
    assert len(out) == 3


def test_load_prompt_set_jsonl_rejects_missing_prompt(tmp_path: Path):
    p = tmp_path / "rlvr_prompts.jsonl"
    p.write_text(json.dumps({"answer": "4"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt"):
        load_prompt_set_jsonl(p)


def test_load_prompt_set_jsonl_rejects_missing_answer(tmp_path: Path):
    p = tmp_path / "rlvr_prompts.jsonl"
    p.write_text(json.dumps({"prompt": "Q"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="answer"):
        load_prompt_set_jsonl(p)


def test_load_prompt_set_jsonl_rejects_invalid_json(tmp_path: Path):
    p = tmp_path / "rlvr_prompts.jsonl"
    p.write_text("{not valid json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_prompt_set_jsonl(p)


# =============================================================================
# check_reward_variance — P2 preflight, the operational realization of the
# BASELINE.md "low diversity" warning.
# =============================================================================

def test_check_reward_variance_high_variance_passes():
    """Mixed correct/wrong rollouts → variance is well above threshold."""
    rewards = [
        [1.05, 0.05, 1.05, 0.05, 1.05, 0.05, 1.05, 0.05],   # 50/50 mix
        [1.05, 1.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    ]
    passed, var = check_reward_variance(rewards)
    assert passed
    assert var > REWARD_VARIANCE_THRESHOLD


def test_check_reward_variance_all_equal_fails():
    """All rollouts get identical reward → per-prompt variance is exactly
    zero. GRPO's advantage is 0/0 noise; refuse to start."""
    rewards = [
        [1.05] * 8,
        [1.05] * 8,
    ]
    passed, var = check_reward_variance(rewards)
    assert not passed
    assert var == 0.0


def test_check_reward_variance_all_zero_fails():
    """No rollout boxed correctly OR even produced a box → reward 0 across
    the board → variance 0 → no signal. Same failure mode."""
    rewards = [[0.0] * 8 for _ in range(3)]
    passed, var = check_reward_variance(rewards)
    assert not passed
    assert var == 0.0


def test_check_reward_variance_empty_raises():
    with pytest.raises(ValueError):
        check_reward_variance([])


def test_check_reward_variance_single_rollout_per_prompt_raises():
    """Variance over a single sample is undefined; reject explicitly."""
    with pytest.raises(ValueError, match="≥2 rollouts"):
        check_reward_variance([[1.0]])


def test_check_reward_variance_threshold_boundary():
    """At exactly the threshold, variance check passes (>=, not >)."""
    # Construct rollouts whose per-prompt variance equals threshold exactly.
    # var = mean((r - mean)^2). With two rollouts at +d and -d around mean=0,
    # var = d². Pick d such that d² == threshold.
    d = REWARD_VARIANCE_THRESHOLD ** 0.5
    rewards = [[d, -d]]
    passed, var = check_reward_variance(rewards)
    assert passed
    # Float tolerance — the var should equal threshold within ~1e-12.
    assert abs(var - REWARD_VARIANCE_THRESHOLD) < 1e-9


# =============================================================================
# _is_kl_spike — P3 monitor, off-by-one and threshold-equality boundaries.
# =============================================================================

def test_is_kl_spike_above_threshold_in_window():
    assert _is_kl_spike(0.6, step=10)
    assert _is_kl_spike(KL_SPIKE_THRESHOLD + 0.01, step=0)


def test_is_kl_spike_below_threshold_in_window():
    assert not _is_kl_spike(0.3, step=10)
    assert not _is_kl_spike(KL_SPIKE_THRESHOLD, step=10)   # equality is NOT a spike


def test_is_kl_spike_outside_window_never_fires():
    """After ``window`` steps the alert is suppressed — Dang & Ngo 2025
    only frames the early-training KL trajectory as the instability signal."""
    assert not _is_kl_spike(10.0, step=KL_SPIKE_WINDOW_STEPS)
    assert not _is_kl_spike(10.0, step=KL_SPIKE_WINDOW_STEPS + 1)


def test_is_kl_spike_window_boundary():
    """Step ``window-1`` is the last in-window step that can fire; ``window``
    is the first that cannot. This pins the off-by-one direction."""
    assert _is_kl_spike(0.6, step=KL_SPIKE_WINDOW_STEPS - 1)
    assert not _is_kl_spike(0.6, step=KL_SPIKE_WINDOW_STEPS)


# =============================================================================
# validate_max_new_tokens — advisory warnings only; pin which conditions
# trigger which warnings.
# =============================================================================

def test_validate_max_new_tokens_in_bounds_silent():
    warns = validate_max_new_tokens(4096, training_seq_length=4096, ci_eval_cap=4096)
    assert warns == []


def test_validate_max_new_tokens_warns_above_ci_cap():
    warns = validate_max_new_tokens(8192, training_seq_length=4096, ci_eval_cap=4096)
    assert len(warns) >= 1
    assert any("CI eval cap" in w for w in warns)


def test_validate_max_new_tokens_warns_above_training_seq():
    warns = validate_max_new_tokens(8192, training_seq_length=4096, ci_eval_cap=8192)
    assert any("max_seq_length" in w for w in warns)


# =============================================================================
# grpo_config_kwargs — pin the values that the trainer actually consumes.
# =============================================================================

def test_grpo_config_kwargs_passes_d5_defaults():
    args = _parse_args(["--output-dir", "/tmp/x"])
    yaml_dict = {"max_seq_length": 4096}
    out = grpo_config_kwargs(
        args=args, yaml_dict=yaml_dict, precision="bf16",
        run_name="rlvr-test", use_wandb=False,
    )
    assert out["learning_rate"] == 3e-6
    assert out["beta"] == 0.04
    assert out["temperature"] == 0.8
    assert out["num_generations"] == 8
    assert out["max_completion_length"] == 4096
    assert out["per_device_train_batch_size"] == 1
    assert out["gradient_accumulation_steps"] == 8
    assert out["num_train_epochs"] == 1
    assert out["seed"] == 42
    assert out["bf16"] is True
    assert out["fp16"] is False
    assert out["report_to"] == "none"
    # Default --max-prompt-length=1024 leaves 4096 tokens of completion
    # room. Combined sequence (5120) exceeds the SFT max_seq_length=4096
    # but stays well under Qwen3-1.7B's actual ~32k context — see the
    # docstring on grpo_config_kwargs for why the SFT-era cap doesn't
    # bind RLVR rollouts.
    assert out["max_prompt_length"] == 1024


def test_grpo_config_kwargs_fp16_path():
    args = _parse_args(["--output-dir", "/tmp/x"])
    out = grpo_config_kwargs(
        args=args, yaml_dict={"max_seq_length": 4096}, precision="fp16",
        run_name="rlvr-test", use_wandb=False,
    )
    assert out["bf16"] is False
    assert out["fp16"] is True


def test_grpo_config_kwargs_wandb_routing():
    args = _parse_args(["--output-dir", "/tmp/x"])
    out = grpo_config_kwargs(
        args=args, yaml_dict={"max_seq_length": 4096}, precision="bf16",
        run_name="rlvr-test", use_wandb=True,
    )
    assert out["report_to"] == "wandb"


# =============================================================================
# default_run_name — shape only.
# =============================================================================

def test_default_run_name_shape():
    name = default_run_name(now=_dt.datetime(2026, 5, 9, 14, 30))
    assert name == "rlvr-20260509-1430"
