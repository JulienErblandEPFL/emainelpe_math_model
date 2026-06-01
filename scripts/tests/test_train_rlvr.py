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
  - sys.path bootstrap (regression: ``python scripts/train_rlvr.py --help``
    must succeed because P2 imports ``scripts.reward_fn`` at runtime)
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.train_rlvr import (
    CHAT_TEMPLATE_OPEN_MARKER,
    KL_SPIKE_THRESHOLD,
    KL_SPIKE_WINDOW_STEPS,
    REWARD_SIGNAL_ERROR_STEP,
    REWARD_SIGNAL_WARN_STEP,
    REWARD_SIGNAL_WARN_THRESHOLD,
    REWARD_VARIANCE_THRESHOLD,
    RewardSignalCallback,
    THINK_PREFIX,
    _is_kl_spike,
    _parse_args,
    assert_prompts_are_chat_templated,
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
    args = _parse_args(["--output-dir", "/tmp/x", "--adapter-dir", "/tmp/a", "--prompt-set", "/tmp/p.jsonl"])
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
    args = _parse_args(["--output-dir", "/tmp/x", "--adapter-dir", "/tmp/a", "--prompt-set", "/tmp/p.jsonl", "--dry-run"])
    assert args.dry_run is True


def test_parse_args_length_bonus_defaults_off():
    """The two new length-shaping flags must default to OFF (0.0) and 1024.

    Backward-compat contract: an existing invocation that does NOT pass
    --length-bonus-weight has the exact same training behavior as before
    the flag existed. main() will set reward_fn.LENGTH_BONUS_WEIGHT=0.0
    from this default, which short-circuits the length term inside
    reward_fn.compute_reward.
    """
    args = _parse_args(["--output-dir", "/tmp/x", "--adapter-dir", "/tmp/a", "--prompt-set", "/tmp/p.jsonl"])
    assert args.length_bonus_weight == 0.0
    assert args.target_length_tokens == 1024


def test_parse_args_length_bonus_custom_values():
    """When passed, the two flags parse as float / int respectively.

    Catches regressions like accidentally typing ``type=int`` on
    --length-bonus-weight (0.1 would silently become 0).
    """
    args = _parse_args([
        "--output-dir", "/tmp/x",
        "--adapter-dir", "/tmp/a",
        "--prompt-set", "/tmp/p.jsonl",
        "--length-bonus-weight", "0.1",
        "--target-length-tokens", "512",
    ])
    assert args.length_bonus_weight == 0.1
    assert args.target_length_tokens == 512
    assert isinstance(args.length_bonus_weight, float)
    assert isinstance(args.target_length_tokens, int)


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
    args = _parse_args(["--output-dir", "/tmp/x", "--adapter-dir", "/tmp/a", "--prompt-set", "/tmp/p.jsonl"])
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
    # max_prompt_length intentionally absent — TRL 0.19.1's GRPOConfig
    # rejects it (verified empirically on the course image 2026-05-12).
    # Prompt-length truncation defers to the tokenizer's own limits.
    assert "max_prompt_length" not in out


def test_grpo_config_kwargs_fp16_path():
    args = _parse_args(["--output-dir", "/tmp/x", "--adapter-dir", "/tmp/a", "--prompt-set", "/tmp/p.jsonl"])
    out = grpo_config_kwargs(
        args=args, yaml_dict={"max_seq_length": 4096}, precision="fp16",
        run_name="rlvr-test", use_wandb=False,
    )
    assert out["bf16"] is False
    assert out["fp16"] is True


def test_grpo_config_kwargs_wandb_routing():
    args = _parse_args(["--output-dir", "/tmp/x", "--adapter-dir", "/tmp/a", "--prompt-set", "/tmp/p.jsonl"])
    out = grpo_config_kwargs(
        args=args, yaml_dict={"max_seq_length": 4096}, precision="bf16",
        run_name="rlvr-test", use_wandb=True,
    )
    assert out["report_to"] == "wandb"


# =============================================================================
# Rescue-config knobs — added 2026-05-13 after the retry3 starvation incident.
# The "do no harm" contract: with no new CLI flags set, the kwargs dict must
# reflect the pre-rescue defaults that match the failed run's config.
# =============================================================================

def test_grpo_config_kwargs_defaults_preserve_old_behavior():
    """Without the new flags, kwargs must match the failed-run config so the
    patch is byte-stable for existing invocations.

    Pre-rescue values (= TRL 0.19.1 defaults that the retry3 run used):
      loss_type='dapo', use_vllm=False, vllm_gpu_memory_utilization=0.3,
      mask_truncated_completions=False, log_completions=False.
    """
    args = _parse_args(["--output-dir", "/tmp/x", "--adapter-dir", "/tmp/a", "--prompt-set", "/tmp/p.jsonl"])
    out = grpo_config_kwargs(
        args=args, yaml_dict={"max_seq_length": 4096}, precision="bf16",
        run_name="rlvr-test", use_wandb=False,
    )
    assert out["loss_type"] == "dapo"
    assert out["use_vllm"] is False
    assert out["vllm_gpu_memory_utilization"] == 0.3
    assert out["mask_truncated_completions"] is False
    assert out["log_completions"] is False


def test_grpo_config_kwargs_includes_new_fields():
    """All 5 rescue knobs flow from CLI flags into the kwargs dict."""
    args = _parse_args([
        "--output-dir", "/tmp/x",
        "--adapter-dir", "/tmp/a",
        "--prompt-set", "/tmp/p.jsonl",
        "--loss-type", "grpo",
        "--use-vllm",
        "--vllm-gpu-memory-utilization", "0.45",
        "--mask-truncated-completions",
        "--log-completions",
    ])
    out = grpo_config_kwargs(
        args=args, yaml_dict={"max_seq_length": 4096}, precision="bf16",
        run_name="rlvr-rescue", use_wandb=False,
    )
    assert out["loss_type"] == "grpo"
    assert out["use_vllm"] is True
    assert out["vllm_gpu_memory_utilization"] == 0.45
    assert out["mask_truncated_completions"] is True
    assert out["log_completions"] is True


# =============================================================================
# Liger Kernel — same primary OOM mitigation as SFT. Added 2026-05-13.
# Plumbing-only tests; the kernel itself runs on the cluster.
# =============================================================================

def test_grpo_config_includes_use_liger_kernel_true_by_default():
    """Default --use-liger-kernel=True must flow into GRPOConfig kwargs.
    The failed retry3 run's W&B config showed use_liger_kernel=False,
    leaving the same OOM door open for RL as v4 SFT hit at step 1514."""
    args = _parse_args(["--output-dir", "/tmp/x", "--adapter-dir", "/tmp/a", "--prompt-set", "/tmp/p.jsonl"])
    out = grpo_config_kwargs(
        args=args, yaml_dict={"max_seq_length": 4096}, precision="bf16",
        run_name="rlvr-test", use_wandb=False,
    )
    assert "use_liger_kernel" in out, (
        "use_liger_kernel must be set explicitly in GRPOConfig kwargs; "
        "TRL's default is False, which leaves the OOM door open."
    )
    assert out["use_liger_kernel"] is True, (
        f"use_liger_kernel={out['use_liger_kernel']}; must be True by "
        "default. See CLAUDE.md → OOM mitigations."
    )


def test_grpo_config_use_liger_kernel_disables_when_cli_overrides():
    """`--no-use-liger-kernel` flips GRPOConfig's field to False for
    A/B comparison."""
    args = _parse_args(["--output-dir", "/tmp/x", "--adapter-dir", "/tmp/a", "--prompt-set", "/tmp/p.jsonl", "--no-use-liger-kernel"])
    out = grpo_config_kwargs(
        args=args, yaml_dict={"max_seq_length": 4096}, precision="bf16",
        run_name="rlvr-test", use_wandb=False,
    )
    assert out["use_liger_kernel"] is False


def test_parse_args_use_liger_kernel_default_is_true_rlvr():
    """argparse default for --use-liger-kernel is True so the default
    RLVR invocation gets the OOM-safe path."""
    args = _parse_args(["--output-dir", "/tmp/x", "--adapter-dir", "/tmp/a", "--prompt-set", "/tmp/p.jsonl"])
    assert args.use_liger_kernel is True


def test_parse_args_no_use_liger_kernel_flips_to_false_rlvr():
    args = _parse_args(["--output-dir", "/tmp/x", "--adapter-dir", "/tmp/a", "--prompt-set", "/tmp/p.jsonl", "--no-use-liger-kernel"])
    assert args.use_liger_kernel is False


# =============================================================================
# RewardSignalCallback — pure-Python state machine. Tests drive it directly,
# no transformers/TrainerCallback import needed.
# =============================================================================

def _feed_window(cb: RewardSignalCallback, value: float, count: int) -> None:
    """Pre-fill the rolling window with `count` log entries at `value`.

    Each call uses global_step=0 so it accumulates into the window without
    triggering the warn check (which requires step >= WARN_STEP).
    """
    for _ in range(count):
        cb.on_log({"frac_reward_zero_std": value}, global_step=0)


def test_reward_signal_callback_warns_at_step_100(caplog):
    """At step 100 with rolling mean > 0.5, the callback fires a WARNING.

    The window is pre-filled with 1.0 values (the retry3 signature). The
    threshold trigger happens on the on_log() call at step=100, which is
    when the WARN_STEP gate first opens.
    """
    cb = RewardSignalCallback(hard_kill=False)
    _feed_window(cb, value=1.0, count=49)
    caplog.clear()
    with caplog.at_level("WARNING", logger="train_rlvr"):
        cb.on_log({"frac_reward_zero_std": 1.0}, global_step=REWARD_SIGNAL_WARN_STEP)
    warns = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warns) == 1
    assert "SIGNAL WEAK" in warns[0].message
    assert "rolling mean" in warns[0].message
    assert f"step {REWARD_SIGNAL_WARN_STEP}" in warns[0].message


def test_reward_signal_callback_escalates_at_step_200(caplog):
    """At step 200 with rolling mean still > 0.5, the callback escalates
    from WARNING to ERROR (without raising — hard_kill=False)."""
    cb = RewardSignalCallback(hard_kill=False)
    _feed_window(cb, value=1.0, count=49)
    caplog.clear()
    with caplog.at_level("ERROR", logger="train_rlvr"):
        cb.on_log({"frac_reward_zero_std": 1.0}, global_step=REWARD_SIGNAL_ERROR_STEP)
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1
    assert "SIGNAL WEAK" in errors[0].message
    assert f"step {REWARD_SIGNAL_ERROR_STEP}" in errors[0].message


def test_reward_signal_callback_hard_kill_when_enabled(caplog):
    """With hard_kill=True, the step-200 escalation also raises
    RuntimeError so the cluster job aborts cleanly rather than burning
    wall-clock on a starved run. The ERROR log is emitted *before* the
    raise, so operators see the diagnostic in pod logs.

    With hard_kill=False (default), the same signal only logs — does not
    raise. This is the documented difference between the two modes.
    """
    cb_kill = RewardSignalCallback(hard_kill=True)
    _feed_window(cb_kill, value=1.0, count=49)
    caplog.clear()
    with caplog.at_level("ERROR", logger="train_rlvr"):
        with pytest.raises(RuntimeError, match="SIGNAL WEAK"):
            cb_kill.on_log(
                {"frac_reward_zero_std": 1.0},
                global_step=REWARD_SIGNAL_ERROR_STEP,
            )
    # The ERROR log MUST have fired before the raise.
    assert any(r.levelname == "ERROR" and "SIGNAL WEAK" in r.message
               for r in caplog.records)

    # Same input, hard_kill=False: no raise, just an ERROR log.
    cb_soft = RewardSignalCallback(hard_kill=False)
    _feed_window(cb_soft, value=1.0, count=49)
    # No exception expected:
    cb_soft.on_log(
        {"frac_reward_zero_std": 1.0},
        global_step=REWARD_SIGNAL_ERROR_STEP,
    )


def test_reward_signal_callback_silent_when_signal_healthy(caplog):
    """With rolling mean < 0.3 (well below WARN_THRESHOLD), no warnings
    fire at any step — even past ERROR_STEP. This is the do-not-bother-me
    case: a normally curated prompt set sits at frac_reward_zero_std
    around 0.1-0.3 after warmup.
    """
    cb = RewardSignalCallback(hard_kill=True)  # even with hard_kill on
    _feed_window(cb, value=0.2, count=49)
    caplog.clear()
    with caplog.at_level("WARNING", logger="train_rlvr"):
        cb.on_log({"frac_reward_zero_std": 0.2}, global_step=REWARD_SIGNAL_WARN_STEP)
        cb.on_log({"frac_reward_zero_std": 0.2}, global_step=REWARD_SIGNAL_ERROR_STEP)
        cb.on_log({"frac_reward_zero_std": 0.2}, global_step=500)
    # No warnings, no errors.
    assert all(r.levelname not in ("WARNING", "ERROR") for r in caplog.records)


def test_reward_signal_callback_ignores_logs_without_the_key():
    """Trainer fires on_log for many event types — most don't carry the
    GRPO reward stats. The callback must be tolerant of None/empty logs
    and logs missing the frac_reward_zero_std key."""
    cb = RewardSignalCallback()
    cb.on_log(None, global_step=150)  # no-op, no raise
    cb.on_log({}, global_step=150)
    cb.on_log({"loss": 0.5}, global_step=150)  # different metric
    assert cb.window == []  # nothing recorded


def test_reward_signal_callback_silent_before_warn_step(caplog):
    """Even with rolling mean = 1.0, the callback stays silent until
    global_step crosses WARN_STEP. The first 50 steps of training are
    warmup and the rolling window hasn't stabilized."""
    cb = RewardSignalCallback()
    _feed_window(cb, value=1.0, count=50)
    caplog.clear()
    with caplog.at_level("WARNING", logger="train_rlvr"):
        cb.on_log({"frac_reward_zero_std": 1.0},
                  global_step=REWARD_SIGNAL_WARN_STEP - 1)
    assert all(r.levelname != "WARNING" for r in caplog.records)


# =============================================================================
# TRL API drift — instantiate GRPOConfig with the actual kwargs.
#
# The 2026-05-12 RLVR pod crash was a silent API-drift bug: TRL 0.19.1
# rejected the ``max_prompt_length`` kwarg that grpo_config_kwargs() had
# always returned. The dict-shape test above pins the *absence* of
# max_prompt_length, but only a real GRPOConfig(**kwargs) call can catch
# future drift on other kwargs. We gate on importorskip so the test
# runs in any environment where TRL is installed (CI, RCP pod) and is
# silently skipped on the user's CPU-only laptop.
# =============================================================================

def test_grpoconfig_accepts_all_kwargs(tmp_path: Path):
    """Every key in ``grpo_config_kwargs()`` must be a valid GRPOConfig
    ``__init__`` parameter.

    Skipped on environments where TRL is not installed (e.g., the user's
    laptop). On the course image (TRL 0.19.1) and any future-version
    pod, this test fires and would have caught the 2026-05-12
    ``max_prompt_length`` regression before the pod ran.

    We deliberately compare against ``inspect.signature(...).parameters``
    instead of attempting full ``GRPOConfig(**kwargs)`` instantiation
    because the latter triggers ``transformers.TrainingArguments``
    runtime checks (bf16-requires-GPU, etc.) that fail on CPU-only dev
    boxes — those are environment validations, not API-drift signals.
    """
    import inspect
    trl = pytest.importorskip("trl")
    valid_params = set(
        inspect.signature(trl.GRPOConfig.__init__).parameters.keys()
    )

    args = _parse_args([
        "--output-dir", str(tmp_path),
        "--adapter-dir", "/tmp/a",
        "--prompt-set", "/tmp/p.jsonl",
    ])
    kwargs = grpo_config_kwargs(
        args=args, yaml_dict={"max_seq_length": 4096}, precision="bf16",
        run_name="rlvr-test", use_wandb=False,
    )
    unexpected = set(kwargs.keys()) - valid_params
    assert not unexpected, (
        f"grpo_config_kwargs() emits keys that TRL "
        f"{getattr(trl, '__version__', '?')} GRPOConfig rejects: "
        f"{sorted(unexpected)}. The 2026-05-12 crash had "
        f"{{'max_prompt_length'}} here — fix grpo_config_kwargs to use "
        f"the current TRL kwarg names."
    )


# =============================================================================
# default_run_name — shape only.
# =============================================================================

def test_default_run_name_shape():
    name = default_run_name(now=_dt.datetime(2026, 5, 9, 14, 30))
    assert name == "rlvr-20260509-1430"


# =============================================================================
# Chat-template sanity check — defends against the 2026-05-12 retry2
# degenerate-rollout incident (raw prompts → 100% token-cap clipping →
# reward_std=0 → no learning signal).
# =============================================================================

_TEMPLATED_ROW = {
    "prompt": (
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    ),
    "answer": "4",
}
_RAW_ROW = {
    "prompt": r"Let $\mathbf{a} = \langle x, y\rangle$. Compute ...",
    "answer": "42",
}
# Chat-templated but missing the THINK_PREFIX — the 2026-05-12 retry3
# bug shape. Has <|im_start|> wrapping but ends at ``assistant\n``.
_NO_THINK_ROW = {
    "prompt": (
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "answer": "4",
}


def test_assert_prompts_are_chat_templated_accepts_templated_prompts():
    assert CHAT_TEMPLATE_OPEN_MARKER in _TEMPLATED_ROW["prompt"]
    assert _TEMPLATED_ROW["prompt"].endswith(THINK_PREFIX)
    # No exception.
    assert_prompts_are_chat_templated([_TEMPLATED_ROW, _TEMPLATED_ROW])


def test_assert_prompts_are_chat_templated_rejects_raw_prompts():
    with pytest.raises(RuntimeError, match="NOT chat-templated"):
        assert_prompts_are_chat_templated([_RAW_ROW])


def test_assert_prompts_are_chat_templated_rejects_no_think_prefix():
    """Retry3 incident — chat-templated but missing the <think>\\n
    suffix. The model needs the prefix to stay in the trained regime;
    without it, rollouts at temp=0.8 drop <think> and never terminate."""
    assert CHAT_TEMPLATE_OPEN_MARKER in _NO_THINK_ROW["prompt"]
    assert not _NO_THINK_ROW["prompt"].endswith(THINK_PREFIX)
    with pytest.raises(RuntimeError, match=r"<think>"):
        assert_prompts_are_chat_templated([_NO_THINK_ROW])


def test_assert_prompts_are_chat_templated_empty_list_is_noop():
    """Empty prompt sets are blocked earlier in main() with a clearer
    error; this helper should not double-fire on empty input."""
    assert_prompts_are_chat_templated([])


def test_assert_prompts_are_chat_templated_samples_only_first_n():
    """A long valid prefix followed by a stray raw row beyond sample_size
    should not trip the check (we trust the curation invariant on the
    bulk of the file; the sanity check is a cheap pre-GPU smoke)."""
    rows = [_TEMPLATED_ROW] * 5 + [_RAW_ROW]
    # Default sample_size=5 — raw row at idx 5 is beyond the window.
    assert_prompts_are_chat_templated(rows)
    # But with sample_size=6 the raw row is in scope and trips.
    with pytest.raises(RuntimeError, match="NOT chat-templated"):
        assert_prompts_are_chat_templated(rows, sample_size=6)


# =============================================================================
# sys.path bootstrap regression — exercises script-mode import resolution.
#
# RLVR run on 2026-05-12 crashed at preflight P2 with:
#   ModuleNotFoundError: No module named 'scripts'
# because ``python scripts/train_rlvr.py`` puts scripts/ (not the repo
# root) on sys.path, so the deferred ``from scripts.reward_fn import
# compute_reward`` at line 399 / 649 could not resolve. The fix is the
# canonical sys.path.insert snippet at the top of train_rlvr.py — same
# idiom as run_eval.py and prepare_rlvr.py.
#
# This test catches the regression cheaply: ``--help`` exits before any
# GPU/TRL/torch code runs, but still triggers module-level imports.
# =============================================================================

def test_script_mode_help_succeeds_without_importerror():
    """``python scripts/train_rlvr.py --help`` must exit 0.

    Runs from the repo root so the invocation matches how submit_rlvr.sh
    launches it on the cluster. ``--help`` short-circuits past the heavy
    ML imports — we are only proving that the module-level imports +
    sys.path bootstrap don't blow up.
    """
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "train_rlvr.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"exit={result.returncode}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "ModuleNotFoundError" not in result.stderr
    # argparse --help output should mention at least one D5 flag.
    assert "--learning-rate" in result.stdout


def test_deferred_scripts_reward_fn_import_resolves_in_script_mode():
    """Direct regression test for the 2026-05-12 P2 crash.

    Simulates ``python scripts/train_rlvr.py`` (which puts ``scripts/``
    on sys.path[0] but NOT the repo root), then loads train_rlvr.py via
    importlib so its module body — including the sys.path bootstrap —
    runs. After that, the deferred ``from scripts.reward_fn import
    compute_reward`` used inside ``reward_variance_preflight_p2`` must
    resolve. If anyone removes the sys.path snippet at the top of
    train_rlvr.py, this test fails with ModuleNotFoundError.
    """
    repo_root = Path(__file__).resolve().parents[2]
    inline = (
        "import sys, pathlib, importlib.util\n"
        f"repo = pathlib.Path({str(repo_root)!r})\n"
        # Mimic `python scripts/train_rlvr.py`: scripts/ on sys.path,
        # repo root deliberately absent.
        "sys.path.insert(0, str(repo / 'scripts'))\n"
        "spec = importlib.util.spec_from_file_location("
        "'train_rlvr', str(repo / 'scripts' / 'train_rlvr.py'))\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        # The deferred import that crashed on 2026-05-12 at P2.
        "from scripts.reward_fn import compute_reward\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", inline],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"exit={result.returncode}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "ModuleNotFoundError" not in result.stderr
    assert result.stdout.strip().endswith("OK")
