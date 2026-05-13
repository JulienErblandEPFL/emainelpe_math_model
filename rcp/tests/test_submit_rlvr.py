"""CPU tests for ``rcp/submit_rlvr.sh``.

Same shape as ``test_submit_train.py``: shell out with ``--dry-run`` and
assert properties of the assembled ``runai submit`` command. No cluster
access required. Each test starts from a clean env so an outer shell that
exported HF_TOKEN/etc. cannot mask a regression.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "rcp" / "submit_rlvr.sh"

# Variables submit_rlvr.sh reads from the environment. Stripped before
# every test so inherited values cannot influence the assembled command.
SCRIPT_ENV_VARS = (
    "GASPAR", "GROUP", "HF_TOKEN", "WANDB_API_KEY",
    "IMAGE", "SCRATCH_USER", "REPO_DIR", "DATA_OUT_DIR",
    "ADAPTER_DIR", "PROMPT_SET", "SFT_MODEL",
    "MAX_PROMPTS", "POOL_SIZE", "TARGET_SIZE",
    "LEARNING_RATE", "KL_COEF", "ROLLOUT_TEMP", "MAX_NEW_TOKENS",
    "SKIP_CURATION", "SKIP_PREFLIGHTS",
    # Rescue-config knobs (added 2026-05-13).
    "LOSS_TYPE", "USE_VLLM", "VLLM_GPU_MEM_UTIL",
    "MASK_TRUNCATED", "LOG_COMPLETIONS", "HARD_KILL_ON_WEAK_SIGNAL",
    "DIFFICULTY_MIN", "DIFFICULTY_MAX",
)


def _run(env_overrides: dict, args: list[str], expect_exit: int = 0):
    env = os.environ.copy()
    for k in SCRIPT_ENV_VARS:
        env.pop(k, None)
    env.update(env_overrides)
    result = subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == expect_exit, (
        f"exit={result.returncode}, expected {expect_exit}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result


def _extract_pod_cmd(dry_run_stdout: str) -> str:
    """Pull the assembled pod command out of --dry-run output. See the
    twin helper in test_submit_train.py for the rationale."""
    lines = dry_run_stdout.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "-lc" and i + 1 < len(lines):
            return lines[i + 1]
    raise AssertionError(
        "Could not locate the '-lc' marker in dry-run output. "
        "submit_rlvr.sh may have changed its print_args_masked layout."
    )


# =============================================================================
# Placeholder validation — same protections as submit_train.sh.
# =============================================================================

def test_placeholder_gaspar_exits_one():
    result = _run(
        {"GASPAR": "gaspar", "GROUP": "g65"},
        ["--dry-run"],
        expect_exit=1,
    )
    assert "GASPAR" in result.stderr


def test_placeholder_group_exits_one():
    result = _run(
        {"GASPAR": "erbland", "GROUP": "gXX"},
        ["--dry-run"],
        expect_exit=1,
    )
    assert "GROUP" in result.stderr


# =============================================================================
# Dry-run pipeline body — should run prepare_rlvr THEN train_rlvr,
# wired with the right CLI flags. Output paths must be absolute under
# /scratch/${SCRATCH_USER}/, never repo-relative.
# =============================================================================

def test_dry_run_chains_curation_then_training():
    """The pod command is `prepare_rlvr.py && train_rlvr.py`. Both must
    appear, in that order, with absolute output paths under /scratch."""
    result = _run(
        {"GASPAR": "erbland", "GROUP": "g65"},
        ["--dry-run"],
    )
    assert "python data/prepare_rlvr.py" in result.stdout
    assert "python scripts/train_rlvr.py" in result.stdout
    # prepare_rlvr.py runs first
    idx_curation = result.stdout.index("python data/prepare_rlvr.py")
    idx_train = result.stdout.index("python scripts/train_rlvr.py")
    assert idx_curation < idx_train, "curation must run before training"
    # Output paths absolute
    assert "/scratch/Julien/data_out/rlvr_prompts.jsonl" in result.stdout
    assert "/scratch/Julien/runs/cs552-erbland-g65-rlvr-" in result.stdout


def test_dry_run_default_hyperparameters():
    """D5 hyperparameters are the dry-run defaults."""
    result = _run(
        {"GASPAR": "erbland", "GROUP": "g65"},
        ["--dry-run"],
    )
    assert "--learning-rate 3e-6" in result.stdout
    assert "--kl-coef 0.04" in result.stdout
    assert "--rollout-temp 0.8" in result.stdout
    assert "--max-prompts 5000" in result.stdout


def test_dry_run_default_adapter_dir():
    """Default ADAPTER_DIR points at the v1 SFT run from 2026-05-08."""
    result = _run(
        {"GASPAR": "erbland", "GROUP": "g65"},
        ["--dry-run"],
    )
    assert "cs552-erbland-g65-train-20260508-150203/final" in result.stdout


# =============================================================================
# Env var customization — operator can override any default.
# =============================================================================

def test_skip_curation_drops_prepare_rlvr_from_pipeline():
    """Setting SKIP_CURATION=1 makes the pod command go straight to
    train_rlvr.py, using whatever PROMPT_SET already exists. Useful when
    iterating on training hyperparameters without re-curating."""
    result = _run(
        {"GASPAR": "erbland", "GROUP": "g65", "SKIP_CURATION": "1"},
        ["--dry-run"],
    )
    assert "python data/prepare_rlvr.py" not in result.stdout
    assert "python scripts/train_rlvr.py" in result.stdout


def test_skip_preflights_forwarded_to_train_rlvr():
    """SKIP_PREFLIGHTS=1 should add --skip-preflights to the train CLI.
    This is dangerous for real runs but useful for trainer-wiring debug."""
    result = _run(
        {"GASPAR": "erbland", "GROUP": "g65", "SKIP_PREFLIGHTS": "1"},
        ["--dry-run"],
    )
    assert "--skip-preflights" in result.stdout


def test_custom_learning_rate_propagates():
    result = _run(
        {"GASPAR": "erbland", "GROUP": "g65", "LEARNING_RATE": "1e-6"},
        ["--dry-run"],
    )
    assert "--learning-rate 1e-6" in result.stdout
    assert "--learning-rate 3e-6" not in result.stdout


def test_custom_prompt_set_propagates():
    """An ablation run targeting a different prompt set should not
    regenerate the file (skip curation) AND use the override path."""
    result = _run(
        {
            "GASPAR": "erbland",
            "GROUP": "g65",
            "PROMPT_SET": "/scratch/Julien/data_out/rlvr_prompts_v2.jsonl",
            "SKIP_CURATION": "1",
        },
        ["--dry-run"],
    )
    assert "rlvr_prompts_v2.jsonl" in result.stdout
    assert "python data/prepare_rlvr.py" not in result.stdout


def test_run_name_contains_rlvr_suffix():
    result = _run(
        {"GASPAR": "erbland", "GROUP": "g65"},
        ["--dry-run"],
    )
    # Default suffix is "rlvr", so the runai --name should reflect it.
    assert "cs552-erbland-g65-rlvr-" in result.stdout


def test_token_masking_in_dry_run():
    """The dry-run output must NEVER contain the actual token values
    even if the operator set them. This makes the dry-run safe to paste
    into a Slack message or PR comment."""
    result = _run(
        {
            "GASPAR": "erbland",
            "GROUP": "g65",
            "HF_TOKEN": "hf_secret_token_value",
            "WANDB_API_KEY": "wandb_secret_value",
        },
        ["--dry-run"],
    )
    assert "hf_secret_token_value" not in result.stdout
    assert "wandb_secret_value" not in result.stdout
    assert "HF_TOKEN=<set>" in result.stdout
    assert "WANDB_API_KEY=<set>" in result.stdout


# =============================================================================
# Shell-quoting regression guard — see test_submit_train.py for rationale.
# =============================================================================

def test_pod_cmd_passes_bash_syntax_check():
    result = _run(
        {"GASPAR": "erbland", "GROUP": "g65"},
        ["--dry-run"],
    )
    pod_cmd = _extract_pod_cmd(result.stdout)
    syntax = subprocess.run(
        ["bash", "-n", "-c", pod_cmd],
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, (
        f"POD_CMD failed bash -n syntax check:\n"
        f"  exit:   {syntax.returncode}\n"
        f"  stderr: {syntax.stderr}\n"
        f"  POD_CMD: {pod_cmd}"
    )


def test_pod_cmd_liger_sanity_check_uses_safe_quoting():
    """Pin the corrected outer-double/inner-single quoting form."""
    result = _run(
        {"GASPAR": "erbland", "GROUP": "g65"},
        ["--dry-run"],
    )
    pod_cmd = _extract_pod_cmd(result.stdout)
    expected = (
        "python -c \"import liger_kernel; "
        "from liger_kernel.transformers import apply_liger_kernel_to_qwen3; "
        "print('liger_kernel imported OK (Qwen3 patch available)')\""
    )
    assert expected in pod_cmd, (
        f"Liger sanity check is not in the expected outer-double/inner-"
        f"single-quote form. POD_CMD slice:\n  {pod_cmd}"
    )
