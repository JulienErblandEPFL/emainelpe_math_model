"""CPU tests for ``rcp/submit_train_v4.sh`` — v4 SFT submit script.

Mirrors the shape of test_submit_train.py / test_submit_rlvr.py: shell
out with ``--dry-run`` and assert properties of the assembled command.
No cluster access. Each test strips inherited env vars so outer-shell
exports cannot mask a regression.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "rcp" / "submit_train_v4.sh"

# Env vars submit_train_v4.sh consults — stripped before each test.
SCRIPT_ENV_VARS = (
    "GASPAR", "GROUP", "HF_TOKEN", "WANDB_API_KEY",
    "IMAGE", "SCRATCH_USER", "REPO_DIR", "DATA_OUT_DIR",
    "SKIP_PREP", "EPOCHS", "LEARNING_RATE", "INIT_FROM_ADAPTER",
    "V4_OMI2_COUNT", "V4_INTALG_COUNT", "V4_PRECALC_COUNT",
    "V4_LEVEL45_COUNT", "V4_LEVEL13_COUNT", "V4_NUMINAMATH_COUNT",
    "V4_MAX_FORMATTED_TOKENS",
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


# =============================================================================
# Placeholder validation (mirrors test_submit_train.py).
# =============================================================================

def test_placeholder_gaspar_exits_one():
    result = _run(
        {"GASPAR": "gaspar", "GROUP": "g65"},
        ["fresh", "--dry-run"],
        expect_exit=1,
    )
    assert "GASPAR" in result.stderr


def test_unknown_mode_exits_two():
    """Positional arg must be one of: fresh, resume. Anything else is rejected."""
    result = _run(
        {"GASPAR": "erbland", "GROUP": "g65"},
        ["bogus", "--dry-run"],
        expect_exit=2,
    )
    assert "unknown mode" in result.stderr


# =============================================================================
# Mode-derived defaults — fresh vs resume.
# =============================================================================

def test_submit_train_v4_fresh_dry_run():
    """Fresh mode: LR=1e-4 (v3 SFT default), no --init-from-adapter,
    suffix=v4-fresh, runs prepare_sft.py with --source v4-mix."""
    result = _run(
        {"GASPAR": "erbland", "GROUP": "g65"},
        ["fresh", "--dry-run"],
    )
    # Banner shows fresh mode.
    assert "MODE              : fresh" in result.stdout
    # LR = 1e-4 (v3 SFT default for fresh training).
    assert "LEARNING_RATE     : 1e-4" in result.stdout
    # No --init-from-adapter in fresh mode.
    assert "INIT_FROM_ADAPTER : <unset>" in result.stdout
    # Run name uses v4-fresh suffix.
    assert "v4-fresh-" in result.stdout
    # Prepare uses --source v4-mix.
    assert "--source v4-mix" in result.stdout
    # Composition flags present.
    assert "--omi2-count 40000" in result.stdout
    assert "--math-intermediate-algebra-count 12000" in result.stdout
    assert "--math-precalculus-count 7000" in result.stdout
    assert "--math-level45-count 18000" in result.stdout
    assert "--math-level13-count 13000" in result.stdout
    assert "--numinamath-count 5000" in result.stdout
    # Train script invoked with the resolved LR.
    assert "--learning-rate 1e-4" in result.stdout
    # No --init-from-adapter in the python invocation either.
    assert "--init-from-adapter" not in result.stdout


def test_submit_train_v4_resume_dry_run():
    """Resume mode: LR=5e-5 (gentler), --init-from-adapter set to the
    v3 final/ adapter, suffix=v4-resume."""
    result = _run(
        {"GASPAR": "erbland", "GROUP": "g65"},
        ["resume", "--dry-run"],
    )
    assert "MODE              : resume" in result.stdout
    assert "LEARNING_RATE     : 5e-5" in result.stdout
    # INIT_FROM_ADAPTER points to the v3 adapter dir (default).
    assert "INIT_FROM_ADAPTER : /scratch/Julien/runs/" in result.stdout
    assert "v3-omi2-fix2" in result.stdout  # the v3 winner adapter
    # Run name uses v4-resume suffix.
    assert "v4-resume-" in result.stdout
    # Train invoked with --init-from-adapter pointing at v3 adapter.
    assert "--init-from-adapter /scratch/Julien/runs/" in result.stdout
    # LR forwarded correctly.
    assert "--learning-rate 5e-5" in result.stdout
    # Same v4-mix data composition.
    assert "--source v4-mix" in result.stdout
