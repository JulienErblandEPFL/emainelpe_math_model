"""CPU tests for ``rcp/submit_train.sh``.

These tests shell out to the bash script with ``--dry-run`` and assert
properties of the assembled ``runai submit`` command. They never invoke
``runai`` itself, so they pass on the user's laptop without any cluster
access.

Each test starts from a clean env (the relevant submit_train.sh inputs
are stripped from ``os.environ.copy()``) so an outer shell that already
exported HF_TOKEN/WANDB_API_KEY/etc. cannot mask a regression.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "rcp" / "submit_train.sh"

# Variables submit_train.sh reads from the environment. Stripped before
# every test so inherited values cannot influence the assembled command.
SCRIPT_ENV_VARS = (
    "GASPAR", "GROUP", "HF_TOKEN", "WANDB_API_KEY",
    "RESUME", "IMAGE", "SCRATCH_USER", "N_SAMPLES", "EPOCHS", "REPO_DIR",
    "DATA_OUT_DIR", "SKIP_PREP",
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


def test_placeholder_gaspar_exits_one():
    """``GASPAR=gaspar`` is the placeholder string from RCP_GUIDE; the
    script must refuse rather than submit a job named after the
    placeholder. Same protection covers an unset GASPAR."""
    result = _run(
        {"GASPAR": "gaspar", "GROUP": "g65"},
        ["--dry-run"],
        expect_exit=1,
    )
    assert "GASPAR" in result.stderr


def test_dry_run_contains_python_invocations():
    """The dry-run output must show the chained pipeline that will run
    inside the pod: prepare_sft, then train_sft, with the locked
    50000/2 defaults applied AND with all output paths absolute under
    /scratch/${SCRATCH_USER}/ — never repo-relative.

    Output paths are deliberately outside REPO_DIR so the git tree stays
    clean. A regression that re-points --output-dir or --train-file at a
    repo-relative path would silently mix versioned code with operational
    artifacts; this test catches that."""
    result = _run(
        {"GASPAR": "erbland", "GROUP": "g65"},
        ["--dry-run"],
    )
    assert "python data/prepare_sft.py" in result.stdout
    assert "python scripts/train_sft.py" in result.stdout
    assert "--n-samples 50000" in result.stdout
    assert "--epochs 2" in result.stdout
    # prepare_sft writes to /scratch/${SCRATCH_USER}/data_out
    assert "--output-dir /scratch/Julien/data_out" in result.stdout
    # train_sft reads from the same dir
    assert "--train-file /scratch/Julien/data_out/train.jsonl" in result.stdout
    assert "--eval-file /scratch/Julien/data_out/eval.jsonl" in result.stdout
    # train_sft writes to /scratch/${SCRATCH_USER}/runs/${RUN_NAME}
    # (RUN_NAME contains a timestamp, so just match the prefix)
    assert "--output-dir /scratch/Julien/runs/cs552-erbland-g65-" in result.stdout
    # No repo-relative output paths anywhere in the assembled command.
    assert " data_out/" not in result.stdout
    assert " runs/cs552-" not in result.stdout


def test_resume_set_injects_resume_flag():
    """When the operator exports ``RESUME=latest`` (the train_sft.py
    sentinel for "newest checkpoint under output-dir"), the submit
    script must forward that flag to the python invocation."""
    result = _run(
        {"GASPAR": "erbland", "GROUP": "g65", "RESUME": "latest"},
        ["--dry-run"],
    )
    assert "--resume latest" in result.stdout


def test_resume_unset_omits_resume_flag():
    """``${RESUME:+ --resume ${RESUME}}`` must produce nothing when
    RESUME is unset. Catches a regression where someone writes
    ``${RESUME:-...}`` (default-IF-empty) by accident, which would
    silently inject a literal default value as the resume target."""
    result = _run(
        {"GASPAR": "erbland", "GROUP": "g65"},
        ["--dry-run"],
    )
    assert "--resume" not in result.stdout


def test_default_repo_dir_uses_scratch_user_not_gaspar():
    """Default REPO_DIR must resolve to /scratch/Julien/emainelpe_math_model.
    Catches a regression where someone "fixes" the default and accidentally
    re-conflates SCRATCH_USER with GASPAR (which would yield
    /scratch/erbland/... — wrong; the team's scratch convention uses the
    first name, and inside the pod $USER is "root")."""
    result = _run(
        {"GASPAR": "erbland", "GROUP": "g65"},
        ["--dry-run"],
    )
    assert "/scratch/Julien/emainelpe_math_model" in result.stdout
    assert "/scratch/erbland/" not in result.stdout


def test_data_out_dir_override_redirects_both_prep_and_train():
    """When DATA_OUT_DIR is set, BOTH the prepare_sft --output-dir AND the
    train_sft --train-file/--eval-file must point at the override path —
    not the default /scratch/${SCRATCH_USER}/data_out. Catches a
    regression where only one of the two reads picked up the env var."""
    result = _run(
        {
            "GASPAR": "erbland",
            "GROUP": "g65",
            "DATA_OUT_DIR": "/scratch/Julien/data_out_v2",
        },
        ["--dry-run"],
    )
    assert "--output-dir /scratch/Julien/data_out_v2" in result.stdout
    assert "--train-file /scratch/Julien/data_out_v2/train.jsonl" in result.stdout
    assert "--eval-file /scratch/Julien/data_out_v2/eval.jsonl" in result.stdout
    # The default v1 path must not leak in alongside the override.
    assert "/scratch/Julien/data_out/train.jsonl" not in result.stdout


def test_skip_prep_omits_prepare_sft_call():
    """SKIP_PREP=1 must drop the in-pod `python data/prepare_sft.py` call
    entirely while leaving train_sft.py untouched. Operators rely on this
    when DATA_OUT_DIR points at v2/v3 data prepared offline — running v1
    prep would clobber the carefully-prepped JSONL."""
    result = _run(
        {
            "GASPAR": "erbland",
            "GROUP": "g65",
            "DATA_OUT_DIR": "/scratch/Julien/data_out_v3",
            "SKIP_PREP": "1",
        },
        ["--dry-run"],
    )
    assert "python data/prepare_sft.py" not in result.stdout
    assert "python scripts/train_sft.py" in result.stdout
    # Train flags still point at the override path.
    assert "--train-file /scratch/Julien/data_out_v3/train.jsonl" in result.stdout


def test_skip_prep_unset_keeps_prepare_sft_call():
    """The default (SKIP_PREP unset) must keep emitting prepare_sft.py —
    sister-test to test_skip_prep_omits_prepare_sft_call. Without this
    explicit positive assertion, a regression that always-skips would
    only fail when someone happens to set SKIP_PREP, slipping past CI."""
    result = _run(
        {"GASPAR": "erbland", "GROUP": "g65"},
        ["--dry-run"],
    )
    assert "python data/prepare_sft.py" in result.stdout
    assert "python scripts/train_sft.py" in result.stdout
