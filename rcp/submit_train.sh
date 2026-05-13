#!/usr/bin/env bash
# rcp/submit_train.sh — submit the SFT pipeline (data prep + LoRA train) to RCP.
#
# Usage:
#   GASPAR=erbland GROUP=g65 ./rcp/submit_train.sh             # real submit
#   GASPAR=erbland GROUP=g65 ./rcp/submit_train.sh --dry-run   # print only
#   GASPAR=erbland GROUP=g65 ./rcp/submit_train.sh smoke50k    # custom suffix
#
# Required env vars (refused with placeholder values):
#   GASPAR   EPFL username (must NOT be the literal "gaspar")
#   GROUP    Team number, e.g. g65 (must NOT be "gXX" or "g00")
#
# Recommended env vars (warn-only if unset):
#   HF_TOKEN          HF Hub token; needed for any push and for gated datasets
#   WANDB_API_KEY     W&B token; without it train_sft.py logs to stdout only
#
# Optional env vars (have defaults):
#   IMAGE         Course Docker image. Default: ayushkumartarun/course-cs-552-standard:v1
#   SCRATCH_USER  First-level directory name under /scratch where the team
#                 repo lives. Default: Julien. INTENTIONALLY SEPARATE FROM
#                 GASPAR — the scratch path component is whatever name the
#                 team picked (first name, in our case) and is NOT derivable
#                 from the EPFL username. Inside the pod $USER resolves to
#                 "root", which would also be wrong as a path component.
#   REPO_DIR      Repo path inside the pod.
#                 Default: /scratch/${SCRATCH_USER}/emainelpe_math_model
#   DATA_OUT_DIR  Where train.jsonl / eval.jsonl live (and where prepare_sft.py
#                 writes, unless SKIP_PREP is set). Default:
#                 /scratch/${SCRATCH_USER}/data_out. Override to point at v2
#                 (mixed) or v3 (pure OMI2) data, e.g.
#                 DATA_OUT_DIR=/scratch/Julien/data_out_v2.
#   SKIP_PREP     If non-empty, skip the in-pod prepare_sft.py call entirely
#                 and go straight to train_sft.py. Required when DATA_OUT_DIR
#                 points at v2/v3 data prepared offline — otherwise the
#                 default v1 DART prep would overwrite it.
#   N_SAMPLES     prepare_sft.py --n-samples. Default: 50000. Ignored when
#                 SKIP_PREP is set.
#   EPOCHS        train_sft.py --epochs.    Default: 2
#   RESUME        Forwarded to train_sft.py --resume when non-empty.
#                 Use "latest" to resume from the newest checkpoint under
#                 runs/${RUN_NAME}, or pass an explicit checkpoint path.

set -euo pipefail

# ----- Defaults ----------------------------------------------------------
IMAGE="${IMAGE:-ayushkumartarun/course-cs-552-standard:v1}"
# SCRATCH_USER is the first-level dir under /scratch where the team repo
# lives. NOT derivable from GASPAR (EPFL username); the team picked
# "Julien" by convention. Inside the pod $USER is "root", so we cannot
# infer it at runtime either — the operator must override this if the
# convention changes.
SCRATCH_USER="${SCRATCH_USER:-Julien}"
N_SAMPLES="${N_SAMPLES:-50000}"
EPOCHS="${EPOCHS:-2}"
RESUME="${RESUME:-}"

# ----- Argument parsing --------------------------------------------------
DRY_RUN=0
SUFFIX="train"
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,40p' "$0"
      exit 0
      ;;
    -*)
      echo "ERROR: unknown flag: $1" >&2
      exit 2
      ;;
    *) SUFFIX="$1" ;;
  esac
  shift
done

# ----- Placeholder validation -------------------------------------------
if [[ -z "${GASPAR:-}" || "${GASPAR}" == "gaspar" ]]; then
  echo "ERROR: set GASPAR=<your-EPFL-username> (got: '${GASPAR:-}')" >&2
  exit 1
fi
if [[ -z "${GROUP:-}" || "${GROUP}" =~ ^g(XX|00)$ ]]; then
  echo "ERROR: set GROUP=g<NN> (got: '${GROUP:-}')" >&2
  exit 1
fi

# Token warnings — don't exit; the operator may have set them another way
# (e.g. baked into the image, or for a no-push smoke test).
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN unset — gated HF resources will fail to download" >&2
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARN: WANDB_API_KEY unset — loss curves go to stdout only" >&2
fi

# ----- Names + paths -----------------------------------------------------
# Kubernetes object names (which Run:AI inherits for jobs) are capped at
# 63 characters. The pattern below produces ~38–45 chars for typical
# inputs (e.g. cs552-erbland-g65-train-20260508-013045 = 39), leaving
# headroom. An unusually long GASPAR or SUFFIX could overflow; if
# `runai submit` rejects with `must be no more than 63 characters`,
# shorten one of them.
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-${TIMESTAMP}"
REPO_DIR="${REPO_DIR:-/scratch/${SCRATCH_USER}/emainelpe_math_model}"

# ----- The in-pod command -----------------------------------------------
# Output paths live OUTSIDE the repo tree so version-controlled code and
# operational artifacts don't share a directory. `git pull`/`git diff` in
# REPO_DIR stay clean; `data_out/` and `runs/` are just sibling dirs
# under /scratch/${SCRATCH_USER}/. The .gitignore safety net is no longer
# the only line of defense.
#
# DATA_OUT_DIR override semantics: set the env var to point at v2/v3 data
# (e.g. DATA_OUT_DIR=/scratch/Julien/data_out_v2). Set SKIP_PREP=1 to skip
# the in-pod prepare_sft.py step (assumes data already exists at
# DATA_OUT_DIR). Without SKIP_PREP, prepare_sft.py runs with v1 defaults
# and would clobber any v2/v3 data already at that path.
DATA_OUT_DIR="${DATA_OUT_DIR:-/scratch/${SCRATCH_USER}/data_out}"
RUN_OUT_DIR="/scratch/${SCRATCH_USER}/runs/${RUN_NAME}"

# `${RESUME:+ --resume ${RESUME}}` expands to "" when RESUME is unset/empty
# and " --resume <value>" when it is set. NOT `${RESUME:-...}`, which
# would default IN a value when empty (the wrong direction).
TRAIN_FLAGS="--train-file ${DATA_OUT_DIR}/train.jsonl"
TRAIN_FLAGS+=" --eval-file ${DATA_OUT_DIR}/eval.jsonl"
TRAIN_FLAGS+=" --output-dir ${RUN_OUT_DIR}"
TRAIN_FLAGS+=" --run-name ${RUN_NAME}"
TRAIN_FLAGS+=" --epochs ${EPOCHS}"
TRAIN_FLAGS+="${RESUME:+ --resume ${RESUME}}"

# Built as a single string passed to `bash -lc` inside the pod. The `\$`
# escape keeps `$(command -v python3)` literal in the string we send to
# the pod, where it is evaluated by the pod's shell (the course image's
# `python` symlink is inconsistent across versions; per RCP_GUIDE).
POD_CMD="ln -sf \"\$(command -v python3)\" /usr/local/bin/python"
POD_CMD+=" && cd ${REPO_DIR}"
POD_CMD+=" && pip install -r requirements.txt"
# Liger Kernel sanity check: fail fast if the install ever breaks (the
# primary OOM mitigation depends on this import succeeding before
# train_sft.py launches). See submit_train_v4.sh for the rationale.
# liger-kernel 0.8.0 doesn't expose __version__; the Qwen3-patch import
# validates the model-specific entry point we actually use.
POD_CMD+=" && python -c \"import liger_kernel; from liger_kernel.transformers import apply_liger_kernel_to_qwen3; print('liger_kernel imported OK (Qwen3 patch available)')\""
# Skip prepare_sft.py when SKIP_PREP is set — used when DATA_OUT_DIR
# already contains v2/v3 data prepared offline; running the default v1
# prep would clobber it.
if [[ -z "${SKIP_PREP:-}" ]]; then
  POD_CMD+=" && python data/prepare_sft.py --output-dir ${DATA_OUT_DIR} --n-samples ${N_SAMPLES}"
fi
POD_CMD+=" && python scripts/train_sft.py ${TRAIN_FLAGS}"

# ----- Compose the runai args -------------------------------------------
RUNAI_ARGS=(
  runai submit
  --name "${RUN_NAME}"
  -p "course-cs-552-${GASPAR}"
  --image "${IMAGE}"
  --gpu 1
  --large-shm
  --node-pools a100-40g
  --working-dir /scratch
  --environment "HF_HOME=/scratch/hf_cache"
  --environment "HF_TOKEN=${HF_TOKEN:-}"
  --environment "WANDB_API_KEY=${WANDB_API_KEY:-}"
  --environment "WANDB_PROJECT=emainelpe-math"
  # PyTorch CUDA caching-allocator: expandable_segments coalesces freed
  # blocks; belt-and-suspenders to Liger Kernel against fragmentation
  # accumulation over long SFT runs.
  --environment "PYTORCH_ALLOC_CONF=expandable_segments:True"
  --existing-pvc "claimname=course-cs-552-scratch-${GROUP},path=/scratch"
  --existing-pvc "claimname=course-cs-552-shared-ro,path=/shared-ro"
  --existing-pvc "claimname=course-cs-552-shared-rw,path=/shared-rw"
  --command --
  /bin/bash -lc "${POD_CMD}"
)

# ----- Print or submit --------------------------------------------------
# Mask token values in dry-run output so the assembled command is safe to
# paste into a slack message or an issue. The masking is presence-only
# (we say <set>/<unset>) — the actual token never appears.
print_args_masked() {
  local arg
  for arg in "${RUNAI_ARGS[@]}"; do
    case "$arg" in
      HF_TOKEN=*)
        if [[ -n "${HF_TOKEN:-}" ]]; then echo "HF_TOKEN=<set>"; else echo "HF_TOKEN=<unset>"; fi
        ;;
      WANDB_API_KEY=*)
        if [[ -n "${WANDB_API_KEY:-}" ]]; then echo "WANDB_API_KEY=<set>"; else echo "WANDB_API_KEY=<unset>"; fi
        ;;
      *) echo "$arg" ;;
    esac
  done
}

if (( DRY_RUN )); then
  echo "=== submit_train.sh --dry-run ==="
  echo "RUN_NAME    : ${RUN_NAME}"
  echo "REPO_DIR    : ${REPO_DIR}"
  echo "IMAGE       : ${IMAGE}"
  echo "DATA_OUT_DIR: ${DATA_OUT_DIR}"
  echo "SKIP_PREP   : ${SKIP_PREP:-<unset>}"
  echo "N_SAMPLES   : ${N_SAMPLES}"
  echo "EPOCHS      : ${EPOCHS}"
  echo "RESUME      : ${RESUME:-<unset>}"
  echo
  echo "Assembled command (one arg per line, secrets masked):"
  print_args_masked
  exit 0
fi

# Real submission.
"${RUNAI_ARGS[@]}"

cat <<EOF

=== Job submitted ===

Job name:        ${RUN_NAME}
Code (RO-ish):   ${REPO_DIR}
Data outputs:    ${DATA_OUT_DIR}/{train,eval}.jsonl
Train outputs:   ${RUN_OUT_DIR}/{checkpoint-*,final/}

Follow logs:        runai logs -f ${RUN_NAME}
Inspect status:     runai describe job ${RUN_NAME}
Shell into pod:     runai bash ${RUN_NAME}
Delete when done:   runai delete job ${RUN_NAME}

Estimated runtime: ~8–12h on 1×A100 40g (${N_SAMPLES} examples × ${EPOCHS} epochs).
EOF
