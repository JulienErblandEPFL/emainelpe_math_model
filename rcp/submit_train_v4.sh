#!/usr/bin/env bash
# rcp/submit_train_v4.sh — submit the v4 SFT pipeline to RCP.
#
# v4 is a targeted SFT run designed to fix v3's diagnosed coverage gaps:
# Intermediate Algebra (pass@1=0.296), Precalculus (pass@1=0.339), Level 5
# (pass@1=0.213) on the MATH-500 diagnostic. The training mix
# (--source v4-mix) is OMI2 + Hendrycks MATH-train (per-subject + per-level
# buckets) + NuminaMath olympiad subset. See CLAUDE.md → "v4 training plan"
# for the full design rationale.
#
# Two variants from the same data:
#
#   ./submit_train_v4.sh fresh    # Train from base Qwen3-1.7B, LR=1e-4
#   ./submit_train_v4.sh resume   # Init from v3 adapter, LR=5e-5
#
# The better of the two becomes the math expert.
#
# Usage:
#   GASPAR=erbland GROUP=g65 ./rcp/submit_train_v4.sh                  # fresh
#   GASPAR=erbland GROUP=g65 ./rcp/submit_train_v4.sh fresh
#   GASPAR=erbland GROUP=g65 ./rcp/submit_train_v4.sh resume
#   GASPAR=erbland GROUP=g65 ./rcp/submit_train_v4.sh --dry-run        # print only
#
# Required env vars (refused with placeholder values):
#   GASPAR   EPFL username (must NOT be the literal "gaspar")
#   GROUP    Team number, e.g. g65 (must NOT be "gXX" or "g00")
#
# Recommended env vars (warn-only if unset):
#   HF_TOKEN          HF Hub token; needed for gated datasets
#   WANDB_API_KEY     W&B token; without it train_sft.py logs to stdout only
#
# Optional env vars (have defaults):
#   IMAGE               Course Docker image. Default: ayushkumartarun/course-cs-552-standard:v1
#   SCRATCH_USER        First-level /scratch dir name. Default: Julien
#   REPO_DIR            Repo path inside the pod.
#                       Default: /scratch/${SCRATCH_USER}/emainelpe_math_model
#   DATA_OUT_DIR        Where prepare_sft writes train.jsonl / eval.jsonl.
#                       Default: /scratch/${SCRATCH_USER}/data_out_v4
#   SKIP_PREP           If non-empty, skip the in-pod prepare_sft.py call.
#   EPOCHS              train_sft.py --epochs. Default: 2
#   LEARNING_RATE       train_sft.py --learning-rate. Default: 1e-4 (fresh)
#                       or 5e-5 (resume); set by the mode arg unless overridden.
#   INIT_FROM_ADAPTER   Path to a v3 adapter dir for --init-from-adapter.
#                       Set automatically when the mode is "resume"; ignored
#                       in "fresh" mode.
#   V4_OMI2_COUNT             prepare_sft.py --omi2-count. Default: 40000
#   V4_INTALG_COUNT           --math-intermediate-algebra-count. Default: 12000
#   V4_PRECALC_COUNT          --math-precalculus-count. Default: 7000
#   V4_LEVEL45_COUNT          --math-level45-count. Default: 18000
#   V4_LEVEL13_COUNT          --math-level13-count. Default: 13000
#   V4_NUMINAMATH_COUNT       --numinamath-count. Default: 5000
#   V4_MAX_FORMATTED_TOKENS   --max-formatted-tokens. Default: empty (auto 2900
#                             via prepare_sft.py when --source v4-mix).

set -euo pipefail

# ----- Defaults ----------------------------------------------------------
IMAGE="${IMAGE:-ayushkumartarun/course-cs-552-standard:v1}"
SCRATCH_USER="${SCRATCH_USER:-Julien}"
EPOCHS="${EPOCHS:-2}"
V4_OMI2_COUNT="${V4_OMI2_COUNT:-40000}"
V4_INTALG_COUNT="${V4_INTALG_COUNT:-12000}"
V4_PRECALC_COUNT="${V4_PRECALC_COUNT:-7000}"
V4_LEVEL45_COUNT="${V4_LEVEL45_COUNT:-18000}"
V4_LEVEL13_COUNT="${V4_LEVEL13_COUNT:-13000}"
V4_NUMINAMATH_COUNT="${V4_NUMINAMATH_COUNT:-5000}"
V4_MAX_FORMATTED_TOKENS="${V4_MAX_FORMATTED_TOKENS:-}"

# v3 adapter dir for --init-from-adapter (resume mode).
DEFAULT_V3_ADAPTER="/scratch/${SCRATCH_USER}/runs/cs552-erbland-g65-v3-omi2-fix2-20260511-152150/final"

# ----- Argument parsing --------------------------------------------------
DRY_RUN=0
MODE="fresh"
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,55p' "$0"
      exit 0
      ;;
    fresh|resume) MODE="$1" ;;
    -*)
      echo "ERROR: unknown flag: $1" >&2
      exit 2
      ;;
    *)
      echo "ERROR: unknown mode '$1' (expected: fresh|resume)" >&2
      exit 2
      ;;
  esac
  shift
done

# ----- Mode-derived defaults -------------------------------------------
# The two variants differ in LR + whether to init from v3's adapter.
# Defaults below preserve the design: fresh = LR=1e-4 (v3 SFT default),
# resume = LR=5e-5 (gentler continuation on existing weights). Both
# can be overridden via LEARNING_RATE env var.
if [[ "${MODE}" == "resume" ]]; then
  LEARNING_RATE="${LEARNING_RATE:-5e-5}"
  INIT_FROM_ADAPTER="${INIT_FROM_ADAPTER:-${DEFAULT_V3_ADAPTER}}"
  SUFFIX="v4-resume"
else
  LEARNING_RATE="${LEARNING_RATE:-1e-4}"
  INIT_FROM_ADAPTER="${INIT_FROM_ADAPTER:-}"
  SUFFIX="v4-fresh"
fi

# ----- Placeholder validation -------------------------------------------
if [[ -z "${GASPAR:-}" || "${GASPAR}" == "gaspar" ]]; then
  echo "ERROR: set GASPAR=<your-EPFL-username> (got: '${GASPAR:-}')" >&2
  exit 1
fi
if [[ -z "${GROUP:-}" || "${GROUP}" =~ ^g(XX|00)$ ]]; then
  echo "ERROR: set GROUP=g<NN> (got: '${GROUP:-}')" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN unset — gated HF resources will fail to download" >&2
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARN: WANDB_API_KEY unset — loss curves go to stdout only" >&2
fi

# ----- Names + paths ----------------------------------------------------
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-${TIMESTAMP}"
REPO_DIR="${REPO_DIR:-/scratch/${SCRATCH_USER}/emainelpe_math_model}"
DATA_OUT_DIR="${DATA_OUT_DIR:-/scratch/${SCRATCH_USER}/data_out_v4}"
RUN_OUT_DIR="/scratch/${SCRATCH_USER}/runs/${RUN_NAME}"

# ----- Compose prepare_sft flags ----------------------------------------
PREP_FLAGS="--source v4-mix"
PREP_FLAGS+=" --output-dir ${DATA_OUT_DIR}"
PREP_FLAGS+=" --omi2-count ${V4_OMI2_COUNT}"
PREP_FLAGS+=" --math-intermediate-algebra-count ${V4_INTALG_COUNT}"
PREP_FLAGS+=" --math-precalculus-count ${V4_PRECALC_COUNT}"
PREP_FLAGS+=" --math-level45-count ${V4_LEVEL45_COUNT}"
PREP_FLAGS+=" --math-level13-count ${V4_LEVEL13_COUNT}"
PREP_FLAGS+=" --numinamath-count ${V4_NUMINAMATH_COUNT}"
PREP_FLAGS+="${V4_MAX_FORMATTED_TOKENS:+ --max-formatted-tokens ${V4_MAX_FORMATTED_TOKENS}}"

# ----- Compose train flags ----------------------------------------------
TRAIN_FLAGS="--train-file ${DATA_OUT_DIR}/train.jsonl"
TRAIN_FLAGS+=" --eval-file ${DATA_OUT_DIR}/eval.jsonl"
TRAIN_FLAGS+=" --output-dir ${RUN_OUT_DIR}"
TRAIN_FLAGS+=" --run-name ${RUN_NAME}"
TRAIN_FLAGS+=" --epochs ${EPOCHS}"
TRAIN_FLAGS+=" --learning-rate ${LEARNING_RATE}"
TRAIN_FLAGS+="${INIT_FROM_ADAPTER:+ --init-from-adapter ${INIT_FROM_ADAPTER}}"

# ----- The in-pod command -----------------------------------------------
POD_CMD="ln -sf \"\$(command -v python3)\" /usr/local/bin/python"
POD_CMD+=" && cd ${REPO_DIR}"
POD_CMD+=" && pip install -r requirements.txt"
# Liger Kernel sanity check: fail fast if the install ever breaks (the
# primary OOM mitigation depends on this import succeeding before
# train_sft.py launches). Also imports apply_liger_kernel_to_qwen3 — the
# model-specific patch we actually need — so a wheel that loses Qwen3
# support trips here, not 30 min later at the train_sft.py preflight.
# liger-kernel 0.8.0 does NOT expose a __version__ attribute, so we
# check importability only.
POD_CMD+=" && python -c \"import liger_kernel; from liger_kernel.transformers import apply_liger_kernel_to_qwen3; print('liger_kernel imported OK (Qwen3 patch available)')\""
if [[ -z "${SKIP_PREP:-}" ]]; then
  POD_CMD+=" && python data/prepare_sft.py ${PREP_FLAGS}"
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
  # blocks instead of pinning them at their original size, which lowers
  # long-run fragmentation. Belt-and-suspenders to Liger Kernel: Liger
  # is the structural fix for the logits-tensor OOM; this knob is the
  # fallback that helps if some other allocation grows over training.
  --environment "PYTORCH_ALLOC_CONF=expandable_segments:True"
  --existing-pvc "claimname=course-cs-552-scratch-${GROUP},path=/scratch"
  --existing-pvc "claimname=course-cs-552-shared-ro,path=/shared-ro"
  --existing-pvc "claimname=course-cs-552-shared-rw,path=/shared-rw"
  --command --
  /bin/bash -lc "${POD_CMD}"
)

# ----- Print or submit --------------------------------------------------
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
  echo "=== submit_train_v4.sh --dry-run ==="
  echo "MODE              : ${MODE}"
  echo "RUN_NAME          : ${RUN_NAME}"
  echo "REPO_DIR          : ${REPO_DIR}"
  echo "IMAGE             : ${IMAGE}"
  echo "DATA_OUT_DIR      : ${DATA_OUT_DIR}"
  echo "SKIP_PREP         : ${SKIP_PREP:-<unset>}"
  echo "EPOCHS            : ${EPOCHS}"
  echo "LEARNING_RATE     : ${LEARNING_RATE}"
  echo "INIT_FROM_ADAPTER : ${INIT_FROM_ADAPTER:-<unset>}"
  echo
  echo "--- v4-mix composition ---"
  echo "V4_OMI2_COUNT             : ${V4_OMI2_COUNT}"
  echo "V4_INTALG_COUNT           : ${V4_INTALG_COUNT}"
  echo "V4_PRECALC_COUNT          : ${V4_PRECALC_COUNT}"
  echo "V4_LEVEL45_COUNT          : ${V4_LEVEL45_COUNT}"
  echo "V4_LEVEL13_COUNT          : ${V4_LEVEL13_COUNT}"
  echo "V4_NUMINAMATH_COUNT       : ${V4_NUMINAMATH_COUNT}"
  echo "V4_MAX_FORMATTED_TOKENS   : ${V4_MAX_FORMATTED_TOKENS:-<unset, auto 2900 via prepare_sft>}"
  echo
  echo "Assembled command (one arg per line, secrets masked):"
  print_args_masked
  exit 0
fi

# Real submission.
"${RUNAI_ARGS[@]}"

cat <<EOF

=== v4 job submitted (mode=${MODE}) ===

Job name:        ${RUN_NAME}
Code (RO-ish):   ${REPO_DIR}
Data outputs:    ${DATA_OUT_DIR}/{train,eval}.jsonl
Train outputs:   ${RUN_OUT_DIR}/{checkpoint-*,final/}
Adapter init:    ${INIT_FROM_ADAPTER:-<none, fresh init>}

Follow logs:        runai logs -f ${RUN_NAME}
Inspect status:     runai describe job ${RUN_NAME}
Shell into pod:     runai bash ${RUN_NAME}
Delete when done:   runai delete job ${RUN_NAME}

Estimated runtime: ~10-14h on 1×A100 40g (~95k v4-mix examples × ${EPOCHS} epochs).
EOF
