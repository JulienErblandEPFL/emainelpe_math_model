#!/usr/bin/env bash
# rcp/submit_rlvr.sh — submit the RLVR pipeline (prompt curation + GRPO train) to RCP.
#
# Mirrors rcp/submit_train.sh; differs only in:
#   - Pipeline body: data/prepare_rlvr.py → scripts/train_rlvr.py
#   - Default run-name suffix: "rlvr"
#   - New env vars: ADAPTER_DIR, PROMPT_SET, MAX_PROMPTS (no N_SAMPLES)
#   - Skips DART data prep — Stage 1's train.jsonl is read by prepare_rlvr.py.
#
# Usage:
#   GASPAR=erbland GROUP=g65 ./rcp/submit_rlvr.sh             # real submit
#   GASPAR=erbland GROUP=g65 ./rcp/submit_rlvr.sh --dry-run   # print only
#   GASPAR=erbland GROUP=g65 ./rcp/submit_rlvr.sh smoke       # custom suffix
#
# Required env vars (refused with placeholder values):
#   GASPAR   EPFL username (must NOT be the literal "gaspar")
#   GROUP    Team number, e.g. g65 (must NOT be "gXX" or "g00")
#
# Recommended env vars (warn-only if unset):
#   HF_TOKEN          HF Hub token
#   WANDB_API_KEY     W&B token; without it train_rlvr.py logs to stdout only
#                     (reward variance / KL trajectory get hidden in pod logs)
#
# Optional env vars (have defaults):
#   IMAGE         Course Docker image. Default: ayushkumartarun/course-cs-552-standard:v1
#   SCRATCH_USER  Same convention as submit_train.sh. Default: Julien
#   REPO_DIR      Repo path inside the pod.
#                 Default: /scratch/${SCRATCH_USER}/emainelpe_math_model
#   ADAPTER_DIR   Trained SFT adapter dir to continue training from.
#                 Default: /scratch/${SCRATCH_USER}/runs/cs552-erbland-g65-train-20260508-150203/final
#   PROMPT_SET    Curated RLVR prompts JSONL produced by prepare_rlvr.py.
#                 Default: /scratch/${SCRATCH_USER}/data_out/rlvr_prompts.jsonl
#   SFT_MODEL     Merged SFT checkpoint used to score difficulty during
#                 prompt curation. Default: /scratch/${SCRATCH_USER}/merged/math_model_v1
#   MAX_PROMPTS   train_rlvr.py --max-prompts. Default: 5000
#   POOL_SIZE     prepare_rlvr.py --pool-size. Default: 10000
#   TARGET_SIZE   prepare_rlvr.py --target-size. Default: 5000
#   LEARNING_RATE train_rlvr.py --learning-rate. Default: 3e-6
#   KL_COEF       train_rlvr.py --kl-coef. Default: 0.04
#   ROLLOUT_TEMP  train_rlvr.py --rollout-temp. Default: 0.8
#   SKIP_CURATION If "1", skip prepare_rlvr.py (use existing PROMPT_SET).
#                 Default: empty (run curation).
#   SKIP_PREFLIGHTS  If "1", forward --skip-preflights to train_rlvr.py.
#                 Default: empty. ONLY for trainer-wiring debugging.

set -euo pipefail

# ----- Defaults ----------------------------------------------------------
IMAGE="${IMAGE:-ayushkumartarun/course-cs-552-standard:v1}"
SCRATCH_USER="${SCRATCH_USER:-Julien}"
MAX_PROMPTS="${MAX_PROMPTS:-5000}"
POOL_SIZE="${POOL_SIZE:-10000}"
TARGET_SIZE="${TARGET_SIZE:-5000}"
LEARNING_RATE="${LEARNING_RATE:-3e-6}"
KL_COEF="${KL_COEF:-0.04}"
ROLLOUT_TEMP="${ROLLOUT_TEMP:-0.8}"
SKIP_CURATION="${SKIP_CURATION:-}"
SKIP_PREFLIGHTS="${SKIP_PREFLIGHTS:-}"

# ----- Argument parsing --------------------------------------------------
DRY_RUN=0
SUFFIX="rlvr"
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,53p' "$0"
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

# Token warnings — don't exit; the operator may have set them another way.
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN unset — gated HF resources will fail to download" >&2
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARN: WANDB_API_KEY unset — reward variance / KL trajectory only in stdout" >&2
fi

# ----- Names + paths -----------------------------------------------------
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-${TIMESTAMP}"
REPO_DIR="${REPO_DIR:-/scratch/${SCRATCH_USER}/emainelpe_math_model}"

DATA_OUT_DIR="/scratch/${SCRATCH_USER}/data_out"
RUN_OUT_DIR="/scratch/${SCRATCH_USER}/runs/${RUN_NAME}"
ADAPTER_DIR="${ADAPTER_DIR:-/scratch/${SCRATCH_USER}/runs/cs552-erbland-g65-train-20260508-150203/final}"
PROMPT_SET="${PROMPT_SET:-${DATA_OUT_DIR}/rlvr_prompts.jsonl}"
SFT_MODEL="${SFT_MODEL:-/scratch/${SCRATCH_USER}/merged/math_model_v1}"

# ----- Compose train flags ----------------------------------------------
TRAIN_FLAGS="--adapter-dir ${ADAPTER_DIR}"
TRAIN_FLAGS+=" --prompt-set ${PROMPT_SET}"
TRAIN_FLAGS+=" --output-dir ${RUN_OUT_DIR}"
TRAIN_FLAGS+=" --run-name ${RUN_NAME}"
TRAIN_FLAGS+=" --learning-rate ${LEARNING_RATE}"
TRAIN_FLAGS+=" --kl-coef ${KL_COEF}"
TRAIN_FLAGS+=" --rollout-temp ${ROLLOUT_TEMP}"
TRAIN_FLAGS+=" --max-prompts ${MAX_PROMPTS}"
TRAIN_FLAGS+="${SKIP_PREFLIGHTS:+ --skip-preflights}"

CURATION_FLAGS="--input-jsonl ${DATA_OUT_DIR}/train.jsonl"
CURATION_FLAGS+=" --sft-model-path ${SFT_MODEL}"
CURATION_FLAGS+=" --output-jsonl ${PROMPT_SET}"
CURATION_FLAGS+=" --pool-size ${POOL_SIZE}"
CURATION_FLAGS+=" --target-size ${TARGET_SIZE}"

# ----- The in-pod command -----------------------------------------------
POD_CMD="ln -sf \"\$(command -v python3)\" /usr/local/bin/python"
POD_CMD+=" && cd ${REPO_DIR}"
POD_CMD+=" && pip install -r requirements.txt"
if [[ -z "${SKIP_CURATION}" ]]; then
  POD_CMD+=" && python data/prepare_rlvr.py ${CURATION_FLAGS}"
fi
POD_CMD+=" && python scripts/train_rlvr.py ${TRAIN_FLAGS}"

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
  echo "=== submit_rlvr.sh --dry-run ==="
  echo "RUN_NAME       : ${RUN_NAME}"
  echo "REPO_DIR       : ${REPO_DIR}"
  echo "IMAGE          : ${IMAGE}"
  echo "ADAPTER_DIR    : ${ADAPTER_DIR}"
  echo "PROMPT_SET     : ${PROMPT_SET}"
  echo "SFT_MODEL      : ${SFT_MODEL}"
  echo "MAX_PROMPTS    : ${MAX_PROMPTS}"
  echo "POOL_SIZE      : ${POOL_SIZE}"
  echo "TARGET_SIZE    : ${TARGET_SIZE}"
  echo "LEARNING_RATE  : ${LEARNING_RATE}"
  echo "KL_COEF        : ${KL_COEF}"
  echo "ROLLOUT_TEMP   : ${ROLLOUT_TEMP}"
  echo "SKIP_CURATION  : ${SKIP_CURATION:-<unset>}"
  echo "SKIP_PREFLIGHTS: ${SKIP_PREFLIGHTS:-<unset>}"
  echo
  echo "Assembled command (one arg per line, secrets masked):"
  print_args_masked
  exit 0
fi

# Real submission.
"${RUNAI_ARGS[@]}"

cat <<EOF

=== RLVR job submitted ===

Job name:        ${RUN_NAME}
Code (RO-ish):   ${REPO_DIR}
Adapter input:   ${ADAPTER_DIR}
Prompt set:      ${PROMPT_SET}
Train outputs:   ${RUN_OUT_DIR}/{checkpoint-*,final/}

Follow logs:        runai logs -f ${RUN_NAME}
Inspect status:     runai describe job ${RUN_NAME}
Shell into pod:     runai bash ${RUN_NAME}
Delete when done:   runai delete job ${RUN_NAME}

V0 — UNTRAINED. Expect to iterate on hyperparameters multiple times before
a stable run lands. Watch reward_std (P2 signal) and kl (P3 signal) early.
EOF
