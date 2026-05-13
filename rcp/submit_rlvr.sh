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
#   DATA_OUT_DIR  Stage 1 SFT output dir; supplies train.jsonl as the
#                 curation pool. Default: /scratch/${SCRATCH_USER}/data_out
#                 Override for v2/v3, e.g. DATA_OUT_DIR=/scratch/Julien/data_out_v3
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
#   MAX_NEW_TOKENS    train_rlvr.py --max-new-tokens. Default: 4096
#   SKIP_CURATION If "1", skip prepare_rlvr.py (use existing PROMPT_SET).
#                 Default: empty (run curation).
#   SKIP_PREFLIGHTS  If "1", forward --skip-preflights to train_rlvr.py.
#                 Default: empty. ONLY for trainer-wiring debugging.
#
# Rescue-config env vars (added 2026-05-13 after the retry3 starvation
# incident; see CLAUDE.md → "RLVR rescue plan"). All defaults preserve the
# pre-rescue behavior so existing invocations are byte-stable.
#   LOSS_TYPE             train_rlvr.py --loss-type {grpo,dapo}. Default: dapo
#   USE_VLLM              If set, forward --use-vllm. Default: empty (False).
#   VLLM_GPU_MEM_UTIL     train_rlvr.py --vllm-gpu-memory-utilization.
#                         Default: 0.3 (only consulted when USE_VLLM is set).
#   MASK_TRUNCATED        If set, forward --mask-truncated-completions.
#                         Default: empty (False).
#   LOG_COMPLETIONS       If set, forward --log-completions. Default: empty.
#   HARD_KILL_ON_WEAK_SIGNAL  If set, forward --hard-kill-on-weak-signal.
#                         Default: empty (callback only logs ERROR).
#   DIFFICULTY_MIN        prepare_rlvr.py --difficulty-lo. Default: 0.2
#   DIFFICULTY_MAX        prepare_rlvr.py --difficulty-hi. Default: 0.8

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
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
SKIP_CURATION="${SKIP_CURATION:-}"
SKIP_PREFLIGHTS="${SKIP_PREFLIGHTS:-}"

# ----- Rescue-config defaults (preserve pre-rescue behavior). ------------
LOSS_TYPE="${LOSS_TYPE:-dapo}"
USE_VLLM="${USE_VLLM:-}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.3}"
MASK_TRUNCATED="${MASK_TRUNCATED:-}"
LOG_COMPLETIONS="${LOG_COMPLETIONS:-}"
HARD_KILL_ON_WEAK_SIGNAL="${HARD_KILL_ON_WEAK_SIGNAL:-}"
DIFFICULTY_MIN="${DIFFICULTY_MIN:-0.2}"
DIFFICULTY_MAX="${DIFFICULTY_MAX:-0.8}"

# ----- Argument parsing --------------------------------------------------
DRY_RUN=0
SUFFIX="rlvr"
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,56p' "$0"
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

# DATA_OUT_DIR override semantics: set the env var to point at v2/v3 SFT
# outputs (e.g. DATA_OUT_DIR=/scratch/Julien/data_out_v3) so prepare_rlvr.py
# scores the matching pool. Matches the submit_train.sh:118 pattern.
DATA_OUT_DIR="${DATA_OUT_DIR:-/scratch/${SCRATCH_USER}/data_out}"
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
TRAIN_FLAGS+=" --max-new-tokens ${MAX_NEW_TOKENS}"
TRAIN_FLAGS+=" --loss-type ${LOSS_TYPE}"
TRAIN_FLAGS+=" --vllm-gpu-memory-utilization ${VLLM_GPU_MEM_UTIL}"
TRAIN_FLAGS+="${USE_VLLM:+ --use-vllm}"
TRAIN_FLAGS+="${MASK_TRUNCATED:+ --mask-truncated-completions}"
TRAIN_FLAGS+="${LOG_COMPLETIONS:+ --log-completions}"
TRAIN_FLAGS+="${HARD_KILL_ON_WEAK_SIGNAL:+ --hard-kill-on-weak-signal}"
TRAIN_FLAGS+="${SKIP_PREFLIGHTS:+ --skip-preflights}"

CURATION_FLAGS="--input-jsonl ${DATA_OUT_DIR}/train.jsonl"
CURATION_FLAGS+=" --sft-model-path ${SFT_MODEL}"
CURATION_FLAGS+=" --output-jsonl ${PROMPT_SET}"
CURATION_FLAGS+=" --pool-size ${POOL_SIZE}"
CURATION_FLAGS+=" --target-size ${TARGET_SIZE}"
CURATION_FLAGS+=" --difficulty-lo ${DIFFICULTY_MIN}"
CURATION_FLAGS+=" --difficulty-hi ${DIFFICULTY_MAX}"

# ----- The in-pod command -----------------------------------------------
POD_CMD="ln -sf \"\$(command -v python3)\" /usr/local/bin/python"
POD_CMD+=" && cd ${REPO_DIR}"
POD_CMD+=" && pip install -r requirements.txt"
# Liger Kernel sanity check: same fail-fast as the SFT submit scripts.
# GRPO rollouts can hit the same logits-tensor OOM as SFT batches, so
# Liger is the primary OOM mitigation here too. liger-kernel 0.8.0
# doesn't expose __version__; the Qwen3-patch import validates the
# model-specific entry point we actually use.
POD_CMD+=" && python -c 'import liger_kernel; from liger_kernel.transformers import apply_liger_kernel_to_qwen3; print(\"liger_kernel imported OK (Qwen3 patch available)\")'"
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
  # Belt-and-suspenders fragmentation mitigation; Liger Kernel is the
  # primary OOM fix. See CLAUDE.md → "OOM mitigations" for rationale.
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
  echo "=== submit_rlvr.sh --dry-run ==="
  echo "RUN_NAME       : ${RUN_NAME}"
  echo "REPO_DIR       : ${REPO_DIR}"
  echo "IMAGE          : ${IMAGE}"
  echo "DATA_OUT_DIR   : ${DATA_OUT_DIR}"
  echo "ADAPTER_DIR    : ${ADAPTER_DIR}"
  echo "PROMPT_SET     : ${PROMPT_SET}"
  echo "SFT_MODEL      : ${SFT_MODEL}"
  echo "MAX_PROMPTS    : ${MAX_PROMPTS}"
  echo "POOL_SIZE      : ${POOL_SIZE}"
  echo "TARGET_SIZE    : ${TARGET_SIZE}"
  echo "LEARNING_RATE  : ${LEARNING_RATE}"
  echo "KL_COEF        : ${KL_COEF}"
  echo "ROLLOUT_TEMP   : ${ROLLOUT_TEMP}"
  echo "MAX_NEW_TOKENS : ${MAX_NEW_TOKENS}"
  echo "SKIP_CURATION  : ${SKIP_CURATION:-<unset>}"
  echo "SKIP_PREFLIGHTS: ${SKIP_PREFLIGHTS:-<unset>}"
  echo
  echo "--- Rescue config (defaults preserve pre-rescue behavior) ---"
  echo "LOSS_TYPE                : ${LOSS_TYPE}"
  echo "USE_VLLM                 : ${USE_VLLM:-<unset>}"
  echo "VLLM_GPU_MEM_UTIL        : ${VLLM_GPU_MEM_UTIL}"
  echo "MASK_TRUNCATED           : ${MASK_TRUNCATED:-<unset>}"
  echo "LOG_COMPLETIONS          : ${LOG_COMPLETIONS:-<unset>}"
  echo "HARD_KILL_ON_WEAK_SIGNAL : ${HARD_KILL_ON_WEAK_SIGNAL:-<unset>}"
  echo "DIFFICULTY_MIN           : ${DIFFICULTY_MIN}"
  echo "DIFFICULTY_MAX           : ${DIFFICULTY_MAX}"
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
