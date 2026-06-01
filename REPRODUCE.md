# REPRODUCE — CS-552 Math Model

Reproduction guide for every experiment reported in `REPORT.md`. Commands
are paste-ready against the current `data/`, `scripts/`, and `evaluate/`
modules. All paths are repo-relative.

Commands marked **[GPU]** require a CUDA GPU (training/merge/eval).
The rest run on CPU.

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- `liger-kernel>=0.8.0` is **required**. `scripts/train_sft.py` and
  `scripts/train_rlvr.py` default `--use-liger-kernel=True` and hard-fail
  at startup if the wheel is unavailable. Pass `--no-use-liger-kernel`
  only for A/B comparison; the stock HF cross-entropy path OOMs on
  near-max-length batches with Qwen3-1.7B.
- Base model `Qwen/Qwen3-1.7B` and all datasets (`hkust-nlp/dart-math-uniform`,
  `nvidia/OpenMathInstruct-2`, `EleutherAI/hendrycks_math`,
  `AI-MO/NuminaMath-CoT`) are pulled from the Hugging Face Hub on first
  use. No manual download is needed; set `HF_HOME` to control cache location.
- `HF_TOKEN` is only required for `scripts/push.py` (the optional final
  upload step). Training, merging, and eval need no HF auth.
- `WANDB_API_KEY` is optional. Training silently disables W&B when unset.
- GPU: 1× A100 40 GB used in our runs. Approximate single-GPU wall-clock:
  - SFT 50 k rows, 2 epochs: ~4–6 h
  - SFT 100 k rows, 2 epochs: ~7–9 h
  - SFT 200 k rows, 2 epochs: ~16–18 h
  - Merge + smoke: ~3 min
  - Local eval (`validation_samples/math.jsonl`, n=8): ~3–5 min
  - RLVR rescue (1 epoch, ~3.9 k prompts, vLLM rollouts): ~15–17 h

## 2. Name mapping

| Report name   | Internal | `--source`         | Train size | LR    | Epochs | Notes                                       |
|---------------|----------|--------------------|-----------:|------:|-------:|---------------------------------------------|
| DART-50k      | v1       | `dart`             |     50 000 | 1e-4  |     2  | DART-Math-Uniform only                      |
| Mixed-50k     | v2       | `mixed`            |     50 000 | 1e-4  |     2  | 50/50 DART + OMI2                           |
| OMI-50k       | v3       | `openmathinstruct` |     50 000 | 1e-4  |     2  | Base for RLVR rescue                        |
| **OMI-100k**  | **v5**   | `openmathinstruct` |    100 000 | 1e-4  |     2  | **FINAL deployed model on team HF**         |
| OMI-200k      | v6       | `openmathinstruct` |    200 000 | 1e-4  |     2  | Local lift, regressed on CI; rolled back    |
| v4-fresh      | v4-fresh | `v4-mix` (bucket-composed) | —  | 1e-4  |     2  | Fresh init from Qwen3-1.7B base             |
| v4-resume     | v4-resume| `v4-mix` (bucket-composed) | —  | 5e-5  |     2  | Resume from OMI-50k adapter (`--init-from-adapter`) |
| RLVR-rescue   | —        | (GRPO)             |   ~3.9 k prompts | 3e-6 |  1   | On OMI-50k adapter; signal-band filter      |

`--source v4-mix` composes OMI2 + Hendrycks MATH-train (per-subject and
per-level buckets) + NuminaMath-CoT (olympiad-filtered). Bucket targets
are diagnostic-driven defaults in `data/prepare_sft.py`.

**v4-mix sizing note.** The v4-mix train-set size is fixed by internal
bucket-count constants (`V4_DEFAULT_OMI2_COUNT`,
`V4_DEFAULT_MATH_INTALG_COUNT`, etc. in `data/prepare_sft.py`), which
yield roughly **~67 k rows after the per-question cap** (95 k composed
pre-cap). Do **not** pass `--train-size` for `--source v4-mix` — the
flag is OMI2/DART-only semantics and would override the bucket budgets.
The two v4 variants (fresh, resume) train on the same `data_out/v4-mix/`
dataset; only the initialization and learning rate differ.

## 3. Pipeline (dependency order)

```
prepare_sft.py  →  train_sft.py  →  merge.py  →  run_eval.py   →  diagnose.py   →  push.py
   (data prep)    (LoRA SFT, GPU)    (GPU)       (quick GPU)      (full GPU)       (final only)
                                                  validation       reported nums
                                                  pass@8 check     (MATH-500, ...)
```

Each step's output is the next step's input.
- `scripts/run_eval.py` produces the headline validation pass@1 / pass@8
  on the N=10 `validation_samples/math.jsonl` snapshot — fast, ~3–5 min.
- `scripts/diagnose.py` produces the **reported** in-distribution and
  MATH-500 numbers with per-subject / per-level / failure-mode
  breakdowns. MATH-500 is auto-downloaded from HF
  (`HuggingFaceH4/MATH-500` with `hendrycks/competition_math` fallback);
  no local file is needed. The `indist` target requires the
  `eval.jsonl` produced by `prepare_sft.py` *for that model's source*;
  pass it via `--indist-file`. `--model` is required.
- `push.py` is only run for the final deployed checkpoint (OMI-100k in
  our submission).

## 4. Per-experiment reproduction

`--seed 42` is the CI contract seed and is the default everywhere; we
pass it explicitly below for clarity. `--train-size N` writes `N` train
rows and `--eval-size` rows (default 500) of held-out eval; the two
flags are mutually exclusive with the legacy `--n-samples` flag.

`scripts/train_sft.py` writes the final LoRA adapter to
`<output-dir>/final/`. `scripts/merge.py` consumes that `final/` path.

### 4.1 DART-50k (v1)

```bash
# 1. Prepare data (CPU)
python data/prepare_sft.py \
    --source dart --train-size 50000 \
    --output-dir data_out/dart-50k --seed 42

# 2. Train [GPU]
python scripts/train_sft.py \
    --train-file data_out/dart-50k/train.jsonl \
    --eval-file  data_out/dart-50k/eval.jsonl \
    --output-dir runs/dart-50k \
    --learning-rate 1e-4 --epochs 2 --seed 42

# 3. Merge [GPU]
python scripts/merge.py \
    --adapter-dir runs/dart-50k/final \
    --output-dir  merged/dart-50k \
    --temperature 0.4 --top-p 0.95 --top-k 20

# 4. Quick validation pass@1 / pass@8 check [GPU]
python scripts/run_eval.py \
    --model merged/dart-50k \
    --eval-file validation_samples/math.jsonl \
    --output-dir runs/eval/dart-50k

# 5. Diagnostic eval — produces the reported MATH-500 + in-distribution
#    numbers + per-subject / per-level / failure-mode breakdowns. [GPU]
python scripts/diagnose.py \
    --model merged/dart-50k \
    --target all \
    --indist-file data_out/dart-50k/eval.jsonl \
    --output-dir runs/diagnostics/dart-50k

# MATH-500 only (skip the indist surface; no --indist-file needed):
python scripts/diagnose.py \
    --model merged/dart-50k --target math_test \
    --output-dir runs/diagnostics/dart-50k                   # [GPU]
```

### 4.2 Mixed-50k (v2)

```bash
python data/prepare_sft.py \
    --source mixed --train-size 50000 \
    --output-dir data_out/mixed-50k --seed 42

python scripts/train_sft.py \
    --train-file data_out/mixed-50k/train.jsonl \
    --eval-file  data_out/mixed-50k/eval.jsonl \
    --output-dir runs/mixed-50k \
    --learning-rate 1e-4 --epochs 2 --seed 42                # [GPU]

python scripts/merge.py \
    --adapter-dir runs/mixed-50k/final \
    --output-dir  merged/mixed-50k --temperature 0.4         # [GPU]

python scripts/run_eval.py \
    --model merged/mixed-50k \
    --eval-file validation_samples/math.jsonl \
    --output-dir runs/eval/mixed-50k                         # [GPU]

# Diagnostic eval — MATH-500 + in-distribution surfaces. [GPU]
python scripts/diagnose.py \
    --model merged/mixed-50k --target all \
    --indist-file data_out/mixed-50k/eval.jsonl \
    --output-dir runs/diagnostics/mixed-50k
```

### 4.3 OMI-50k (v3)

```bash
python data/prepare_sft.py \
    --source openmathinstruct --train-size 50000 \
    --output-dir data_out/omi-50k --seed 42

python scripts/train_sft.py \
    --train-file data_out/omi-50k/train.jsonl \
    --eval-file  data_out/omi-50k/eval.jsonl \
    --output-dir runs/omi-50k \
    --learning-rate 1e-4 --epochs 2 --seed 42                # [GPU]

python scripts/merge.py \
    --adapter-dir runs/omi-50k/final \
    --output-dir  merged/omi-50k --temperature 0.4           # [GPU]

python scripts/run_eval.py \
    --model merged/omi-50k \
    --eval-file validation_samples/math.jsonl \
    --output-dir runs/eval/omi-50k                           # [GPU]

# Diagnostic eval — MATH-500 + in-distribution surfaces. [GPU]
python scripts/diagnose.py \
    --model merged/omi-50k --target all \
    --indist-file data_out/omi-50k/eval.jsonl \
    --output-dir runs/diagnostics/omi-50k
```

### 4.4 OMI-100k (v5) — **the deployed model**

```bash
python data/prepare_sft.py \
    --source openmathinstruct --train-size 100000 \
    --output-dir data_out/omi-100k --seed 42

python scripts/train_sft.py \
    --train-file data_out/omi-100k/train.jsonl \
    --eval-file  data_out/omi-100k/eval.jsonl \
    --output-dir runs/omi-100k \
    --learning-rate 1e-4 --epochs 2 --seed 42                # [GPU]

python scripts/merge.py \
    --adapter-dir runs/omi-100k/final \
    --output-dir  merged/omi-100k --temperature 0.4          # [GPU]

python scripts/run_eval.py \
    --model merged/omi-100k \
    --eval-file validation_samples/math.jsonl \
    --output-dir runs/eval/omi-100k                          # [GPU]

# Diagnostic eval — produces the deployed model's headline MATH-500
# pass@1 = 0.516 (§7) and per-subject / per-level breakdowns. [GPU]
python scripts/diagnose.py \
    --model merged/omi-100k --target all \
    --indist-file data_out/omi-100k/eval.jsonl \
    --output-dir runs/diagnostics/omi-100k

# 5. Optional: upload the final checkpoint to a HF repo.
# Requires HF_TOKEN (or prior `huggingface-cli login`).
python scripts/push.py \
    --model-dir merged/omi-100k \
    --hf-repo   <your-org>/math_model
```

### 4.5 OMI-200k (v6)

```bash
python data/prepare_sft.py \
    --source openmathinstruct --train-size 200000 \
    --output-dir data_out/omi-200k --seed 42

python scripts/train_sft.py \
    --train-file data_out/omi-200k/train.jsonl \
    --eval-file  data_out/omi-200k/eval.jsonl \
    --output-dir runs/omi-200k \
    --learning-rate 1e-4 --epochs 2 --seed 42                # [GPU]

python scripts/merge.py \
    --adapter-dir runs/omi-200k/final \
    --output-dir  merged/omi-200k --temperature 0.4          # [GPU]

python scripts/run_eval.py \
    --model merged/omi-200k \
    --eval-file validation_samples/math.jsonl \
    --output-dir runs/eval/omi-200k                          # [GPU]

# Diagnostic eval — MATH-500 + in-distribution surfaces. [GPU]
python scripts/diagnose.py \
    --model merged/omi-200k --target all \
    --indist-file data_out/omi-200k/eval.jsonl \
    --output-dir runs/diagnostics/omi-200k
```

### 4.6 v4-mix (shared dataset prep — run once)

Both `v4-fresh` (§4.7) and `v4-resume` (§4.8) train on the same
bucket-composed v4-mix dataset. Build it once:

```bash
# v4-mix dataset prep (CPU). Bucket-composed: do NOT pass --train-size.
# The composition is driven by internal V4_DEFAULT_*_COUNT constants
# (~67 k rows after the per-question cap).
python data/prepare_sft.py \
    --source v4-mix \
    --output-dir data_out/v4-mix --seed 42
```

### 4.7 v4-fresh (fresh init from Qwen3-1.7B, LR 1e-4)

```bash
python scripts/train_sft.py \
    --train-file data_out/v4-mix/train.jsonl \
    --eval-file  data_out/v4-mix/eval.jsonl \
    --output-dir runs/v4-fresh \
    --learning-rate 1e-4 --epochs 2 --seed 42                # [GPU]

python scripts/merge.py \
    --adapter-dir runs/v4-fresh/final \
    --output-dir  merged/v4-fresh --temperature 0.4          # [GPU]

python scripts/run_eval.py \
    --model merged/v4-fresh \
    --eval-file validation_samples/math.jsonl \
    --output-dir runs/eval/v4-fresh                          # [GPU]

# Diagnostic eval — MATH-500 + in-distribution surfaces. [GPU]
python scripts/diagnose.py \
    --model merged/v4-fresh --target all \
    --indist-file data_out/v4-mix/eval.jsonl \
    --output-dir runs/diagnostics/v4-fresh
```

### 4.8 v4-resume (resume from OMI-50k adapter, LR 5e-5)

Continues training from the **OMI-50k adapter** with a gentler learning
rate via `--init-from-adapter`; run AFTER §4.3 (OMI-50k) and after the
shared v4-mix prep in §4.6.

```bash
python scripts/train_sft.py \
    --train-file data_out/v4-mix/train.jsonl \
    --eval-file  data_out/v4-mix/eval.jsonl \
    --output-dir runs/v4-resume \
    --init-from-adapter runs/omi-50k/final \
    --learning-rate 5e-5 --epochs 2 --seed 42                # [GPU]

python scripts/merge.py \
    --adapter-dir runs/v4-resume/final \
    --output-dir  merged/v4-resume --temperature 0.4         # [GPU]

python scripts/run_eval.py \
    --model merged/v4-resume \
    --eval-file validation_samples/math.jsonl \
    --output-dir runs/eval/v4-resume                         # [GPU]

# Diagnostic eval — MATH-500 + in-distribution surfaces. [GPU]
python scripts/diagnose.py \
    --model merged/v4-resume --target all \
    --indist-file data_out/v4-mix/eval.jsonl \
    --output-dir runs/diagnostics/v4-resume
```

## 5. RLVR rescue run

Reported result. Curates a difficulty-filtered prompt set from the
OMI-50k training data using the OMI-50k merged checkpoint to score each
prompt's empirical solve rate, then runs GRPO from the OMI-50k adapter.
Must be run AFTER §4.3.

```bash
# 1. Curate prompts in the [0.2, 0.8] difficulty band (GPU scoring pass).
python data/prepare_rlvr.py \
    --input-jsonl    data_out/omi-50k/train.jsonl \
    --sft-model-path merged/omi-50k \
    --output-jsonl   data_out/rlvr_prompts.jsonl \
    --difficulty-lo 0.2 --difficulty-hi 0.8 --seed 42        # [GPU]

# 2. GRPO training [GPU]
python scripts/train_rlvr.py \
    --adapter-dir runs/omi-50k/final \
    --prompt-set  data_out/rlvr_prompts.jsonl \
    --output-dir  runs/rlvr-rescue \
    --learning-rate 3e-6 --kl-coef 0.04 \
    --num-generations 8 --rollout-temp 0.8 \
    --epochs 1 --seed 42

# 3. Merge a chosen checkpoint and evaluate [GPU]
python scripts/merge.py \
    --adapter-dir runs/rlvr-rescue/checkpoint-650 \
    --output-dir  merged/rlvr-rescue-ckpt650 --temperature 0.4
python scripts/run_eval.py \
    --model merged/rlvr-rescue-ckpt650 \
    --eval-file validation_samples/math.jsonl \
    --output-dir runs/eval/rlvr-rescue-ckpt650
```

**Observed outcome.** The full-epoch run exhibits late-run policy
collapse despite healthy in-flight monitoring (`frac_reward_zero_std`,
KL, reward variance all nominal). Checkpoints are written every 50
steps; `save_total_limit=20` retains a 20-checkpoint history per the
Tina methodology. The pre-collapse `checkpoint-650` (epoch ≈ 0.165) was
recoverable and evaluates at **OMI-50k-level noise** — no measurable
RLVR lift over SFT at 1.7B. The negative result is what we report.

## 6. Length-aware reward (design extension)

`scripts/reward_fn.py` includes an optional conciseness bonus, gated on
correctness so a short wrong answer can never beat a long right one.
Exposed via `scripts/train_rlvr.py` as:

```bash
python scripts/train_rlvr.py \
    --adapter-dir runs/omi-50k/final \
    --prompt-set  data_out/rlvr_prompts.jsonl \
    --output-dir  runs/rlvr-length-bonus \
    --learning-rate 3e-6 --kl-coef 0.04 \
    --num-generations 8 --rollout-temp 0.8 \
    --epochs 1 --seed 42 \
    --length-bonus-weight 0.1 --target-length-tokens 256     # [GPU]
```

This is a **designed and partially tested extension**, not a benchmarked
result. We did not run it to a fully evaluated checkpoint; no numbers
are reported for it.

## 7. Expected results

Headline metrics from `REPORT.md`. Each column below names the command
that reproduces it from a fresh clone.

| Column                            | How it is measured                                                                  | Reproducible locally? |
|-----------------------------------|-------------------------------------------------------------------------------------|-----------------------|
| validation pass@8 @0.4            | `run_eval.py --model merged/<name> --eval-file validation_samples/math.jsonl`       | yes                   |
|                                   | (or equivalently `diagnose.py --model merged/<name> --target validation`)           | yes                   |
| in-dist pass@1 / pass@4 (N=500)   | `diagnose.py --model merged/<name> --target indist --indist-file data_out/<name>/eval.jsonl` | yes            |
| MATH-500 pass@1 / pass@4 + per-subject / per-level | `diagnose.py --model merged/<name> --target math_test` (HF auto-download) | yes                   |
| CI nightly pass@8                 | Course's nightly CI against its **secret** problem set                              | **NO — not locally reproducible** |

Validation pass@8 is measured on the N=10 `validation_samples/math.jsonl`
snapshot at temperature 0.4 with the standard n=8 contract.
In-distribution pass@k is on the held-out N=500 split of the training
pool produced by `prepare_sft.py` for that model's source — so the
`--indist-file` for OMI-100k is `data_out/omi-100k/eval.jsonl`, not a
shared file. MATH-500 is the public MATH test set; `diagnose.py`
auto-downloads it from `HuggingFaceH4/MATH-500` (falling back to
`hendrycks/competition_math` if needed). Per-subject and per-level
breakdowns are in `runs/diagnostics/<name>/math_test/summary.json`
after the diagnostic run.

CI grades are the course's nightly secret-set scores — **not
reproducible from this repo alone**. They are reported here for
reference only and were observed via the leaderboard on
`cs-552-2026-emainelpe/math_model`.

| Model      | val pass@8 @0.4 | in-dist pass@1 (N=500) | in-dist pass@4 (N=500) | MATH-500 pass@1 | CI pass@8       |
|------------|----------------:|-----------------------:|-----------------------:|----------------:|----------------:|
| Qwen3-1.7B (bare baseline) | 0.200 | — | — | — | — |
| OMI-50k    |           0.400 |                  0.408 |                  0.628 |           0.514 | 0.32 – 0.35     |
| **OMI-100k (deployed)** | **0.500** (n=8) / 0.39 (n=16) | **0.456** | **0.686** | **0.516** | **0.34** |
| OMI-200k   |           0.300 |                  0.456 |                  0.678 |           0.525 | 0.31 (rollback) |
| v4-fresh   |           0.30  |                  0.395 |                  0.608 |           0.413 | — (not pushed)  |
| v4-resume  |           0.40  |                  0.408 |                  0.634 |           0.431 | — (not pushed)  |
| RLVR-ckpt650 | 0.300 (noise) |                ~0.43 |                     — |   0.519 | — (not pushed)  |

The 1.7B model is **capacity-bound**: scaling OMI2 from 50k → 100k →
200k yields only +1.1 pp on MATH-500 pass@1, and local lifts on
in-distribution / per-subject diagnostics did not consistently transfer
to the CI's secret set. The deployed checkpoint (OMI-100k) is the
operating point where local lift first appeared without a CI
regression.
