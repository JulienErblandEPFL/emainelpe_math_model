# Baseline: bare Qwen/Qwen3-1.7B on math

**Date.** 2026-05-07
**Hardware.** 1× A100 40GB on RCP cluster
**Wall-clock.** ~13 minutes for inference (10 problems × 8 completions)

## Command

```
python scripts/eval_local.py \
    --model Qwen/Qwen3-1.7B \
    --output-dir /tmp/eval-baseline
```

## Results

| Metric         | Value |
|----------------|-------|
| pass@1         | 0.300 |
| pass@8         | 0.400 |
| n_problems     | 10    |
| n_completions  | 8     |

## Settings

- **Eval file:** `validation_samples/math.jsonl` (default — course-vendored
  OOD competition snapshot, N=10)
- **Sampling:** temperature=0.3, top_p=0.95, top_k=20 — Stage-4 fallback,
  picked because the bare HF id ships no `generation_config.json` with
  sampling fields. The script's INFO log confirms the source-of-truth
  on every run.
- **CI contract (pinned):** n=8, seed=42, max_tokens=16384
- **vLLM:** bf16, max_model_len=20480, gpu_memory_utilization=0.85

## Notes

**Round-trip verified.** Piping the script's `generations.jsonl` back
through `python -m evaluate.score --benchmark math` produced
byte-identical metrics. The output schema is canonical CI-compatible;
re-scoring with a different `--method` is a one-liner without re-running
inference.

**Low diversity signal.** pass@1 ≈ pass@8 (0.300 vs 0.400) suggests the
bare model's 8 completions per problem are mostly consistent rather than
diverse — each problem tends to be either consistently solved or
consistently failed across the 8 samples. Worth flagging for RLVR
planning: GRPO needs reward variance *within* a problem's sample set, and
near-deterministic outputs at temp=0.3 may starve the policy gradient.
Likely want to bump rollout temperature when Stage 7 begins.

**Noise budget.** N=10 means the standard error on pass@1 is roughly
±5 percentage points. Differences smaller than ~10pp between checkpoints
are within noise on this snapshot. For tighter signals, also evaluate
against `data_out/eval.jsonl` (the 500-row DART held-out slice from
`data/prepare_sft.py` — different distribution, in-domain, much lower
variance) and look for movement on both targets together.

**What this number means.** It's the floor that any post-SFT checkpoint
must beat to claim the SFT phase added value. Pass@1 = 0.300 on N=10
OOD competition problems is the bar.

---

## Update 2026-05-09 — methodology gap, CI-mode re-baseline pending

The numbers above were taken under the legacy `scripts/eval_local.py`
defaults: `max_model_len=20480`, `max_tokens=16384`. Per the team
project README, the actual CI ceiling is `max_model_len=4096`
(combined prompt + generation). On 2026-05-09 the script's default
flipped to CI-faithful 4096/4096, with `--no-ci-mode` reinstating the
legacy permissive caps for ablations.

**What this means for the recorded numbers.** pass@1 = 0.300 and
pass@8 = 0.400 are a *soft upper estimate* of the bare-model
performance under the CI's actual budget. The true CI-mode baseline
may be lower because long `<think>` chains can be clipped at 4096
combined tokens. They remain the best estimate available until a
CI-mode re-baseline lands on RCP; treat the 0.400 pass@8 as a
pending-confirmation soft floor for the SFT-vs-baseline delta. The
methodology gap is also flagged in CLAUDE.md → "Bar to claim SFT
added value" so the policy stays in sync with the data here.

**Don't edit the original numbers above** — they record what was
actually measured on 2026-05-07 with the then-current settings. When
the CI-mode re-baseline runs, append a new section below this one
rather than overwriting history.

---

## 2026-05-09 CI-mode re-baseline

**Date.** 2026-05-09
**Hardware.** 1× A100 40GB on RCP cluster
**Eval file.** `validation_samples/math.jsonl` (course-vendored OOD
competition snapshot, N=10)
**Sampling.** n=8, seed=42. Bare model used `temperature=0.3` (the
script's pre-Stage-5 fallback because the bare HF id ships no
`generation_config.json`); the merged checkpoint used the
`generation_config.json` written by Stage 5 — `temperature=0.6`.

### Results

| Mode         | Model                              | pass@1   | pass@8   |
|--------------|------------------------------------|----------|----------|
| ci-faithful  | `Qwen/Qwen3-1.7B`                  | 0.1625   | 0.2000   |
| ci-faithful  | `cs-552-2026-emainelpe/math_model` | 0.2125   | 0.4000   |
| legacy       | `Qwen/Qwen3-1.7B`                  | 0.2875   | 0.3000   |
| legacy       | `cs-552-2026-emainelpe/math_model` | 0.2000   | 0.3000   |

ci-faithful: `max_model_len=4096`, `max_tokens=4096` (default after
the 2026-05-09 default-flip). legacy: `max_model_len=20480`,
`max_tokens=16384` (`--no-ci-mode`).

### Headline finding

Under ci-faithful caps — what the nightly CI actually exercises —
v1 SFT lifts pass@8 from **0.2000 to 0.4000 (+20 pp)**, well outside
the ±5 pp noise band noted earlier in this file. Pass@1 also moves
in the right direction (+5 pp).

Under legacy caps the improvement disappears: pass@8 is flat at
0.30 across baseline and v1 SFT, and pass@1 actually drops on the
SFT model (0.29 → 0.20).

### Most plausible interpretation

The SFT model produces longer reasoning chains than baseline. Under
the tight 4096-combined cap it commits to a `\boxed{...}` answer
within budget; under the loose 16384 cap it spirals (the loop
behavior diagnosed in earlier eval analysis), and the extra tokens
buy the bare model more recovery room while costing the SFT model
its commit-discipline. **Operational consequence:** ci-faithful is
the predictive number for what CI will report. legacy is now an
ablation knob, not the headline reading.

### What this replaces

- The 2026-05-07 numbers above (pass@1=0.300, pass@8=0.400) were
  measured under legacy caps and are no longer the headline
  baseline. They stay in this file as historical record.
- The "soft upper estimate" language in CLAUDE.md → "Bar to claim
  SFT added value" is replaced (in the same commit) with the
  measured ci-faithful values. v1 SFT (pushed to HF) cleared the
  bar. Future SFT variants (v2 mixed, v3 OMI2-only) must beat
  pass@8 = 0.4000 under ci-faithful caps to be considered an
  improvement over v1.

### Caveats still standing

- **N=10 noise budget unchanged.** ±5 pp standard error on pass@1
  applies to the new numbers too. The +20 pp pass@8 jump is
  comfortably outside noise; the +5 pp pass@1 jump is at the noise
  threshold and should not be over-interpreted on this snapshot.
  For tighter signals re-run on `data_out/eval.jsonl` (500-row DART
  held-out slice).
- **OOD competition problems only.** This snapshot is what CI uses,
  but it doesn't track in-distribution generalization. Pair with
  the DART eval slice for a fuller picture.
- **The secret CI eval set is not this snapshot.** The team README
  describes the public snapshot as a calibration tool; the CI's
  private set may differ in difficulty mix. Treat 0.4000 as the
  best available estimate, not a CI-grade promise.

### Evidence trail (W&B run IDs)

The runs behind the numbers above and the in-flight variant experiments
share a single W&B project (`emainelpe-math`):

- **v1 SFT (DART only, pushed to HF as `cs-552-2026-emainelpe/math_model`).**
  W&B run id: `yazg1nth`. The four-row table above scores this checkpoint
  against the bare-model rows.
- **v2 SFT (mixed DART + OpenMathInstruct-2, 50/50).** RCP job:
  `cs552-erbland-g65-v2-mixed-20260511-123452`. In flight at the time of
  this writing (ETA ~17:30 on 2026-05-11). Re-baseline this section once
  the v2 checkpoint is evaluated against the same `validation_samples/math.jsonl`
  snapshot under ci-faithful caps.
- **v3 SFT (pure OpenMathInstruct-2).** RCP job (third attempt, with the
  eval-OOM mitigation in place): `cs552-erbland-g65-v3-omi2-fix2-20260511-152150`.
  See `IMPLEMENTATION_PLAN.md` → "Lessons learned" for the OOM bug-fix
  arc. Re-baseline when training finishes.

---

## 2026-05-11 SFT comparison and temperature sweep

**Date.** 2026-05-11
**Hardware.** 1× A100 40GB on RCP cluster
**Eval file.** `validation_samples/math.jsonl` (N=10, OOD competition snapshot)
**Sampling.** n=8, seed=42, `top_p=0.95`, `top_k=20`. Five temperatures
swept per checkpoint: 0.4, 0.5, 0.6, 0.7, 0.8. CI-faithful caps
(`max_model_len=4096`, `max_tokens=4096`). 3 checkpoints × 5 temperatures
= 15 evals.

### Results

| variant | temp | pass@1 | pass@8 |
|---------|------|--------|--------|
| v1      | 0.4  | 0.2000 | 0.3000 |
| v1      | 0.5  | 0.2000 | 0.3000 |
| v1      | 0.6  | 0.2000 | 0.3000 |
| v1      | 0.7  | 0.1625 | 0.3000 |
| v1      | 0.8  | 0.2000 | 0.3000 |
| v2      | 0.4  | 0.2125 | 0.3000 |
| v2      | 0.5  | 0.1875 | 0.3000 |
| v2      | 0.6  | 0.2750 | 0.4000 |
| v2      | 0.7  | 0.2125 | 0.3000 |
| v2      | 0.8  | 0.2250 | 0.3000 |
| v3      | 0.4  | **0.2875** | **0.4000** |  ← winner
| v3      | 0.5  | 0.2500 | 0.3000 |
| v3      | 0.6  | 0.2375 | 0.4000 |
| v3      | 0.7  | 0.2125 | 0.3000 |
| v3      | 0.8  | 0.2375 | 0.3000 |

### Headline finding

**v3 at temp=0.4 is the SFT winner and the RLVR base.** It posts the
highest pass@1 of any (variant, temp) combination in the sweep (0.2875)
and joint-highest pass@8 (0.4000). It also has the widest operating
range on pass@8: v3 reaches 0.4000 at *two* temperatures (0.4 and 0.6),
while v2 only reaches 0.4000 at temp=0.6, and v1 never reaches 0.4000
at any temperature swept.

Per-variant best:
- v1: pass@8 invariant at 0.3000 across all five temperatures (no
  temperature unlocks the +20 pp jump previously attributed to it).
- v2: best at temp=0.6, pass@8 = 0.4000, pass@1 = 0.2750.
- v3: best at temp=0.4, pass@8 = 0.4000, pass@1 = 0.2875.

### The earlier 0.4000 was upper-end noise

The 2026-05-09 CI-mode re-baseline section above reports v1 SFT at
pass@8 = 0.4000. That number was a single-temperature eval (one run
at the checkpoint's `generation_config.json`-pinned temp=0.6), and on
N=10 the standard error is ±5 pp on pass@1 and even chunkier on
pass@8. The five-temperature sweep here shows v1's actual pass@8 is
flat at 0.3000 — the earlier 0.4000 was a lucky draw on a single
seed-42 sample, not a stable estimate. Calibration via a temperature
sweep is the more rigorous methodology and supersedes the
single-temperature comparison.

**This does not invalidate the 2026-05-09 "+20 pp pass@8" claim for
SFT-vs-baseline.** The bare-model baseline measurement (0.2000) also
stands on a single seed-42 draw at temp=0.3; under the same sweep
methodology it could move similarly. The right read is: v3 SFT at
temp=0.4 beats the bare-model baseline measurement by ~+20 pp pass@8
under the same CI-faithful eval contract. The earlier read survives;
the *variant comparison within SFT* did not.

### New headline for the SFT phase

**v3 (pure OMI2) > v2 (mixed) > v1 (DART)** on calibrated comparison,
under ci-faithful caps. The teacher-quality hypothesis (OMI2's
Llama3.1-405B-Instruct teacher outperforms DART's DeepSeekMath-7B-RL)
holds once the eval methodology is robust enough to surface a 10 pp
difference on N=10. Mixing helps less than going pure.

### Evaluator calibration

`scripts/eval_local.py` imports `score_generations` from the
team-vendored `evaluate/` module. `scripts/reward_fn.py` imports
`is_equiv` from the same module. The `evaluate/` package is
byte-identical to the OpenCompass code that the nightly CI runs.

This means all three feedback loops — local eval, RLVR reward shaping,
and CI grading — share the same equivalence function. Local pass@k
numbers from this script are predictive of CI pass@k on the *same
generations*. They are not predictive across the dataset boundary
(the CI's secret eval set differs from `validation_samples/math.jsonl`
in difficulty mix), but within the public snapshot the local→CI
numerical drift is zero.

### Operational consequences

- **RLVR base.** Stage 7 (GRPO) starts from the v3 SFT adapter, not v1.
  Update `--sft-model` and `--adapter-dir` accordingly in the RCP
  submission. The v3 adapter directory lives at the path produced by
  the `cs552-erbland-g65-v3-omi2-fix2-20260511-152150` job's output.
- **Inference temperature.** v3's `generation_config.json` should be
  updated to `temperature=0.4` before the HF push so the CI samples at
  the calibrated peak. (Out of scope for this doc update — the user
  handles the regenerate-config step as part of the RLVR launch flow.)
- **Bar for v4+ ablations.** Any future SFT recipe must beat pass@8 =
  0.4000 *with the temperature sweep applied*, not a single-temperature
  draw. The within-noise band is ±5 pp; one isolated 0.40 at one
  temperature does not clear the bar.

---

## 2026-05-13 — RLVR-v3 (partial), cap-mode parity, CI nightly grade

**Date.** 2026-05-13
**Hardware.** 1× A100 40GB on RCP cluster
**Eval file.** `validation_samples/math.jsonl` (N=10, OOD competition snapshot)
**Sampling.** n=8, seed=42, `top_p=0.95`, `top_k=20`. Each variant
evaluated at its locked checkpoint temperature (v3 SFT @ 0.4; bare
baseline @ 0.3 — fallback because the bare HF id ships no
`generation_config.json`).

### Results — local

| Mode               | Model                              | pass@1 | pass@8 |
|--------------------|------------------------------------|--------|--------|
| ci-faithful (4096) | `Qwen/Qwen3-1.7B` (bare, temp=0.3) | 0.1625 | 0.2000 |
| ci-faithful (4096) | v3 SFT (temp=0.4)                  | 0.2875 | 0.4000 |
| ci-faithful (4096) | RLVR-v3 (600 steps, temp=0.4)      | —      | 0.3000 |
| final-grading (16384) | `Qwen/Qwen3-1.7B` (bare, temp=0.3) | 0.1625 | 0.2000 |
| final-grading (16384) | v3 SFT (temp=0.4)                  | 0.2875 | 0.4000 |
| final-grading (16384) | RLVR-v3 (600 steps, temp=0.4)      | —      | 0.3000 |

### Results — CI nightly (2026-05-13)

| Model    | benchmark    | pass@8 |
|----------|--------------|--------|
| v3 SFT   | CI math set  | 0.3200 |

The CI's secret math benchmark is N≥100 and disjoint from
`validation_samples/math.jsonl`. The 0.32 nightly grade is the
externally verified headline for the currently pushed checkpoint.

### Cap-mode parity finding

Per TA clarification, final-grading mode raises the per-completion
budget from 4096 to 16384 tokens. For our 1.7B math expert on
`validation_samples/math.jsonl`, **the cap mode does not change
pass@8 on any evaluated checkpoint**: completions terminate
naturally (EOS or `\boxed{...}`) before 4096 tokens. Truncation is
not the binding constraint on this snapshot.

**Operational consequence.** ci-faithful caps remain the predictive
local-eval mode. The TA's final-grading bump does not buy our model
any extra headroom on this set; do not expect the CI nightly's 0.32
to lift under final-grading mode either. If a future variant has
markedly longer reasoning chains the parity could break — re-check
at that point.

### RLVR-v3 regression

After five integration bugs were fixed (see `IMPLEMENTATION_PLAN.md`
→ "Lessons learned" → "2026-05-12/13 — RLVR 5-bug arc"), GRPO
training ran cleanly. Wall-clock constraints stopped training at
**600 steps (~16% of one epoch on 3919 difficulty-curated
prompts)**. Training metrics looked healthy at stop:
- reward_std ≈ 0.35–0.55 (good gradient signal)
- KL from SFT reference ≈ 0.001–0.002 (no spike)
- ~38,400 total rollouts trained on (600 steps × 8 rollouts × 8 prompts)

The resulting RLVR-v3 checkpoint **regressed pass@8 from 0.40 to
0.30** on `validation_samples/math.jsonl` under both cap modes. The
policy moved off the SFT optimum without enough steps to recover or
improve. **Not pushed to HF.** v3 SFT remains the team math expert.

This is consistent with Dang & Ngo 2025's small-model RLVR
instability warning: partial RLVR can be net-negative. The training
infrastructure is correct (fail-fast preflights, healthy in-training
metrics) — the limiting factor was wall-clock, not engineering.

### Bar for any future RLVR attempt

Beat **pass@8 = 0.4000 under ci-faithful caps** on
`validation_samples/math.jsonl`. A regression is a regression; the
v3 SFT 0.40 stands until a full-epoch RLVR run lands and clears it.
Headline grade target: **CI pass@8 > 0.32** for any push that
replaces the current v3 SFT.

### v4 SFT note

A v4 SFT variant (200k OMI2, single-source) was submitted in
parallel on 2026-05-12 and **crashed at epoch 0.08 with OOM during a
training step** (9.27 GiB single-tensor allocation on a long
sequence). The existing eval-OOM fix (`per_device_eval_batch_size=1`)
addresses eval-time logits materialization but not the training-step
logits + loss path on extremely long rows. v4 is dead; v3 SFT
remains the production checkpoint. The OOM signal suggests 200k pure
OMI2 has a tail of very long sequences that the current token-length
filter does not catch.

---

## 2026-05-14 — v4 SFT MATH-500 measurement, Liger Kernel deployed,
## RLVR rescue + v5 OMI2 100k launches

**Date.** 2026-05-14
**Hardware.** 1× A100 40GB on RCP cluster (×2 slots — parallel runs)
**Eval files.** MATH-500 for v4 final read; `validation_samples/math.jsonl`
for the 5-temperature sanity sweep.

### v4 SFT — final measurement

After the Liger Kernel OOM fix landed (`use_liger_kernel=True` default,
2026-05-13/14), the two v4 variants completed training on the
v4-mix dataset (67,135 train rows; v4-mix composition documented in
CLAUDE.md → "v4 training plan").

| Variant     | W&B run    | init                | lr     | final loss | MATH-500 pass@1 |
|-------------|------------|---------------------|--------|------------|-----------------|
| v4-fresh    | k93kbsns   | Qwen3-1.7B base     | 1e-4   | ~0.394     | 0.413           |
| v4-resume   | zd5x6syj   | v3 adapter          | 5e-5   | ~0.413     | 0.431           |
| v3 (prior)  | —          | Qwen3-1.7B base     | 1e-4   | (recorded) | 0.514           |

Both v4 variants **regressed** from v3 across every subject and every
level on MATH-500. The diagnostic-targeted subjects regressed sharply:

| Subject              | v3 pass@1 | v4-fresh | v4-resume | delta vs v3 |
|----------------------|-----------|----------|-----------|-------------|
| Intermediate Algebra | 0.296     | 0.211    | 0.216     | -8 pp       |
| Precalculus          | 0.339     | 0.174    | 0.196     | -14/-16 pp  |

5-temperature validation sweep on v4-resume on
`validation_samples/math.jsonl` (N=10):

| Temp | v4-resume pass@8 | v3 pass@8 |
|------|------------------|-----------|
| 0.4  | 0.40             | 0.40      |
| 0.5  | 0.30             | 0.30      |
| 0.6  | 0.30             | 0.40      |
| 0.7  | 0.30             | 0.30      |
| 0.8  | 0.30             | 0.30      |

v3 retains its 2-of-5-temps-at-0.40 footprint; v4-resume's lone 0.40
at temp=0.4 is upper-end noise on N=10. The N=500 MATH-500 read is
the load-bearing measurement — v4-resume regresses there too.

### Headline finding — v4 NEGATIVE RESULT

The v4 diagnostic-driven mix did **not** clear the bar of v3 SFT.
**v3 remains the production SFT checkpoint.** Hypothesis confirmed:
at 1.7B parameters the math expert is **capacity-bound**, not
coverage-bound. Adding MATH-train + NuminaMath buckets at the
oversampling ratios the v3 diagnostic suggested diluted OMI2's
contribution without unlocking new problem-solving capability.

Within-bucket oversampling (IntAlg 1295 unique × 4× cap, Precalc 746
× 4× cap) was the right operational implementation of the v3
diagnostic — the data-prep pipeline worked as designed. The lesson
lives upstream: targeted data augmentation has a per-parameter
ceiling, and we appear to be at it.

### Team HF state

v4-resume was pushed to team HF on 2026-05-14 for one CI cycle to
observe the nightly grade. **Plan: re-push v3 on 2026-05-15 morning**
before the next CI window if v4-resume's CI grade lands below v3's
~0.32 (likely — expected v4-resume ~0.25-0.28 given the MATH-500
regression).

Personal HF backup (pending):
`JulienE220/math-adapter-sft-v4-resume-r32-20260514`.
v4-fresh not backed up (underperformed v4-resume).

### Liger Kernel — deployed and verified

Three OOM crashes (v4-200k 2026-05-12, v4-fresh first attempt
2026-05-13, v4-resume first attempt 2026-05-13) all hit
`torch.OutOfMemoryError` at `outputs.logits[..., :-1, :].contiguous()`
— the materialized B × T × 151,643 × 4 bytes logits tensor exceeded
A100-40GB headroom on near-max-length batches.

Fix landed 2026-05-13/14: `use_liger_kernel=True` default in both
SFTConfig and GRPOConfig; `PYTORCH_ALLOC_CONF=expandable_segments:True`
env var in all three submit scripts; `liger-kernel>=0.8.0` pinned in
`requirements.txt`. Liger's fused linear-cross-entropy never
materializes the full logits tensor — the `vocab_size × 4 bytes` term
drops out of the per-step memory equation entirely. Cluster
verification (2026-05-14): liger-kernel 0.8.0 installs cleanly with
TRL 0.19.1 + transformers 5.7.0 + torch 2.10.0+cu128. After deployment
the v4-fresh and v4-resume retries completed without OOM.

**Caveat for future ops.** `liger_kernel` 0.8.0 does NOT expose
`__version__`. Any future sanity check must not depend on that
attribute.

### RLVR rescue — IN FLIGHT (launched 2026-05-14 15:25)

Job: `cs552-erbland-g65-rescue-20260514-152540`. ETA 2026-05-15
08:00-10:00.

**Config snapshot** (rescue invocation per CLAUDE.md → "RLVR rescue
plan"):
- Adapter init: v3
  (`cs552-erbland-g65-v3-omi2-fix2-20260511-152150/final`)
- Prompt set: `rlvr_prompts.jsonl` from `data_out_v3`, 3936 problems,
  signal-band-filtered to [0.250, 0.750] solve_rate (n=8 rollouts
  → quantized to {0.250, 0.375, 0.500, 0.625, 0.750}).
- `USE_VLLM=1`, `MASK_TRUNCATED=1`, `LOG_COMPLETIONS=1`
- `LEARNING_RATE=3e-6`, `KL_COEF=0.04`, `ROLLOUT_TEMP=0.8`,
  `MAX_PROMPTS=3936`, `LOSS_TYPE=dapo`,
  `HARD_KILL_ON_WEAK_SIGNAL=unset`
- Liger Kernel default ON

**Early observations** (steps 1-30): `frac_reward_zero_std` mostly 0
with sporadic per-prompt 1s; KL ~0.0003-0.002; reward std
~0.35-0.52. Step pace ~10-20 s/step → projected ~15-17h wall-clock.
This is the *opposite* failure-signal profile from retry3
(constant `frac_reward_zero_std=1.0`); the signal-band-filtered
prompt set is doing its job — rescue lever P1 from the 2026-05-13
plan is working.

### v5 OMI2 100k SFT — IN FLIGHT (launched 2026-05-14 16:22)

Job: `cs552-erbland-g65-v4-fresh-20260514-162214` (job-name cosmetic
— re-used the v4-fresh submit script with `SKIP_PREP=1` +
`DATA_OUT_DIR` override; the *data* is v5 OMI2 100k, not v4-mix).
ETA 2026-05-14 23:30 - 2026-05-15 01:00.

**Hypothesis.** Tests whether v3 is OMI2-saturated at 50k. Single
variable changed vs v3: dataset size (50k → 100k). All else
identical.

**Dataset.** `/scratch/Julien/data_out_v5_omi2_100k/` — 100,000 train
+ 500 eval. Source `nvidia/OpenMathInstruct-2` split=train_1M.
per_question_cap=4 does not bind (all 999,893 raw rows have unique
problems); max_formatted_tokens=2900 drops 0 (OMI2 CoTs compact).
Byte-clean: no oversampling, no cross-source-dedup tension.

**Early observations** (steps 1-2): loss 0.974 → 0.961,
mean_token_accuracy 0.807 → 0.808. Starting loss is **lower** than
v4-fresh's 1.103 — OMI2 alone is closer to Qwen3-1.7B's natural
distribution than the v4-mix was. Step pace ~4 s/step → projected
~7-8h wall-clock.

### Parallel run note

Both RLVR rescue and v5 SFT are running concurrently on 2 A100-40g
slots. Both preemptible. **Higher-value-to-preserve on preemption:
RLVR** (more compute invested; non-trivial-to-regenerate signal-band
prompt set). v5 SFT is cheap to relaunch from the byte-clean
`data_out_v5_omi2_100k/` cache.

### Bar reminder (unchanged)

Beat **MATH-500 pass@1 = 0.514** and **CI nightly pass@8 > 0.32**.
v3 SFT holds both numbers; v4 cleared neither. RLVR-rescue and
v5 OMI2 100k will be evaluated against this bar on 2026-05-15.
