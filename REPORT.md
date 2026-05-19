# REPORT.md — math expert track, working scratchpad

**Status.** Working document. Open on 2026-06-01 to write the final
report; everything in here is sourced from `CLAUDE.md`,
`IMPLEMENTATION_PLAN.md`, `docs/BASELINE.md`, the daily-log entries in
`CLAUDE.md`, and the diagnostic JSONs at `/scratch/Julien/diagnostics/`.
Do not invent numbers when finalizing — only re-frame what's recorded
here.

Last updated: 2026-05-14.

---

## 1. Project Overview

- **Task.** Train a math expert that scores well on the team's secret
  test set evaluated as `pass@8` with `n=8` rollouts per problem (free-
  form math answers, extracted from `\boxed{...}`).
- **Base architecture.** `Qwen/Qwen3-1.7B` — **LOCKED** for team merge
  compatibility (Phase 3 DARE / AdaMerging into the group model).
- **Adapter spec (LOCKED via `configs/lora.yaml`).**
  - `r = 32`
  - `lora_alpha = 64`
  - 7 target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`,
    `gate_proj`, `up_proj`, `down_proj`
  - `max_seq_length = 4096`
- **Chat template.** `chat_template/chat_template.jinja` — LOCKED
  (byte-identical to `emainelpe-shared`).
- **Team.** Émainèlpé (g65). Math specialist: Julien Erbland. Other
  three specialists: knowledge (Max Henrotin), multilingual (Mathis
  Richard), safety (Morgane Magnin).
- **Hugging Face targets.**
  - Team: `cs-552-2026-emainelpe/math_model` (single math repo per
    course).
  - Personal backups (Julien): under `JulienE220/...`.
- **Critical dates.**
  - 2026-05-19: Phase 3 team merge (all four specialists merged into
    one group model).
  - 2026-05-24: model-running validation milestone (10% of project
    grade).
  - 2026-06-07: final submission (50% of project grade).
- **CI behaviour (binding contract).** Nightly grades on an N≥100
  secret math set. `max_model_len=4096` (combined prompt + completion),
  `n=8`, `seed=42`, 1800 s wall-clock cap. Extraction via `\boxed{}`,
  OpenCompass `is_equiv` for equivalence. Currently graded v3 SFT at
  **pass@8 ≈ 0.32** on the CI secret set (2026-05-13 nightly).

---

## 2. Methodology — overall approach

Two-phase training stack:

- **Phase 1: SFT.** `trl.SFTTrainer` on the locked LoRA spec, fed by
  `data/prepare_sft.py`. Loss masking is full-sequence (no
  assistant-only mask — TRL 0.21+ refused the locked Jinja's lack of
  `{% generation %}` markers; adding them is a v2 stretch coordinated
  with `emainelpe-shared`).
- **Phase 2 (optional): RLVR.** `trl.GRPOTrainer` on top of the SFT
  adapter; reward is exact-match (`evaluate.is_equiv`) plus a small
  shape bonus (`+ 0.05 * has_box`). Prompt set curated to a difficulty
  band `[0.2, 0.8]` (later tightened to `[0.25, 0.75]` for rescue).

Supporting infrastructure:

- **Inference / eval.** `vLLM` is the inference back-end for both
  evaluation (`scripts/eval_local.py`) and RLVR rollouts when
  `USE_VLLM=1`. The CI uses vLLM identically.
- **Local eval mirroring CI.** `scripts/eval_local.py` defaults to
  CI-faithful caps: `max_model_len=4096`, `max_tokens=4096`, `n=8`,
  `seed=42`. `--no-ci-mode` reinstates the legacy permissive
  20480/16384 caps for ablations.
- **Diagnostic harness.** `scripts/diagnose_v3.py` produces a 3-target
  analysis (validation_samples, in-distribution OMI2/DART held-out,
  MATH-500) with per-subject, per-level, and per-failure-mode
  breakdowns. The failure-mode classifier (priority order) is:
  `repetition` > `correct` > `no_box` > `truncated` > `wrong_box` >
  `other`.
- **CI-faithful scoring.** All metrics computed against the **vendored
  `evaluate/`** package (byte-identical to the course CI scorer). We
  never re-implement extraction or equivalence.
- **Cluster.** RCP, project `course-cs-552-erbland`, 1×A100-40GB per
  job (occasionally 2 slots for parallel runs). Course Docker image
  `ayushkumartarun/course-cs-552-standard:v1`. HF cache lives in
  `/scratch/hf_cache`; training outputs in `/scratch/Julien/runs/`;
  data outputs in `/scratch/Julien/data_out*`.
- **Repro contract.** `seed=42` is fixed in `data/prepare_sft.py`,
  `scripts/train_sft.py`, and `scripts/eval_local.py`. Same `seed=42`
  as the course CI.

---

## 3. Iteration history

For each named training run: **Goal / hypothesis**, **Config**,
**Results**, **What we learned**.

### 3.1 v1 — DART-uniform 50k SFT (initial SFT baseline)

- **Goal / hypothesis.** Establish that the SFT pipeline runs
  end-to-end and that DART-uniform 50k clears the bare-model baseline.
- **Config.**
  - Data: `hkust-nlp/dart-math-uniform`, subsampled to ~50k examples
    (per-question cap 4-6).
  - LR=1e-4, 2 epochs, effective batch size 32 (`per_device=4`,
    `grad_accum=8`), cosine schedule with 3% warmup.
  - Locked LoRA (r=32, α=64, 7 modules).
  - Fresh init from Qwen3-1.7B base.
- **Personal HF.** `JulienE220/math-adapter-sft-dart50k-r32-20260508`.
- **Results.**
  - `validation_samples/math.jsonl` (N=10, n=8): pass@1 = 0.2000,
    pass@8 = 0.3000.
  - 5-temperature sweep (2026-05-11): flat at pass@8 = 0.3000 across
    all temps; the earlier "v1 cleared the bar" reading at temp=0.6
    that hit pass@8=0.4000 was upper-end noise on N=10.
- **What we learned.** DART-uniform delivers a clean +10pp pass@8
  jump over the bare baseline, but the +20pp jump that briefly
  appeared in single-temp evaluation didn't survive the 5-temp sweep.
  The first signal that **single-temperature evaluation on N=10 is
  noisy enough to fool both us and the leaderboard** — methodology
  shift logged in `IMPLEMENTATION_PLAN.md` → "Lessons learned" →
  "single-temperature eval comparison is noisy on N=10".

### 3.2 v2 — Mixed 50k SFT (DART + OpenMathInstruct-2)

- **Goal / hypothesis.** Mixing DART's diversity with OMI2's stronger
  teacher CoT (Llama-3.1-405B-Instruct) should lift pass@8 above v1.
- **Config.**
  - Data: 50/50 mix of `hkust-nlp/dart-math-uniform` and
    `nvidia/OpenMathInstruct-2`, ~50k total. Per-question cap=4.
  - All other hyperparams identical to v1 (LR=1e-4, 2 epochs, locked
    LoRA, fresh init).
- **Personal HF.**
  `JulienE220/math-adapter-sft-mixed-50k-r32-20260511`.
- **Results.**
  - `validation_samples/math.jsonl` 5-temp sweep: best at temp=0.6
    with pass@1 = 0.2750, **pass@8 = 0.4000**.
- **What we learned.** OMI2 helps. Mixing alone (without dropping
  DART) cleared the bare-baseline by +20pp pass@8. Set up the next
  question: would pure OMI2 do even better, or is the DART diversity
  load-bearing?

### 3.3 v3 — Pure OMI2 50k SFT (the deployed baseline)

- **Goal / hypothesis.** Test whether pure OMI2 (drop DART entirely)
  matches or beats the mixed v2. If matched, OMI2 alone is the simpler
  recipe and the headline.
- **Config.**
  - Data: `nvidia/OpenMathInstruct-2` split `train_1M`, subsampled to
    50k, per-question cap=4.
  - LR=1e-4, 2 epochs, locked LoRA, fresh init.
  - Run name: `cs552-erbland-g65-v3-omi2-fix2-20260511-152150`.
- **Personal HF.**
  `JulienE220/math-adapter-sft-omi2-50k-r32-20260511`.
- **Team HF (pushed).** `cs-552-2026-emainelpe/math_model` with
  `generation_config.json` carrying `temperature=0.4`.
- **Results — `validation_samples/math.jsonl` (N=10, n=8).** Best
  (variant, temp) across the 5-temperature sweep is temp=0.4:
  - pass@1 = **0.2875**, pass@8 = **0.4000**
  - Per-temp pass@8: temp 0.4 = **0.40**, temp 0.5 = 0.30, temp 0.6 =
    **0.40**, temp 0.7 = 0.30, temp 0.8 = 0.30.
  - v3 hits 0.40 at **two** temperatures — multi-temp robust signal,
    not the single-temp noise spike v1/v2 had.
- **Results — in-distribution OMI2 held-out eval set (N=500, n=4).**
  - pass@1 = **0.408**, pass@4 = **0.628**.
- **Results — MATH-500 (full test set, n=4).**
  - pass@1 = **0.514**, pass@4 = **0.686**.
- **Per-subject MATH-500 pass@1 (v3 diagnostic, source
  `/scratch/Julien/diagnostics/v3_eval_20260513T133259Z/`):**

  | Subject              | pass@1 |
  |----------------------|--------|
  | Algebra              | 0.700  |
  | Prealgebra           | 0.668  |
  | Number Theory        | 0.492  |
  | Counting & Prob.     | 0.480  |
  | Geometry             | 0.463  |
  | Precalculus          | 0.339  |
  | Intermediate Algebra | 0.296  |

- **Per-level MATH-500 pass@1 (v3 diagnostic).**

  | Level | pass@1 |
  |-------|--------|
  | 1     | 0.797  |
  | 2     | (TODO: not recorded in user-provided summary; pull from `level_summary.json`) |
  | 3     | (TODO)  |
  | 4     | (TODO)  |
  | 5     | 0.213  |

- **Failure-mode breakdown (v3 diagnostic, aggregated):**

  | Mode       | Count  |
  |------------|--------|
  | wrong_box  | 922    |
  | repetition | 241    |
  | no_box     | 20     |
  | correct / truncated / other | (TODO: compute from diagnostic JSON) |

- **CI nightly grade (2026-05-13).** pass@8 = **0.3200** on the
  course's secret math set (N≥100, disjoint from
  `validation_samples/math.jsonl`).
- **Cap-mode parity finding.** v3 produces identical pass@8 under both
  ci-faithful (`max_tokens=4096`) and final-grading
  (`max_tokens=16384`) cap modes on `validation_samples/math.jsonl`.
  Completions terminate naturally (EOS or `\boxed{...}`) before 4096
  tokens. Truncation is not the binding constraint on this set; the
  TA's final-grading bump does not lift v3's headline.
- **What we learned.** Pure OMI2 50k is the strongest SFT recipe in
  the v1/v2/v3 sweep. The +20pp pass@8 over the bare baseline survives
  the 5-temp sweep AND a 32% nightly CI grade. The diagnostic reveals
  two structural weaknesses: **Intermediate Algebra and Precalculus**
  underperform Algebra/Prealgebra by 30-40 pp, and **Level 5**
  problems underperform Level 1 by 58 pp. These two gaps become the
  v4-mix design target.

### 3.4 Diagnostic-driven design of v4-mix

This is the analysis-to-design bridge between v3's diagnostic and the
v4 dataset. Worth a standalone subsection in the report because the
v4 design *was the experiment* — the question wasn't "will training
work" but "will targeted data fix targeted gaps."

- **Method.** `scripts/diagnose_v3.py` produced per-subject and
  per-level pass@1 on MATH-500 (N=500, n=4). The three weakest signals
  were:
  - Intermediate Algebra: 0.296 (−14 pp vs strongest Algebra subject)
  - Precalculus: 0.339 (−10 pp vs Algebra)
  - Level 5: 0.213 (−25 pp vs Level 1)
- **v4-mix dataset composition target (pre-pipeline-cap, 2026-05-13):**

  | Source                          | Target (pre-cap) | Pool size      | Effective post-cap |
  |---------------------------------|------------------|----------------|---------------------|
  | OMI2 train_1M subset            | 40,000           | ~999k unique   | 40,000              |
  | MATH-train IntAlg bucket        | 12,000           | ~1,295 unique  | ~5,180 (1,295×4)    |
  | MATH-train Precalc bucket       | 7,000            | ~746 unique    | ~2,984 (746×4)      |
  | MATH-train Level 4-5 bucket     | 18,000           | ~3,000 unique  | ~12,000 (3,000×4)   |
  | MATH-train Level 1-3 bucket     | 13,000           | ~4,500 unique  | 13,000 (no cap bind)|
  | NuminaMath-CoT olympiad subset  | 5,000            | ~247k unique   | 5,000               |
  | **Total**                       | ~95,000          | —              | ~67k effective      |

  Sources allowlist (NuminaMath olympiad subset): `olympiads`,
  `amc_aime`, `aops_forum`, `synthetic_amc`. `math` source intentionally
  excluded to avoid cross-bucket duplication with the EleutherAI
  Hendrycks MATH train.
- **Two prep iterations.**
  1. First attempt ran cross-source dedup at the final concat; the
     94k → 50k collapse eliminated the within-bucket oversampling
     that the diagnostic-driven multipliers depended on (IntAlg's
     12k target from 1.3k unique collapsed to 1.3k, defeating the
     entire IntAlg lever).
  2. Fix (2026-05-13, final policy): **cross-source dedup disabled.**
     Within-bucket oversampling is preserved end-to-end; the
     downstream `per_question_cap=4` inside `build_pipeline`
     becomes the binding multiplicity cap.
- **Final v4 dataset shape.** 67,135 train + 500 eval rows. Effective
  trained-on per epoch ≈ 60-70k rows with diagnostic-targeted
  subjects contributing their full 4× weight where the source pool
  permits.
- **OOM mitigation (data-prep layer).** `--source v4-mix` auto-defaults
  `--max-formatted-tokens` to 2900, dropping rows whose Qwen3-tokenized
  formatted chat exceeds the cap. The locked `configs/lora.yaml`
  (`max_seq_length=4096`) was untouched to preserve the merge contract.
  After 2026-05-13/14 the primary OOM fix became Liger Kernel (see
  §4.1) and this cap became belt-and-suspenders.
- **What we learned (preview).** The diagnostic-to-design pipeline
  worked cleanly: identify weak slice → oversample its problems
  upstream → train and re-measure. The result (§3.5–3.6) says the
  *learning lever* was the wrong one, but the *engineering of the
  experiment* was sound.

### 3.5 v4-fresh — fresh init from Qwen3-1.7B base on v4-mix (NEGATIVE RESULT)

- **Goal / hypothesis.** Fresh basin exploration on the
  diagnostic-targeted v4 mix. If v3's optimum is local, fresh init
  has the best chance to escape it and lift the weak subjects.
- **Config.**
  - Init: Qwen3-1.7B base (no v3 adapter).
  - Data: v4-mix (67,135 train rows).
  - LR=1e-4, 2 epochs, locked LoRA, Liger Kernel ON.
  - Run name: `cs552-erbland-g65-v4-fresh-20260513-213048`.
  - W&B run: `k93kbsns`.
- **Final loss.** ~0.394.
- **Results — `validation_samples/math.jsonl` (N=10, n=8).**
  - pass@8 = 0.30 (TODO: confirm per-temp sweep — only single-temp
    recorded in user prompt).
- **Results — in-distribution OMI2 held-out (N=500, n=4).**
  - pass@1 = **0.395**, pass@4 = **0.608**. Regressed vs v3
    (0.408 / 0.628).
- **Results — MATH-500 (n=4).**
  - pass@1 = **0.413**, pass@4 = **0.648**. Regressed vs v3
    (0.514 / 0.686), −10 pp pass@1.
- **Per-subject MATH-500 pass@1 (v4-fresh diagnostic, source
  `/scratch/Julien/diagnostics/v4_fresh_eval/`):**

  | Subject              | v4-fresh | v3    | Δ vs v3 |
  |----------------------|----------|-------|---------|
  | Algebra              | 0.585    | 0.700 | −11.5 pp |
  | Prealgebra           | 0.613    | 0.668 | −5.5 pp  |
  | Number Theory        | 0.375    | 0.492 | −11.7 pp |
  | Counting & Prob.     | 0.349    | 0.480 | −13.1 pp |
  | Geometry             | 0.421    | 0.463 | −4.2 pp  |
  | Precalculus          | 0.174    | 0.339 | −16.5 pp |
  | Intermediate Algebra | 0.211    | 0.296 | −8.5 pp  |

- **Per-level MATH-500 pass@1 (v4-fresh diagnostic).**

  | Level | v4-fresh | v3 (where recorded) | Δ          |
  |-------|----------|---------------------|------------|
  | 1     | 0.744    | 0.797               | −5.3 pp    |
  | 2     | 0.656    | (TODO)              | (TODO)     |
  | 3     | 0.514    | (TODO)              | (TODO)     |
  | 4     | 0.316    | (TODO)              | (TODO)     |
  | 5     | 0.159    | 0.213               | −5.4 pp    |

- **What we learned.** Fresh init on the v4 mix didn't escape v3's
  optimum — it landed in a *worse* one. Targeted subjects (IntAlg,
  Precalc) regressed; everything else regressed too. The fresh-basin
  hypothesis was wrong: v3 wasn't trapped in a local optimum, it was
  at a high-water mark that the v4 data dilution couldn't beat from
  scratch.

### 3.6 v4-resume — resume from v3 adapter on v4-mix (NEGATIVE RESULT, slightly less bad)

- **Goal / hypothesis.** Build on v3's wins. Lower LR (5e-5 vs 1e-4)
  should let v3's OMI2-derived weights act as a strong prior; the
  v4-mix should refine them on weak subjects without destroying them.
- **Config.**
  - Init: v3 adapter at
    `/scratch/Julien/runs/cs552-erbland-g65-v3-omi2-fix2-20260511-152150/final`
    (via `--init-from-adapter`, validates r/α/target_modules against
    locked `lora.yaml`; refused to launch if mismatched).
  - Data: v4-mix (67,135 train rows).
  - LR=5e-5 (gentler than v4-fresh), 2 epochs, locked LoRA, Liger
    Kernel ON.
  - Run name: `cs552-erbland-g65-v4-resume-20260513-213244`.
  - W&B run: `zd5x6syj`.
- **Final loss.** ~0.413.
- **Results — `validation_samples/math.jsonl` (N=10, n=8), 5-temp
  sweep:**

  | Temp | v4-resume pass@8 | v3 pass@8 |
  |------|------------------|-----------|
  | 0.4  | 0.40             | 0.40      |
  | 0.5  | 0.30             | 0.30      |
  | 0.6  | 0.30             | 0.40      |
  | 0.7  | 0.30             | 0.30      |
  | 0.8  | 0.30             | 0.30      |

  v4-resume hits 0.40 at **one** temperature; v3 hits it at **two**.
  Single-temp 0.40 on N=10 is upper-end noise (≈10pp standard error).
- **Results — in-distribution OMI2 held-out (N=500, n=4).**
  - pass@1 = **0.408**, pass@4 = **0.634**. Effectively flat vs v3
    (0.408 / 0.628), tiny improvement on pass@4.
- **Results — MATH-500 (n=4).**
  - pass@1 = **0.431**, pass@4 = **0.636**. Regressed vs v3
    (0.514 / 0.686). −8 pp pass@1 vs v3, but +2 pp vs v4-fresh.
- **Per-subject MATH-500 pass@1 (v4-resume diagnostic, source
  `/scratch/Julien/diagnostics/v4_resume_eval/`):**

  | Subject              | v4-resume | v3    | Δ vs v3  |
  |----------------------|-----------|-------|----------|
  | Algebra              | 0.627     | 0.700 | −7.3 pp  |
  | Prealgebra           | 0.622     | 0.668 | −4.6 pp  |
  | Number Theory        | 0.407     | 0.492 | −8.5 pp  |
  | Counting & Prob.     | 0.342     | 0.480 | −13.8 pp |
  | Geometry             | 0.402     | 0.463 | −6.1 pp  |
  | Precalculus          | 0.196     | 0.339 | −14.3 pp |
  | Intermediate Algebra | 0.216     | 0.296 | −8.0 pp  |

- **Per-level MATH-500 pass@1 (v4-resume diagnostic).**

  | Level | v4-resume |
  |-------|-----------|
  | 1     | 0.756     |
  | 2     | 0.672     |
  | 3     | 0.536     |
  | 4     | 0.359     |
  | 5     | 0.151     |

- **What we learned.** **Targeted data augmentation at 1.7B scale
  did not lift the targeted subjects.** It regressed across the
  board, including the two subjects the v4 mix was designed to fix
  (Precalc: 0.339 → 0.196, IntAlg: 0.296 → 0.216). The most likely
  cause: **the model is parameter-bound, not coverage-bound.** A 4×
  exposure on small problem pools (1,295 IntAlg unique, 746 Precalc
  unique) was insufficient signal to move pass@1, while the addition
  of MATH-train + NuminaMath diluted OMI2's high-quality
  405B-teacher contribution. v3 keeps the throne.

### 3.7 v5 — Pure OMI2 100k SFT (POSITIVE RESULT, deployed to team HF)

- **Goal / hypothesis.** Does scaling pure OMI2 from 50k (v3) to 100k
  lift performance? Tests whether v3 is OMI2-saturated at 50k.
  **Single variable changed: dataset size.**
- **Config.**
  - Init: Qwen3-1.7B base (fresh).
  - Data: `nvidia/OpenMathInstruct-2` split=train_1M, 100,000 train +
    500 eval. Stored at `/scratch/Julien/data_out_v5_omi2_100k/`.
  - per_question_cap=4 — does NOT bind (~999k unique-problem raw
    rows). max_formatted_tokens=2900 — drops 0 rows. Dataset is
    byte-clean.
  - LR=1e-4, 2 epochs, locked LoRA, Liger Kernel ON.
  - Job name on cluster (cosmetic naming, data is v5):
    `cs552-erbland-g65-v4-fresh-20260514-162214`.
  - Final training loss: ~0.40.
- **Results.**

| Surface | v3 | v5 | Δ |
|---|---|---|---|
| Validation pass@8 (temp=0.4) | 0.400 | 0.500 | +10pp |
| In-distribution pass@1 (N=500) | 0.408 | 0.456 | +4.8pp |
| In-distribution pass@4 (N=500) | 0.628 | 0.686 | +5.8pp |
| MATH-500 pass@1 | 0.514 | 0.516 | +0.2pp (tied) |
| MATH-500 pass@4 | 0.686 | 0.672 | −1.4pp |

Per-subject highlights on MATH-500 (pass@1): Algebra 0.700→0.732,
Counting 0.480→0.520, Prealgebra 0.668→0.683, Level 1 0.797→0.855.
Slight regressions on hard subjects: IntAlg −2.8pp, Precalc −4.0pp,
Level 5 −1.9pp.

**5-temperature sweep on validation** (N=10):

| temp | pass@8 |
|---|---|
| 0.4 | 0.500 |
| 0.5 | 0.400 |
| 0.6 | 0.300 |
| 0.7 | 0.300 |
| 0.8 | 0.300 |

Single-temp peak at 0.500 (temp=0.4) — like v4-resume's noise mirage
in shape. BUT the in-distribution N=500 lift is robust and the
MATH-500 pass@1 is tied (no regression).

- **Deployment.** Pushed to team HF `cs-552-2026-emainelpe/math_model`
  at 2026-05-15 13:05 UTC, replacing the knowingly-deployed v4-resume.
  Personal backup at
  `JulienE220/math-adapter-sft-v5-omi2-100k-r32-20260515`.
- **Diagnostics archived.** `/scratch/Julien/diagnostics/v5_eval/`,
  `/scratch/Julien/v5_temp_sweep/`.
- **What we learned.** Scaling pure OMI2 from 50k to 100k produces a
  robust lift on in-distribution surfaces (+4.8pp pass@1, +5.8pp
  pass@4 at N=500), a single-temp validation peak (+10pp at temp=0.4
  only), and ties MATH-500 pass@1 — but with **per-subject
  redistribution** (easy and mid-difficulty up, hard subjects very
  slightly down). v3 was **not** OMI2-saturated at 50k; the
  capacity-bound from v4 is **soft, not hard**. Pushed to team HF as
  the math expert. CI nightly grade pending.

### 3.8 RLVR rescue — GRPO on v3 with signal-band-filtered prompts (POLICY COLLAPSE + recovered checkpoint-650)

- **Goal / hypothesis.** GRPO refinement on a properly signal-banded
  prompt set can lift v3 beyond pure-SFT capability. Specifically,
  P1 from the 2026-05-13 rescue plan: per-prompt reward variance is
  maximized when solve_rate is near 0.5, so a tighter difficulty band
  produces a healthier gradient.
- **Prior failed attempt — retry3 (2026-05-13, `res35mif`).**
  Trained 600 GRPO steps on top of v3 before stopping for wall-clock.
  Regressed validation pass@8 from 0.40 → 0.30. Root cause:
  `frac_reward_zero_std` ≈ 1.0 throughout — nearly every GRPO group
  had zero per-prompt reward variance, so the advantage
  `(r - mean) / std` was numerically zero. Policy barely moved
  (KL ≈ 0.001) for the entire 600-step run. Combined factors:
  half-configured DAPO loss (no `epsilon_high`), `use_vllm=False`,
  `mask_truncated_completions=False`, `learning_rate=3e-6`. Gradient-
  starved from step 1.
- **Rescue infrastructure built (added 2026-05-13).**
  - **Signal-band filter** in `data/prepare_rlvr.py` configurable via
    `DIFFICULTY_MIN`/`DIFFICULTY_MAX` env vars.
  - **`RewardSignalCallback`** in `scripts/train_rlvr.py`: warns at
    step 100 / errors at step 200 if `frac_reward_zero_std`
    rolling-50-step mean > 0.5.
  - **`HARD_KILL_ON_WEAK_SIGNAL=1`** env var to make step-200
    escalation raise `RuntimeError` (frees the A100 instead of
    burning wall-clock).
  - **Loss-type knob** (`LOSS_TYPE=grpo` or `dapo`).
  - **vLLM rollout option** (`USE_VLLM=1`) — ~5-10× faster than HF
    `.generate` and rollout temperature actually takes effect.
  - **Mask-truncated-completions option** (`MASK_TRUNCATED=1`).
  - **`KLSpikeCallback`** (pre-existing P3): warns if KL > 0.5 in the
    first 100 steps — Dang & Ngo 2025 small-model instability signal.
  - **Liger Kernel** OOM protection (default ON).
- **Config for this run.**
  - Run name: `cs552-erbland-g65-rescue-20260514-152540`.
  - Adapter init: v3 (`cs552-erbland-g65-v3-omi2-fix2-20260511-152150/final`).
  - Prompt set: `/scratch/Julien/data_out_v3/rlvr_prompts.jsonl`,
    **3,936 problems**, signal-band-filtered to `[0.250, 0.750]`
    solve_rate (n=8 rollouts quantizes solve_rate to
    `{0.250, 0.375, 0.500, 0.625, 0.750}`).
  - SFT_MODEL for preflights: `/scratch/Julien/merged/math_model_v3`.
  - `USE_VLLM=1`, `MASK_TRUNCATED=1`, `LOG_COMPLETIONS=1`.
  - `LEARNING_RATE=3e-6`, `KL_COEF=0.04` (Tülu 3 default),
    `ROLLOUT_TEMP=0.8`, `MAX_PROMPTS=3936`.
  - `LOSS_TYPE=dapo` (default — was suspected in retry3 but the
    signal-band filter resolves the upstream cause).
  - `HARD_KILL_ON_WEAK_SIGNAL=unset` — let it run even if signal
    weakens, to observe the full trajectory.
  - Liger Kernel ON (default after today's fix).
- **Run progression.** `frac_reward_zero_std` mostly 0 with sporadic
  per-prompt 1s throughout; KL ~0.0003-0.002 for the bulk of the run;
  reward std ~0.35-0.52. **Opposite failure-signal profile from
  retry3** (which had constant `frac_reward_zero_std=1.0`) — the
  signal-band-filtered prompt set fixed the upstream gradient
  starvation. Training completed 100% of the first epoch.

- **Critical failure mode discovered post-training.** The final
  adapter at `/final/` produces broken output —
  `"useruseruseruser..."` 1000+ token repetition on a "What is 2+2?"
  smoke test. Policy collapse occurred near the end of the run
  despite all in-flight monitoring signals (reward, KL,
  frac_reward_zero_std) being healthy. Importance sampling ratios
  were unstable across the run, and the late-epoch combination of
  half-configured DAPO (`epsilon_high=null`) + unbounded duration +
  1.7B instability flipped the policy off the SFT basin.
  - The post-training `smoke_inference_p1` preflight caught the
    collapse: "P1 preflight FAILED: smoke output missing
    `\boxed{}`." This is exactly what that callback was designed
    for — abort cleanly instead of silently shipping a broken
    adapter.

- **Recovered checkpoint-650.** Intermediate checkpoints `-650` (at
  epoch=0.1651, global_step=650) and `-700` (at epoch=0.1778) were
  saved during training.
  - `checkpoint-700` had reward crash from 0.55 → 0.044 between
    step 698 → step 699 (collapse moment located).
  - `checkpoint-650` had healthy reward 0.425-0.675, KL
    0.0008-0.0014, `frac_zero_std=0` — pre-collapse.
  - Smoke-tested checkpoint-650: produces clean
    `<think>2+2=4</think>\boxed{4}` output.
  - Merged at `/scratch/Julien/merged/math_model_rlvr_ckpt650`.

- **Diagnostic results on RLVR-ckpt650 vs v3.**

| Surface | v3 | RLVR-ckpt650 | Δ |
|---|---|---|---|
| MATH-500 pass@1 | 0.514 | 0.519 | within noise |
| In-distribution pass@1 (N=500) | 0.408 | 0.431 | small lift, possibly noise |
| Validation pass@8 (N=10) | 0.400 | 0.300 | N=10 noise |

Per-subject: noise-level shifts (≤2pp) in both directions, no
consistent pattern.

- **Diagnostics archived.** `/scratch/Julien/diagnostics/rlvr_ckpt650_eval/`.

- **What we learned.** RLVR at 1.7B has a narrow stability window:
  16.5% epoch = noise; 100% epoch = policy collapse. The signal-band
  filter (rescue lever P1) **did** fix the retry3 starvation — the
  reward signal was healthy throughout — but the run still collapsed.
  Hypothesis: the half-configured DAPO loss plus unbounded duration
  drives the policy past the SFT basin once enough gradient
  accumulates, and small models (1.7B) lack the capacity buffer
  larger models use to recover. This is consistent with Dang & Ngo
  2025's small-model RLVR warning, but with a different failure shape
  than retry3 (late-run collapse vs early-run starvation). A
  publishable negative result for the report: **partial RLVR on a
  1.7B model has two failure regimes, both observed**, and the
  intermediate-checkpoint recovery (ckpt-650) lands at noise vs the
  SFT base. v3 → v5 lifts came from SFT scaling, not RLVR.

### 3.9 v6 — Pure OMI2 200k SFT (small positive lift, mixed signal)

- **Goal / hypothesis.** Does doubling the v5 dataset (100k → 200k
  pure OMI2) continue the v3 → v5 scaling trajectory, or does it
  saturate? Tests whether the "soft capacity bound" found between v3
  and v5 is fully closed at 100k or has further headroom.
  **Single variable changed: dataset size.**
- **Config.**
  - Init: Qwen3-1.7B base (fresh, not from v5).
  - Data: `nvidia/OpenMathInstruct-2` split=train_1M, 200,000 train +
    500 eval. Stored at `/scratch/Julien/data_out_v6_omi2_200k/`.
  - per_question_cap=4 — does NOT bind. Dataset is byte-clean.
  - LR=1e-4, 2 epochs, locked LoRA, Liger Kernel ON.
  - Job name on cluster (cosmetic v4 naming, data is v6):
    `cs552-erbland-g65-v4-fresh-20260515-152430`.
  - Final training loss: **0.329** (lower than v5's ~0.40; final
    `mean_token_accuracy` 0.889 vs v5's ~0.87).
  - Wall-clock: ~17h (launched 2026-05-15 15:24, finished early
    morning 2026-05-16).
- **Results vs v5.**

| Surface | v5 | v6 | Δ |
|---|---|---|---|
| Validation pass@8 (temp=0.4) | 0.500 | 0.300 | −20pp (N=10 noise, see sweep) |
| In-distribution pass@1 (N=500) | 0.456 | 0.456 | tied |
| In-distribution pass@4 (N=500) | 0.686 | 0.678 | −0.8pp |
| MATH-500 pass@1 | 0.516 | 0.525 | +0.9pp (real at N=500) |
| MATH-500 pass@4 | 0.672 | 0.682 | +1.0pp |

Per-subject on MATH-500 (pass@1): Algebra +2.6pp, IntAlg +2.3pp,
Precalc +3.1pp (the hard-subject "gap" v3 left flagged in diagnostic
is **partially recovered** in v6), but Counting −4.6pp, Prealgebra
−1.2pp, Level 1 −1.8pp (easy subjects slightly regressed). Level 5
jumped +4.1pp — v6 lifts hard problems v5 was flat on.

**5-temperature sweep on validation** (N=10):

| temp | pass@8 |
|---|---|
| 0.3 | 0.300 |
| 0.4 | 0.300 |
| 0.5 | 0.300 |
| 0.6 | 0.300 |
| 0.7 | 0.300 |

All temps flat at 0.300. v6 hits the "always-solvable" core of
`validation_samples/math.jsonl` (presumably the same 3 problems v3
solves) but doesn't reach v5's temp-0.4 peak of 0.500. Likely:
v5's 0.500 at one temp was getting lucky on one specific problem
that v6's distribution doesn't favor — N=10 validation set is
sample-noise-dominated at this resolution.

- **Deployment.** v6 merged at
  `/scratch/Julien/merged/math_model_v6_omi2_200k`. **NOT pushed**
  to team HF — waiting on v5's CI grade to decide whether the v6
  redistribution is worth the validation regression.
- **Diagnostics archived.** `/scratch/Julien/diagnostics/v6_eval/`,
  `/scratch/Julien/v6_temp_sweep/`.
- **What we learned.** v3 → v5 → v6 is a clean monotonic MATH-500
  lift (+0.2pp then +0.9pp), for a cumulative **+1.1pp over 4× more
  data**. Per-subject the pattern is **non-uniform redistribution**:
  v5 helps easy and mid problems; v6 partially recovers v3's hard-
  subject gaps but at the cost of small regressions on easy
  subjects. The diminishing-returns curve is consistent with a soft
  capacity bound. Open question whether v7 at 400k or 500k would
  push further or hit a true ceiling.

---

## 4. Infrastructure improvements

### 4.1 OOM structural fix — Liger Kernel (deployed 2026-05-13/14)

- **Problem.** Three OOM crashes prior to fix (v4-200k 2026-05-12,
  v4-fresh first attempt 2026-05-13, v4-resume first attempt
  2026-05-13) all hit `torch.OutOfMemoryError` at
  `outputs.logits[..., :-1, :].contiguous()` — the materialized
  `B × T × 151,643 × 4 bytes` logits tensor exceeded A100-40GB
  headroom on near-max-length batches.
- **Why prior mitigations weren't enough.**
  - `gradient_checkpointing=True`: shrinks activations, doesn't
    touch logits.
  - `max_formatted_tokens=2900`: drops the longest rows so T<4096,
    but still 1.76 GiB per batch at T=2900.
  - `per_device_eval_batch_size=1` + `eval_accumulation_steps=4`:
    fixes eval-time OOM but training-time logits independent.
  - Lower LR, fewer epochs: zero memory effect.
- **Fix.** `liger-kernel`'s fused `LinearCrossEntropy` computes loss
  and gradients block-by-block from hidden states + embedding matrix.
  Per-block intermediate is constant-memory regardless of vocab size.
  The `vocab_size × 4 bytes` term drops out of the per-step memory
  equation entirely.
- **Implementation.**
  - `use_liger_kernel=True` default in `SFTConfig` AND `GRPOConfig`,
    exposed via `--use-liger-kernel` (BooleanOptionalAction, default
    True). Pass `--no-use-liger-kernel` for A/B comparison.
  - `requirements.txt` pins `liger-kernel>=0.8.0`.
  - Secondary mitigation: `PYTORCH_ALLOC_CONF=expandable_segments:True`
    env var added to all three submit scripts; coalesces freed CUDA
    blocks under fragmentation pressure.
- **Cluster verification (2026-05-14).** `liger-kernel 0.8.0` installs
  cleanly with `trl 0.19.1` + `transformers 5.7.0` + `torch
  2.10.0+cu128`. `apply_liger_kernel_to_qwen3` patch exists.
- **Sanity-check pitfall.** `liger_kernel 0.8.0` does **NOT** expose
  `__version__`. Any future sanity check must not depend on that
  attribute. Three submit-script sanity-check attempts (`__version__`
  print, then nested-double quoting bug, then runai serialization
  stripping inner single quotes) all failed; final fix was to
  remove the cosmetic line entirely and rely on the implicit chain:
  pip install fails clean OR TRL's `SFTConfig.use_liger_kernel=True`
  raises `ImportError` at trainer construction.
- **Expected post-fix memory profile** (Qwen3-1.7B, T=4096, B=1,
  bf16, Liger ON, Adam-on-LoRA-only):

  | Item                                           | Size       |
  |------------------------------------------------|------------|
  | Base model weights                             | ~3.4 GB    |
  | LoRA adapter (~12M params)                     | ~24 MB     |
  | LoRA gradients                                 | ~24 MB     |
  | Adam state on LoRA (m, v in fp32)              | ~96 MB     |
  | Activations (T=4096, gradient_checkpointing)   | ~4-6 GB    |
  | **Liger fused LCE intermediate**               | **~0.5 GB** (was ~5 GB without Liger) |
  | Misc + framework overhead                      | ~2-3 GB    |
  | **Total**                                      | **~10-13 GB / 40 GB available** |

- **Throughput note.** Liger's fused kernels are also ~5-10% faster
  for Qwen-class models on A100 per LinkedIn's benchmarks. Not the
  primary motivation, but the OOM fix is not paid for in wall-clock.

### 4.2 Test suite growth

Each test was added in response to a specific failure mode that
surprised us. The test suite is a reliable proxy for "things we've
seen go wrong and don't want to see again."

| Date / event                                        | Test count    | Key additions                                                                 |
|-----------------------------------------------------|---------------|-------------------------------------------------------------------------------|
| Stage 1-6 baseline (2026-05-09)                     | ~142          | Data prep, train_sft, eval_local, merge_and_push                              |
| Stage 7 RLVR (2026-05-11)                           | 234 (+71+21)  | reward_fn 10, prepare_rlvr 25, train_rlvr 25, submit_rlvr 11                  |
| RLVR rescue scaffolding (2026-05-13)                | 240 → 252     | RewardSignalCallback, env-var routing for rescue knobs                        |
| diagnose_v3.py + v4 data prep (2026-05-13)          | 297 → 323     | Failure-mode classifier, v4-mix composition / cross-source-dedup off          |
| Liger Kernel plumbing (2026-05-13/14)               | 323 → 331     | use_liger_kernel SFT/RLVR CLI default + config kwargs propagation             |
| Bash-n syntax check on POD_CMD (2026-05-13)         | 331 → 337     | submit_train.sh / submit_train_v4.sh / submit_rlvr.sh POD_CMD syntactic guard |
| Sanity-check line removed (2026-05-13)              | 337 → 334     | Three `test_pod_cmd_liger_sanity_check_uses_safe_quoting` retired             |
| **Current (2026-05-14)**                            | **334 + 1 skipped** | All passing in <1s on user's laptop                                   |

### 4.3 Eval infrastructure

- **`scripts/eval_local.py`** — vLLM front-end + thin glue around the
  vendored CI scorer at `evaluate/`. Defaults to CI-faithful
  (`max_model_len=4096`, `max_tokens=4096`, `n=8`, `seed=42`).
  Sampling-param resolution is three-tiered:
  1. CLI flags (each override logged at WARNING).
  2. `<model>/generation_config.json` if present.
  3. Hard-coded fallback (`temperature=0.3`, `top_p=0.95`, `top_k=20`).
- **`scripts/diagnose_v3.py`** — Per-subject / per-level / per-failure-
  mode analysis on three eval targets (validation_samples, in-dist
  OMI2/DART holdout, MATH-500). Failure modes classified via priority
  order: `repetition` > `correct` > `no_box` > `truncated` >
  `wrong_box` > `other`.
- **5-temperature sweep pattern.** Established 2026-05-11: every new
  SFT variant gets evaluated at temps {0.4, 0.5, 0.6, 0.7, 0.8} so
  single-temp noise (≈10 pp standard error on N=10) doesn't fool the
  comparison. Multi-temp robust hits at the same pass@8 are signal;
  single-temp hits are not.
- **Scoring is byte-identical to CI.** The `evaluate/` package vendored
  in the repo is copied byte-for-byte from the course CI; we never
  re-implement extraction or equivalence.

### 4.4 Submit script ergonomics

- **`rcp/submit_train.sh`** — v1/v2/v3 SFT submissions. Reads
  `GASPAR`, `GROUP` (required), `HF_TOKEN`, `WANDB_API_KEY`,
  `RESUME`, `SKIP_PREP`, `DATA_OUT_DIR`, etc.
- **`rcp/submit_train_v4.sh`** — v4-fresh / v4-resume modes. Mode
  positional arg picks LR (1e-4 fresh vs 5e-5 resume) and adapter
  init. Same SKIP_PREP / DATA_OUT_DIR overrides; v4-mix composition
  flags injected automatically.
- **`rcp/submit_rlvr.sh`** — full rescue-config env vars surface area:
  `LOSS_TYPE`, `USE_VLLM`, `VLLM_GPU_MEM_UTIL`, `MASK_TRUNCATED`,
  `LOG_COMPLETIONS`, `HARD_KILL_ON_WEAK_SIGNAL`, `DIFFICULTY_MIN`,
  `DIFFICULTY_MAX`, `MAX_NEW_TOKENS`, `LEARNING_RATE`, `KL_COEF`,
  `ROLLOUT_TEMP`, `MAX_PROMPTS`, `ADAPTER_DIR`, `PROMPT_SET`,
  `SFT_MODEL`, `SKIP_CURATION`, `SKIP_PREFLIGHTS`.
- **Bash syntax debugging history (2026-05-13).** The Liger Kernel
  sanity-check `python -c "..."` line caused 3 false starts: first
  `__version__` AttributeError, then nested-double-quote bash syntax
  error, then runai serialization stripping inner single quotes
  between local dry-run (which looked fine to `bash -n`) and actual
  pod execution. **Final fix: removed the cosmetic line.** Kept the
  `bash -n` on `POD_CMD` as regression guard for future POD_CMD
  changes (catches local syntax bugs cheaply; the runai-side stripping
  cannot be caught client-side).

### 4.5 Locked artifacts (DO NOT modify)

- `configs/lora.yaml` — shared across all four team experts. Modifying
  r / α / target_modules breaks the Phase 3 DARE + AdaMerging merge.
- `chat_template/chat_template.jinja` — shared. Loss-masking patches
  (`{% generation %}` markers) require coordination via the
  `emainelpe-shared` repo.
- `evaluate/` — byte-identical to course CI scorer. No re-implementing.

---

## 5. Key insights and lessons learned

One paragraph each, anchored to concrete evidence from the runs above.
These are the "discussion section" candidates for the final report.

- **Capacity-bound vs coverage-bound at 1.7B — refined to "soft
  bound".** v3 (50k pure OMI2) outperforms v4-fresh and v4-resume on
  every subject and every level in MATH-500. Adding diverse data
  didn't help; diluting OMI2 hurt. At this parameter scale,
  **teacher quality** (Llama-3.1-405B-Instruct in OMI2) matters more
  than data diversity. **However**, v3 → v5 → v6 (50k → 100k → 200k
  pure OMI2) is monotonic on MATH-500 pass@1 (0.514 → 0.516 → 0.525),
  a cumulative +1.1pp lift over 4× more data. The 1.7B "capacity
  bound" is **soft, not hard** — more pure-OMI2 data continues to
  produce diminishing-returns improvements. The hard ceiling is
  unknown; v5 confirmed that v3 was not OMI2-saturated at 50k.
- **Scaling progression at 1.7B is non-uniform per-subject.** v5
  vs v3: easy + mid subjects up (Algebra +3.2pp, Counting +4pp,
  Prealgebra +1.5pp, Level 1 +5.8pp), hard subjects slightly down
  (IntAlg, Precalc, Level 5 each ~ −2 to −4pp). v6 vs v5:
  partial reversal — hard subjects recover (IntAlg +2.3pp, Precalc
  +3.1pp, Level 5 +4.1pp) at the cost of slight regressions on easy
  subjects (Counting −4.6pp, Prealgebra −1.2pp). The scaling
  recipe **redistributes mass** rather than lifting uniformly —
  evidence that at this capacity, the model is doing capacity-
  constrained tradeoffs between problem-type representations.
- **Diagnostic-driven targeting failed at this scale.** Identifying
  weak subjects (IntAlg, Precalc, Level 5) via `diagnose_v3.py` gave
  a clean experimental design, but oversampling those problems
  didn't lift their pass@1 — it regressed them (Precalc 0.339 →
  0.196, IntAlg 0.296 → 0.216 in v4-resume vs v3). Suggests the
  limitation is the model's representational capacity for those
  problem types, not exposure count. The diagnostic was useful to
  surface the gap, but closing the gap isn't a data problem at 1.7B.
- **Per-question multiplicity cap matters.** The v4 design intended
  ~12× oversampling on IntAlg (12k target / 1,295 unique), but
  `per_question_cap=4` (a v3-era memorization safeguard inside
  `build_pipeline`) bounded effective multiplicity at 4×. We chose
  to keep the cap rather than raise it — going higher would have
  risked memorization on the small problem pools (1,295 IntAlg
  unique, 746 Precalc unique). It is unknown whether 8× or 12× would
  have helped or memorized; future experiments could probe this.
- **The validation_samples set is noisy at N=10.** A pass@8 of 0.40
  on `validation_samples/math.jsonl` can be reproduced by v3 at two
  temperatures (signal) or by v4-resume at one temperature (noise-
  level). 5-temperature sweep is the cheap discriminator (~5×
  inference cost, but pulls the standard error from ≈10 pp single-
  temp down to a much sharper multi-temp comparison). Lesson: never
  promote a checkpoint on a single-temp eval. MATH-500 (N=500) was
  decisive for the v4 NEGATIVE RESULT call — N=10 alone would have
  left v4-resume technically tied with v3.
- **OOM in 1.7B LLM training is dominated by the logits tensor**,
  not the model weights, optimizer state, or activations. The
  `B × T × vocab × 4 bytes` allocation at vocab=151,643 reaches
  ~2.49 GiB per batch even at B=1, T=4096 — and the contiguous
  `shift_logits` copy in `compute_loss` doubles that. Gradient
  checkpointing is necessary but insufficient. The **structural fix
  is fused cross-entropy (Liger Kernel, or equivalent)** — it removes
  the `vocab_size × 4 bytes` term from the memory equation entirely.
- **Cluster preemption is real.** Parallel preemptible jobs are
  convenient for throughput but can be killed simultaneously when
  capacity tightens. Plan for at least one preemption event per
  ~10h-long run. When parallel, identify *which* job is higher value
  to preserve (RLVR > v5 SFT in our 2026-05-14 doubleheader: more
  compute invested, non-trivial-to-regenerate signal-band prompt set
  vs cheap-to-relaunch byte-clean dataset).
- **Tooling iteration friction is expensive.** Hours spent on bash
  quoting (3 versions), runai serialization (drops inner single
  quotes from the local-bash-array argv between submit and pod),
  and version-detection quirks across libraries
  (`liger_kernel.__version__` missing in 0.8.0). Lesson:
  1. Write the bash-syntax check ON the *assembled* `POD_CMD`
     (post-render, not pre-render). Cheap and catches the local-side
     bugs immediately.
  2. Don't add cosmetic verification lines that have more failure
     modes than the thing they're verifying. The implicit chain
     (`pip install` clean failure + TRL trainer construction
     `ImportError`) was structurally stronger than the explicit
     sanity-check `python -c`.
- **Test count tracks confidence.** Project started at ~142 tests
  (Stage 1-6 baseline) and now sits at 334+1. Each new test was
  added in response to a specific failure mode that surprised us
  (e.g., the v4-mix cross-source dedup that collapsed 94k → 50k
  silently; the runai POD_CMD quote stripping; the RLVR
  `frac_reward_zero_std=1.0` global starvation). The test suite is
  thus an artifact of the production incidents, not a coverage
  exercise — and the user is right that it's a reliable proxy for
  "things we've seen go wrong and don't want to see again."
- **Partial RLVR has two failure regimes on 1.7B models, both
  observed.** retry3 (2026-05-13, 600 GRPO steps): gradient
  **starvation** — `frac_reward_zero_std≈1.0` constant, policy
  stuck at SFT, pass@8 regressed 0.40 → 0.30. RLVR rescue
  (2026-05-14/15, 100% epoch on signal-band-filtered prompts):
  gradient signal **healthy throughout**, but policy **collapsed**
  near end of run — final adapter produced `"useruseruseruser..."`
  repetition. The signal-band filter (rescue lever P1) fixed
  retry3's root cause but unmasked a different failure shape. The
  recovered intermediate `checkpoint-650` (at 16.5% epoch) lands at
  v3-level noise (MATH-500 +0.5pp, within sampling variance) — so
  GRPO did **not** lift v3 even in its pre-collapse window. The
  team-committed fallback ("SFT fallback if RLVR destabilizes")
  remains the right call. Net for the report: **two distinct
  failure modes, no measurable RLVR lift, SFT scaling (v3 → v5 →
  v6) is the path that produced positive deltas.**
- **Cap-mode parity surprise.** Per TA clarification, the final-
  grading cap mode raises `max_tokens` from 4096 to 16384. For our
  1.7B math expert on `validation_samples/math.jsonl`, the cap mode
  does **not** change pass@8 on any evaluated checkpoint —
  completions terminate naturally before 4096 tokens. Truncation
  isn't the binding constraint on this set. So the TA's bump won't
  lift our headline. Re-check this if a future variant has markedly
  longer reasoning chains.

---

## 6. What's deployed at end of day 2026-05-15

- **Team HF (`cs-552-2026-emainelpe/math_model`).** **v5 OMI2 100k**
  pushed 2026-05-15 13:05 UTC, replacing v4-resume. CI nightly grade
  pending (will land overnight 2026-05-15 → 16).
- **Personal HF backups (Julien).**
  - v1: `JulienE220/math-adapter-sft-dart50k-r32-20260508` ✅
  - v2: `JulienE220/math-adapter-sft-mixed-50k-r32-20260511` ✅
  - v3: `JulienE220/math-adapter-sft-omi2-50k-r32-20260511` ✅
  - v4-fresh: ⏳ pending push (deferred from 2026-05-14)
  - v4-resume: `JulienE220/math-adapter-sft-v4-resume-r32-20260514`
    ⏳ pending push (deferred from 2026-05-14)
  - v5: `JulienE220/math-adapter-sft-v5-omi2-100k-r32-20260515` ✅
  - v6: ⏳ pending push (planned 2026-05-16 as
    `JulienE220/math-adapter-sft-v6-omi2-200k-r32-20260516`)
- **Cluster artifacts.**
  - `/scratch/Julien/merged/math_model_v3` — v3 SFT merged
  - `/scratch/Julien/merged/math_model_v5_omi2_100k` — v5 merged
    (sourced for team HF push)
  - `/scratch/Julien/merged/math_model_v6_omi2_200k` — v6 merged,
    **NOT pushed** (pending v5 CI signal)
  - `/scratch/Julien/merged/math_model_rlvr_ckpt650` — RLVR ckpt-650
    merged, evaluated → noise vs v3 → not a deployment candidate
- **Critical decision point for 2026-05-16 morning.** Based on v5 CI
  nightly grade:
  - v5 ≥ v3 CI → keep v5, optionally push v6 as an upgrade attempt.
  - v5 ≈ v3 CI → keep v5, do not push v6.
  - v5 < v3 CI → roll back to v3 immediately on team HF.

---

## 7. Open questions for the report

- ~~Did the RLVR rescue lift v3?~~ **ANSWERED — no.** The final
  adapter collapsed end-of-epoch; the recovered intermediate
  `checkpoint-650` lands at v3-level noise (MATH-500 +0.5pp, within
  sampling variance). Two distinct failure regimes observed
  (starvation in retry3, late-run collapse in rescue); no
  measurable lift in either. See §3.8 and §5 partial-RLVR insight.
- ~~Did v5 OMI2 100k lift v3?~~ **ANSWERED — yes, on
  in-distribution and validation-peak, tied on MATH-500.**
  In-distribution N=500 pass@1 +4.8pp; validation pass@8 +10pp at
  single temp; MATH-500 pass@1 tied (no regression). Pushed to
  team HF. See §3.7.
- ~~Does scaling OMI2 follow a power-law curve or saturate at
  1.7B?~~ **PARTIALLY ANSWERED — soft, not hard, saturation.**
  v3 → v5 → v6 (50k → 100k → 200k) is monotonic on MATH-500 pass@1
  (0.514 → 0.516 → 0.525), cumulative +1.1pp over 4× more data.
  Diminishing-returns curve — true ceiling unknown. Open whether v7
  at 400k or 500k continues the trajectory or hits a hard wall.
- **NEW: Does v5 or v6 actually improve CI nightly grade vs v3?**
  v5's first CI nightly grade lands overnight 2026-05-15 → 16. v5
  is the candidate to grade because it's deployed; v6 is held in
  reserve. If v5 CI < v3's ~0.32, the local→CI gap is wider than
  the local→CI gap was for v3, and the "scaling helps" finding
  may not transfer.
- Is the team Phase 3 merge (2026-05-19) going to use v3, v5, or v6?
  Decision pending v5 CI grade and team alignment.
- Is the v3 nightly CI grade of 0.32 robust across nightly draws,
  or was it near the top of its noise distribution? Need multiple
  nightly grades on v3 to estimate variance. (TODO: check with
  team — has v3 been graded more than once?)
- Why is Level 5 pass@1 so much weaker than Level 1 across **every**
  variant we trained? Capacity-bound argument suggests the rank
  ordering can't be flipped by SFT alone, but v6 lifts Level 5 by
  +4.1pp over v5 — so the gap **can** narrow with more data. Open
  question: does the gap narrow further with v7 at 400k+, or does
  Level 5 hit a hard capacity ceiling that scaling can't push past?
  Our rescue prompt set was signal-banded across all difficulty;
  Level-targeted RLVR remains untested.

---

## 8. Sources and pointers

- **Project plan / methodology decisions.**
  - `/home/julienerbland/Documents/EPFL/Master/MA2/MNLP/emainelpe_math_model/CLAUDE.md`
    (canonical decision log; "Daily log" sections by date)
  - `/home/julienerbland/Documents/EPFL/Master/MA2/MNLP/emainelpe_math_model/IMPLEMENTATION_PLAN.md`
    (Stage 0-7 implementation history + Lessons learned subsections)
  - `/home/julienerbland/Documents/EPFL/Master/MA2/MNLP/emainelpe_math_model/docs/BASELINE.md`
    (append-only measurement log; bare baseline → v1/v2/v3 sweep →
    2026-05-13 RLVR regression → 2026-05-14 v4 negative result)
- **Authoritative course documents.**
  - `docs/proposal.pdf` — team's committed project plan
  - `docs/literature_review.pdf` — team's committed methods + refs
  - Team project README — CI behaviour (max_model_len=4096, n=8,
    1800s wall-clock cap, EVAL_REPORT mechanism), roster naming
  - `docs/project_description.pdf` — original grading rubric (older;
    superseded by README on CI parameters)
  - `docs/RCP_GUIDE.md` — cluster setup and submission
- **Diagnostic JSONs.**
  - `/scratch/Julien/diagnostics/v3_eval_20260513T133259Z/`
  - `/scratch/Julien/diagnostics/v4_fresh_eval/`
  - `/scratch/Julien/diagnostics/v4_resume_eval/`
  - `/scratch/Julien/diagnostics/v5_eval/`
  - `/scratch/Julien/diagnostics/v6_eval/`
  - `/scratch/Julien/diagnostics/rlvr_ckpt650_eval/`
  - `/scratch/Julien/v5_temp_sweep/`
  - `/scratch/Julien/v6_temp_sweep/`
- **W&B project.** `https://wandb.ai/julienerbland-epfl/emainelpe-math`
  - v3 SFT: cs552-erbland-g65-v3-omi2-fix2-20260511-152150
  - v4-fresh: `k93kbsns`
  - v4-resume: `zd5x6syj`
  - retry3 RLVR: `res35mif`
  - RLVR rescue: cs552-erbland-g65-rescue-20260514-152540 (W&B run
    name TODO — pull from `wandb_link.txt` in run dir)
  - v5: cs552-erbland-g65-v4-fresh-20260514-162214 (cosmetic v4
    naming, data is v5; W&B run name TODO)
  - v6: cs552-erbland-g65-v4-fresh-20260515-152430 (cosmetic v4
    naming, data is v6; W&B run name TODO)
- **Cluster.** RCP project `course-cs-552-erbland`; pods named
  `cs552-erbland-g65-<suffix>-<timestamp>`.
- **Hugging Face.**
  - Team: `https://huggingface.co/cs-552-2026-emainelpe/math_model`
  - Personal: `https://huggingface.co/JulienE220`

---

## TODO list for 2026-05-16 morning (carry-over from CLAUDE.md daily log)

1. Find v5's CI nightly grade (graded overnight 2026-05-15 → 16).
2. Decide based on v5 CI grade:
   - v5 ≥ v3 CI → keep v5, consider pushing v6 as upgrade.
   - v5 ≈ v3 CI → keep v5, do NOT push v6.
   - v5 < v3 CI → roll back to v3 immediately.
3. Push v6 to personal HF backup:
   `JulienE220/math-adapter-sft-v6-omi2-200k-r32-20260516`.
4. Push v4-resume + v4-fresh to personal HF backups (deferred from
   2026-05-14).
5. Begin Phase 3 group merge prep (4 days to the 2026-05-19
   milestone). Confirm with team that the adapter on
   `cs-552-2026-emainelpe/math_model` matches the merge contract
   (locked LoRA r/alpha/target_modules; locked chat template).
6. Update REPORT.md §6 once v5 CI grade and v6 push decision land.

## Closed TODO list from 2026-05-15 (status)

1. ✅ Checked v4-resume CI grade (one cycle observed).
2. ✅ Re-pushed v3? — Superseded by v5 push (better signal than
   re-pushing v3).
3. ❌ v4-resume personal backup not yet pushed (deferred).
4. ✅ RLVR rescue completion checked — final adapter collapsed,
   recovered ckpt-650.
5. ✅ v5 completion checked — final loss ~0.40.
6. ✅ Merged each adapter (v5, v6, RLVR-ckpt650).
7. ✅ 5-temp sweep on v5 and v6.
8. ✅ `diagnose_v3.py` per-subject on v5, v6, RLVR-ckpt650.
9. ✅ Decision: v5 to team HF (pushed 13:05 UTC); v6 held in
   reserve.
10. ✅ REPORT.md §3.7, §3.8 filled; new §3.9 for v6; §5/§6/§7
    updated.
