# CLAUDE.md — Math Expert (CS-552, Team Émainèlpé)

> **Read this entire file at the start of every session.** It encodes the
> team's committed plan and the design decisions that have already been
> made. When in doubt, anchor to this document and the proposal — not to
> general ML knowledge, and not to "what most projects do."

---

## What this repo is

The math expert for the CS-552 Modern NLP final project (EPFL, Spring 2026).
One of four domain specialists trained by Team Émainèlpé. The four
specialists (math, general knowledge, multilingual, safety) will be merged
into a single group model in Phase 3 via DARE + AdaMerging.

- Owner: Julien Erbland
- Other team members: Max Henrotin (knowledge), Mathis Richard (multilingual),
  Morgane Magnin (safety)
- HF target: `cs-552-2026-emainelpe/math_model`

The deliverable: a Qwen3-1.7B + LoRA model that produces high-quality
solutions to competition-style math problems, with the final answer wrapped
in `\boxed{...}` and reasoning enclosed in `<think>...</think>`.

## Authoritative documents

In order of precedence when sources conflict:

1. `docs/proposal.pdf` — the team's committed project plan
2. `docs/literature_review.pdf` — the team's committed methods and references
3. **Team project README** (the CS-552 project starter README, mirrored
   in the team's `emainelpe_group_model` repo). Authority for CI behavior
   (max_model_len, n, wall-clock cap, EVAL_REPORT mechanism) and roster
   naming. More recent than `project_description.pdf`; where the two
   conflict (e.g., `max_new_tokens=16384` vs `max_model_len=4096`),
   the README is binding and the conflict is flagged inline below.
4. `docs/project_description.pdf` — the course's original grading rubric.
   Older. Use for milestone weights and submission policy; do not
   re-cite for CI parameters that the README has updated.
5. `docs/RCP_GUIDE.md` — RCP cluster setup and submission.

Two research drafts (`docs/strategy_v2.pdf`, `docs/lit_review_deep.pdf`) may
appear in this folder. They are useful for ideas but are NOT binding. If
they contradict the proposal, follow the proposal.

## Pipeline (proposal-anchored)

Two phases per expert:

- **Phase 1**: Domain SFT with LoRA on `Qwen/Qwen3-1.7B`
- **Phase 2**: RLVR with exact-match reward (objective domain → verifiable)

Then in week 4 (team work, not part of this repo): DARE → AdaMerging merge.

## Settled design decisions — DO NOT RELITIGATE

| Decision | Choice | Source |
|---|---|---|
| Base model | `Qwen/Qwen3-1.7B` | Course requirement |
| Adapter type | LoRA (no full FT anywhere) | Required for the merge |
| LoRA rank `r` | 32 | Team decision; closes LoRA-vs-FFT gap |
| LoRA alpha | 64 | Team decision; standard 2×r heuristic |
| LoRA target modules | q/k/v/o, gate/up/down (all 7) | Team decision |
| SFT dataset | `hkust-nlp/dart-math-uniform` | Lit review §1.1; Dang & Ngo stability |
| Subsample size | 40k–50k examples | Lit review (Yuan, Toshniwal): diversity > volume |
| Per-question cap | max 4–6 solutions | Forces solution diversity |
| SFT learning rate | 1e-4 | Standard for LoRA per TRL docs |
| SFT epochs | 2 | Avoid overfit at r=32 |
| Effective batch size | 32 (4 per-device × 8 grad-accum) | Fits A100 40GB |
| Eval-time batch (Trainer) | `per_device_eval_batch_size=1`, `eval_accumulation_steps=4` | 2026-05-11 v3 OOM: pure-OMI2 eval rows are token-dense; the per-batch `(B × T × V × 2B)` logits tensor + its contiguous `shift_logits` copy in `compute_loss` requested 13.77 GiB on a 40 GB A100. `eval_accumulation_steps=4` ALONE is insufficient — it only chunks cross-batch accumulation, not the per-batch allocation. Both knobs needed. |
| Sequence length | 4096 | Matches CI eval cap |
| LR schedule | Cosine, 3% warmup | Standard |
| Gradient checkpointing | ON | Memory headroom |
| Thinking mode | ON, baked into chat template | CI does NOT pass enable_thinking |
| Loss masking | Full sequence (no assistant-only mask) | Stage 3 smoke (2026-05-07): TRL 0.21+ refused to auto-patch the locked Jinja because it lacks `{% generation %}` markers. Adding markers is a v2 stretch (requires emainelpe-shared coordination) |
| RLVR verifier | Exact-match | Proposal commitment; SymPy is v2 stretch |
| Inference temperature | 0.4 | Locked after the 2026-05-11 five-temperature sweep (0.4 / 0.5 / 0.6 / 0.7 / 0.8) on v1/v2/v3 SFT checkpoints under ci-faithful caps: v3 maximizes pass@8=0.4000 at temp=0.4 with the highest pass@1 (0.2875) of any (variant, temp) combination. Trained checkpoints carry `temperature=0.4` in `generation_config.json` so the CI samples at the calibrated peak. See `docs/BASELINE.md` → "2026-05-11 SFT comparison and temperature sweep". |
| RLVR experiment outcome | **v3 SFT remains the production checkpoint** | Trained 600 GRPO steps (~16% of one epoch on 3919 difficulty-curated prompts) on 2026-05-13 before stopping for wall-clock. Resulting RLVR-v3 regressed pass@8 from 0.40 → 0.30 on `validation_samples/math.jsonl`. Not pushed to HF. Five integration bugs were fixed during the run (sys.path, TRL 0.19.1 GRPOConfig API drift, raw-prompt JSONL, missing `<think>\n` prefix, false-alarm preflight reading) — all surfaced as fail-fast preflight assertions and regression tests (240 → 252 tests). See `IMPLEMENTATION_PLAN.md` → Stage 7 + "Lessons learned" → "2026-05-12/13 — RLVR 5-bug arc". |
| Cap-mode parity | ci-faithful (max_tokens=4096) ≡ final-grading (max_tokens=16384) on our checkpoints | 2026-05-13: all three evaluated models (bare, v3 SFT, RLVR-v3) produce identical pass@8 under both cap modes on `validation_samples/math.jsonl`. Completions terminate naturally before 4096 tokens; truncation is not the binding constraint on this set. The TA's final-grading bump does not lift our headline. Re-check if a future variant has markedly longer reasoning chains. |

## RLVR rescue plan (post-retry3 starvation, 2026-05-13)

The 2026-05-13 RLVR run `res35mif` regressed the SFT base from pass@8=0.40
to 0.30 over 600 GRPO steps. **Root cause**: `frac_reward_zero_std`
stayed at ≈1.0 throughout — nearly every GRPO group had zero per-prompt
reward variance, so the advantage `(r - mean) / std` was numerically
zero on most steps. The policy barely moved (KL ≈ 0.001) for the entire
600-step run. Combined with `loss_type=dapo` (with `epsilon_high=null` —
half-configured DAPO masks gradients further), `use_vllm=False`,
`mask_truncated_completions=False`, and `learning_rate=3e-6`, the run was
gradient-starved from step 1.

This patch (2026-05-13) makes the rescue config invocable via env vars
without changing any defaults. The five rescue levers, in priority order:

1. **Tighten the difficulty band** (`DIFFICULTY_MIN=0.35 DIFFICULTY_MAX=0.65`).
   Root-cause fix for `frac_reward_zero_std≈1.0`: curated prompts cluster
   around ~50% solve rate where per-prompt reward variance is highest.
   Prompts at solve_rate 0.2 or 0.8 contribute almost nothing to the
   gradient under n=8 rollouts.
2. **Switch to vLLM rollouts** (`USE_VLLM=1`). ~5-10× faster than the HF
   `.generate` path and the rollout `temperature` actually takes effect
   (the HF path silently ignores it under some TRL configurations).
   Wall-clock unlock; also fixes the rollout-temp drift hypothesis.
3. **Mask truncated completions** (`MASK_TRUNCATED=1`). When most
   rollouts hit the token cap (as in retry3), the gradient is being
   computed against arbitrary mid-reasoning suffixes. Masking them
   restricts gradient to finished rollouts only.
4. **Use plain GRPO loss** (`LOSS_TYPE=grpo`). DAPO needs `epsilon_high`
   configured to work; the half-configured DAPO in retry3 masked
   additional gradients. Switching to plain GRPO until `epsilon_high`
   is intentionally tuned.
5. **Raise learning rate** (`LEARNING_RATE=1e-5`). At 3e-6 with a starved
   gradient, 600 steps of training moved policy weights almost nowhere
   (KL≈0.001). A 3× LR bump gives the policy a chance to move past the
   SFT optimum under a healthier reward signal.

### Two preflight callbacks watching opposite failure modes

- `KLSpikeCallback` (P3, pre-existing): WARNs if KL > 0.5 within the
  first 100 steps — Dang & Ngo 2025's policy-explosion signal.
- `RewardSignalCallback` (added 2026-05-13): WARNs at step 100 / ERRORs
  at step 200 if `frac_reward_zero_std` rolling-50-step mean > 0.5 —
  the retry3 starvation signal. With `HARD_KILL_ON_WEAK_SIGNAL=1` the
  step-200 escalation raises RuntimeError to abort the run cleanly
  (free the A100 instead of burning wall-clock on a dead run).

The two callbacks are independent. A healthy run should see neither.
A polluted prompt set (or a half-configured DAPO loss) trips the
reward-signal callback. An over-aggressive learning rate or KL coefficient
trips the KL callback.

### Exact rescue invocation (paste-ready)

```bash
USE_VLLM=1 \
VLLM_GPU_MEM_UTIL=0.4 \
MASK_TRUNCATED=1 \
LOG_COMPLETIONS=1 \
LOSS_TYPE=grpo \
LEARNING_RATE=1e-5 \
DIFFICULTY_MIN=0.35 \
DIFFICULTY_MAX=0.65 \
MAX_NEW_TOKENS=2048 \
GASPAR=erbland GROUP=g65 ./rcp/submit_rlvr.sh rescue
```

The `rescue` positional arg becomes the run-name suffix
(`cs552-erbland-g65-rescue-<timestamp>`), so the rescue run is
distinguishable from prior `rlvr-*` attempts in W&B and HF.

**Hard-kill is OFF for the first rescue run.** The
`RewardSignalCallback` still logs WARN @ step 100 and ERROR @ step 200
if `frac_reward_zero_std` rolling mean stays > 0.5, but it does NOT
abort the job. This lets us observe the full `frac_reward_zero_std`
trajectory across a complete run — we need that healthy baseline before
we know what threshold value is operationally tight vs spurious. Future
rescue runs can append `HARD_KILL_ON_WEAK_SIGNAL=1` once we've seen the
trajectory on a known-good run and the 0.5 threshold is calibrated.

**Expected curation yield.** Under the tighter `[0.35, 0.65]` band,
expect roughly **1500–2500 prompts** kept from a 10k pool (down from
~3900 at the proposal's `[0.2, 0.8]` band). If the curation pass yields
fewer than 1000 in-band prompts, either raise `POOL_SIZE=20000` to
double the candidate set or relax the band to `[0.30, 0.70]`. Below
~1000 prompts the run is short on diversity and the gradient signal
becomes a function of which specific problems landed in-band, not the
ability v3 SFT has to learn from the regime.

### Curation/rollout alignment

`data/prepare_rlvr.py` scores difficulty at `SCORING_NUM_GENERATIONS=8`
— byte-identical to `train_rlvr.py`'s `--num-generations 8` default.
Curation-time solve_rate is therefore a direct predictor of
in-training per-prompt reward variance at the same n. **Do not break
this alignment** by setting different `n` values on the two scripts;
the difficulty band's empirical guarantee depends on it.

---

## v4 training plan (2026-05-13)

The v3 diagnostic on MATH-500 (scripts/diagnose_v3.py) surfaced three
specific coverage gaps:

| Slice | v3 pass@1 | gap vs strongest slice |
|---|---|---|
| Intermediate Algebra | 0.296 | -0.14 vs Algebra |
| Precalculus | 0.339 | -0.10 vs Algebra |
| Level 5 (any subject) | 0.213 | -0.25 vs Level 1 |
| Level 4 | flat-ish | (informational) |

v4 is a **targeted SFT run** designed to fix these gaps without losing
v3's OMI2-driven base. The mix is composed from three sources via
`data/prepare_sft.py --source v4-mix`:

### v4-mix composition

| Source | Target | Rationale |
|---|---|---|
| OMI2 (train_1M subset) | 40k | Carries v3's strongest signal forward — Llama3.1-405B teacher; the source v3 won on. Continuation, not replacement. |
| MATH-train IntAlg bucket | 12k | Direct fix for v3's IntAlg gap. Source has ~1.3k unique IntAlg problems; the 12k target oversamples ~10x BEFORE dedup. After in-source dedup, contributes ~1.3k unique problems. |
| MATH-train Precalc bucket | 7k | Direct fix for v3's Precalc gap. Source has ~750 unique Precalc problems; ~10x oversample pre-dedup; ~750 unique post-dedup. |
| MATH-train Level 4-5 bucket | 18k | Direct fix for v3's Level 5 gap. Source has ~3k unique Lvl4-5 problems across all subjects; ~6x oversample pre-dedup. |
| MATH-train Level 1-3 bucket | 13k | Anchor against catastrophic forgetting on easy problems. Source has ~4.5k unique Lvl1-3 problems; ~3x oversample pre-dedup. |
| NuminaMath-CoT (olympiad subset) | 5k | Distribution-diverse hard problems from the four-source allowlist `(olympiads, amc_aime, aops_forum, synthetic_amc)` — ~247k available problems total in the underlying dataset; we sample 5k. Complements MATH-train with competition-style breadth. The `math` source (~7.5k) is intentionally excluded to avoid cross-bucket duplicates with the EleutherAI/hendrycks_math bucket. |

**Total before downstream caps**: ~95k. **No cross-source dedup**
(2026-05-13 final policy). The first cut of v4-mix ran cross-source
dedup at the final concat and measured 94k → 50k collapse — that
collapse eliminated the within-bucket oversampling the diagnostic
multipliers depend on (IntAlg 12k target from 1.3k unique = collapses
to 1.3k, defeating the lever). Disabling cross-source dedup accepts a
small rate of true cross-source overlap as a tolerable cost; the
downstream `per_question_cap=4` inside `build_pipeline` caps the
multiplicity for any single query, so the effective training count
for a small-pool bucket like Precalc lands at ~3k rows (750 unique
problems × 4 copies each), not 7k or 750.

Within-bucket oversampling now flows end-to-end:

  - IntAlg bucket: 12k oversampled → after per_question_cap=4 →
    ~5.2k rows (1.3k × 4).
  - Precalc bucket: 7k oversampled → ~3k rows (~750 × 4).
  - Level 4-5 bucket: 18k oversampled → bounded by 4 × the unique
    L4-5 problem count (~3k unique → ~12k rows).
  - Level 1-3 bucket: 13k oversampled → bounded by 4 × ~4.5k unique
    → already-larger pool, no cap effect.

The effective total trained-on per epoch is on the order of 60-70k
rows, with the diagnostic-targeted subjects (IntAlg, Precalc, L4-5)
contributing their full 4× weight where the source pool permits.

### Two variants from the same data

Both variants train on the same v4-mix dataset. They differ in
**initialization** and **learning rate**:

| Variant | Init from | LR | When it's the right call |
|---|---|---|---|
| **v4-fresh** | base Qwen3-1.7B | 1e-4 | Clean slate. The v3 SFT optimization may have driven the policy into a local optimum that v4-mix can't escape; fresh init explores a fresh basin. |
| **v4-resume** | v3's adapter (via `--init-from-adapter`) | 5e-5 | Build on v3's wins. Use when you expect v3's OMI2-derived capabilities to be a net positive that the new MATH/NuminaMath data refines, not contradicts. Gentler LR avoids erasing v3's learned weights. |

**Pick the better of the two for the math expert.** Train both,
evaluate via `scripts/eval_local.py` + `scripts/diagnose_v3.py`,
publish whichever wins on `validation_samples/math.jsonl` pass@8 with
non-regressing per-subject diagnostic numbers.

### OOM fix: data-prep cap, not yaml change

The v4-200k OMI2 attempt (2026-05-12) crashed at epoch 0.08 with a
9.27 GiB single-tensor allocation on a long training sequence. The
natural fix would be to lower `lora.yaml.max_seq_length` from 4096
to 2900 — but **`configs/lora.yaml` is locked across all four team
experts** for the Phase 3 DARE + AdaMerging merge. Any divergence
there would silently break the merge.

The chosen fix lives entirely at the data-prep layer:
`--source v4-mix` auto-defaults `--max-formatted-tokens` to 2900,
which drops rows from `train.jsonl` whose Qwen3-tokenized formatted
chat exceeds the cap. The locked yaml is untouched; the training step
never sees a long sequence; the merge contract is preserved.

Override knob (use sparingly): pass `V4_MAX_FORMATTED_TOKENS=3500` to
`submit_train_v4.sh` to restore the v2/v3 default and accept the OOM
risk on long-sequence outliers.

### Pre-launch verification (both datasets verified accessible 2026-05-13)

Before launching, confirm the two non-OMI2 HF datasets are still
reachable and emit the expected schema. Both were verified on
2026-05-13 and the commands below should reproduce the same output
on the cluster pod (where the script will actually run).

**a. Verify MATH train (`EleutherAI/hendrycks_math`).** The dataset
ships per-subject configs; we load the `algebra` config to confirm
schema and count without pulling all 7 subjects.

```bash
python3 -c "from datasets import load_dataset; d = load_dataset('EleutherAI/hendrycks_math', 'algebra', split='train'); print(len(d), list(d[0].keys()))"
```

Expected output:

```
1744 ['problem', 'level', 'type', 'solution']
```

If `len(d)` differs or the schema is missing any of those four keys,
do NOT launch — the v4-mix MATH bucket composer expects exactly that
schema. The total train rows across all 7 subjects is ~7.5k; the
algebra config alone is the largest single subject.

**b. Verify NuminaMath (`AI-MO/NuminaMath-CoT`).** Confirms the
`source` field values match the four-entry olympiad allowlist
(`olympiads`, `amc_aime`, `aops_forum`, `synthetic_amc`).

```bash
python3 -c "from datasets import load_dataset; import collections; d = load_dataset('AI-MO/NuminaMath-CoT', split='train'); print('Schema:', list(d[0].keys())); print('Sources:', collections.Counter(r['source'] for r in d).most_common(10))"
```

Expected schema: `['source', 'problem', 'solution', 'messages']`

Expected top sources (counts approximate):

```
cn_k12       (~277k)
synthetic_math (~168k)
orca_math    (~153k)
olympiads    (~151k)   ← in allowlist
synthetic_amc (~62k)   ← in allowlist
aops_forum   (~30k)    ← in allowlist
math         (~7.5k)
gsm8k        (~7.3k)
amc_aime     (~4k)     ← in allowlist
...
```

The four-entry allowlist totals ~247k available problems
(olympiads 151k + synthetic_amc 62k + aops_forum 30k + amc_aime 4k).
We sample 5k of them. If the `source` field stops matching this
spelling (HF datasets occasionally rename), update
`NUMINAMATH_OLYMPIAD_SOURCES` in `data/prepare_sft.py` and rerun the
test suite before launching.

### Launch commands (paste-ready)

```bash
# v4-fresh — start from base Qwen3-1.7B, LR=1e-4
GASPAR=erbland GROUP=g65 \
  HF_TOKEN=$HF_TOKEN WANDB_API_KEY=$WANDB_API_KEY \
  ./rcp/submit_train_v4.sh fresh

# v4-resume — start from v3's adapter, LR=5e-5
GASPAR=erbland GROUP=g65 \
  HF_TOKEN=$HF_TOKEN WANDB_API_KEY=$WANDB_API_KEY \
  ./rcp/submit_train_v4.sh resume
```

Each launches a `runai submit` job with the v4-mix data prep + LoRA
training pipeline. Estimated wall-clock: ~10-14h on 1×A100 40g per
variant. Run them in parallel if cluster quota permits.

### `--init-from-adapter` mechanism (v4-resume specifics)

`scripts/train_sft.py --init-from-adapter PATH` does NOT reload
optimizer + LR scheduler state (unlike `--resume`). It only loads the
adapter's LoRA weights via `PeftModel.from_pretrained(base, PATH)`,
then trains fresh on the new data with a fresh optimizer / scheduler.
The adapter's `r` / `lora_alpha` / `target_modules` are validated
against `configs/lora.yaml` before training launches — a mismatched
adapter is refused with a clear error to keep the Phase 3 merge safe.

The two flags are mutually exclusive (argparse-enforced):
- `--resume`: continue interrupted training on the SAME data
- `--init-from-adapter`: continue training on NEW data, fresh optimizer

---

## Locked shared files

`configs/lora.yaml` and `chat_template/chat_template.jinja` are copied from
the team's `emainelpe-shared` repo. They MUST stay byte-identical to the
shared source for the Phase 3 merge to work. Treat both as read-only.
If a change is genuinely needed, propose it in the shared repo first, get
team sign-off, then update.

## Eval contract — what the CI runs against the pushed checkpoint

Encoded from the team project README and pinned to the vendored copy of
the CI scoring code at `evaluate/`. These are NOT design choices we
made; they are the parameters the course CI exercises against the model
on HF, and our local eval (`scripts/eval_local.py`, Stage 4) MUST mirror
them to be predictive of CI scores.

| Parameter | Value | Notes |
|---|---|---|
| Framework | OpenCompass (vendored at `evaluate/`) | Byte-identical copy of the CI's scoring code. See `evaluate/README.md`. |
| Extraction | `\boxed{...}` or `\fbox{...}`, last occurrence, brace-balanced | `evaluate.extract_answer.extract_boxed_answer` with `strip_double_curly_brace=True` (peels one extra `{...}` layer). No box → counted wrong. |
| Equivalence | OpenCompass `is_equiv` (multi-stage) | NOT pure exact-match. `evaluate.extract_answer.is_equiv` runs `strip_string` → `normalize_final_answer` → fallback `==`. Aggressive math-specific: unit removal, `\text{}` peeling, `100,000 ↔ 100000`, `0.5 ↔ \frac{1}{2}`, TeX shorthand like `\fracab → \frac{a}{b}`. |
| Seed | 42 | Fixed by the CI; same seed used in `data/prepare_sft.py` and `scripts/train_sft.py` for end-to-end reproducibility |
| Completions per problem | n = 8 | Sampled with the model's `generation_config.json` (temp/top_p/top_k come from the pushed config — Stage 5). |
| `max_model_len` | **4096** (combined prompt + generated tokens) | Per the team README: "Max model length 4096. The generation stops once the `\boxed{...}` answer is generated, or the model reaches an EoS token, or the maximum length is reached." This is vLLM's combined-context cap; an effective per-completion ceiling once you subtract the prompt. |
| `max_new_tokens` | **Effectively ≤ 4096 − \|prompt\|** under the README's cap | **Conflict surfaced.** `docs/project_description.pdf` (older, page 3) explicitly says `Max new tokens: 16384`. The README is more recent and binding for CI behavior; under `max_model_len=4096`, a 16384 generation cap is unreachable. We treat 4096 as the conservative working ceiling. `scripts/eval_local.py` defaults to CI-faithful 4096/4096 and exposes `--no-ci-mode` as the legacy 20480/16384 escape hatch. NOT a training choice — `lora.yaml:max_seq_length=4096` is the training cap, and it happens to coincide with the README's eval cap, but the two are independent settings. |
| Wall-clock cap | 1800 s per model | Per the README. n=8 generations × ~10 problems must fit. Slow checkpoints (e.g., chronic OOM-thrash, mis-tuned `gpu_memory_utilization`) can hit this cap and get partial credit only. |
| Metrics | pass@1 and pass@8 (math headline = **pass@8**) | Unbiased Chen-et-al-2021 estimator (`evaluate.pass_at_k.pass_at_k`): `pass@k = 1 - C(n-c, k) / C(n, k)`. With n=8: pass@1 = mean(c/8) across problems; pass@8 = mean(any-of-8). **Per the README, math is graded on pass@8 (free-form).** Knowledge / Safety / Multilinguality are pass@1 MC — context only; not this repo's concern. **Measured ci-faithful baseline (2026-05-09):** bare `Qwen/Qwen3-1.7B` reports pass@1 = 0.1625, pass@8 = 0.2000 on `validation_samples/math.jsonl`. The 0.400 figure that appeared earlier was a *legacy-cap* number and is no longer the headline baseline (see `docs/BASELINE.md` → "2026-05-09 CI-mode re-baseline"). |

**Use the vendored `evaluate/` directly.** Do not re-implement extraction,
`is_equiv`, or pass@k. The CI runs byte-identical code, and any
re-implementation will silently drift. Stage 4's `scripts/eval_local.py`
is a vLLM front-end whose only job is to produce a generations JSONL
that `evaluate.score.score_generations` then scores.

Note on `max_model_len=4096` vs the trained 4096 seq-length: these are
two *independent* 4096 caps that happen to match. `lora.yaml:max_seq_length`
is the *training* sequence length; the README's 4096 is the *inference*
context window. A future stretch goal could de-couple them (longer
inference window via a different `max_model_len` once the README's
constraint loosens), but today they coincide.

## What the CI actually does

The team project README spells out a 5-step nightly flow. Reproduced
here so the operational picture lives next to our config:

1. **Freshness check.** `huggingface_hub.repo_info(...).last_modified`
   per roster repo. Unchanged repos are skipped — push to trigger a re-eval.
2. **Validation.** Repo exists; `generation_config.json` present;
   tokenizer has `chat_template`; vLLM can load the model.
3. **Inference.** vLLM batch generation, **n=8** completions per
   problem, with the model's chat template + `generation_config.json`,
   bounded by `max_model_len=4096` and the **1800 s** wall-clock cap.
4. **Scoring.** `\boxed{...}` extraction, OpenCompass `is_equiv`
   normalization, `pass@1` and `pass@8` (math headline = pass@8).
5. **Reporting.** Public leaderboard updates **and** an automatic PR on
   the model's HF repo (Community tab) that adds/replaces
   `EVAL_REPORT.md` at the repo root. PR is non-blocking — read it for
   debug, no need to merge.

Roster naming is locked by the README: the five repos under the team
org must be exactly `cs-552-2026-<org>/{group_model, math_model,
general_knowledge_model, safety_model, multilingual_model}`. Our push
target `cs-552-2026-emainelpe/math_model` matches.

## Bar to claim "SFT added value"

The headline metric for math is **pass@8** (free-form), per the team
README. SFT adds value when the post-Stage-3 checkpoint's pass@8 on
`validation_samples/math.jsonl` exceeds the bare-model baseline run
under the same context caps (`max_model_len=4096`, `max_tokens=4096`).

**Measured (2026-05-11 calibrated comparison, on RCP).** Five-temperature
sweep (0.4 / 0.5 / 0.6 / 0.7 / 0.8) on each of v1/v2/v3 SFT checkpoints
under ci-faithful caps. Best (variant, temp) per row:

| Model                                       | best temp | pass@1   | pass@8   |
|---------------------------------------------|-----------|----------|----------|
| `Qwen/Qwen3-1.7B` (bare baseline, 2026-05-09)| 0.3 (single) | 0.1625 | 0.2000 |
| v1 SFT (DART only, pushed to HF)            | invariant | 0.2000   | 0.3000   |
| v2 SFT (mixed DART + OMI2)                  | 0.6       | 0.2750   | 0.4000   |
| **v3 SFT (pure OMI2)** — winner             | **0.4**   | **0.2875** | **0.4000** |

**v3 SFT at temp=0.4 cleared the bar.** Pass@8 = 0.4000 vs the bare
baseline's 0.2000 (+20 pp, comfortably outside the ±5 pp noise band on
N=10). Pass@1 = 0.2875 vs 0.1625 (+12 pp, also outside noise). This is
the new headline. v3 is the SFT winner and the RLVR base.

**This methodology supersedes the earlier 2026-05-09 "v1 cleared the
bar" reading.** That measurement was a single-temperature draw at
temp=0.6 and v1 happened to land at pass@8 = 0.4000 on the seed-42
sample — upper-end noise on N=10. The five-temperature sweep shows v1
is flat at pass@8 = 0.3000 across all temperatures; the +20 pp jump
*does* still survive, but it belongs to v3, not v1. See `docs/BASELINE.md`
→ "2026-05-11 SFT comparison and temperature sweep" for the full
15-eval table and write-up.

**The bar for future SFT variants.** Any new SFT recipe must beat
**pass@8 = 0.4000 under ci-faithful caps with the temperature sweep
applied** — not on a single-temperature draw. One isolated 0.40 at one
temperature is within noise on N=10 and does not clear the bar.
Re-run on `data_out/eval.jsonl` (500-row DART held-out slice) for
a tighter read when temperature-sweep ranking is ambiguous.

**Legacy-cap reading is an ablation knob, not the headline.**
Under `--no-ci-mode` the pass@8 collapses on every variant (older
write-up in `docs/BASELINE.md` → "2026-05-09 CI-mode re-baseline"). The
CI grades under tight caps, so legacy numbers are not predictive.

Pass@1 is reported alongside as a secondary diagnostic — a pass@1 jump
with flat pass@8 means the model became more consistent but isn't
unlocking new problems; useful for ablation reads, not for grading.

## Milestone strategy

- **May 24 — model-running validation (10% of project grade).** The CI just
  needs to confirm the model loads in vLLM and emits `\boxed{}` answers.
  Aim: a working SFT checkpoint pushed to HF, CI green. RLVR is NOT required
  for this milestone.
- **June 7 — final submission (50% of project grade).** RLVR'd checkpoint
  if RLVR helps; SFT fallback if RLVR destabilizes (Dang & Ngo 2025 warns
  this is a real risk on small models). Document either outcome in the report.

## Hard constraints

- **Do not change the LoRA config.** Reading `configs/lora.yaml` is fine.
  Modifying r, alpha, or target_modules will break the Phase 3 merge.
- **Do not change the chat template** without the team agreeing first.
- **Do not use full fine-tuning.** LoRA only.
- **Do not push to HF until local eval shows non-trivial pass@1.** Each
  push triggers a CI re-evaluation; we don't want to waste those on broken runs.
- **Do not add stages from the strategy_v2.pdf draft.** No format cold-start
  Stage 0, no English-pivot, no DOOR-DPO. Those are not in the team's plan.
- **Do not build a SymPy or hybrid verifier yet.** Exact-match per the
  proposal. The hybrid verifier is documented as a v2 stretch goal in
  IMPLEMENTATION_PLAN.md and may or may not happen.

## Working environment

- **User's laptop**: CPU only. Used for editing code, running fast unit
  tests on small synthetic data, inspecting JSONL files, reading docs.
  Anything ML training-related must NOT run locally.
- **RCP cluster**: 1× A100 40GB. Used for everything that touches the
  model: dataset download (HF cache lives in `/scratch/hf_cache`),
  training, inference, eval.
- **Hugging Face Hub**: target org `cs-552-2026-emainelpe`.

## When uncertain — STOP AND ASK

If you encounter a decision that isn't covered in this file, the proposal,
or the literature review, **stop and ask the user before coding**. Examples:

- Choosing between two reasonable hyperparameters not in the table above
- Picking a specific eval metric or threshold
- Adding any new dependency
- Modifying anything related to the merge (Phase 3) — that's a team decision
- Anything that touches `configs/lora.yaml`, `chat_template/`, or `generation_config.json`

Surfacing the decision is always better than picking a default that
contradicts the proposal.

## Working loop per session

1. Read `CLAUDE.md` (this file) + `IMPLEMENTATION_PLAN.md` + any files
   directly relevant to the task.
2. If the task touches a settled decision, follow the decision.
3. If the task touches an unsettled decision, ask before coding.
4. Write code. Reuse values from `configs/lora.yaml` rather than hardcoding.
5. Add or update CPU-runnable unit tests where it makes sense. Tests
   should run in <30s on the user's laptop.
6. Update `README.md` if user-facing behavior changed.
7. At the end of the session, summarize:
   - What was implemented
   - What was deferred
   - Any flagged follow-ups or new uncertainties

## Sanity checks before any HF push

- [ ] `generation_config.json` present at repo root with chosen temp/top_p/top_k
- [ ] Tokenizer has `chat_template` set (verify byte-identical to `emainelpe-shared`)
- [ ] Local eval (`scripts/eval_local.py`) shows non-zero pass@1 on validation
- [ ] Sample model output actually contains `\boxed{...}`
- [ ] vLLM can load the merged checkpoint without error
