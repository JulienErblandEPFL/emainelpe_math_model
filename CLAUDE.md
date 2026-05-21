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
| RLVR experiment outcome | **SFT scaling won; RLVR did not lift** | Trained 600 GRPO steps (~16% of one epoch on 3919 difficulty-curated prompts) on 2026-05-13: regressed pass@8 0.40 → 0.30. Rescue run (2026-05-14/15) on signal-band-filtered prompts ran 100% epoch but ended in policy collapse; recovered checkpoint-650 (pre-collapse) lands at v3-level noise. Two distinct failure regimes observed (early-run starvation, late-run collapse), no measurable RLVR lift in either. Five integration bugs were fixed during the runs — all surfaced as fail-fast preflight assertions and regression tests. See `IMPLEMENTATION_PLAN.md` → Stage 7. |
| Final deployed math expert | **v5 OMI2 100k SFT** (2026-05-19, after v6 rollback) | v3 (50k) → v5 (100k) → v6 (200k) is monotonic +1.1pp MATH-500 pass@1 but v6 → CI regressed (-3pp vs v5). v5 stayed deployed; v6 rolled back same day. RLVR-checkpoint650 evaluated → noise vs v3. Multi-adapter weight-space merges (v3+v5+v6, linear and DARE drop=0.2) failed format discipline (empty `<think>`, no `\boxed{}`). Teacher distillation infra built but not deployed (context-budget bind under CI's 4096 cap). See `REPORT.md` §3.7-§3.11. |
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

## OOM mitigations (2026-05-13)

The v4 SFT runs hit `torch.OutOfMemoryError` at step 1514, and the earlier
v4-200k attempt died at epoch 0.08 — both on the same allocation. The
fundamental cause is structural, not configurational:

**The logits tensor is the dominant memory consumer.** For Qwen3-1.7B the
vocab is 151,643. At sequence length T=4096 and per-batch B=1 the
`(B × T × vocab × 4 bytes)` logits tensor is **2.49 GiB by itself**, and
the contiguous `shift_logits` copy inside `compute_loss` doubles that to
~5 GiB. With B=2 it hits ~10 GiB, with B=4 it hits ~20 GiB. On a 40 GB
A100, after model weights (~3.4 GB in bf16), gradients (~3.4 GB), Adam
optimizer state (~6.8 GB at fp32 moments + ~3.4 GB scratch), and
activations (multi-GB at T=4096 with gradient_checkpointing), the
remaining headroom is ~10-15 GB — which means even per_device_batch=1
can OOM when activations balloon on a near-max-length sequence.

### Why prior mitigations weren't enough

The pre-2026-05-13 mitigations all reduce OTHER memory consumers; the
logits tensor stays put.

- `gradient_checkpointing=True`: shrinks activations by recomputing them
  in the backward pass. Big help, but logits are not activations.
- `max_formatted_tokens=2900`: drops the longest training rows so T<4096
  for the worst cases. Helps the long tail, doesn't fix the headline
  allocation — still 1.76 GiB per batch at T=2900.
- `per_device_eval_batch_size=1` + `eval_accumulation_steps=4`: fixes
  eval-time OOM specifically by making eval B=1 and moving predictions
  to CPU. Training-time logits are independent of this.
- Lower `learning_rate`, fewer `epochs`: orthogonal — no memory effect.

The closer the policy gets to the trained max length, the higher the
probability that one batch sees a near-max-length composition and trips
the same OOM. Statistical mitigation can defer the failure but cannot
prevent it.

### Why Liger Kernel is the structural fix

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) provides a
fused `LinearCrossEntropy` that NEVER materializes the full logits
tensor. Loss and gradients are computed block-by-block directly from
the hidden states and the embedding matrix; the per-block intermediate
fits in a small constant memory budget regardless of vocab size. This
removes `vocab_size × 4 bytes` from the per-batch memory equation
entirely — `vocab_size` simply stops mattering.

TRL 0.19.1's `SFTConfig` and `GRPOConfig` both accept `use_liger_kernel`.
`liger_kernel.apply_liger_kernel_to_qwen3` exists directly (no auto
fallback needed). Verified on the course image 2026-05-13: `liger-kernel
0.8.0` installs cleanly alongside `trl 0.19.1`, `transformers 5.7.0`,
the image's torch build. The saved adapter is byte-identical to a
non-Liger run (Liger affects the loss compute path, not the LoRA
weights), so the Phase 3 merge contract is unchanged.

**Default ON** (both `scripts/train_sft.py` and `scripts/train_rlvr.py`):
`--use-liger-kernel` is the default; pass `--no-use-liger-kernel` for
A/B comparison. We rely on the implicit chain to surface broken installs:
the pod's `pip install -r requirements.txt` dies with a clear error if
the wheel is unavailable, and TRL's `SFTConfig.use_liger_kernel=True`
import-and-patches Qwen3 at trainer construction — an ImportError there
crashes the run within a few seconds of model loading, long before any
real training cost is incurred.

### Secondary mitigation: `PYTORCH_ALLOC_CONF=expandable_segments:True`

PyTorch's caching allocator pins freed blocks at their original size,
which over a 10-14h run accumulates fragmentation: a request for 3 GiB
can fail even when 8 GiB is free if it's spread across non-coalescing
blocks. `expandable_segments:True` lets the allocator coalesce freed
regions, lowering peak-resident memory under fragmentation pressure.
The submit scripts now pass this via `--environment` so it covers both
SFT and RLVR jobs. This is belt-and-suspenders to Liger — Liger removes
the headline allocation; this knob helps with everything else.

### Post-fix memory profile (expected)

At T=4096, B=1, gradient_checkpointing on, bf16, Liger Kernel on,
Adam-on-LoRA-only (PEFT trains ~12M parameters, not 1.7B):

| Item | bf16 size |
|---|---|
| Base model weights | ~3.4 GB |
| LoRA adapter (~12M params) | ~24 MB |
| LoRA gradients | ~24 MB |
| Adam state on LoRA (m, v in fp32) | ~96 MB |
| Activations (T=4096, gradient_checkpointing) | ~4-6 GB |
| **Liger fused LCE intermediate** | **~0.5 GB** (was ~5 GB without Liger) |
| Misc + framework overhead | ~2-3 GB |
| **Total** | **~10-13 GB / 40 GB available** |

The 30 GB headroom is intentional: it survives long-tail sequences
(when an OMI2 row tokenizes close to 4096) without hitting the wall.
Pre-fix, the equivalent budget was ~13-18 GB used + a logits spike that
could reach 30-35 GB on a long sequence — right against the 40 GB cap.

### Throughput note

Liger Kernel's fused kernels are also moderately faster — LinkedIn's
benchmarks show ~5-10% step-time improvement for Qwen-class models on
A100. Not a primary motivation, but the OOM fix is not paid for in
wall-clock.

---

## 2026-05-14 Daily Log

### v4 SFT experiments — NEGATIVE RESULT

Two v4 variants trained successfully on the v4-mix dataset (67,135 train
rows, no cross-source dedup, per_question_cap=4 binding for IntAlg /
Precalc oversampling):

- **v4-fresh** (`cs552-erbland-g65-v4-fresh-20260513-213048`, W&B
  `k93kbsns`): fresh init from Qwen3-1.7B base, lr=1e-4, final loss
  ~0.394, MATH-500 pass@1 = 0.413
- **v4-resume** (`cs552-erbland-g65-v4-resume-20260513-213244`, W&B
  `zd5x6syj`): initialized from v3's adapter, lr=5e-5, final loss
  ~0.413, MATH-500 pass@1 = 0.431

Both regressed from v3 (MATH-500 pass@1 = 0.514) on every subject and
level. Including the targeted subjects: IntAlg 0.296 → 0.211 / 0.216
(−8 pp), Precalc 0.339 → 0.174 / 0.196 (−14 to −16 pp). The
5-temperature validation sweep on v4-resume showed pass@8 = 0.40 only
at temp=0.4 (single-temp), vs v3 hitting 0.40 at temps 0.4 AND 0.6
(multi-temp). Conclusion: noise-level mirage on the N=10 validation
set; real regression on MATH-500 (N=500).

**Lesson.** At 1.7B parameters, targeted data augmentation via the v4
mix did not lift performance. Within-bucket oversampling for IntAlg
(1295 unique problems × 4× cap) and Precalc (746 × 4× cap) wasn't
enough signal to move pass@1, while the addition of MATH-train +
NuminaMath diluted OMI2's contribution. Hypothesis confirmed: the
model is capacity-bound, not coverage-bound, at this scale. The v4
diagnostic-driven multipliers were the right call given the v3
diagnostic gaps, but the gap *is the gap a 1.7B parameter count
imposes*, not one that more aligned data closes.

Personal HF backups (created 2026-05-14):
- `JulienE220/math-adapter-sft-v4-resume-r32-20260514` (pending push)
- v4-fresh not backed up — underperformed v4-resume

Team HF repo state at end of day: **v4-resume pushed knowingly for one
CI cycle** to observe the nightly grade. Plan: re-push v3 tomorrow
morning before the next CI window if v4-resume's grade comes in below
v3's ~0.32.

### Liger Kernel deployed — OOM structural fix (verified on cluster)

Three OOM crashes prior to fix (v4-200k, v4-fresh first attempt,
v4-resume first attempt) all hit `torch.OutOfMemoryError` at
`outputs.logits[..., :-1, :].contiguous()` — the materialized
B × T × 151,643 × 4 bytes logits tensor exceeded A100-40GB headroom on
near-max-length batches.

Fix: enabled `use_liger_kernel=True` in both SFTConfig and GRPOConfig
via `--use-liger-kernel` BooleanOptionalAction (default True). Liger
Kernel's fused cross-entropy never materializes the full logits
tensor — loss is computed chunk-by-chunk on hidden states directly.
Plus added `PYTORCH_ALLOC_CONF=expandable_segments:True` env var to
all three submit scripts as secondary fragmentation mitigation. Plus
added `liger-kernel>=0.8.0` to `requirements.txt`.

Verified on cluster (2026-05-14): `liger-kernel 0.8.0` installs
cleanly with TRL 0.19.1 + transformers 5.7.0 + torch 2.10.0+cu128.
`apply_liger_kernel_to_qwen3` patch exists. **Note: `liger_kernel`
0.8.0 does NOT expose `__version__`** — any future sanity check
should not depend on that attribute. All 8 new CLI plumbing tests
pass. Total test count after fix: 334 passed + 1 skipped.

A bash quoting bug in the submit scripts' Liger Kernel sanity check
caused 3 false-start submissions ("syntax error near unexpected
token `('"). Root cause: the bash escape `\"` was correctly emitted
by the submit script, but `runai workload submit` stripped escape
characters when serializing the pod command. The `bash -n` test on
the assembled POD_CMD only caught the LOCAL form, not what arrived
at the pod. Final fix: removed the cosmetic sanity-check line
entirely. The implicit chain is structurally stronger — pip install
failure or TRL trainer construction will raise a clean `ImportError`
if Liger Kernel isn't available. Kept the generic `bash -n` test as
regression guard for future POD_CMD changes.

### RLVR rescue — currently running (launched 2026-05-14 15:25)

Job: `cs552-erbland-g65-rescue-20260514-152540`. W&B run name pending.

**Config**:
- Adapter init: v3
  (`/scratch/Julien/runs/cs552-erbland-g65-v3-omi2-fix2-20260511-152150/final`)
- Prompt set: `/scratch/Julien/data_out_v3/rlvr_prompts.jsonl` (3936
  problems, signal-band-filtered to [0.250, 0.750] solve_rate per
  v3's scoring — uses k=8 rollouts so solve_rate is quantized to
  {0.250, 0.375, 0.500, 0.625, 0.750})
- SFT_MODEL (for preflights): `/scratch/Julien/merged/math_model_v3`
- `USE_VLLM=1` (asynchronous vLLM rollouts)
- `MASK_TRUNCATED=1` (don't penalize rollouts hitting `max_new_tokens`)
- `LOG_COMPLETIONS=1` (visibility for debugging via W&B)
- `LEARNING_RATE=3e-6`, `KL_COEF=0.04` (Tülu 3 default),
  `ROLLOUT_TEMP=0.8`, `MAX_PROMPTS=3936`
- `LOSS_TYPE=dapo` (default — was suspected in retry3, but the
  signal-band filter resolves the upstream cause)
- `HARD_KILL_ON_WEAK_SIGNAL=unset` — let the run proceed even if
  signal weakens, to observe the full trajectory
- Liger Kernel enabled (default True after today's fix)

**Early observations** (steps 1-30): `frac_reward_zero_std` mostly 0
(sporadically 1 on prompts that have converged to always-pass /
always-fail individually). KL tiny (~0.0003-0.002). Rewards varying
healthily with std ~0.35-0.52. Step pace ~10-20 s/step average →
projected ~15-17h total wall-clock. ETA: 2026-05-15 08:00-10:00.

**Critical insight.** The retry3 failure mode was *global* signal
starvation (`frac_reward_zero_std=1.0` constant). This run shows
*sporadic* per-prompt saturation (some prompts always-pass or
always-fail) but global signal still flows. The signal-band-filtered
prompt set is doing its job — exactly the rescue lever P1 from the
2026-05-13 plan.

### v5 OMI2 100k SFT — currently running (launched 2026-05-14 16:22)

Job: `cs552-erbland-g65-v4-fresh-20260514-162214`. The job-name
convention is cosmetic (re-used the v4-fresh submit script with
`SKIP_PREP=1` + override `DATA_OUT_DIR`); the *data* is v5 OMI2 100k,
NOT v4-mix.

**Hypothesis.** At 1.7B scale, scaling pure OMI2 from 50k (v3) to
100k might lift performance. Tests whether v3 is OMI2-saturated at
50k. Single variable changed: dataset size. If v5 > v3, scaling
works at this parameter count. If v5 ≈ v3, the parameter-bound
hypothesis from v4 holds.

**Dataset**: `/scratch/Julien/data_out_v5_omi2_100k/` — 100,000 train
+ 500 eval. Source: `nvidia/OpenMathInstruct-2` split=train_1M (same
as v3). Filters: per_question_cap=4 (no binding at this scale, all
999,893 raw rows have unique problems), max_formatted_tokens=2900
(0 dropped — OMI2 CoTs are compact). Dataset is byte-clean: no
oversampling, no cross-source-dedup tension.

**Config.** Identical to v4-fresh runtime (Liger Kernel on, lr=1e-4,
fresh init from base Qwen3-1.7B, 2 epochs).

**Early observations** (steps 1-2): loss 0.974 → 0.961,
mean_token_accuracy 0.807 → 0.808. Starting loss is *lower* than
v4-fresh's 1.103 — OMI2's well-curated 405B-teacher CoTs are closer
to Qwen3-1.7B's natural distribution than the v4-mix was. Token
accuracy already 2 pp higher than v4-fresh at the same step. Step
pace ~4 s/step → projected ~7-8h wall-clock. ETA: 2026-05-14
23:30 - 2026-05-15 01:00.

### Parallel run note

Both RLVR and v5 SFT are running in parallel on 2 GPUs (the cluster
gave us a second slot). Each is preemptible. **If preemption hits,
RLVR is the higher-value job to preserve** — more compute invested,
the signal-band prompt set is non-trivial to re-generate. v5 SFT is
cheap to relaunch.

### Pending tasks for tomorrow (2026-05-15 ~09:00)

1. Check CI nightly grade on v4-resume (deployed in team HF overnight
   by design).
2. Re-push v3 to team HF if v4-resume CI grade is worse than v3's
   ~0.32 (likely — expected v4-resume CI ~0.25-0.28).
3. Push v4-resume backup to personal HF:
   `JulienE220/math-adapter-sft-v4-resume-r32-20260514`.
4. Check RLVR completion and the final reward trajectory in W&B.
5. Check v5 OMI2 100k completion and final eval loss.
6. Merge each completed adapter (`merge_and_push.py` dry-run; no push).
7. Run the 5-temperature sweep on each via `eval_local.py`.
8. Run `diagnose_v3.py --target all --model <merged>` against each
   for per-subject comparison vs v3.
9. Decide which adapter (RLVR-rescue, v5, or v3) to push to team HF
   as the math expert for the Phase 3 merge.
10. Update CLAUDE.md with results from this comparison.

---

## 2026-05-15 Daily Log

Four experiments closed out today: two overnight runs from 2026-05-14
(v5 OMI2 100k SFT, RLVR rescue) finished in the early morning, plus a
fresh v6 OMI2 200k SFT run that ran through the day and finished early
on 2026-05-16. v5 was diagnosed, smoke-tested, and pushed to team HF
at 13:05 UTC; RLVR rescue suffered end-of-run policy collapse and the
recoverable checkpoint-650 turned out to be v3-equivalent noise; v6
landed a small-but-real MATH-500 lift over v5 but failed the
5-temperature sweep on validation.

### v5 OMI2 100k SFT — POSITIVE RESULT (deployed to team HF)

- Run name: `cs552-erbland-g65-v4-fresh-20260514-162214` (cosmetic v4
  naming; data is v5 OMI2 100k).
- Dataset: `/scratch/Julien/data_out_v5_omi2_100k` (100k train + 500
  eval, pure OMI2 from `train_1M` split).
- Config: fresh init from `Qwen3-1.7B` base, lr=1e-4, 2 epochs, Liger
  Kernel ON.
- Final training loss ~0.40.
- Personal HF backup: `JulienE220/math-adapter-sft-v5-omi2-100k-r32-20260515`.
- Team HF deployment: pushed 2026-05-15 13:05 UTC, replacing v4-resume
  (which had been knowingly deployed for one CI cycle to observe the
  nightly grade).

**Diagnostic results vs v3 baseline.**

| Surface | v3 | v5 | Δ |
|---|---|---|---|
| Validation pass@8 (temp=0.4) | 0.400 | 0.500 | +10pp |
| In-distribution pass@1 (N=500) | 0.408 | 0.456 | +4.8pp |
| In-distribution pass@4 (N=500) | 0.628 | 0.686 | +5.8pp |
| MATH-500 pass@1 | 0.514 | 0.516 | +0.2pp (tied) |
| MATH-500 pass@4 | 0.686 | 0.672 | −1.4pp |

Per-subject highlights (MATH-500 pass@1): Algebra 0.700→0.732,
Counting 0.480→0.520, Prealgebra 0.668→0.683, Level 1 0.797→0.855.
Slight regressions on hard subjects: IntAlg −2.8pp, Precalc −4.0pp,
Level 5 −1.9pp.

**5-temperature sweep on validation.** pass@8 = 0.500 only at
temp=0.4 (single-temp peak), 0.400 at temp=0.5, 0.300 at temps
0.6/0.7/0.8. The 0.500 is single-temp like v4-resume's noise mirage,
but the in-distribution N=500 lift (+4.8pp pass@1, +5.8pp pass@4) is
robust and the MATH-500 pass@1 is tied (no regression). Net: pushed
to team HF as the math expert.

**Verdict.** v5 lifts in-distribution robustly, validation peak only
at one temperature, MATH-500 tied. Better than v3 in mid-difficulty
and easy-difficulty problems, slightly worse on the hardest subjects.
Pushed to team HF as the math expert pending nightly CI signal.

### RLVR rescue — POLICY COLLAPSE + RECOVERED CHECKPOINT-650

- Run name: `cs552-erbland-g65-rescue-20260514-152540`.
- Config: SFT_MODEL=v3, USE_VLLM=1, MASK_TRUNCATED=1,
  LOG_COMPLETIONS=1, LR=3e-6, KL_COEF=0.04, ROLLOUT_TEMP=0.8,
  MAX_PROMPTS=3936, LOSS_TYPE=dapo, Liger Kernel ON.

**Critical failure mode discovered post-training.**
- Training completed at 100% epoch with all monitoring signals
  healthy: `frac_reward_zero_std` mostly 0, KL tiny, reward varying
  with std ~0.35-0.52.
- BUT: the final adapter at `/final/` produces broken output —
  `"useruseruseruser..."` 1000+ token repetition on a "What is 2+2?"
  smoke test.
- Policy collapse occurred near the end of training despite healthy
  in-flight monitoring; importance sampling ratios were unstable
  across the run.
- The script's post-training smoke check (`smoke_inference_p1`)
  caught the collapse: "P1 preflight FAILED: smoke output missing
  `\boxed{}`."

**Recovered checkpoint-650.**
- Saved `checkpoint-650` at epoch=0.1651 (16.5% epoch,
  global_step=650), `checkpoint-700` at epoch=0.1778.
- `checkpoint-700` had reward crash from 0.55 → 0.044 between step
  698 → step 699 (collapse moment located).
- `checkpoint-650` had healthy reward 0.425-0.675, KL 0.0008-0.0014,
  frac_zero_std=0 — pre-collapse.
- Smoke-tested checkpoint-650: produces clean
  `<think>2+2=4</think>\boxed{4}` output.
- Merged at `/scratch/Julien/merged/math_model_rlvr_ckpt650`.

**Diagnostic on RLVR-ckpt650 vs v3.**

| Surface | v3 | RLVR-ckpt650 | Δ |
|---|---|---|---|
| MATH-500 pass@1 | 0.514 | 0.519 | within noise |
| In-distribution pass@1 | 0.408 | 0.431 | small lift, possibly noise |
| Validation pass@8 | 0.400 | 0.300 | N=10 noise |

Per-subject: noise-level shifts (≤2pp) in both directions, no
consistent pattern. **Verdict**: essentially v3 with noise. 16.5%
epoch of GRPO refinement → no measurable lift.

**Lesson.** RLVR at 1.7B has a narrow stability window: 16.5% epoch
= noise; 100% epoch = policy collapse. The signal-band-filtered
prompt set fixed the retry3 starvation (rescue P1 worked), but the
combination of `LOSS_TYPE=dapo` (still half-configured —
`epsilon_high=null`), unbounded training duration, and 1.7B-scale
instability produced a different failure mode (late-run policy
collapse instead of gradient starvation). Publishable negative result
for the report.

### v6 OMI2 200k SFT — positive but mixed signal

- Run name: `cs552-erbland-g65-v4-fresh-20260515-152430` (cosmetic v4
  naming; data is v6 OMI2 200k).
- Dataset: `/scratch/Julien/data_out_v6_omi2_200k` (200k train + 500
  eval, pure OMI2 from `train_1M` split).
- Config: fresh init from `Qwen3-1.7B` base, lr=1e-4, 2 epochs, Liger
  Kernel ON.
- Final training loss: **0.329** (lower than v5's ~0.40; final
  `mean_token_accuracy` 0.889 vs v5's ~0.87).
- Wall-clock: ~17h (launched 2026-05-15 15:24, finished early morning
  2026-05-16).

**Diagnostic results vs v5.**

| Surface | v5 | v6 | Δ |
|---|---|---|---|
| Validation pass@8 (temp=0.4) | 0.500 | 0.300 | −20pp (N=10 noise but consistent with sweep) |
| In-distribution pass@1 (N=500) | 0.456 | 0.456 | tied |
| In-distribution pass@4 (N=500) | 0.686 | 0.678 | −0.8pp |
| MATH-500 pass@1 | 0.516 | 0.525 | +0.9pp (real at N=500) |
| MATH-500 pass@4 | 0.672 | 0.682 | +1.0pp |

Per-subject (MATH-500 pass@1): Algebra +2.6pp, IntAlg +2.3pp,
Precalc +3.1pp (hard subjects partially recovered), but Counting
−4.6pp, Prealgebra −1.2pp, Level 1 −1.8pp (easy subjects slightly
regressed). Level 5 jumped +4.1pp — v6 lifts hard problems v5 was
flat on.

**5-temperature sweep on validation.** All temps flat at pass@8 = 0.300:

| temp | pass@8 |
|---|---|
| 0.3 | 0.300 |
| 0.4 | 0.300 |
| 0.5 | 0.300 |
| 0.6 | 0.300 |
| 0.7 | 0.300 |

v6 hits the "always-solvable" core of `validation_samples/math.jsonl`
but doesn't reach v5's temp-0.4 peak of 0.500. Likely explanation:
v5's 0.500 at temp=0.4 was getting lucky on one specific problem
that v6's distribution doesn't favor. `validation_samples` is N=10,
dominated by noise at this resolution.

**Verdict on v6.** Small but real lift on MATH-500 (+0.9pp pass@1,
+1.0pp pass@4), redistribution of per-subject performance (hard
subjects up, easy subjects slightly down). Validation regression is
likely N=10 noise. **NOT yet pushed to team HF** — waiting for v5's
CI grade to inform the decision.

### Scaling progression at 1.7B (key finding)

| Variant | Data | MATH-500 pass@1 | Δ vs v3 |
|---|---|---|---|
| v3 | 50k OMI2 | 0.514 | — |
| v5 | 100k OMI2 | 0.516 | +0.2pp (tied) |
| v6 | 200k OMI2 | 0.525 | +1.1pp |

Cumulative +1.1pp pass@1 for 4× more data. Per-subject pattern is
non-uniform: 100k lifts easy/mid, 200k partially recovers hard
subjects. Consistent with a **soft capacity bound** at 1.7B that more
data partially overcomes — not hard saturation. Open question
whether v7 at 400k or 500k would push further or hit a true ceiling.

### Diagnostics archived

- `/scratch/Julien/diagnostics/v5_eval/`
- `/scratch/Julien/diagnostics/rlvr_ckpt650_eval/`
- `/scratch/Julien/diagnostics/v6_eval/`
- `/scratch/Julien/v5_temp_sweep/`
- `/scratch/Julien/v6_temp_sweep/`

### Deployment state end of 2026-05-15

- **Team HF (`cs-552-2026-emainelpe/math_model`)**: v5 OMI2 100k
  (pushed 13:05 UTC, awaiting CI grade).
- **Personal HF backups (Julien)**: v1, v2, v3, v5 done. v4-resume,
  v4-fresh, v6 backup pending push.
- **v6 merged** at `/scratch/Julien/merged/math_model_v6_omi2_200k`,
  NOT pushed (pending v5 CI signal).

### Pending tasks for 2026-05-16

1. Find v5's CI nightly grade (graded overnight 2026-05-15 → 16).
2. Decide based on v5 CI grade:
   - v5 ≥ v3 CI → keep v5, consider pushing v6 as upgrade.
   - v5 ≈ v3 CI → keep v5, do NOT push v6.
   - v5 < v3 CI → roll back to v3 immediately.
3. Push v6 to personal HF backup:
   `JulienE220/math-adapter-sft-v6-omi2-200k-r32-20260516`.
4. Push v4-resume + v4-fresh to personal HF backups (deferred from
   yesterday).
5. Begin Phase 3 group merge prep (4 days to the 2026-05-19
   milestone).

---

## 2026-05-19 Daily Log

The 2026-05-16 v5 CI grade landed at **0.34** (within v3's 0.32/0.35
band — no significant CI lift, no regression). v5 stayed deployed.
On 2026-05-19, four lines of work ran in parallel: a v6 production
test, a v5 re-measurement (pass@16 + low-temp sweep), a multi-adapter
weight-space merge smoke test, and a teacher-distillation
infrastructure build.

### v6 deployed to team HF, regressed on CI, rolled back

- 2026-05-19 morning: pushed v6 OMI2 200k to team HF as an upgrade
  attempt. v6 had **+0.9pp MATH-500 pass@1** vs v5 (0.525 vs 0.516)
  with hard-subject recoveries (IntAlg +2.3pp, Precalc +3.1pp, L5
  +4.1pp) — a "scaling helps hard problems" curve.
- 2026-05-19 nightly CI grade: **pass@8 = 0.31**. Regression of -3pp
  vs v5's 0.34.
- Decision: **rolled back to v5** on team HF the same day. The CI
  distribution clearly leans easy/mid where v6 regressed (Counting
  -4.6pp, Prealgebra -1.2pp, L1 -1.8pp) more than hard where v6
  lifted. Per-subject redistribution at this capacity is **zero-sum**
  in absolute pass@1 terms.

**Lesson — local-eval lift does NOT transfer to CI at 1.7B (new).**
v5 also had ~0pp CI lift over v3 despite +4.8pp in-distribution
pass@1; v6 had -3pp CI despite +0.9pp MATH-500 and per-subject
recoveries on the diagnostic hard slices. **MATH-500 is necessary
but not sufficient as a CI predictor.** The CI secret set's
distribution rewards a different operating point than MATH-500's;
without access to the CI set composition, we can only observe the
gap, not close it.

### v5 pass@16 measurement + low-temperature sweep

Re-measured v5 on `validation_samples/math.jsonl` to anchor the
n=8 pass@8 = 0.500 reading from 2026-05-15 (which looked suspicious
once v5's CI grade came in at v3-level).

- **pass@16 (16 completions × 10 problems at temp=0.4): reported
  pass@8 from n=16 = 0.390.** The n=8 reading (0.500) was the
  upper-tail of Chen-2021 estimator noise on N=10; the n=16 anchor
  places v5's **true pass@8 ≈ 0.39 on this set**, not 0.50.
- Per-problem solve pattern: **4-5 of 10 reliably solvable**
  (solve_rate ~0.7-1.0), **5-6 at-or-beyond the 1.7B reasoning
  frontier** (solve_rate ~0.0-0.3). pass@8 on this set has a hard
  ceiling near 0.5 that no SFT scaling we've tried has moved.
- **Low-temperature sweep (n=8, extends prior 0.4-0.8 sweep down):**

  | temp | pass@1 | pass@8 |
  |------|--------|--------|
  | 0.20 | 0.238  | 0.300  |
  | 0.25 | 0.188  | 0.300  |
  | 0.30 | 0.263  | 0.400  |
  | 0.35 | 0.288  | 0.400  |
  | 0.40 | 0.288  | 0.500  |

  Best is temp=0.40 (already locked in v5's `generation_config.json`).
  The low-temp tail (0.20-0.25) underperforms — too greedy on a
  10-row set leaves diversity unused.

**Lesson — n=8 single-temp pass@8 on N=10 has ~10pp std error.**
Don't promote a checkpoint on a single-temp n=8 reading near a
ceiling; anchor on pass@16 (or larger N) when stakes are high.

### Multi-adapter weight-space merge experiments (NEGATIVE RESULT)

Built `scripts/merge_adapters.py` (CPU-only LoRA weight-space merge
w/ optional DARE drop; 5 unit tests). Tested two configurations
on `(v3, v5, v6)`:

- **Linear** at `(0.2, 0.5, 0.3)` weights.
- **DARE drop=0.2** at the same weight triple.

Both produced coherent math on a smoke prompt ("What is 2+2?" →
"2+2=4" was correct), but **both lost format-emission discipline**:
empty `<think>...</think>` blocks, missing `\boxed{...}` answer
wrapper. DARE drop=0.2 did not rescue.

**Lesson — LoRA weight-space linear blending breaks format
discipline at 1.7B under our r=32 spec (new).** Same-task adapters
(all math, identical chat template, identical LoRA spec) lose the
discrete format conventions under linear interpolation in the LoRA
delta space. **Implication for the Phase 3 four-expert merge
(2026-05-19): the cross-task merge (math + knowledge + multilingual
+ safety) faces a similar format-preservation risk.** The team
should run a format-preservation smoke test on the merged group
model output before relying on its `\boxed{}` outputs. Math side
contributes v5 alone to the merge, no merged-adapter candidate.

### Teacher distillation infrastructure built, not deployed

Built five scripts with CPU-only unit tests:

| Script                             | Tests | Status                    |
|------------------------------------|-------|---------------------------|
| `scripts/merge_adapters.py`        | 5     | used for §3.10 smoke test |
| `scripts/sample_failures.py`       | 6     | built, not yet run        |
| `scripts/teacher_smoke.py`         | 5     | smoke-tested teacher      |
| `scripts/teacher_distill.py`       | 6     | built, not yet run        |
| `scripts/extract_math_level45.py`  | 20    | built, not yet run        |

Teacher smoke (Qwen3-32B-AWQ in thinking mode):

| Problem set                            | N  | format_rate | pass_rate |
|----------------------------------------|----|-------------|-----------|
| `validation_samples/math.jsonl` (olympiad) | 10 | 0.15        | 0.15      |
| MATH-train Level 4-5                   | 10 | 0.70        | 0.45      |

**The context-budget bind.** Qwen3-32B-AWQ in thinking mode emits
multi-thousand-token thinking traces; the formatted output then
truncates beyond `max_model_len=4096` before `\boxed{...}`. On
olympiad-difficulty problems (the kind v5/v6 fail on) only 15%
produced format-valid + correct traces. On MATH-train Level 4-5
the rate rose to 45% — still less than half.

**Decision — NOT committing to overnight 4000-problem distillation.**
Expected yield at 45% rate over 4k pool ≈ 1,800 usable
(problem, trace) pairs. Folding into a v7 SFT projects to **+1-3pp
MATH-500 pass@1** — same band where v6's local lift failed to
transfer to CI. Combined with overnight cluster runtime
(~12-15h on 1×A100) and non-zero format-regression risk on the SFT
side, payoff doesn't justify cost. Infrastructure preserved for
future re-use if a longer-context teacher or relaxed CI cap
becomes available.

**Lesson — available open quantized teachers face a context-budget
bind at 4096 (new).** Qwen3-32B-AWQ specifically; same may apply
to other quantized 30B-class teachers in thinking mode on
competition-difficulty problems.

### Final deployed math expert (end of 2026-05-19)

- **Team HF (`cs-552-2026-emainelpe/math_model`).** v5 OMI2 100k SFT.
  Re-deployed after v6 rollback. CI re-grade pending overnight
  2026-05-19 → 20.
- **Personal HF backups (Julien).** v1, v2, v3, v5 done; v4-resume,
  v4-fresh, v6 backups still pending push (deferred again).
- **Cluster artifacts.**
  - `/scratch/Julien/merged/math_model_v3` — v3 merged
  - `/scratch/Julien/merged/math_model_v5_omi2_100k` — v5 merged
    (sourced for team HF push, currently deployed)
  - `/scratch/Julien/merged/math_model_v6_omi2_200k` — v6 merged,
    pushed and rolled back same day
  - `/scratch/Julien/merged/math_model_rlvr_ckpt650` — RLVR ckpt-650
    merged, evaluated → noise vs v3
- **Diagnostics added 2026-05-19.**
  - `/scratch/Julien/v5_pass16/` (n=16 anchor)
  - `/scratch/Julien/v5_temp_sweep_lowtemp/` (0.20-0.40 sweep)
- **CI grade history (final view).**

  | Date           | Variant on team HF | CI pass@8 |
  |----------------|--------------------|-----------|
  | 2026-05-13 04:17 | v3 (SFT, OMI2 50k) | 0.32      |
  | 2026-05-13 23:30 | v3 (re-grade)      | 0.35      |
  | 2026-05-16 04:57 | v5 (SFT, OMI2 100k) | **0.34**  |
  | 2026-05-19       | v6 (SFT, OMI2 200k) | **0.31** (rollback trigger) |
  | 2026-05-20 (pending) | v5 (re-pushed after rollback) | TBD |

### Pending tasks for 2026-05-20

1. Observe v5 CI re-grade after the v6 rollback (confirm v5 ≈ 0.34
   is stable across nightly draws).
2. Push v6 to personal HF backup as
   `JulienE220/math-adapter-sft-v6-omi2-200k-r32-20260516`.
3. Push v4-resume + v4-fresh personal HF backups (continuing defer
   is acceptable if storage isn't urgent).
4. Coordinate with team on Phase 3 group merge (today's milestone):
   confirm v5 on `cs-552-2026-emainelpe/math_model` is the math
   contribution; budget a format-preservation smoke test on the
   merged group model (see today's §3.10 finding in REPORT.md).
5. Decide whether to git-commit the 5 new scripts
   (`merge_adapters.py`, `sample_failures.py`, `teacher_smoke.py`,
   `teacher_distill.py`, `extract_math_level45.py`). All have CPU-
   only passing tests; not yet in git.

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
