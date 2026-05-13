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
