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
3. `docs/project_description.pdf` — the course's grading rubric
4. `docs/RCP_GUIDE.md` — RCP cluster setup and submission

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
| Sequence length | 4096 | Matches CI eval cap |
| LR schedule | Cosine, 3% warmup | Standard |
| Gradient checkpointing | ON | Memory headroom |
| Thinking mode | ON, baked into chat template | CI does NOT pass enable_thinking |
| RLVR verifier | Exact-match | Proposal commitment; SymPy is v2 stretch |

## Locked shared files

`configs/lora.yaml` and `chat_template/chat_template.jinja` are copied from
the team's `emainelpe-shared` repo. They MUST stay byte-identical to the
shared source for the Phase 3 merge to work. Treat both as read-only.
If a change is genuinely needed, propose it in the shared repo first, get
team sign-off, then update.

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
