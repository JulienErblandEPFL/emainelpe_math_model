# IMPLEMENTATION_PLAN.md — Math Expert

This plan breaks the math expert work into stages. Each stage is sized to
fit roughly one focused Claude Code session. The "Done when" criterion
for each stage is what tells you (the user) the stage is finished and
safe to move on from.

Stages are numbered and labeled with their phase from the proposal
(Phase 1 = SFT, Phase 2 = RLVR). Some stages are pure setup with no
phase label.

---

## Stage 0 — Repo skeleton

**Goal.** Create the directory structure and stub files described in
`CLAUDE.md`. No working code yet; just the scaffolding.

**Tasks.**
- Create `configs/`, `chat_template/`, `data/`, `data/tests/`, `scripts/`,
  `rcp/`, `docs/` directories
- Create `requirements.txt` with the locked dependencies (see below)
- Create empty stub `README.md` describing what this repo is
- Create `.gitignore` excluding: `runs/`, `data_out/`, `*.safetensors`,
  `*.bin`, `wandb/`, `__pycache__/`, `.venv/`, `.env`, `*.parquet`
- Copy `configs/lora.yaml` and `chat_template/chat_template.jinja` from
  the team's `emainelpe-shared` repo (the user will provide these)

**Requirements file contents (lock at this stage):**
```
trl>=0.21.0
transformers>=4.51.0
peft>=0.13.0
accelerate>=0.34.0
datasets>=3.0.0
bitsandbytes>=0.43.0
vllm>=0.6.0
pyyaml
huggingface_hub>=0.25.0
wandb
pytest
```

**Done when.** The directory tree matches the layout in `CLAUDE.md`,
`requirements.txt` exists, the two shared files are in place, and
`git status` shows a clean initial commit ready to make.

---

## Stage 1 — Data preparation (Phase 1)

**Goal.** A script that loads `hkust-nlp/dart-math-uniform`, subsamples to
~40-50k examples with a per-question cap of 4-6 solutions, wraps each
example in `<think>...</think>\n\n\boxed{...}` format, and writes a JSONL
ready for TRL's `SFTTrainer`.

**Files to create:**
- `data/prepare_sft.py` — main script
- `data/tests/test_prepare_sft.py` — CPU-only unit tests on synthetic data

**Key requirements for `prepare_sft.py`:**
- DART-Math-Uniform schema: `query` (problem) and `response` (solution
  ending in `\boxed{...}`)
- For each row: split the response into `(reasoning_before_box, final_answer)`
  using regex on the LAST `\boxed{...}` match
- Wrap as: `<think>\n{reasoning}\n</think>\n\n\\boxed{{{answer}}}`
- Output JSONL with `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`
- Subsample with a fixed seed (default 42) for reproducibility
- Apply per-question cap (default 4 solutions per unique query)
- Drop rows where the boxed answer cannot be extracted
- Drop rows whose response exceeds a length cap (8000 chars by default,
  proxy for token count)
- Hold out a small eval split (default 500 examples) before writing train

**Tests should cover:**
- Splitting a response with a single `\boxed{...}` at the end
- Splitting a response with multiple `\boxed{...}` (use the last one)
- Dropping a response with no `\boxed{...}`
- The per-question cap actually caps
- Output JSONL is valid JSON, one example per line

**Done when.**
- `pytest data/tests/test_prepare_sft.py` passes on the user's laptop
- A 200-row dry run on RCP produces a valid `train.jsonl` and `eval.jsonl`
- A spot-check of 3 random output rows shows correctly formatted
  `<think>...</think>\n\n\boxed{...}` structure

**Notes on dataset behavior (RCP dry-run findings).**
- DART-Math-Uniform is *mixed format*. About 50% of rows end with
  `\boxed{...}` (the canonical convention) and the rest use plain
  "The answer is: $X$" or are corrupted token-salad. The pipeline
  filters by presence of `\boxed{...}` and drops the rest. **A ~52% drop
  rate on DART is expected behavior, not a bug.** With the dataset's
  ~590k rows, ~280k survive — well above the 50k subsample target.
- Purity filters layered on top of `\boxed{}` extraction:
  - `--min-reasoning-chars` (default 150): drops rows whose cleaned
    reasoning is too short to demonstrate step-by-step thinking.
  - `--max-answer-chars` (default 200): drops rows with pathologically
    long boxed payloads (token-salad with a box).
- Trailing-fragment cleanup: after `extract_last_boxed`, the prose
  before `\boxed{...}` often ends in orphan tokens like `$`, `$$`,
  `\[`, or "The answer is:". `strip_trailing_preamble` removes a
  conservative set of these so the `<think>` block reads cleanly. The
  strip-rate is logged at INFO with the prefix `[prepare_sft]` so we
  can verify the cleanup is firing on real data.
- Pipeline ordering: extract_last_boxed → max_answer_chars →
  strip_trailing_preamble → min_reasoning_chars. The strip runs BEFORE
  the length floor so cleaned reasoning is what gets measured.

**Verified on RCP (2026-05-07).** Full DART-Math-Uniform dry run:
kept=282402, dropped_no_box=307133, dropped_too_long=395,
dropped_too_short_reasoning=770, dropped_too_long_answer=5.
Strip-rate `[prepare_sft]` diagnostic: 94.8% (268448/283172 rows
reaching the strip step). Spot-check of 5 random surviving rows: 4/5
perfectly clean, 1/5 with a tiny inert cosmetic artifact (orphan
"The"). The 94.8% is the real measured rate, not an estimate.

### v2 (mixed DART + OpenMathInstruct-2) — implemented 2026-05-09

**Goal.** A second dataset variant that adds `nvidia/OpenMathInstruct-2`
(OMI2) as a parallel source to DART-Math-Uniform. Final v2 size matches
v1 (~50k examples) at ~50/50 mix.

**Why mix, not replace.** OMI2's teacher (Llama3.1-405B-Instruct) is
substantially stronger than DART's (DeepSeekMath-7B-RL) — that's the
core motivation for adding it. DART contributes diversity and per-
problem multi-solution coverage that OMI2's augmented variants don't
match cleanly. Mixing keeps both signals.

**Locked decisions (D1–D5).**

| ID | Decision |
|---|---|
| D1 | **MIX, not REPLACE.** v2 = 50/50 split DART + OMI2, ~25k each, mixed and shuffled with seed=42. |
| D2 | **OMI2 boxing strategy.** Append `\boxed{expected_answer}` to the cleaned `generated_solution`. `extract_last_boxed` takes the LAST box, so the appended answer wins; any mid-text `\boxed{}` in the model's CoT is preserved as reasoning. Same chat-format output as DART. |
| D3 | **Per-source per-problem cap.** Max 4 solutions per unique `problem` string, applied INDEPENDENTLY to each source. Same DART rule, same `apply_per_question_cap`. |
| D4 | **Token-length filter on formatted chat.** New step: drop rows whose Qwen3-tokenized formatted chat exceeds 3500 tokens. Auto-default for `--source openmathinstruct` and `--source mixed`; off for `--source dart` (preserves v1 byte-stable). The OpenMathInstruct-2 paper explicitly warns that "excessive verbosity is detrimental to SFT". |
| D5 | **Subsample seed = 42.** Same as DART resampling; the mix is reproducible end-to-end. |

**Architecture.** Additive to `data/prepare_sft.py`, no refactor of
`build_pipeline`. The OMI2 path normalizes raw rows into the DART
`{query, response}` shape via `normalize_openmathinstruct_row` (append
the boxed answer), then feeds through the existing extract → strip →
cap → format pipeline. The token filter is opt-in via two new kwargs
(`max_formatted_tokens` + `tokenize_fn`); both default to `None` and the
filter is a no-op unless both are set, so v1 callers and tests are
unaffected.

The `transformers.AutoTokenizer` import that backs the token filter
lives in `main()` only, behind the `--max-formatted-tokens` resolution.
CPU unit tests inject a fake `tokenize_fn` so the laptop suite stays
under 0.1s.

**Files changed.**
- `data/prepare_sft.py` — new helpers `normalize_openmathinstruct_row`,
  `resolve_n_samples`; new kwargs `max_formatted_tokens` + `tokenize_fn`
  on `build_pipeline`; new CLI flags `--source`, `--train-size`,
  `--max-formatted-tokens`, `--dart-fraction`, `--openmathinstruct-name`,
  `--openmathinstruct-split`, `--chat-template`. Existing DART-only
  defaults unchanged.
- `data/tests/test_prepare_sft.py` — 20 new tests covering OMI2
  normalization, the token filter (with mock `tokenize_fn`),
  `resolve_n_samples` semantics, mixed-mode shuffle invariants.
  Existing 36 v1 tests pass byte-stable.
- `README.md` — new "Data prep" section documenting v1 (default) and
  v2 (`--source mixed`) invocations.

**Mutually exclusive CLI flags.**
- `--n-samples X` (v1 semantics): X total rows after filtering, BEFORE
  the train/eval split. The existing tests + the existing
  `rcp/submit_train.sh` keep working unchanged.
- `--train-size X` (v2 semantics): X rows in `train.jsonl` AFTER the
  split. Internally translates to `n_samples = X + eval_size`.
- Passing both → CLI error.

**Done when.**
- Existing DART tests pass byte-stable (✅ 36/36).
- New v2 tests pass on the laptop in <0.1s (✅ 20/20).
- An RCP dry run with `--source mixed` produces a non-empty
  `data_out_v2/{train,eval}.jsonl` and the diagnostic logs show
  per-source kept/dropped counts (operational; not yet done).

**Not in v2 scope.**
- No change to `scripts/train_sft.py` — the v2 JSONL is consumed exactly
  like v1.
- No new `--source` for RLVR. Stage 7's prompt-set curation still
  reads from `data_out/train.jsonl` (DART-shaped); to feed v2 prompts
  to RLVR, point `--input-jsonl` at `data_out_v2/train.jsonl` — same
  schema.
- No streaming-mode loader. OMI2 `train_1M` loads in full; ~1GB in
  memory on RCP. Streaming would be cleaner but requires rework of
  `build_pipeline` to consume an `IterableDataset`.

---

## Stage 2 — Local chat-template verification

**Goal.** Confirm the shared chat template renders correctly when applied
to the actual Qwen3-1.7B tokenizer, and produces what the CI expects.

**This is the only stage that runs on RCP before any training.**

**Tasks.**
- Write a small `scripts/verify_chat_template.py` that:
  - Loads `Qwen/Qwen3-1.7B` tokenizer
  - Sets `tokenizer.chat_template` to the contents of `chat_template/chat_template.jinja`
  - Calls `apply_chat_template([{"role":"user","content":"What is 2+2?"}], tokenize=False, add_generation_prompt=True)`
  - Prints the output
  - Asserts: ends with `<|im_start|>assistant\n` AND does NOT contain `<think>\n\n</think>` (which would indicate thinking is OFF)
  - Then runs the same call with `add_generation_prompt=False` on a full conversation including an assistant turn with `<think>` tags, and asserts the round-trip looks correct

**Done when.**
- The script runs on RCP and prints expected output
- Both assertions pass
- The user has manually inspected the output and confirms it looks right

**If this stage fails:** stop. Do not proceed to training. The chat
template is the foundation for everything else; a broken template means
broken training, broken inference, and a broken CI check. Fix the Jinja
template (which lives in `emainelpe-shared`, so the fix needs to propagate
to all four expert repos).

---

## Stage 3 — SFT training script (Phase 1)

**Goal.** A TRL-based SFT script that trains a LoRA adapter on the JSONL
produced in Stage 1, using the locked LoRA config.

**Files to create:**
- `scripts/train_sft.py`

**Key requirements:**
- Read `configs/lora.yaml` for r, alpha, target_modules, max_seq_length
- Load tokenizer from base model and overwrite chat_template with the
  locked Jinja
- Use `TRL.SFTTrainer` with `peft_config=LoraConfig(...)` and
  `processing_class=tokenizer`
- `assistant_only_loss=False` — TRL 0.21+ refuses to auto-patch the
  locked Qwen3 Jinja because it lacks `{% generation %}` markers
  (discovered Stage 3 smoke run, 2026-05-07). Loss is computed over the
  full sequence (user + assistant tokens). Adding the markers requires a
  team-coordinated update to `emainelpe-shared` and is filed as a v2
  stretch goal below.
- `packing=False` (the `<think>` blocks are long; packing also rules
  out a future flip back to assistant-only-loss without re-considering)
- bf16 if supported, else fp16
- Gradient checkpointing ON
- CLI args for `--train_file`, `--eval_file`, `--output_dir`,
  `--epochs`, `--learning_rate`, `--per_device_train_batch_size`,
  `--gradient_accumulation_steps`, `--run_name`
- Saves the final adapter + tokenizer (with chat template) to
  `<output_dir>/final/`

**Done when.**
- The script runs end-to-end on RCP on a tiny smoke set (200 examples,
  1 epoch) without errors
- The resulting adapter directory contains `adapter_model.safetensors`,
  `adapter_config.json`, `tokenizer_config.json`, and the chat template
- W&B logs (or stdout if W&B is not set up) show declining loss

**Verified end-to-end on RCP (2026-05-07).** Larger smoke (1000 ex,
2 epochs, 60 steps): train_loss 0.9999 → 0.5678 → 0.6005, eval_loss
0.6227 (close to train, no overfit), token accuracy 78% → 83%,
token-length filter dropped 0/1000 rows, `chat_template` round-trip
byte-identical after save. Smoke inference for "What is 2+2?" returned
well-formed `<think>\n2+2=4\n</think>\n\n\boxed{4}` — exact shape of
the training-data format produced by `data/prepare_sft.format_response`.

**Note.** The actual full SFT run is a separate operational task — submit
via `rcp/submit_train.sh`, monitor logs, expect 8-12 hours. That's not
something Claude Code does; that's the user. The script just has to be
correct.

---

## Stage 4 — Local eval (mimicking the CI)

**Goal.** A vLLM front-end that loads a merged checkpoint, runs n=8
completions per problem, dumps a generations JSONL in the schema
`evaluate/score.py` expects, and pipes it through
`evaluate.score.score_generations` to report pass@1 and pass@8.

**Mirrors the CI contract by reuse, not re-implementation.** The
`evaluate/` package vendored at the repo root is the CI's scoring code,
byte-identical. Stage 4 wraps it; it does not replicate it. All sampling
parameters below are pinned to match the "Eval contract" section in
`CLAUDE.md`.

**Files to create:**
- `scripts/eval_local.py`
- `scripts/tests/test_eval_local.py`

**Key requirements (CI-mirrored values are bolded):**
- Input: a JSONL file with `{"prompt": "...", "answer": "..."}` per line.
  - Default `--eval-file validation_samples/math.jsonl` (the course's
    vendored snapshot, 10 rows, OOD competition problems — matches what
    the CI scores against).
  - Secondary `--eval-file data_out/eval.jsonl` (the DART held-out slice
    produced by `data/prepare_sft.py` — larger N, in-distribution, lower
    variance). Same script, no new data prep required.
- Outputs (under `--output-dir`):
  - `generations.jsonl` — input rows with `completions` (list of n=8
    strings) appended. Matches `evaluate/score.py`'s expected schema, so
    re-scoring with a different `--method` is a one-liner without
    re-running inference.
  - `scored.json` — `evaluate.score.score_generations`'s full output:
    pass@k metrics + per-problem detailed results (extracted answers,
    per-completion correctness flags).
  - stdout: one-line pass@1 / pass@8 summary.
- Use vLLM with bf16. **Default (CI-faithful): `max_model_len=4096`,
  `max_tokens=4096`** — matches the team README's combined-context
  cap; the binding ceiling for any CI prediction. **`--no-ci-mode`
  escape hatch: `max_model_len=20480`, `max_tokens=16384`** — tracks
  the older `docs/project_description.pdf` (page 3: `Max new tokens:
  16384`), more permissive than CI; numbers measured under it
  *overstate* what CI will report. Use only for ablations where the
  longer generation budget matters. See CLAUDE.md "Eval contract" for
  the conflict write-up. Runtime assertion in `main()`: refuse to run
  if `AutoConfig.from_pretrained(model).max_position_embeddings <
  max_model_len`.
- Apply chat template via `tokenizer.apply_chat_template(..., add_generation_prompt=True)` AFTER overwriting `tokenizer.chat_template` with the locked Jinja (same idiom as `scripts/verify_chat_template.py` and `scripts/train_sft.py:smoke_inference`). vLLM receives pre-rendered prompt strings.
- SamplingParams:
  - **`n=8`** (CI contract)
  - **`max_tokens`**: 4096 by default (CI-faithful, matches the team
    README's `max_model_len=4096`); 16384 under `--no-ci-mode` (legacy,
    tracks `docs/project_description.pdf` page 3, more permissive
    than CI). NOT the training-time `max_seq_length=4096` from
    `lora.yaml` — the training cap and the README's inference cap are
    independent settings that happen to coincide at 4096.
  - **`seed=42`** (CI contract; same seed as data prep and SFT)
  - `temperature` / `top_p` / `top_k` default to the merged checkpoint's
    `generation_config.json` (set in Stage 5). Pre-Stage-5 fallback
    defaults: `temp=0.3, top_p=0.95, top_k=20`. CLI override is
    available for sweeps; the script logs a WARNING if any sampling
    param is overridden so the operator notices the drift from the
    pushed-checkpoint contract.
- Scoring: import `evaluate.score.score_generations` and call with
  `method="boxed"`. Do NOT write own extraction, normalization, or
  pass@k math — the CI uses byte-identical code, and re-implementing
  invites silent drift.

**Done when.**
- CPU unit tests for the pure helpers (prompt construction given a fake
  tokenizer, generations-dump JSONL shape, sampling-params override
  warning, max_model_len assertion logic, `resolve_context_caps` —
  CI-faithful default + `--no-ci-mode` legacy escape hatch + explicit
  overrides) pass on the user's laptop in <5s without vLLM imports.
- The script runs on `validation_samples/math.jsonl` with bare
  `Qwen/Qwen3-1.7B` and produces non-trivial pass@1 (i.e., not 0% and
  not 100%).
- `generations.jsonl` is well-formed: re-feeding it through
  `python -m evaluate.score --generations <file> --benchmark math`
  reproduces the script's reported metrics byte-for-byte.

**Bar to claim "SFT added value" (post-Stage-3 checkpoint vs baseline).**
The team README makes **pass@8 the headline metric for math** (free-form,
graded on pass@8). The recorded bare-model numbers in `BASELINE.md`
(pass@1 = 0.300, pass@8 = 0.400, 2026-05-07) were measured under the
*legacy* 20480/16384 caps — more permissive than CI. They are a soft
upper estimate of the true CI-faithful baseline; a re-baseline under
the new default (`max_model_len=4096`, `max_tokens=4096`) is pending
on RCP. Once it lands, the primary criterion to beat is **CI-mode
pass@8 above the CI-mode bare baseline**. Until then, treat
pass@8 = 0.400 as the soft floor with the caveat that it may drop
under the 4096 cap (long `<think>` chains can clip). Report pass@1
alongside as secondary signal — a pass@1 jump with flat pass@8 means
the model became more consistent but isn't unlocking new problems;
that's a useful diagnostic but isn't what the CI grades.

**Noise budget on the default snapshot.** N=10 means the standard error
on pass@1 is roughly ±5 percentage points; pass@8 is binary per problem
and even chunkier. A 4-point swing between two checkpoints is within
noise, not a real signal. For tighter signals, also run against
`data_out/eval.jsonl` (500 rows from the DART held-out slice — different
distribution, in-domain, much lower variance) and look for movement on
both targets together.

**Why this matters.** Local eval is the feedback loop. Without it you're
flying blind between HF pushes (which only get evaluated nightly).

---

## Stage 5 — Merge adapter and push to HF (Phase 1 deliverable)

**Goal.** Take the trained LoRA adapter, merge it into the base model
weights, write `generation_config.json`, and push the full checkpoint to
`cs-552-2026-emainelpe/math_model`.

**Sets the eval-time contract.** The CI samples with the values written
into `generation_config.json` here (see "Eval contract" in `CLAUDE.md`).
Defaults to consider for math: `temperature=0.3, top_p=0.95, top_k=20`.
Run a 50-sample temperature sweep on the clean eval before the June 7
final push to confirm these values — small change, high leverage on
pass@1.

**Files to create:**
- `scripts/merge_and_push.py`

**Key requirements:**
- Load `Qwen/Qwen3-1.7B` in bf16
- Load the trained adapter via `PeftModel.from_pretrained`
- Call `merge_and_unload()` to fold the LoRA into the base weights
- Save merged weights with `safe_serialization=True`
- Save tokenizer (carrying the chat template) to the same dir
- Write `generation_config.json` with: temperature, top_k, top_p (from
  CLI args, defaults conservative for math: temp=0.3, top_p=0.95, top_k=20),
  `do_sample=True`, the standard Qwen3 EOS/BOS/pad token IDs, and
  `transformers_version: "4.51.0"`
- Pre-flight checks: assert `config.json` and `generation_config.json`
  exist, assert `tokenizer.chat_template` is set; refuse to push otherwise
- Push to HF only if `--push` flag is set
- Push: weights, tokenizer, generation_config (re-uploaded explicitly to
  guard against transformers version differences)

**Done when.**
- A merged checkpoint exists locally and `vllm.LLM(<path>).generate(...)`
  works on a sample prompt
- `--push` to a test branch succeeds and the resulting HF repo has all
  required files at root: `config.json`, `generation_config.json`,
  `*.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`
- After the push, the course CI (running nightly at 23:59) re-evaluates
  the checkpoint (freshness check passes because `lastModified` advanced)
  and opens or updates an automatic Pull Request on the model repo's
  Hugging Face Community tab, adding/replacing `EVAL_REPORT.md` at the
  repo root. The PR is non-blocking — read it for debug, no need to
  merge. See team project README → "Automatic evaluation reports" for
  the canonical wording.

**This stage produces the May 24 milestone deliverable.**

---

## Stage 6 — RCP submission script

**Goal.** A bash script that submits a non-interactive training job to RCP
running Stages 1 → 3 → 5 in sequence.

**Files to create:**
- `rcp/submit_train.sh`

**Key requirements:**
- `runai submit` with: 1× A100 40g, large-shm, the team's project name,
  the course's PVCs, environment vars for HF_TOKEN and WANDB_API_KEY
- The job's command: `cd /scratch/<repo>; pip install -r requirements.txt;
  python data/prepare_sft.py ...; python scripts/train_sft.py ...`
- Configurable via env vars: `GASPAR`, `GROUP`, `IMAGE`, `REPO_DIR`
- Refuse to run with placeholder values (e.g., `GROUP=gXX`)
- Print clear instructions at end: how to follow logs, how to delete

**Done when.**
- Running locally with team-correct env vars submits a job that reaches
  `Running` state on RCP
- Logs show the data prep + training kicking off without errors
- The user can `runai delete job <name>` cleanly

---

## Stage 7 — RLVR (Phase 2) — V0 IMPLEMENTED 2026-05-09, NOT YET TRAINED

**Goal.** Add GRPO training on top of the SFT checkpoint.

**Why we're doing this now.** Stage 5 SFT post-merge pass@8 on
`validation_samples/math.jsonl` is 0.30 vs the bare-model baseline of
0.40 (BASELINE.md). The SFT model produces correct format consistently
but doesn't beat baseline on this small noisy set. RLVR is the team's
committed lever to close that gap; from `CLAUDE.md`:

> RLVR'd checkpoint if RLVR helps; SFT fallback if RLVR destabilizes
> (Dang & Ngo 2025 warns this is a real risk on small models).

**Decisions locked (2026-05-09):**

| ID | Decision | Choice |
|----|---|---|
| D1 | RL framework | TRL `GRPOTrainer` (consistency with the SFT pipeline) |
| D2 | Starting point | Continue training the SFT LoRA adapter on top of `Qwen3-1.7B` base (Phase 3-merge-compatible) |
| D3 | Prompt set | Score DART-Math-Uniform with the SFT model; keep the [0.2, 0.8] empirical solve-rate band |
| D4 | Reward | `reward = 1.0 * correct + 0.05 * has_box`, via `evaluate.is_equiv` |
| D5 | Hyperparameters | lr=3e-6, beta(KL)=0.04, num_generations=8, rollout_temp=0.8, max_prompts=5000, max_new_tokens=4096, max_prompt_length=1024, per_device_batch=1, grad_accum=8, epochs=1, seed=42 |
| Online eval | Eval-during-training? | No — manual post-training run of `scripts/eval_local.py`. W&B reward variance + KL trajectory are the in-flight diagnostics. |

**Files implemented.**

- `scripts/reward_fn.py` — `compute_reward(generation, gold) -> float`,
  delegating to `evaluate.is_equiv`. Stays TRL-agnostic so it's
  testable on the user's laptop.
- `data/prepare_rlvr.py` — D3 prompt curation. Loads Stage 1 train
  JSONL, runs n=8 rollouts at temp=0.8 via vLLM against the merged SFT
  checkpoint, computes empirical solve rate, filters to
  `[0.2, 0.8]`, writes `{prompt, answer, solve_rate}` JSONL.
- `scripts/train_rlvr.py` — TRL `GRPOTrainer` driver. Loads base +
  SFT adapter trainable (D2). P1 smoke + P2 reward-variance + P3 KL-
  spike preflights. Saves trained adapter to `<output-dir>/final/`,
  byte-compatible with `merge_and_push.py`.
- `rcp/submit_rlvr.sh` — RCP submission script. Mirrors
  `submit_train.sh`. New env vars: `ADAPTER_DIR`, `PROMPT_SET`,
  `SFT_MODEL`, `MAX_PROMPTS`, `LEARNING_RATE`, `KL_COEF`,
  `ROLLOUT_TEMP`, `SKIP_CURATION`, `SKIP_PREFLIGHTS`.

**Critical preflights (enforced in `scripts/train_rlvr.py`).**

- **P1.** Starting adapter must produce well-formed output (`<think>`,
  `\boxed{}`). Smoke run before training. Abort on regression.
- **P2.** Per-prompt reward variance must clear `0.01` on a 10-prompt
  × 8-rollout sample. Without variance, GRPO advantage `(r-mean)/std`
  is 0/0 and training is silent garbage. BASELINE.md flagged this as
  a real risk for our checkpoint at low temperatures. Threshold and
  preflight prompt count are CLI-overridable.
- **P3.** KL divergence callback warns (does not abort) when KL > 0.5
  in the first 100 optimizer steps — Dang & Ngo 2025 small-model
  instability signal.

**Tests (CPU-only, full suite passes in <1s on the user's laptop):**
- `scripts/tests/test_reward_fn.py` — 10 tests covering the 4 input
  combinations + `is_equiv` corner cases (`\frac{1}{2}` ↔ `0.5`, unit
  stripping, last-box-wins).
- `data/tests/test_prepare_rlvr.py` — 25 tests on `difficulty_filter`,
  `extract_prompt_and_gold`, schema validation, JSONL round-trip,
  CLI defaults.
- `scripts/tests/test_train_rlvr.py` — 25 tests on D5 defaults,
  prompt-set loading, P2 variance check (incl. boundary), P3 KL spike
  helper (incl. window/threshold edges), `grpo_config_kwargs`,
  `validate_max_new_tokens`.
- `rcp/tests/test_submit_rlvr.py` — 11 tests on placeholder
  rejection, dry-run pipeline composition, `SKIP_CURATION` /
  `SKIP_PREFLIGHTS` env-var routing, token masking.

**v0 → v1 transition criterion.** First RCP run lands cleanly through
P1+P2 preflights, training completes without KL spike alerts, and the
post-train smoke output still emits `<think>`+`\boxed{}`. Eval the
saved adapter via Stage 4 against `validation_samples/math.jsonl`
(default snapshot) AND `data_out/eval.jsonl` (lower-variance DART
held-out slice). RLVR adds value when **CI-mode pass@8 on
validation_samples** rises above the post-Stage-5 SFT pass@8 (0.30
under the legacy 20480/16384 caps; the CI-faithful re-baseline is
pending and may be lower).

**Done when.**
- v0 implementation passes the CPU test suite (✅ 2026-05-09).
- A first RCP dry-run completes the curation pass and produces a
  non-empty `rlvr_prompts.jsonl` (operational; not done yet).
- A first RCP training run clears P1+P2 preflights and saves a
  `final/` adapter without KL spike alerts (operational; not done yet).

**Uncertainty disclosure.** RLVR on small models is genuinely fragile.
The conservative defaults (low LR, modest prompt set, KL floor at
0.04) are a starting point, not a recipe for guaranteed gains. The
user should expect to iterate on hyperparameters multiple times before
a stable run lands. v0 explicitly does NOT include: SymPy/hybrid
verifier (still v2 stretch), DPO, multi-stage RL curriculum, reward
model training. Eval-during-training is also out of scope at v0.

---

## v2 stretch goals (only if time permits)

These are explicitly NOT in the May 24 → June 7 plan. They become real
options only if the main path lands ahead of schedule.

- **SymPy / hybrid verifier.** Replace exact-match with regex →
  normalize → SymPy equivalence → LLM-judge fallback. Recovers ~15% of
  unjustly rejected correct answers per Huang et al. 2025. Worth ~1–2
  days of work; only useful if exact-match is observed to be the bottleneck.
- **DART-style difficulty re-filtering.** Use the SFT checkpoint to score
  candidate problems, keep only the 20–60% pass@1 band for RLVR. Becomes
  relevant only if doing RLVR in Stage 7.
- **Generation config sweep.** Run a 50-sample temperature sweep
  (0.1 / 0.3 / 0.6) on the clean eval before committing the final
  generation_config to the checkpoint pushed for June 7.
- **Assistant-only loss masking.** Add `{% generation %}` markers to
  `chat_template/chat_template.jinja` so TRL's `SFTTrainer` can mask
  non-assistant tokens out of the loss. Currently disabled because the
  locked template lacks the markers (TRL 0.21+ refuses to auto-patch).
  Requires a coordinated update to `emainelpe-shared` and re-verification
  on all four experts. Marginal expected win on math accuracy; worth
  ~1 day. If taken on, also flip `assistant_only_loss=True` in
  `scripts/train_sft.py:sft_config_kwargs` and update the coupled
  unit test in `scripts/tests/test_train_sft_io.py`.

---

## Status — update at the end of each session

```
Stage 0 — Repo skeleton:                 DONE (2026-05-05)
Stage 1 — Data preparation:              DONE (2026-05-07)
Stage 2 — Chat-template verification:    DONE (2026-05-07)
Stage 3 — SFT training script:           DONE (2026-05-07)
Stage 4 — Local eval:                    DONE (2026-05-07)
Stage 5 — Merge and push:                NOT STARTED
Stage 6 — RCP submission script:         DONE (2026-05-08)
Stage 7 — RLVR:                          V0 IMPLEMENTED 2026-05-09 (NOT YET TRAINED)
```
