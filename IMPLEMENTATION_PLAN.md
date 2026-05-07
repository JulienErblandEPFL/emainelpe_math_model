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
- `assistant_only_loss=True` (TRL auto-patches the Qwen3 chat template)
- `packing=False` (the `<think>` blocks are long; packing breaks
  assistant-only-loss masks)
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

**Note.** The actual full SFT run is a separate operational task — submit
via `rcp/submit_train.sh`, monitor logs, expect 8-12 hours. That's not
something Claude Code does; that's the user. The script just has to be
correct.

---

## Stage 4 — Local eval (mimicking the CI)

**Goal.** A vLLM-based eval that loads a merged checkpoint, runs n=8
completions per problem, extracts answers via `\boxed{...}`, and reports
pass@1 and pass@8.

**Files to create:**
- `scripts/eval_local.py`

**Key requirements:**
- Input: a JSONL file with `{"prompt": "...", "answer": "..."}` per line
  (matches the course's validation snapshot format)
- Output: pass@1 and pass@8 numbers, plus optional per-question dump
- Use vLLM with bf16, max_model_len=4096
- Apply chat template via `tokenizer.apply_chat_template(..., add_generation_prompt=True)`
- SamplingParams: n=8, max_tokens=4096, temperature and top_p from CLI args
  (defaults: temperature=0.7, top_p=0.95)
- Boxed extraction: the LAST `\boxed{...}` in the completion, with
  brace-balancing for nested expressions
- Answer normalization: strip whitespace, strip trailing periods
- Comparison: exact string match after normalization

**Done when.**
- Running on the course's `validation_samples/math.jsonl` with a known
  base model produces sensible numbers (i.e., not 0% and not 100%)
- The script handles edge cases: completion with no `\boxed{}` (counts as
  wrong), completion with malformed `\boxed{}` (counts as wrong)

**Why this matters.** Local eval is the feedback loop. Without it you're
flying blind between HF pushes (which only get evaluated nightly).

---

## Stage 5 — Merge adapter and push to HF (Phase 1 deliverable)

**Goal.** Take the trained LoRA adapter, merge it into the base model
weights, write `generation_config.json`, and push the full checkpoint to
`cs-552-2026-emainelpe/math_model`.

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
  and posts an `EVAL_REPORT.md` PR

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

## Stage 7 — RLVR (Phase 2) — DEFERRED

**Goal.** Add GRPO training on top of the SFT checkpoint.

**Status when this is written.** Not started. Whether to do this at all
depends on Stage 5 results. From `CLAUDE.md`:
> RLVR'd checkpoint if RLVR helps; SFT fallback if RLVR destabilizes
> (Dang & Ngo 2025 warns this is a real risk on small models).

**Decision criteria — assess after Stage 5 finishes:**
- If SFT pass@8 on a held-out math eval is meaningfully below the
  proposal target → try RLVR
- If SFT pass@8 is already strong → consider skipping RLVR; spend the
  June 7 budget on the merge phase and the report
- If team timeline is tight → SFT-only is acceptable per the proposal

**When this stage is taken on:** open a separate IMPLEMENTATION_PLAN_RLVR.md
and break GRPO into its own stages (verifier, prompt set construction,
GRPOTrainer config, eval-during-training). Do NOT inline RLVR into this
plan; it's a substantial body of work and deserves its own context.

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

---

## Status — update at the end of each session

```
Stage 0 — Repo skeleton:                 DONE (2026-05-05)
Stage 1 — Data preparation:              DONE (2026-05-07)
Stage 2 — Chat-template verification:    DONE (2026-05-07)
Stage 3 — SFT training script:           NOT STARTED
Stage 4 — Local eval:                    NOT STARTED
Stage 5 — Merge and push:                NOT STARTED
Stage 6 — RCP submission script:         NOT STARTED
Stage 7 — RLVR:                          DEFERRED
```
