# math_model

Math expert for the CS-552 Modern NLP final project (EPFL, Spring 2026), Team Émainèlpé.

See `CLAUDE.md` for project context and design decisions, and `IMPLEMENTATION_PLAN.md` for the staged work plan.

## Where the CI contract lives

The course's authoritative description of the CI evaluation pipeline
(roster naming, `max_model_len=4096`, n=8, 1800 s wall-clock cap, the
`EVAL_REPORT.md` PR mechanism) is the **team project README** in the
`emainelpe_group_model` repo (a fork of the `epfl-nlp/cs-552-modern-nlp`
project starter). When the README and `docs/project_description.pdf`
disagree (e.g., the README's `max_model_len=4096` vs the PDF's
`Max new tokens: 16384`), the README is the more recent and binding
source. CLAUDE.md flags the conflict in the "Eval contract" section.

## What to expect from local eval

`scripts/eval_local.py` runs in two modes:

- **Default (CI-faithful)** — `max_model_len=4096`, `max_tokens=4096`.
  Matches the team README's combined-context cap, so local pass@1 /
  pass@8 are calibrated against what CI will report.
- **`--no-ci-mode` (legacy escape hatch)** — `max_model_len=20480`,
  `max_tokens=16384`. Tracks `docs/project_description.pdf` page 3,
  the older course doc. More permissive than CI; numbers measured
  under it *overstate* CI scores. Use only for ablations where the
  longer generation budget matters (e.g., probing whether a longer
  `<think>` chain would have produced a `\boxed{...}`).

## Data prep

`data/prepare_sft.py` produces the JSONL files `scripts/train_sft.py` consumes. Two dataset variants are supported:

### v1 (DART-Math-Uniform only) — default

Unchanged from Stage 1 (2026-05-07). The default invocation reproduces the v1 dataset byte-for-byte:

```bash
python data/prepare_sft.py \
    --output-dir data_out \
    --n-samples 50000 \
    --eval-size 500 \
    --seed 42
```

### v2 (mixed DART + OpenMathInstruct-2)

Added 2026-05-09. Mixes `hkust-nlp/dart-math-uniform` 50/50 with `nvidia/OpenMathInstruct-2` (`train_1M` slice). Motivation: OMI2's solutions come from Llama3.1-405B-Instruct, a substantially stronger teacher than DART's DeepSeekMath-7B-RL — so OMI2 brings stronger CoT while DART contributes diversity and per-problem multi-solution coverage.

```bash
python data/prepare_sft.py --source mixed \
    --output-dir data_out_v2 \
    --train-size 50000 \
    --eval-size 500 \
    --seed 42
```

Locked v2 design decisions (also in `IMPLEMENTATION_PLAN.md` Stage 1 v2):

- **Mix ratio** — 50/50, controllable via `--dart-fraction` (default 0.5). Each source is run through `build_pipeline` independently, then concatenated and shuffled with the same seed.
- **OMI2 boxing strategy** — append `\boxed{expected_answer}` to the cleaned `generated_solution`. The `evaluate.extract_answer` helper takes the LAST `\boxed{}`, so any mid-text box in the model's CoT is preserved as reasoning while the appended one is the gold answer.
- **Per-source caps** — same DART rule (max 4 solutions per unique problem) applied independently to each source, before concatenation.
- **Token-length cap** — drops rows whose formatted Qwen3-tokenized chat exceeds 3500 tokens (set via `--max-formatted-tokens`, auto-defaults to 3500 when `--source` is `openmathinstruct` or `mixed`). Bites mostly on OMI2 because Llama3.1-405B solutions are sometimes verbose; the OpenMathInstruct-2 paper notes "excessive verbosity is detrimental to SFT".
- **Reproducibility** — seed=42 for both subsamples and the post-mix shuffle.

CPU unit tests for both v1 and v2 helpers: `pytest data/tests/test_prepare_sft.py`. The `transformers` import that backs the token filter happens only inside `main()`, so the tests inject a fake tokenize_fn and run in <0.1s without the wheel.

## Stage 5: Merge and push

`scripts/merge_and_push.py` folds the trained LoRA adapter into `Qwen/Qwen3-1.7B`, writes the eval-time `generation_config.json`, runs file + chat-template + vLLM smoke preflights, and (with `--push`) uploads the merged checkpoint to `cs-552-2026-emainelpe/math_model`.

Dry-run (default — does everything except the HF push):

```bash
python scripts/merge_and_push.py \
    --adapter-dir /scratch/Julien/runs/<run-name>/final \
    --output-dir  /scratch/Julien/merged/math_model_v1
```

Push to the team org repo:

```bash
python scripts/merge_and_push.py \
    --adapter-dir /scratch/Julien/runs/<run-name>/final \
    --output-dir  /scratch/Julien/merged/math_model_v1 \
    --push
```

Sampling defaults are `temperature=0.3 / top_p=0.95 / top_k=20` (the BASELINE.md fallback). Override with `--temperature`, `--top-p`, `--top-k` for a future tuning re-push.

CPU-only unit tests for the pure helpers (`build_generation_config`, the chat-template byte diff, the file preflight): `pytest data/tests/test_merge_and_push.py`.
