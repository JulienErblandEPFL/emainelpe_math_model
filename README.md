# math_model

Math expert for the CS-552 Modern NLP final project (EPFL, Spring 2026), Team Émainèlpé.

See `CLAUDE.md` for project context and design decisions, and `IMPLEMENTATION_PLAN.md` for the staged work plan.

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
