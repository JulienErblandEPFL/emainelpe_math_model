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
