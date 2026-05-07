# Standalone evaluator

Score your own generations locally with the **same** answer-extraction,
fallback, and equivalence logic the nightly CI uses. Reports `pass@1` and
`pass@8` per benchmark. No vLLM, no HuggingFace datasets — just `numpy`.

Use this to debug why your nightly score disagrees with what you expected, or
to gate checkpoints before pushing.

## Install

```bash
pip install numpy
```

That's the only dependency.

## Input format

One JSON object per line, one row per problem:

```json
{"prompt": "What is 2+2?", "answer": "4", "completions": ["...\\boxed{4}", "...\\boxed{5}", "...\\boxed{4}"]}
```

| Field | Required | Notes |
|---|---|---|
| `answer` | yes | Gold reference. `reference` is accepted as a synonym. |
| `completions` | yes | Non-empty list of model outputs. The list length is the `n` used for pass@k; **all rows must have the same length**. |
| `prompt` | no | Echoed in detailed output if present. |

`pass@8` requires `n >= 8` completions per row. With `n < 8`, only `pass@1` is
reported.

## Producing the input from your model

Append `completions` to each row of the validation snapshot:

```python
import json

with open("validation/math.jsonl") as fin, open("my_math_gens.jsonl", "w") as fout:
    for line in fin:
        row = json.loads(line)
        row["completions"] = my_model.generate(row["prompt"], n=8)  # your inference
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
```

## Run

From the `student-starter/` directory:

```bash
python -m evaluate.score --generations my_math_gens.jsonl              --benchmark math
python -m evaluate.score --generations my_knowledge_gens.jsonl         --benchmark knowledge
python -m evaluate.score --generations my_multilingual_gens.jsonl      --benchmark multilingual
python -m evaluate.score --generations my_safety_gens.jsonl            --benchmark safety
```

Output:

```
pass@1=0.4500, pass@8=0.7200 (n_problems=10, n_completions=8, method=boxed)
```

Add `--output scored.json` to also write per-problem details (which completion
produced which extracted answer, whether each was judged correct).

## What `--benchmark` selects

The flag picks the extraction method that matches `config/benchmarks.yaml` in
the CI repo:

| `--benchmark` | Extraction method | Notes |
|---|---|---|
| `math` | `boxed` | Last `\boxed{...}` only; LaTeX/math equivalence on the inner string. |
| `knowledge` | `knowledge` | Tries `\boxed{...}`, falls back to text. For single-letter gold answers, extracts a choice label (`A`–`Z`). For free-form gold (including JSON arrays of acceptable aliases), normalizes and matches. |
| `multilingual` | `boxed` | Same as `math`. |
| `safety` | `boxed` | Same as `math`. |

If extraction returns `None` (no `\boxed{...}`, no usable letter, etc.), that
completion is judged incorrect.

`--method {boxed|knowledge|exact}` is an escape-hatch override if you want to
score with a different extraction method than the default for the benchmark.

## What this does not do

- It does **not** load the official eval datasets — you score whatever rows
  you pass in. The CI dataset includes a small reproducible subsample
  (`random_exclude_n`) that this tool does not mirror.
- It does **not** run your model — bring your own completions.
- It does **not** report `mean@8` (the public leaderboard uses `pass@1` and
  `pass@8`).
