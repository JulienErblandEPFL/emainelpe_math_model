"""Diagnostic eval for the v3 SFT checkpoint.

Characterizes v3's failure modes across three eval targets so the team can
design a targeted v4 dataset to fix coverage gaps. NOT a CI-faithful
headline score — that's ``scripts/eval_local.py``'s job. This script's
job is to surface *where* v3 breaks (per subject, per level, per failure
mode) at scale.

Eval targets (run all three by default):

  1. ``validation_samples/math.jsonl`` (OOD, N=10, n=8). Pipeline
     sanity check before committing to longer runs.
  2. ``/scratch/Julien/data_out_v3/eval.jsonl`` (in-distribution DART
     held-out, N=500, n=4).
  3. HuggingFace MATH test set (~5000 problems, subject-tagged, n=4).
     Tries ``HuggingFaceH4/MATH-500`` first, falls back to
     ``hendrycks/competition_math``.

Scoring is byte-identical to the nightly CI: the vendored ``evaluate/``
package (``extract_benchmark_answer`` + ``is_correct_benchmark_answer``
with ``method="boxed"``) is the single source of truth for both answer
extraction and equivalence checking.

Failure-mode classification (priority order — first matching rule wins):

  1. ``repetition``     — any 100-char substring repeats ≥3 times.
                          Priority 1 (above ``correct``) because a looping
                          completion is broken generation regardless of
                          whether its boxed payload happens to be right;
                          we want the pathology surfaced in the breakdown.
  2. ``correct``        — extracted box passes is_equiv against gold
  3. ``no_box``         — no \\boxed{...} anywhere
  4. ``truncated``      — token length ≥ 4090 AND no \\boxed in last
                          800 chars (proxy for "no box in last 200 tokens")
  5. ``wrong_box``      — has \\boxed but is_equiv returns False
  6. ``other``          — anything that falls through (should be rare)

Pure helpers (``classify_failure_mode``, ``detect_repetition``,
``aggregate_per_problem``, ``aggregate_target_summary``, the loaders) live
at module scope and are CPU-testable. Heavy imports (``vllm``,
``transformers``, ``datasets``) are deferred into runtime helpers.

Resumability: each target writes a ``completed.marker`` empty file when
done. Reruns skip completed targets unless ``--force`` is passed.

Typical cluster invocation (after ``git pull`` on RCP):

    # smoke (validation only, 2 problems)
    python scripts/diagnose_v3.py --target validation --limit 2

    # full run (all three targets, ~6h wall-clock)
    python scripts/diagnose_v3.py --target all
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Pure-Python imports (cheap; safe at module scope, kept out of vllm path).
from evaluate.benchmarks import (
    extract_benchmark_answer,
    is_correct_benchmark_answer,
)
from evaluate.extract_answer import extract_boxed_answer
from evaluate.pass_at_k import pass_at_k

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants (eval contract — see CLAUDE.md "Eval contract" + Stage 4)
# -----------------------------------------------------------------------------

# CI-faithful caps (same as scripts/eval_local.py default).
MAX_MODEL_LEN = 4096
MAX_NEW_TOKENS = 4096
# Truncation threshold: within 6 of cap (token-level). Tokens >= this AND no
# box near the end of the completion → truncated. See module docstring.
TRUNCATION_TOKEN_THRESHOLD = 4090
# Character window at the end of the completion for the "no box in last 200
# tokens" proxy. ~4 chars/token → 800 chars ≈ 200 tokens for typical math text.
TRUNCATION_NO_BOX_TAIL_CHARS = 800

# Sampling (v3's pinned generation_config.json values, locked 2026-05-11).
TEMPERATURE = 0.4
TOP_P = 0.95
TOP_K = 20
SEED = 42

# Repetition detection (character-based for laptop testability).
REPETITION_WINDOW_CHARS = 100
REPETITION_MIN_COUNT = 3

# MATH dataset canonical subjects (HuggingFaceH4/MATH-500 and
# hendrycks/competition_math both use these — match exactly).
MATH_SUBJECTS: tuple[str, ...] = (
    "Algebra",
    "Counting & Probability",
    "Geometry",
    "Intermediate Algebra",
    "Number Theory",
    "Prealgebra",
    "Precalculus",
)
MATH_LEVELS: tuple[str, ...] = (
    "Level 1", "Level 2", "Level 3", "Level 4", "Level 5",
)

# Per-target completion counts.
N_COMPLETIONS_VALIDATION = 8
N_COMPLETIONS_INDIST = 4
N_COMPLETIONS_MATH_TEST = 4

# Failure-mode labels.
FM_CORRECT = "correct"
FM_REPETITION = "repetition"
FM_NO_BOX = "no_box"
FM_TRUNCATED = "truncated"
FM_WRONG_BOX = "wrong_box"
FM_OTHER = "other"
ALL_FAILURE_MODES: tuple[str, ...] = (
    FM_NO_BOX, FM_WRONG_BOX, FM_TRUNCATED, FM_REPETITION, FM_OTHER,
)
# Order for per-problem JSON `failure_modes` dict.
PER_PROBLEM_FM_KEYS: tuple[str, ...] = (
    "no_box", "wrong_box", "truncated", "repetition", "other",
)

DEFAULT_MODEL = "/scratch/Julien/merged/math_model_v3"
DEFAULT_INDIST_PATH = "/scratch/Julien/data_out_v3/eval.jsonl"
DEFAULT_OUTPUT_ROOT = "/scratch/Julien/diagnostics"
DEFAULT_VALIDATION_PATH = str(REPO_ROOT / "validation_samples" / "math.jsonl")

ALL_TARGETS: tuple[str, ...] = ("validation", "indist", "math_test")


# =============================================================================
# Pure helpers — CPU-testable; no vllm / transformers / datasets imports.
# =============================================================================

def detect_repetition(
    text: str,
    window: int = REPETITION_WINDOW_CHARS,
    min_count: int = REPETITION_MIN_COUNT,
) -> bool:
    """Return True if any ``window``-char substring of ``text`` occurs at
    least ``min_count`` times.

    Sliding-window scan with a Counter. O(len(text)) time and memory.
    Empty or short strings return False trivially.
    """
    if len(text) < window * min_count:
        return False
    counts: Counter[str] = Counter()
    for i in range(len(text) - window + 1):
        substr = text[i : i + window]
        counts[substr] += 1
        if counts[substr] >= min_count:
            return True
    return False


def classify_failure_mode(
    completion_text: str,
    gold: str,
    *,
    completion_token_len: int,
) -> tuple[str, str | None]:
    """Classify one completion against gold; return (label, extracted_answer).

    Priority order (first match wins) — see module docstring.
    The extracted_answer is None when no \\boxed{...} was found.
    """
    extracted = extract_benchmark_answer(completion_text, "boxed", gold)

    # 1. repetition (highest priority — catches looping even when the loop's
    # box payload happens to evaluate correct. We want the generation
    # pathology surfaced in the failure-mode breakdown so we can design v4
    # data to fix it; a lucky correct-box inside a loop should not mask the
    # underlying brokenness).
    if detect_repetition(completion_text):
        return FM_REPETITION, extracted

    # 2. correct
    if extracted is not None and is_correct_benchmark_answer(
        extracted, gold, "boxed"
    ):
        return FM_CORRECT, extracted

    # 3. no_box
    if extracted is None:
        return FM_NO_BOX, None

    # 4. truncated (has a box, but it's not near the end and total length is
    # at the cap → model was still reasoning when cut off).
    if completion_token_len >= TRUNCATION_TOKEN_THRESHOLD and (
        "\\boxed" not in completion_text[-TRUNCATION_NO_BOX_TAIL_CHARS:]
        and "\\fbox" not in completion_text[-TRUNCATION_NO_BOX_TAIL_CHARS:]
    ):
        return FM_TRUNCATED, extracted

    # 5. wrong_box (has a box that didn't pass is_equiv).
    return FM_WRONG_BOX, extracted


def aggregate_per_problem(
    *,
    problem_id: str,
    target: str,
    subject: str | None,
    level: str | None,
    problem: str,
    gold_answer: str,
    per_completion_rows: list[dict],
) -> dict:
    """Build one ``per_problem.jsonl`` row from its per-completion rows.

    ``per_completion_rows`` are the rows for this problem only; the caller is
    responsible for slicing. Counts and solve_rate are derived; failure_modes
    is the breakdown over failed (non-``correct``) completions per the
    PER_PROBLEM_FM_KEYS ordering.
    """
    n = len(per_completion_rows)
    n_correct = sum(1 for r in per_completion_rows if r["is_correct"])
    fm_counts = {k: 0 for k in PER_PROBLEM_FM_KEYS}
    for r in per_completion_rows:
        mode = r["failure_mode"]
        if mode == FM_CORRECT:
            continue
        if mode in fm_counts:
            fm_counts[mode] += 1
        else:
            fm_counts["other"] += 1
    return {
        "problem_id": problem_id,
        "target": target,
        "subject": subject,
        "level": level,
        "problem": problem,
        "gold_answer": gold_answer,
        "n_completions": n,
        "n_correct": n_correct,
        "solve_rate": (n_correct / n) if n > 0 else 0.0,
        "failure_modes": fm_counts,
    }


def aggregate_target_summary(
    target: str,
    per_problem_rows: list[dict],
    per_completion_rows: list[dict],
    *,
    n_completions: int,
) -> dict:
    """Build the per-target summary.json dict.

    Always emits pass@1 and pass@{n_completions}. For target='math_test'
    additionally emits per-subject and per-level pass@1/pass@k breakdowns
    and failure-mode tables.
    """
    n = n_completions
    n_problems = len(per_problem_rows)

    if n_problems == 0:
        return {
            "target": target,
            "n_problems": 0,
            "n_completions": n,
            "metrics": {"pass@1": 0.0, f"pass@{n}": 0.0},
            "failure_mode_distribution": {k: 0 for k in ALL_FAILURE_MODES},
        }

    pass_1 = sum(pass_at_k(n, r["n_correct"], 1) for r in per_problem_rows) / n_problems
    pass_k = sum(pass_at_k(n, r["n_correct"], n) for r in per_problem_rows) / n_problems

    # Failure-mode distribution over all FAILED completions (correct excluded).
    fm_dist = Counter(
        r["failure_mode"] for r in per_completion_rows if not r["is_correct"]
    )
    # Ensure every label is keyed (zero if absent).
    fm_dist_full = {k: int(fm_dist.get(k, 0)) for k in ALL_FAILURE_MODES}

    summary: dict[str, Any] = {
        "target": target,
        "n_problems": n_problems,
        "n_completions": n,
        "metrics": {"pass@1": float(pass_1), f"pass@{n}": float(pass_k)},
        "failure_mode_distribution": fm_dist_full,
    }

    if target == "math_test":
        summary["per_subject"] = _aggregate_by_field(
            per_problem_rows, per_completion_rows, "subject", n,
        )
        summary["per_level"] = _aggregate_by_field(
            per_problem_rows, per_completion_rows, "level", n,
        )

    return summary


def _aggregate_by_field(
    per_problem_rows: list[dict],
    per_completion_rows: list[dict],
    field: str,
    n: int,
) -> dict:
    """Group rows by ``field`` (e.g., 'subject', 'level') and compute
    pass@1, pass@k, and failure-mode breakdown per group."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in per_problem_rows:
        key = r.get(field) or "Unknown"
        groups[key].append(r)

    completion_groups: dict[str, list[dict]] = defaultdict(list)
    pid_to_field = {r["problem_id"]: (r.get(field) or "Unknown") for r in per_problem_rows}
    for c in per_completion_rows:
        key = pid_to_field.get(c["problem_id"], "Unknown")
        completion_groups[key].append(c)

    out: dict[str, dict] = {}
    for key, rows in groups.items():
        n_p = len(rows)
        if n_p == 0:
            continue
        p1 = sum(pass_at_k(n, r["n_correct"], 1) for r in rows) / n_p
        pk = sum(pass_at_k(n, r["n_correct"], n) for r in rows) / n_p
        fm = Counter(
            c["failure_mode"]
            for c in completion_groups.get(key, [])
            if not c["is_correct"]
        )
        out[key] = {
            "n_problems": n_p,
            "pass@1": float(p1),
            f"pass@{n}": float(pk),
            "failure_modes": {k: int(fm.get(k, 0)) for k in ALL_FAILURE_MODES},
        }
    return out


# -----------------------------------------------------------------------------
# Loaders — pure where possible. The HF loader is gated behind a runtime
# import; the wrapping logic is testable with mocks.
# -----------------------------------------------------------------------------

def load_validation_problems(path: Path) -> list[dict]:
    """Load validation_samples/math.jsonl. Schema: {prompt, answer} per line.

    Returns rows shaped: {problem_id, problem, gold_answer,
                          subject=None, level=None}. ``problem_id`` is
    sequential across kept rows (blank lines do not advance the counter).
    """
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rows.append({
                "problem_id": f"validation_{idx}",
                "problem": str(raw["prompt"]),
                "gold_answer": str(raw["answer"]),
                "subject": None,
                "level": None,
            })
            idx += 1
    return rows


def load_indist_problems(path: Path) -> list[dict]:
    """Load /scratch/Julien/data_out_v3/eval.jsonl. Messages schema.

    Each row: {"messages": [user_msg, assistant_msg]}. Extracts the user
    content as the problem and the LAST \\boxed{...} from the assistant
    content as the gold answer.

    Rows whose assistant content has no extractable box are dropped with a
    WARNING.
    """
    rows: list[dict] = []
    n_dropped = 0
    with open(path, encoding="utf-8") as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            messages = raw.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                n_dropped += 1
                continue
            user_msg, asst_msg = messages[0], messages[1]
            if user_msg.get("role") != "user" or asst_msg.get("role") != "assistant":
                n_dropped += 1
                continue
            gold = extract_boxed_answer(
                str(asst_msg["content"]), strip_double_curly_brace=True
            )
            if gold is None:
                n_dropped += 1
                continue
            rows.append({
                "problem_id": f"indist_{idx}",
                "problem": str(user_msg["content"]),
                "gold_answer": str(gold),
                "subject": None,
                "level": None,
            })
            idx += 1
    if n_dropped:
        logger.warning(
            "indist loader dropped %d malformed rows (no extractable gold)",
            n_dropped,
        )
    return rows


# Schema-assertion contract: every row of a usable MATH-test dataset MUST
# carry at least one alternative from each of these field groups. Listed as
# a labeled dict so the error message can name the missing semantic field,
# not just the raw key.
MATH_TEST_SCHEMA_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "problem": ("problem",),
    "gold (one of)": ("solution", "answer"),
    "subject (one of)": ("subject", "type"),
}


def _assert_math_test_schema(dataset_path: str, sample_row: dict) -> None:
    """Fail-fast schema check for a HF MATH-test dataset row.

    Verifies the row has at least one alternative from each required field
    group (see ``MATH_TEST_SCHEMA_REQUIREMENTS``). Raises ``RuntimeError``
    with a precise diff if any group is missing — prevents a fallback HF
    path with a different schema from silently producing garbage downstream.
    """
    present_keys = set(sample_row.keys())
    missing = [
        label
        for label, alts in MATH_TEST_SCHEMA_REQUIREMENTS.items()
        if not any(a in present_keys for a in alts)
    ]
    if missing:
        raise RuntimeError(
            f"MATH test dataset at {dataset_path} has unexpected schema. "
            f"Expected fields: {dict(MATH_TEST_SCHEMA_REQUIREMENTS)}. "
            f"Found fields: {sorted(present_keys)}. "
            f"Missing: {missing}."
        )


def normalize_math_test_row(raw: dict, idx: int) -> dict | None:
    """Pure-Python normalizer for one HF MATH-test row.

    Handles both ``HuggingFaceH4/MATH-500`` schema (problem, solution,
    answer, subject, level — int 1..5) and ``hendrycks/competition_math``
    schema (problem, solution, type, level — string "Level X").

    Returns the unified row dict or None if the row can't be normalized
    (no problem text, no extractable gold).
    """
    problem = raw.get("problem")
    if not problem:
        return None

    # Gold: prefer 'answer' field if present, else extract from 'solution'.
    gold = raw.get("answer")
    if gold is None:
        sol = raw.get("solution")
        if not sol:
            return None
        gold = extract_boxed_answer(str(sol), strip_double_curly_brace=True)
        if gold is None:
            return None

    # Subject: 'subject' (HF MATH-500) or 'type' (competition_math).
    subject = raw.get("subject") or raw.get("type")
    if subject and subject not in MATH_SUBJECTS:
        # Some forks use 'Counting and Probability' vs '& Probability'.
        # Normalize the common variant explicitly.
        if subject == "Counting and Probability":
            subject = "Counting & Probability"

    # Level: int 1..5 or "Level X" string.
    level_raw = raw.get("level")
    level: str | None
    if isinstance(level_raw, int):
        level = f"Level {level_raw}"
    elif isinstance(level_raw, str) and level_raw:
        level = level_raw if level_raw.startswith("Level ") else f"Level {level_raw}"
    else:
        level = None

    return {
        "problem_id": f"math_test_{idx}",
        "problem": str(problem),
        "gold_answer": str(gold),
        "subject": subject,
        "level": level,
    }


# -----------------------------------------------------------------------------
# Resumability + I/O helpers (pure-ish — file system only).
# -----------------------------------------------------------------------------

def target_is_complete(target_dir: Path) -> bool:
    """Return True iff ``target_dir/completed.marker`` exists."""
    return (target_dir / "completed.marker").exists()


def write_completed_marker(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "completed.marker").write_text("", encoding="utf-8")


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Stdout summary formatter (pure — testable).
# -----------------------------------------------------------------------------

def format_target_block(target: str, summary: dict, *, per_problem_rows: list[dict] | None = None) -> str:
    """Render one target's summary block in the spec's canonical format."""
    n = summary["n_completions"]
    p1 = summary["metrics"].get("pass@1", 0.0)
    pk = summary["metrics"].get(f"pass@{n}", 0.0)
    fm = summary["failure_mode_distribution"]
    fm_total = sum(fm.values())
    dominant = max(fm, key=lambda k: fm[k]) if fm_total > 0 else "(none)"

    if target == "validation":
        rows = per_problem_rows or []
        never = [r["problem_id"].split("_")[-1] for r in rows if r["n_correct"] == 0]
        always = [r["problem_id"].split("_")[-1] for r in rows if r["n_correct"] == r["n_completions"]]
        return (
            f"Validation (N={summary['n_problems']}, n={n}):\n"
            f"  pass@1: {p1:.3f}\n"
            f"  pass@{n}: {pk:.3f}\n"
            f"  Never-solved (0/{n}): {never}\n"
            f"  Always-solved ({n}/{n}): {always}\n"
            f"  Dominant failure mode: {dominant}\n"
        )

    if target == "indist":
        return (
            f"In-distribution (N={summary['n_problems']}, n={n}):\n"
            f"  pass@1: {p1:.3f}\n"
            f"  pass@{n}: {pk:.3f}\n"
            f"  Failure mode breakdown: {dict(fm)}\n"
        )

    if target == "math_test":
        per_subject = summary.get("per_subject", {})
        per_level = summary.get("per_level", {})

        # Build subject lines, weakest 2 by pass@1.
        subj_lines = []
        for s in MATH_SUBJECTS:
            entry = per_subject.get(s)
            if entry:
                subj_lines.append(
                    f"    {s + ':':<26}{entry['pass@1']:.3f} (n={entry['n_problems']})"
                )
            else:
                subj_lines.append(f"    {s + ':':<26}— (n=0)")

        ranked = sorted(
            ((s, per_subject[s]["pass@1"]) for s in per_subject),
            key=lambda x: x[1],
        )
        weakest = [s for s, _ in ranked[:2]]

        level_parts = []
        for lv in MATH_LEVELS:
            entry = per_level.get(lv)
            level_parts.append(f"{lv}: {entry['pass@1']:.3f}" if entry else f"{lv}: —")
        level_line = ", ".join(level_parts)

        subj_dom = {
            s: max(per_subject[s]["failure_modes"], key=lambda k: per_subject[s]["failure_modes"][k])
            if sum(per_subject[s]["failure_modes"].values()) > 0 else "(none)"
            for s in per_subject
        }
        dom_table = "\n".join(
            f"    {s + ':':<26}{subj_dom[s]}" for s in MATH_SUBJECTS if s in subj_dom
        )

        return (
            f"MATH test (full, n={n}):\n"
            f"  pass@1: {p1:.3f}\n"
            f"  pass@{n}: {pk:.3f}\n"
            f"  Per-subject pass@1:\n" + "\n".join(subj_lines) + "\n"
            f"  Per-level pass@1:\n    {level_line}\n"
            f"  Weakest 2 subjects: {', '.join(weakest) if weakest else '(none)'}\n"
            f"  Per-subject dominant failure mode:\n{dom_table}\n"
        )

    return f"{target}: (unknown target)\n"


def format_full_summary(
    summaries: dict[str, dict],
    per_problem_by_target: dict[str, list[dict]],
) -> str:
    parts = ["=== v3 DIAGNOSTIC SUMMARY ===\n"]
    if "validation" in summaries:
        parts.append(format_target_block(
            "validation", summaries["validation"],
            per_problem_rows=per_problem_by_target.get("validation"),
        ))
    if "indist" in summaries:
        parts.append(format_target_block("indist", summaries["indist"]))
    if "math_test" in summaries:
        parts.append(format_target_block("math_test", summaries["math_test"]))
    return "\n".join(parts)


# =============================================================================
# Runtime helpers — heavy imports inside.
# =============================================================================

def _hf_load_math_test() -> list[dict]:
    """Load the MATH test split from HuggingFace, normalized.

    Tries datasets in order, falling back when one is gated:
      1. HuggingFaceH4/MATH-500     (smaller, faster)
      2. hendrycks/competition_math (full ~5000)
      3. lighteval/MATH             (last resort)
    """
    from datasets import load_dataset  # type: ignore

    candidates = [
        ("HuggingFaceH4/MATH-500", "test"),
        ("hendrycks/competition_math", "test"),
        ("lighteval/MATH", "test"),
    ]
    last_err: Exception | None = None
    loaded_path: str | None = None
    for name, split in candidates:
        try:
            ds = load_dataset(name, split=split)
            loaded_path = name
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("failed to load %s: %s", name, e)
    else:
        raise RuntimeError(
            f"could not load any MATH test dataset; last error: {last_err}"
        )

    logger.info(
        "Loaded math_test from %s (n=%d, split=test)", loaded_path, len(ds),
    )

    # Fail-fast schema check: if a fallback path returns a different schema
    # we want a clear error here, not garbage downstream.
    if len(ds) > 0:
        _assert_math_test_schema(loaded_path, dict(ds[0]))

    out: list[dict] = []
    for i, raw in enumerate(ds):
        norm = normalize_math_test_row(dict(raw), i)
        if norm is not None:
            out.append(norm)
    logger.info("normalized %d/%d MATH test rows", len(out), len(ds))
    return out


def _load_tokenizer_from_model_dir(model_path: str):
    """Load tokenizer using whatever chat_template ships in the model dir.

    For ``/scratch/Julien/merged/math_model_v3`` this picks up the locked
    Jinja that was baked in at Stage 5 merge time.
    """
    from transformers import AutoTokenizer  # type: ignore

    return AutoTokenizer.from_pretrained(model_path)


def _self_check_chat_template(model_path: str, sample_prompt: str) -> None:
    """Verify the model's bundled chat template renders byte-identically to
    the team-locked Jinja at chat_template/chat_template.jinja.

    Fail-fast: raises RuntimeError on mismatch (with a short diff snippet).
    Catches the bug class where v3 was pushed with a stale chat template.
    """
    from scripts.eval_local import (
        load_tokenizer_with_locked_template,
        render_prompts,
        DEFAULT_CHAT_TEMPLATE,
    )

    # A: model dir's bundled template (whatever Stage 5 pushed).
    tok_a = _load_tokenizer_from_model_dir(model_path)
    # B: team-locked Jinja (the same path eval_local.py uses).
    tok_b = load_tokenizer_with_locked_template(model_path, DEFAULT_CHAT_TEMPLATE)

    items = [{"prompt": sample_prompt, "answer": ""}]
    rendered_a = render_prompts(tok_a, items)[0]
    rendered_b = render_prompts(tok_b, items)[0]
    if rendered_a != rendered_b:
        # Tight diff snippet (first ~120 chars around mismatch).
        for i, (ca, cb) in enumerate(zip(rendered_a, rendered_b)):
            if ca != cb:
                lo = max(0, i - 40)
                hi = i + 80
                raise RuntimeError(
                    "chat-template self-check FAILED: model-bundled template "
                    "differs from team-locked Jinja. Investigate stale chat "
                    f"template in {model_path}.\n"
                    f"first mismatch at char {i}:\n"
                    f"  model-dir : {rendered_a[lo:hi]!r}\n"
                    f"  locked    : {rendered_b[lo:hi]!r}"
                )
        # Length mismatch but common prefix matched.
        raise RuntimeError(
            "chat-template self-check FAILED: length mismatch "
            f"(model-dir={len(rendered_a)}, locked={len(rendered_b)})"
        )
    logger.info("chat-template self-check OK (rendered length=%d)", len(rendered_a))


def _build_llm(model_path: str, gpu_memory_utilization: float):
    """Build the vLLM ``LLM`` once; reused across all targets."""
    from vllm import LLM  # type: ignore

    return LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def _generate(
    llm,
    tokenizer,
    problems: list[dict],
    *,
    n: int,
) -> list[list[dict]]:
    """Render prompts via the locked template, run vLLM, return per-problem
    lists of completion records: [{"text": str, "n_tokens": int}, ...].

    n_tokens is the actual completion-token count from vLLM (used by the
    truncation rule in classify_failure_mode).
    """
    from vllm import SamplingParams  # type: ignore

    # Re-use eval_local.render_prompts so the prompt formatting is byte-identical.
    from scripts.eval_local import render_prompts

    items = [{"prompt": p["problem"], "answer": p["gold_answer"]} for p in problems]
    prompts = render_prompts(tokenizer, items)

    params = SamplingParams(
        n=n,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_NEW_TOKENS,
        seed=SEED,
    )
    outputs = llm.generate(prompts, params)
    result: list[list[dict]] = []
    for out in outputs:
        comps: list[dict] = []
        for co in out.outputs:
            n_tokens = len(co.token_ids) if hasattr(co, "token_ids") and co.token_ids is not None else 0
            comps.append({"text": co.text, "n_tokens": n_tokens})
        result.append(comps)
    return result


def _run_target(
    *,
    target: str,
    problems: list[dict],
    llm,
    tokenizer,
    n_completions: int,
    target_dir: Path,
) -> dict:
    """Generate, classify, aggregate, and persist for one target. Returns
    the target's summary dict."""
    logger.info(
        "target=%s: generating %d completions × %d problems",
        target, n_completions, len(problems),
    )
    completions = _generate(llm, tokenizer, problems, n=n_completions)

    per_problem_rows: list[dict] = []
    per_completion_rows: list[dict] = []
    for problem, comps in zip(problems, completions):
        problem_completion_rows: list[dict] = []
        for ci, comp in enumerate(comps):
            label, extracted = classify_failure_mode(
                comp["text"], problem["gold_answer"],
                completion_token_len=comp["n_tokens"],
            )
            row = {
                "problem_id": problem["problem_id"],
                "completion_idx": ci,
                "completion_text": comp["text"],
                "extracted_answer": extracted,
                "is_correct": label == FM_CORRECT,
                "failure_mode": label,
            }
            problem_completion_rows.append(row)
            per_completion_rows.append(row)
        per_problem_rows.append(aggregate_per_problem(
            problem_id=problem["problem_id"],
            target=target,
            subject=problem.get("subject"),
            level=problem.get("level"),
            problem=problem["problem"],
            gold_answer=problem["gold_answer"],
            per_completion_rows=problem_completion_rows,
        ))

    write_jsonl(per_problem_rows, target_dir / "per_problem.jsonl")
    write_jsonl(per_completion_rows, target_dir / "per_completion.jsonl")
    summary = aggregate_target_summary(
        target, per_problem_rows, per_completion_rows,
        n_completions=n_completions,
    )
    write_json(summary, target_dir / "summary.json")
    write_completed_marker(target_dir)

    # Stash per_problem alongside summary for the stdout block.
    summary["_per_problem_rows"] = per_problem_rows  # consumed by format_full_summary
    return summary


# =============================================================================
# CLI / main
# =============================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Path to merged checkpoint. Default: {DEFAULT_MODEL}",
    )
    p.add_argument(
        "--target",
        choices=(*ALL_TARGETS, "all"),
        default="all",
        help="Which target(s) to run. 'all' runs validation → indist → math_test.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Limit problems per target (debug; default: no limit).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "Output directory. Default: "
            f"{DEFAULT_OUTPUT_ROOT}/v3_eval_<utc-timestamp>"
        ),
    )
    p.add_argument(
        "--validation-file", type=Path, default=Path(DEFAULT_VALIDATION_PATH),
        help=f"validation_samples path. Default: {DEFAULT_VALIDATION_PATH}",
    )
    p.add_argument(
        "--indist-file", type=Path, default=Path(DEFAULT_INDIST_PATH),
        help=f"in-distribution eval path. Default: {DEFAULT_INDIST_PATH}",
    )
    p.add_argument(
        "--gpu-memory-utilization", type=float, default=0.85,
    )
    p.add_argument(
        "--force", action="store_true",
        help="Ignore completed.marker files and re-run completed targets.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def _default_output_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(DEFAULT_OUTPUT_ROOT) / f"v3_eval_{ts}"


def _resolve_targets(arg: str) -> list[str]:
    if arg == "all":
        return list(ALL_TARGETS)
    return [arg]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_dir = args.output_dir or _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output_dir: %s", output_dir)

    # HF cache honoring (set by cluster job): nothing for us to do — the
    # `datasets` library reads HF_HOME directly.
    if "HF_HOME" in os.environ:
        logger.info("HF_HOME=%s", os.environ["HF_HOME"])

    targets = _resolve_targets(args.target)

    # Resumability: skip targets that are already complete unless --force.
    to_run: list[str] = []
    for t in targets:
        td = output_dir / t
        if not args.force and target_is_complete(td):
            logger.info("target=%s already complete (skipping). Pass --force to rerun.", t)
        else:
            to_run.append(t)

    if not to_run:
        logger.info("nothing to do; all requested targets already complete.")
        return 0

    # Self-check chat template (uses a tiny problem; cheap).
    _self_check_chat_template(args.model, "What is 2+2?")

    # Tokenizer (shared across targets) — uses the model-dir's bundled
    # template (self-check confirmed it matches the locked Jinja).
    tokenizer = _load_tokenizer_from_model_dir(args.model)

    # vLLM (shared across targets — amortize the 60-90s startup).
    llm = _build_llm(args.model, args.gpu_memory_utilization)

    summaries: dict[str, dict] = {}
    per_problem_by_target: dict[str, list[dict]] = {}

    for t in to_run:
        target_dir = output_dir / t
        if t == "validation":
            problems = load_validation_problems(args.validation_file)
            n_comp = N_COMPLETIONS_VALIDATION
        elif t == "indist":
            problems = load_indist_problems(args.indist_file)
            n_comp = N_COMPLETIONS_INDIST
        else:  # math_test
            problems = _hf_load_math_test()
            n_comp = N_COMPLETIONS_MATH_TEST

        if args.limit is not None:
            problems = problems[: args.limit]
            logger.info("--limit %d applied; running %d problems", args.limit, len(problems))

        summary = _run_target(
            target=t,
            problems=problems,
            llm=llm,
            tokenizer=tokenizer,
            n_completions=n_comp,
            target_dir=target_dir,
        )
        per_problem_by_target[t] = summary.pop("_per_problem_rows")
        summaries[t] = summary

    # Also load completed-but-not-rerun targets' summaries so the final
    # stdout block reflects everything done so far in this output_dir.
    for t in targets:
        if t in summaries:
            continue
        td = output_dir / t
        sp = td / "summary.json"
        if sp.exists():
            summaries[t] = json.loads(sp.read_text(encoding="utf-8"))
            pp = td / "per_problem.jsonl"
            if pp.exists():
                per_problem_by_target[t] = [
                    json.loads(line) for line in pp.read_text(encoding="utf-8").splitlines() if line.strip()
                ]

    summary_text = format_full_summary(summaries, per_problem_by_target)
    (output_dir / "full_summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
