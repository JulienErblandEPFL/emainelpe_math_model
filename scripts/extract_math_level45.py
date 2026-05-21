"""Extract MATH-train problems at given difficulty levels to a JSONL.

Pulls problems from ``EleutherAI/hendrycks_math`` across the 7 MATH
subjects, filters by level (default Level 4-5) and subject, extracts the
gold answer from each row's ``solution`` via the last ``\\boxed{...}``,
and writes one ``{prompt, answer, subject, level}`` row per problem.

Designed as a small front-end for downstream pipelines that need a
clean problem set without the full ``data/prepare_sft.py`` build:

  - feed ``scripts/teacher_distill.py --problem-set ...``
  - feed ``scripts/sample_failures.py --prompt-set ...``
  - seed a future v7 SFT mix targeted at the hard MATH levels v5 still
    misses (see ``docs/CLAUDE.md`` → 2026-05-15 → "Scaling progression at
    1.7B": Level 5 lift on v6 was +4.1pp but Counting/Prealgebra
    regressed — Level 4-5 problems remain the open coverage gap).

Pure helpers (``extract_last_boxed``, ``parse_level_filter``,
``normalize_level``, ``keep_row``, ``build_problem_row``,
``write_problems_jsonl``, ``format_summary``) live at module scope and
are CPU-testable. ``datasets`` is imported lazily inside ``main()`` so
laptop unit tests run without that wheel.

CLI::

    python scripts/extract_math_level45.py \\
        --output-file /scratch/Julien/math_train_l45/problems.jsonl \\
        --levels "4,5" \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("extract_math_level45")

DEFAULT_DATASET_NAME = "EleutherAI/hendrycks_math"
DEFAULT_SPLIT = "train"
DEFAULT_LEVELS = "4,5"
DEFAULT_SEED = 42

# Locked from data/prepare_sft.py:MATH_TRAIN_SUBJECTS. EleutherAI/
# hendrycks_math ships each MATH subject as a separate HF config; one
# load_dataset call per subject, then concatenate.
DEFAULT_SUBJECTS: tuple[str, ...] = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


# =============================================================================
# Pure helpers — no `datasets` import. CPU-testable.
# =============================================================================

def extract_last_boxed(text: str) -> str | None:
    """Return the contents of the LAST ``\\boxed{...}`` in ``text``.

    Brace-balanced: ``\\boxed{\\frac{1}{2}}`` → ``\\frac{1}{2}``. Returns
    ``None`` if no balanced ``\\boxed{...}`` is found. A naive regex
    ``\\boxed\\{(.+?)\\}`` would stop at the first ``}`` and yield
    ``\\frac{1`` on the example above — common enough in MATH solutions
    that we need a depth counter.
    """
    if not isinstance(text, str):
        return None
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    i = start + len(marker)
    depth = 1
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + len(marker):i]
        i += 1
    return None


def normalize_level(value: object) -> str | None:
    """Normalize a Hendrycks MATH ``level`` field to ``"Level N"``.

    Accepts ``5``, ``"5"``, ``"Level 5"``. Returns ``None`` for missing
    or "Level ?" rows (the dataset has ~2 of these; spec says to skip).
    """
    if value is None:
        return None
    if isinstance(value, int):
        return f"Level {value}"
    s = str(value).strip()
    if not s or s == "Level ?":
        return None
    if s.startswith("Level "):
        suffix = s[len("Level "):].strip()
        if not suffix or suffix == "?":
            return None
        return s
    if s.isdigit():
        return f"Level {s}"
    return None


def parse_level_filter(spec: str) -> tuple[str, ...]:
    """Parse a comma-separated levels filter like ``"4,5"``.

    Returns a tuple of canonical ``"Level N"`` strings. ``"4,5"`` →
    ``("Level 4", "Level 5")``. Raises ``ValueError`` if any token
    doesn't normalize to a valid level.
    """
    out: list[str] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        norm = normalize_level(tok)
        if norm is None:
            raise ValueError(
                f"--levels token {tok!r} is not a valid MATH level "
                f"(expected e.g. '4', '5', or 'Level 5')"
            )
        out.append(norm)
    if not out:
        raise ValueError("--levels parsed to an empty filter")
    return tuple(out)


def keep_row(
    row: dict,
    *,
    levels_filter: tuple[str, ...],
    subjects_filter: tuple[str, ...] | None,
) -> bool:
    """Filter predicate: matches both --levels and --subjects.

    ``subjects_filter=None`` means "accept any subject". The row's
    subject comes from ``row.get("subject") or row.get("type")``;
    matching is case-insensitive and whitespace-tolerant against the
    subjects_filter slugs (e.g., ``"Intermediate Algebra"`` matches
    ``"intermediate_algebra"``).
    """
    level = normalize_level(row.get("level"))
    if level is None or level not in levels_filter:
        return False
    if subjects_filter is None:
        return True
    raw_subject = row.get("subject") or row.get("type")
    if not isinstance(raw_subject, str):
        return False
    slug = raw_subject.strip().lower().replace(" ", "_").replace("&", "and")
    return slug in subjects_filter


def build_problem_row(raw: dict) -> dict | None:
    """Convert one Hendrycks MATH row to the output schema.

    Returns ``{prompt, answer, subject, level}`` or ``None`` if the row
    lacks a usable problem text, level, or extractable boxed answer.
    Logs a WARNING when an otherwise-valid row has no ``\\boxed{}`` in
    its solution (the only "extraction failure" the spec asks to skip).
    """
    problem = raw.get("problem")
    if not isinstance(problem, str) or not problem.strip():
        return None

    level = normalize_level(raw.get("level"))
    if level is None:
        return None

    raw_subject = raw.get("subject") or raw.get("type")
    if not isinstance(raw_subject, str) or not raw_subject.strip():
        return None

    solution = raw.get("solution")
    if not isinstance(solution, str) or not solution.strip():
        logger.warning(
            "skipping row: empty solution (subject=%s, level=%s)",
            raw_subject, level,
        )
        return None

    answer = extract_last_boxed(solution)
    if answer is None or not answer.strip():
        logger.warning(
            "skipping row: no \\boxed{} in solution (subject=%s, level=%s)",
            raw_subject, level,
        )
        return None

    return {
        "prompt": problem.strip(),
        "answer": answer.strip(),
        "subject": raw_subject.strip(),
        "level": level,
    }


def write_problems_jsonl(rows: list[dict], path: Path) -> None:
    """Write rows to a JSONL file, one row per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")


def format_summary(rows: list[dict]) -> str:
    """Human-readable per-subject + per-level counts."""
    subj_counts: dict[str, int] = {}
    level_counts: dict[str, int] = {}
    for r in rows:
        subj_counts[r["subject"]] = subj_counts.get(r["subject"], 0) + 1
        level_counts[r["level"]] = level_counts.get(r["level"], 0) + 1
    parts = [f"total={len(rows)}"]
    parts.append("by_level={" + ", ".join(
        f"{k!r}: {v}" for k, v in sorted(level_counts.items())
    ) + "}")
    parts.append("by_subject={" + ", ".join(
        f"{k!r}: {v}" for k, v in sorted(subj_counts.items())
    ) + "}")
    return " | ".join(parts)


# =============================================================================
# CLI / main — heavy `datasets` import deferred.
# =============================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract MATH-train problems at given levels to JSONL.",
    )
    p.add_argument(
        "--output-file", type=Path, required=True,
        help="Output JSONL path; one row per problem.",
    )
    p.add_argument(
        "--levels", default=DEFAULT_LEVELS,
        help=(
            f"Comma-separated levels to keep. Default: {DEFAULT_LEVELS!r}. "
            f"Examples: '4,5', '5', '1,2,3'."
        ),
    )
    p.add_argument(
        "--subjects", default=",".join(DEFAULT_SUBJECTS),
        help=(
            "Comma-separated MATH subject slugs to keep. Default: all 7. "
            "Slugs match the EleutherAI/hendrycks_math config names: "
            f"{', '.join(DEFAULT_SUBJECTS)}."
        ),
    )
    p.add_argument(
        "--max-problems", type=int, default=None,
        help="Cap on total problems written. Default: no cap.",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Seed for deterministic shuffle. Default: {DEFAULT_SEED}.",
    )
    p.add_argument(
        "--dataset-name", default=DEFAULT_DATASET_NAME,
        help=f"HF dataset name. Default: {DEFAULT_DATASET_NAME}.",
    )
    p.add_argument(
        "--split", default=DEFAULT_SPLIT,
        help=f"HF split. Default: {DEFAULT_SPLIT}.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    levels_filter = parse_level_filter(args.levels)
    subject_slugs = tuple(
        s.strip().lower() for s in args.subjects.split(",") if s.strip()
    )
    if not subject_slugs:
        logger.error("--subjects parsed to an empty filter")
        return 2
    for slug in subject_slugs:
        if slug not in DEFAULT_SUBJECTS:
            logger.error(
                "unknown subject slug %r (expected one of: %s)",
                slug, ", ".join(DEFAULT_SUBJECTS),
            )
            return 2

    logger.info(
        "filters: levels=%s, subjects=%s, max_problems=%s, seed=%d",
        levels_filter, subject_slugs, args.max_problems, args.seed,
    )

    # Lazy import: keep laptop tests free of the `datasets` wheel.
    try:
        from datasets import load_dataset
    except ImportError as e:
        logger.error("`datasets` is required at runtime: %s", e)
        return 3

    kept: list[dict] = []
    per_subject_seen: dict[str, int] = {}
    per_subject_dropped: dict[str, int] = {}

    for subject in subject_slugs:
        logger.info("loading %s config=%s split=%s",
                    args.dataset_name, subject, args.split)
        try:
            ds = load_dataset(args.dataset_name, subject, split=args.split)
        except Exception as e:
            logger.error(
                "failed to load %s/%s (split=%s): %s",
                args.dataset_name, subject, args.split, e,
            )
            return 4
        per_subject_seen[subject] = len(ds)
        dropped = 0
        for raw in ds:
            if not keep_row(raw, levels_filter=levels_filter,
                            subjects_filter=None):
                continue
            row = build_problem_row(raw)
            if row is None:
                dropped += 1
                continue
            kept.append(row)
        per_subject_dropped[subject] = dropped
        logger.info(
            "  subject=%s: seen=%d, kept_so_far=%d, dropped_extraction=%d",
            subject, per_subject_seen[subject], len(kept), dropped,
        )

    rng = random.Random(args.seed)
    rng.shuffle(kept)

    if args.max_problems is not None and args.max_problems >= 0:
        if len(kept) > args.max_problems:
            logger.info("capping %d → %d via --max-problems",
                        len(kept), args.max_problems)
            kept = kept[:args.max_problems]

    write_problems_jsonl(kept, args.output_file)
    logger.info("wrote %d problems to %s", len(kept), args.output_file)
    print(format_summary(kept))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
