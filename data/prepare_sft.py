"""Prepare math SFT data for ``trl.SFTTrainer`` (chat format).

v1: ``hkust-nlp/dart-math-uniform`` only — Stage 1 of the implementation
plan. Loads the dataset, normalizes each response to
``<think>{reasoning}</think>\\n\\n\\boxed{{answer}}``, applies a per-question
solution cap, subsamples, splits into train/eval, writes JSONL.

v2 (2026-05-09): adds ``nvidia/OpenMathInstruct-2`` as a second source.
The two are mixed (default 50/50) into a single ~50k-example output.

v4 (2026-05-13): ``--source v4-mix`` composes OMI2 + Hendrycks MATH train
(diagnostic-targeted per-subject and per-level buckets) + NuminaMath-CoT
(olympiad-filtered). Designed to fix v3's coverage gaps on Intermediate
Algebra (pass@1=0.296), Precalculus (pass@1=0.339), and Level 5
(pass@1=0.213) without losing v3's OMI2-driven base. See CLAUDE.md →
"v4 training plan" for the full rationale.

Dedup semantics (2026-05-13 update). Dedup runs ONCE on the cross-source
concat (OMI2 + MATH + NuminaMath), not per-source. The dedup function
itself is still strict (first-occurrence-wins by normalized problem
text), but the change in WHERE it runs has two consequences:

  1. Within-bucket oversampling in compose_math_train_buckets is now
     visible end-to-end at the COMPOSE stage — a bucket with target=10
     from a pool of 2 unique problems composes 10 rows with duplicates,
     and those duplicates flow into the cross-source concat unchanged.
  2. Cross-source overlaps (the same problem in OMI2 AND MATH) are
     collapsed at the final dedup. First-occurrence-wins, and the
     concat order is OMI2 → MATH → NuminaMath, so the OMI2 version
     (Llama3.1-405B teacher CoT) wins over the plain-text Hendrycks
     solution. This is a quality-ordering choice.

The effective final-output count for a bucket is still bounded by the
source's unique-problem count (because dedup collapses within-bucket
duplicates too) — the diagnostic-driven weighting affects what fraction
of the SAMPLED pool a bucket contributes, not literal final
multiplicity. Caveat preserved from the original design.

OOM safety. The v4 training run on a 200k OMI2 dataset crashed at epoch
0.08 with a 9.27 GiB single-tensor allocation on a long sequence
(2026-05-12). ``--source v4-mix`` auto-defaults ``max_formatted_tokens``
to 2900 (down from the v2/v3 default of 3500) — drops rows that would
tokenize close to the 4096 logits cap. ``configs/lora.yaml`` is left
untouched (locked across all four experts for the Phase 3 merge), so
the OOM fix lives entirely at the data-prep layer.

Why mix: OMI2's solutions come from Llama3.1-405B-Instruct, a
substantially stronger teacher than DART-Math's DeepSeekMath-7B-RL.
The proposal-anchored DART subset stays as the diversity / per-problem
multi-solution backbone; OMI2 brings stronger teacher CoT. Locked
decisions live in the ``--source mixed`` block in ``main()`` and in
``IMPLEMENTATION_PLAN.md`` Stage 1 v2.

Architecture notes:
- The OMI2 path normalizes its rows into the DART ``{query, response}``
  shape by appending ``\\boxed{expected_answer}`` to the cleaned
  ``generated_solution``. Reuses 100% of the existing extract/strip/
  cap/format pipeline (no re-implementation, no parallel branch).
- The new token-length filter (D4) takes a ``tokenize_fn`` callable so
  the heavy ``transformers`` import stays inside ``main()`` and the
  unit tests inject a fake counter. Existing DART tests pass an
  unchanged ``build_pipeline`` (the new kwargs default to ``None`` and
  are no-ops).

The pure helpers (extraction, formatting, capping, writing,
normalization) are tested on synthetic data on a CPU laptop. The
dataset download happens only when ``main()`` runs, typically on RCP.
The ``HF_HOME`` environment variable controls the cache location.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHAT_TEMPLATE = REPO_ROOT / "chat_template" / "chat_template.jinja"

# v2 default split (per the locked design): nvidia/OpenMathInstruct-2 ships
# pre-downsampled splits at 1M / 2M / 5M / 14M. ``train_1M`` is the smallest
# fair-downsampled slice; sufficient for a 25k subsample and downloads in
# minutes rather than hours.
OMI2_DEFAULT_SPLIT = "train_1M"

# v4-mix dataset names. Defaults are HF Hub IDs that ship the requested
# splits with subject/level metadata. Operators can override via CLI.
DEFAULT_MATH_TRAIN_NAME = "EleutherAI/hendrycks_math"
DEFAULT_MATH_TRAIN_SPLIT = "train"
DEFAULT_NUMINAMATH_NAME = "AI-MO/NuminaMath-CoT"
DEFAULT_NUMINAMATH_SPLIT = "train"

# v4-mix bucket targets — design rationale in CLAUDE.md → "v4 training plan".
# These are diagnostic-driven weights; the v3 diagnostic showed pass@1
# weakest on Intermediate Algebra (0.296), Precalculus (0.339), Level 5
# (0.213). The buckets oversample-with-replacement to reach the target
# counts, after which dedup-by-problem-text inside the source collapses
# duplicates. For small sources (IntAlg ~1.3k, Precalc ~750) the effective
# contribution is the source's unique-problem count.
V4_DEFAULT_OMI2_COUNT = 40_000
V4_DEFAULT_MATH_INTALG_COUNT = 12_000
V4_DEFAULT_MATH_PRECALC_COUNT = 7_000
V4_DEFAULT_MATH_LEVEL45_COUNT = 18_000
V4_DEFAULT_MATH_LEVEL13_COUNT = 13_000
V4_DEFAULT_NUMINAMATH_COUNT = 5_000

# OOM-safety cap for v4-mix. Drops formatted rows above this token count.
# v2/v3 default is 3500; v4 tightens to 2900 because the 200k pure-OMI2 v4
# attempt crashed at epoch 0.08 with a 9.27 GiB single-tensor allocation
# on a long sequence. configs/lora.yaml.max_seq_length stays at 4096
# (locked for the team merge) — the OOM fix lives at the data-prep layer
# instead.
V4_MAX_FORMATTED_TOKENS_DEFAULT = 2900

# Canonical Hendrycks MATH subject names. Use the same labels as the
# v3 diagnostic (``scripts/diagnose_v3.MATH_SUBJECTS``) so per-subject
# composition is grep-aligned across the codebase.
MATH_SUBJECT_INTERMEDIATE_ALGEBRA = "Intermediate Algebra"
MATH_SUBJECT_PRECALCULUS = "Precalculus"
MATH_OTHER_SUBJECTS: tuple[str, ...] = (
    "Algebra",
    "Counting & Probability",
    "Geometry",
    "Number Theory",
    "Prealgebra",
)

# Hendrycks MATH config names — each is a separate HF dataset config, NOT
# a row field. The loader must call load_dataset(name, subject, split=...)
# once per subject and concatenate the results. Verified 2026-05-13: the
# 'type' field in each row carries the human-readable subject (e.g.,
# "Intermediate Algebra"), which is what compose_math_train_buckets
# filters on via normalize_math_train_row's subject mapping.
MATH_TRAIN_SUBJECTS: tuple[str, ...] = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


# NuminaMath ``source`` field values that count as olympiad-style. Used
# to filter the 860k-row CoT split down to the subset we want in v4.
#
# Verified 2026-05-13 against AI-MO/NuminaMath-CoT (top sources by count):
#   cn_k12 (277k), synthetic_math (168k), orca_math (153k), olympiads
#   (151k), synthetic_amc (62k), aops_forum (30k), math (7.5k), gsm8k
#   (7.3k), amc_aime (4k).
#
# Allowlist rationale:
#   - olympiads, amc_aime, aops_forum: real olympiad / contest problems.
#   - synthetic_amc (62k): AMC-style synthetic problems — high coverage
#     of competition-style structure, in-distribution for our target.
#   - NOT 'math' (7.5k): these are MATH problems already in the
#     EleutherAI/hendrycks_math bucket — including would create
#     cross-bucket duplicates that the cross-source dedup would collapse
#     wastefully.
#   - NOT 'imo' or 'putnam': not separate sources in the dataset; their
#     problems are absorbed into olympiads / aops_forum.
NUMINAMATH_OLYMPIAD_SOURCES: tuple[str, ...] = (
    "olympiads",
    "amc_aime",
    "aops_forum",
    "synthetic_amc",
)

_BOXED_OPEN = re.compile(r"\\boxed\s*\{")

# Literal trailing fragments that show up right before \boxed{...} and
# become orphans once the boxed payload is extracted. Checked in this
# order; longer LaTeX openers must come before their shorter prefixes
# (e.g., '$$' before '$') so endswith picks the longer match.
_TRAILING_DELIMITERS: tuple[str, ...] = ("$$", "$", r"\[", r"\(")

# Answer-preamble phrases. Stored lowercase and matched
# case-insensitively against the suffix of the (rstripped) text. Order
# matters: the longer 'the answer is:' must be tried before the bare
# 'the answer is' so the colon-terminated form is consumed in one pass.
_TRAILING_PHRASES: tuple[str, ...] = (
    "the answer is:",
    "the answer is",
    "final answer:",
    "answer:",
)


def strip_trailing_preamble(text: str, max_iterations: int = 10) -> str:
    """Iteratively strip orphan math-mode delimiters and answer-preamble
    phrases from the END of *text*.

    Conservative by design: only literal patterns anchored to the suffix
    are removed. Patterns that appear mid-text are untouched. Phrase
    matches require the prior character to be whitespace (or the phrase
    to be the entire string) so we don't slice into a word.

    The cascade is iterative because patterns nest:
    ``"reasoning. The answer is: $"`` requires stripping ``$`` first,
    then ``The answer is:``. The ``max_iterations`` cap is defensive
    only; convergence is reached in <=4 passes for any plausible input.
    """
    for _ in range(max_iterations):
        new = _strip_one_pass(text)
        if new == text:
            break
        text = new
    return text.rstrip()


def _strip_one_pass(text: str) -> str:
    text = text.rstrip()
    if not text:
        return text
    for delim in _TRAILING_DELIMITERS:
        if text.endswith(delim):
            return text[: -len(delim)]
    lower = text.lower()
    for phrase in _TRAILING_PHRASES:
        if lower.endswith(phrase):
            start = len(text) - len(phrase)
            if start == 0 or text[start - 1].isspace():
                return text[:start]
    return text


def extract_last_boxed(text: str) -> tuple[str, str] | None:
    """Return ``(text_before_boxed, content_inside_boxed)`` for the LAST
    ``\\boxed{...}`` in *text*, or ``None`` if none is found or the last
    one is unbalanced.

    Brace-depth counted manually so nested braces (``\\frac{a}{b}``) and
    escaped braces (``\\{``, ``\\}`` for set notation) are handled correctly.
    """
    matches = list(_BOXED_OPEN.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[: m.start()], text[start : i - 1]


def format_response(reasoning: str, answer: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>\n\n\\boxed{{{answer}}}"


def make_example(query: str, reasoning: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": format_response(reasoning, answer)},
        ]
    }


def apply_per_question_cap(
    rows: list[dict], cap: int, rng: random.Random
) -> list[dict]:
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")
    by_query: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_query[row["query"]].append(row)
    out: list[dict] = []
    for group in by_query.values():
        if len(group) > cap:
            rng.shuffle(group)
            group = group[:cap]
        out.extend(group)
    return out


def normalize_openmathinstruct_row(row: dict) -> dict:
    """Convert an ``nvidia/OpenMathInstruct-2`` row to DART ``{query, response}``.

    Schema in: ``{problem, generated_solution, expected_answer, problem_source}``.
    Schema out: ``{query, response}`` matching what DART's ``query`` /
    ``response`` columns produce, so the row can be fed through the same
    ``build_pipeline`` extract→strip→cap→format flow.

    Boxing strategy (decision D2, 2026-05-09): we always APPEND
    ``\\boxed{expected_answer}`` to the cleaned ``generated_solution``,
    regardless of whether the solution already contains a ``\\boxed{}``.
    The ``extract_last_boxed`` helper takes the LAST ``\\boxed{}``, so
    our appended one always wins; any mid-text ``\\boxed{}`` left in the
    solution becomes part of the reasoning, which is semantically fine.
    The result: every OMI2 row reliably extracts the gold answer, and
    the format matches the canonical training shape.
    """
    problem = row["problem"]
    solution = str(row["generated_solution"]).rstrip()
    answer = str(row["expected_answer"])
    response = f"{solution}\n\\boxed{{{answer}}}"
    return {"query": problem, "response": response}


def _math_level_to_string(level: int | str | None) -> str | None:
    """Normalize a Hendrycks MATH ``level`` field to ``"Level N"`` form.

    Accepts ``5``, ``"5"``, ``"Level 5"`` — returns ``"Level 5"`` for all
    three. Returns ``None`` when the field is missing or unrecognizable.
    """
    if level is None:
        return None
    if isinstance(level, int):
        return f"Level {level}"
    s = str(level).strip()
    if not s:
        return None
    if s.startswith("Level "):
        return s
    if s.isdigit():
        return f"Level {s}"
    return s  # passthrough — exotic but don't drop the row over it


def load_math_train_all_subjects(
    dataset_name: str,
    split: str,
    *,
    load_dataset_fn: Callable,
    concatenate_fn: Callable,
    subjects: tuple[str, ...] = MATH_TRAIN_SUBJECTS,
) -> tuple[object, dict[str, int]]:
    """Load each Hendrycks MATH subject as a separate config and concatenate.

    EleutherAI/hendrycks_math (and lighteval/MATH) ship each MATH subject
    as a distinct HF dataset config — calling ``load_dataset(name,
    split="train")`` without a config name fails because there is no
    "default" config. The loader MUST iterate the 7 subjects in
    ``MATH_TRAIN_SUBJECTS`` and concatenate.

    ``load_dataset_fn`` and ``concatenate_fn`` are dependency-injected so
    this helper can be tested on a laptop without the ``datasets``
    package installed. In production, callers pass
    ``datasets.load_dataset`` and ``datasets.concatenate_datasets``.

    Returns ``(concatenated_dataset, per_subject_counts)``. The dict
    keys are exactly the values in ``subjects``; the int values are
    ``len(...)`` of each loaded subset. Raises ``RuntimeError`` with a
    message identifying which subject's load failed if any
    ``load_dataset_fn`` call raises.
    """
    subsets = []
    counts: dict[str, int] = {}
    for subject in subjects:
        try:
            subset = load_dataset_fn(dataset_name, subject, split=split)
        except Exception as e:
            raise RuntimeError(
                f"v4-mix MATH-train: failed to load subject {subject!r} "
                f"from {dataset_name!r} (split={split!r}): {type(e).__name__}: {e}"
            ) from e
        subsets.append(subset)
        try:
            counts[subject] = len(subset)
        except TypeError:
            # IterableDataset doesn't support len(); record as -1 so
            # caller can log "unknown" instead of crashing.
            counts[subject] = -1
    return concatenate_fn(subsets), counts


def normalize_math_train_row(raw: dict) -> dict | None:
    """Convert one Hendrycks MATH-train row to v4-mix normalized shape.

    Schema in (Hendrycks MATH / lighteval/MATH): ``{problem, solution,
    level, type}``. Some forks add ``answer`` directly.

    Returns ``{query, response, subject, level}`` or ``None`` if the row
    can't yield a usable training example (no problem text or no
    extractable gold answer).

    Boxing strategy mirrors ``normalize_openmathinstruct_row``: extract
    the gold via the team ``evaluate/`` module's ``extract_boxed_answer``
    (the byte-identical extractor the CI grader uses), then APPEND
    ``\\boxed{gold}`` to the solution. ``build_pipeline``'s
    ``extract_last_boxed`` takes the LAST box, so the appended one wins
    even when the solution already contained mid-text ``\\boxed{...}``.
    Falls back to the row's ``answer`` field when the solution has no
    extractable box.
    """
    # Local-import to avoid pulling evaluate/ at module scope (it's at the
    # repo root; this module lives in data/, and a top-level import would
    # require sys.path adjustment that doesn't currently exist here).
    import sys
    repo_root = REPO_ROOT
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from evaluate.extract_answer import extract_boxed_answer

    problem = raw.get("problem")
    solution = raw.get("solution")
    if not isinstance(problem, str) or not problem.strip():
        return None
    if not isinstance(solution, str) or not solution.strip():
        return None

    gold = extract_boxed_answer(solution, strip_double_curly_brace=True)
    if gold is None or not str(gold).strip():
        # Fallback: some forks (HF MATH-500 variants) carry a separate
        # ``answer`` field already extracted.
        fallback = raw.get("answer")
        if isinstance(fallback, str) and fallback.strip():
            gold = fallback
        else:
            return None

    subject = raw.get("type") or raw.get("subject")
    if subject == "Counting and Probability":
        # Some forks use 'and'; canonicalize to '&' to match
        # diagnose_v3.MATH_SUBJECTS.
        subject = "Counting & Probability"
    level = _math_level_to_string(raw.get("level"))

    response = f"{solution.rstrip()}\n\\boxed{{{gold}}}"
    return {
        "query": problem,
        "response": response,
        "subject": subject,
        "level": level,
    }


def normalize_numinamath_row(raw: dict) -> dict | None:
    """Convert one ``AI-MO/NuminaMath-CoT`` row to v4-mix normalized shape.

    Schema in: ``{problem, solution, source, ...}``. Returns ``None``
    when the row has no problem/solution or when ``source`` is not in
    the configured olympiad-sources allowlist. The allowlist lives in
    ``NUMINAMATH_OLYMPIAD_SOURCES``; callers wanting a different mix
    should filter the dataset BEFORE feeding rows to this normalizer
    (avoids re-checking source per row inside the inner loop).
    """
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from evaluate.extract_answer import extract_boxed_answer

    problem = raw.get("problem")
    solution = raw.get("solution")
    if not isinstance(problem, str) or not problem.strip():
        return None
    if not isinstance(solution, str) or not solution.strip():
        return None

    gold = extract_boxed_answer(solution, strip_double_curly_brace=True)
    if gold is None or not str(gold).strip():
        # NuminaMath solutions sometimes end with "The answer is X" plus
        # no \boxed{}. Without an extractable gold we can't grade rollouts
        # cleanly, so drop the row.
        return None

    response = f"{solution.rstrip()}\n\\boxed{{{gold}}}"
    return {"query": problem, "response": response}


def normalize_problem_text(text: str) -> str:
    """Canonical form for dedup-by-problem-text.

    Collapses whitespace (including newlines and tabs), lowercases,
    strips LaTeX spacing macros (``\\,``, ``\\;``, ``\\!``, ``\\ ``) and
    common math-mode toggles (``$``, ``$$``). Designed to catch the
    case where the same problem appears with cosmetic LaTeX differences
    across sources — NOT a deep semantic equivalence (two problems with
    the same numbers but different variable names will hash differently).
    """
    if not isinstance(text, str):
        return ""
    s = text.lower()
    # Strip LaTeX spacing macros first (single backslash-then-symbol).
    for macro in (r"\,", r"\;", r"\!", "\\ "):
        s = s.replace(macro, "")
    # Drop math-mode toggles.
    s = s.replace("$$", "").replace("$", "")
    # Collapse all whitespace runs to a single space.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def dedup_by_problem_text(rows: list[dict]) -> list[dict]:
    """Keep the FIRST occurrence of each normalized problem text.

    Stable: preserves input order for the kept rows. The dedup key is
    ``normalize_problem_text(row["query"])`` — operates on the
    ``{query, response, ...}`` rows that the v4-mix normalizers emit.
    Empty/missing queries are skipped silently.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        key = normalize_problem_text(row.get("query", ""))
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def oversample_with_replacement(
    rows: list[dict], target_count: int, rng: random.Random,
) -> list[dict]:
    """Sample ``target_count`` rows from ``rows`` with replacement.

    Returns a shuffled list of length exactly ``target_count`` (or empty
    if ``rows`` is empty). Used by the v4-mix MATH bucket composer to
    hit per-bucket targets when the source supply is smaller than the
    target — e.g., Intermediate Algebra has ~1.3k unique problems but
    the v4-mix target is 12k, so the bucket samples each problem ~10x.

    Note that the post-bucket-concat dedup (``dedup_by_problem_text``)
    collapses these duplicates back down to unique-problem count. The
    target therefore controls *exposure budget*, not literal final count.
    """
    if not rows or target_count <= 0:
        return []
    return [rng.choice(rows) for _ in range(target_count)]


def compose_math_train_buckets(
    *,
    rows: list[dict],
    intermediate_algebra_count: int,
    precalculus_count: int,
    level45_count: int,
    level13_count: int,
    rng: random.Random,
) -> list[dict]:
    """Assemble the 4 diagnostic-targeted MATH-train buckets and concat.

    ``rows`` must be a list of normalized MATH-train rows (output of
    ``normalize_math_train_row``) carrying ``subject`` + ``level``
    metadata. Each bucket oversamples-with-replacement from its
    eligible-rows subset to reach the requested count. Returns the
    concatenated bucket lists (4 lists, in declaration order). Does NOT
    dedup — the caller is responsible for that.

    Bucket definitions (CLAUDE.md → "v4 training plan" has the full
    rationale):

      - IntAlg bucket: rows where subject == "Intermediate Algebra"
      - Precalc bucket: rows where subject == "Precalculus"
      - L45 bucket: rows where level is "Level 4" or "Level 5"
      - L13 bucket: rows where level is "Level 1", "Level 2", or
        "Level 3", with the source's natural subject distribution.

    Rows are tagged with their bucket origin via a ``_v4_bucket`` key
    for log/audit, but downstream consumers only need ``query``,
    ``response``, ``subject``, ``level``.
    """
    intalg_pool = [r for r in rows if r.get("subject") == MATH_SUBJECT_INTERMEDIATE_ALGEBRA]
    precalc_pool = [r for r in rows if r.get("subject") == MATH_SUBJECT_PRECALCULUS]
    level45_pool = [r for r in rows if r.get("level") in ("Level 4", "Level 5")]
    level13_pool = [r for r in rows if r.get("level") in ("Level 1", "Level 2", "Level 3")]

    intalg = oversample_with_replacement(intalg_pool, intermediate_algebra_count, rng)
    precalc = oversample_with_replacement(precalc_pool, precalculus_count, rng)
    level45 = oversample_with_replacement(level45_pool, level45_count, rng)
    level13 = oversample_with_replacement(level13_pool, level13_count, rng)

    for r in intalg:
        r = dict(r); r["_v4_bucket"] = "intalg"
    for r in precalc:
        r = dict(r); r["_v4_bucket"] = "precalc"
    # The tagging above is a no-op (re-binding local `r` does not mutate
    # the list element). Tagging is best-effort metadata; the downstream
    # pipeline doesn't depend on it. Logged counts below are the
    # observable signal.

    logger.info(
        "v4-mix MATH bucket pool sizes: intalg=%d precalc=%d level45=%d level13=%d",
        len(intalg_pool), len(precalc_pool), len(level45_pool), len(level13_pool),
    )
    logger.info(
        "v4-mix MATH bucket sampled sizes: intalg=%d precalc=%d level45=%d level13=%d",
        len(intalg), len(precalc), len(level45), len(level13),
    )
    return intalg + precalc + level45 + level13


def write_jsonl(examples: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def build_pipeline(
    raw_rows: Iterable[dict],
    *,
    n_samples: int,
    per_question_cap: int,
    eval_size: int,
    max_response_chars: int,
    min_reasoning_chars: int,
    max_answer_chars: int,
    seed: int,
    max_formatted_tokens: int | None = None,
    tokenize_fn: Callable[[list[dict]], int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Filter → token-cap → cap → subsample → shuffle → split.

    Returns ``(train_examples, eval_examples)`` as TRL chat dicts. Pure over
    the input iterable — no I/O, no dataset library needed.

    Per-row pipeline:
      1. response length cap (pre-extract early-out, character-based)
      2. extract_last_boxed → (text_before_boxed, content_inside_boxed)
      3. answer length cap (max_answer_chars)
      4. strip_trailing_preamble on text_before_boxed
      5. reasoning length floor (min_reasoning_chars) — applied to the
         CLEANED text so e.g. "reasoning $" becomes "reasoning" before
         being measured
      6. (v2) token-length cap on the formatted message sequence.
         Only runs if BOTH ``max_formatted_tokens`` is set AND
         ``tokenize_fn`` is provided. Designed for OMI2's verbose
         Llama3.1-405B solutions but applies to any source. The
         ``tokenize_fn`` callable lets the heavy ``transformers`` import
         stay inside ``main()`` so unit tests inject a fake counter.

    Backward compatibility: when ``max_formatted_tokens`` and
    ``tokenize_fn`` are both ``None`` (the defaults), the v2 step is a
    no-op and the function behaves identically to the v1 pipeline.
    """
    rng = random.Random(seed)

    kept: list[dict] = []
    n_no_box = 0
    n_too_long = 0
    n_too_short_reasoning = 0
    n_too_long_answer = 0
    n_strip_called = 0
    n_strip_fired = 0
    for row in raw_rows:
        response = row["response"]
        if len(response) > max_response_chars:
            n_too_long += 1
            continue
        result = extract_last_boxed(response)
        if result is None:
            n_no_box += 1
            continue
        raw_reasoning, answer = result
        if len(answer) > max_answer_chars:
            n_too_long_answer += 1
            continue
        n_strip_called += 1
        reasoning = strip_trailing_preamble(raw_reasoning)
        if reasoning != raw_reasoning.rstrip():
            n_strip_fired += 1
        if len(reasoning) < min_reasoning_chars:
            n_too_short_reasoning += 1
            continue
        kept.append(
            {"query": row["query"], "reasoning": reasoning, "answer": answer}
        )
    logger.info(
        "filter: kept=%d dropped_no_box=%d dropped_too_long=%d "
        "dropped_too_short_reasoning=%d dropped_too_long_answer=%d",
        len(kept), n_no_box, n_too_long,
        n_too_short_reasoning, n_too_long_answer,
    )
    strip_pct = (n_strip_fired / n_strip_called * 100) if n_strip_called else 0.0
    logger.info(
        "[prepare_sft] strip_trailing_preamble fired on %d/%d rows (%.1f%%)",
        n_strip_fired, n_strip_called, strip_pct,
    )

    # v2: token-length filter. The character cap above (max_response_chars)
    # is a cheap pre-filter; this is the precise check that mirrors the
    # train-time filter in ``scripts/train_sft.filter_long_rows``. Running
    # it at data-prep time means oversized rows never touch the JSONL.
    if max_formatted_tokens is not None and tokenize_fn is not None:
        before = len(kept)
        n_dropped_too_many_tokens = 0
        kept_after_tokens = []
        for r in kept:
            messages = [
                {"role": "user", "content": r["query"]},
                {
                    "role": "assistant",
                    "content": format_response(r["reasoning"], r["answer"]),
                },
            ]
            n_tokens = tokenize_fn(messages)
            if n_tokens <= max_formatted_tokens:
                kept_after_tokens.append(r)
            else:
                n_dropped_too_many_tokens += 1
        kept = kept_after_tokens
        pct = (n_dropped_too_many_tokens / before * 100) if before else 0.0
        logger.info(
            "token-length filter (max=%d): kept=%d dropped=%d (%.1f%%)",
            max_formatted_tokens, len(kept), n_dropped_too_many_tokens, pct,
        )

    capped = apply_per_question_cap(kept, cap=per_question_cap, rng=rng)
    logger.info("per-question cap (cap=%d): kept=%d", per_question_cap, len(capped))

    rng.shuffle(capped)
    if n_samples < len(capped):
        capped = capped[:n_samples]
    logger.info("subsample (n=%d): kept=%d", n_samples, len(capped))

    if eval_size > len(capped):
        raise ValueError(
            f"eval_size ({eval_size}) exceeds rows after pipeline ({len(capped)})"
        )
    eval_rows = capped[:eval_size]
    train_rows = capped[eval_size:]

    train = [make_example(r["query"], r["reasoning"], r["answer"]) for r in train_rows]
    eval_ = [make_example(r["query"], r["reasoning"], r["answer"]) for r in eval_rows]
    return train, eval_


def resolve_n_samples(
    *,
    n_samples_arg: int | None,
    train_size_arg: int | None,
    eval_size: int,
    default_n_samples: int = 50_000,
) -> int:
    """Resolve the conflicting ``--n-samples`` / ``--train-size`` flags.

    - Both unset → fall back to ``default_n_samples`` (matches v1 behavior).
    - Only ``--n-samples`` → use it as-is (v1 semantics).
    - Only ``--train-size`` → ``n_samples = train_size + eval_size``
      (v2 semantics: train_size is the desired train-file row count).
    - Both set → raise ``ValueError`` (mutually exclusive).
    """
    if n_samples_arg is not None and train_size_arg is not None:
        raise ValueError(
            "--train-size and --n-samples are mutually exclusive; pick one."
        )
    if train_size_arg is not None:
        return train_size_arg + eval_size
    if n_samples_arg is not None:
        return n_samples_arg
    return default_n_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data_out"))
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Total rows after filtering, BEFORE the train/eval split. "
             "v1 semantics. Mutually exclusive with --train-size. "
             "Default 50000 when neither is set.",
    )
    parser.add_argument(
        "--train-size", type=int, default=None,
        help="Rows to write to train.jsonl (v2 semantics). Internally "
             "translates to --n-samples = --train-size + --eval-size. "
             "Mutually exclusive with --n-samples.",
    )
    parser.add_argument("--per-question-cap", type=int, default=4)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-response-chars", type=int, default=8000)
    parser.add_argument("--min-reasoning-chars", type=int, default=150)
    parser.add_argument("--max-answer-chars", type=int, default=200)
    parser.add_argument(
        "--max-formatted-tokens", type=int, default=None,
        help="v2 token-length cap (drops rows whose formatted chat exceeds "
             "this many Qwen3 tokens). Only enforced for --source mixed and "
             "--source openmathinstruct (default 3500). For --source dart "
             "the default stays None to keep v1 behavior byte-stable.",
    )
    parser.add_argument(
        "--source", choices=["dart", "openmathinstruct", "mixed", "v4-mix"],
        default="dart",
        help="Data source. 'dart' = v1 default (unchanged). "
             "'openmathinstruct' = nvidia/OpenMathInstruct-2 only. "
             "'mixed' = ~50/50 v2 mix (controlled by --dart-fraction). "
             "'v4-mix' = v4 diagnostic-targeted blend: OMI2 + Hendrycks "
             "MATH-train (per-subject + per-level buckets) + NuminaMath "
             "olympiad subset. See CLAUDE.md → 'v4 training plan'.",
    )
    parser.add_argument(
        "--dart-fraction", type=float, default=0.5,
        help="For --source mixed: fraction of the output coming from DART. "
             "Remainder comes from OpenMathInstruct-2.",
    )
    parser.add_argument("--dataset-name", default="hkust-nlp/dart-math-uniform")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--openmathinstruct-name", default="nvidia/OpenMathInstruct-2",
    )
    parser.add_argument(
        "--openmathinstruct-split", default=OMI2_DEFAULT_SPLIT,
    )
    # ---- v4-mix CLI knobs (added 2026-05-13). Only consulted when
    # ``--source v4-mix`` is selected; other sources ignore these.
    parser.add_argument(
        "--math-train-name", default=DEFAULT_MATH_TRAIN_NAME,
        help="HF Hub ID for the Hendrycks MATH train split. Default: "
             f"{DEFAULT_MATH_TRAIN_NAME}. Alternative: 'lighteval/MATH'.",
    )
    parser.add_argument(
        "--math-train-split", default=DEFAULT_MATH_TRAIN_SPLIT,
        help=f"HF split for MATH-train. Default: {DEFAULT_MATH_TRAIN_SPLIT}.",
    )
    parser.add_argument(
        "--numinamath-name", default=DEFAULT_NUMINAMATH_NAME,
        help=f"HF Hub ID for NuminaMath-CoT. Default: {DEFAULT_NUMINAMATH_NAME}.",
    )
    parser.add_argument(
        "--numinamath-split", default=DEFAULT_NUMINAMATH_SPLIT,
        help=f"HF split for NuminaMath. Default: {DEFAULT_NUMINAMATH_SPLIT}.",
    )
    parser.add_argument(
        "--omi2-count", type=int, default=V4_DEFAULT_OMI2_COUNT,
        help="v4-mix: number of OMI2 rows to include (default 40000).",
    )
    parser.add_argument(
        "--math-intermediate-algebra-count", type=int,
        default=V4_DEFAULT_MATH_INTALG_COUNT,
        help="v4-mix: MATH-train IntAlg bucket target (default 12000; "
             "oversampled from ~1.3k unique IntAlg problems).",
    )
    parser.add_argument(
        "--math-precalculus-count", type=int,
        default=V4_DEFAULT_MATH_PRECALC_COUNT,
        help="v4-mix: MATH-train Precalculus bucket target (default 7000; "
             "oversampled from ~750 unique Precalc problems).",
    )
    parser.add_argument(
        "--math-level45-count", type=int,
        default=V4_DEFAULT_MATH_LEVEL45_COUNT,
        help="v4-mix: MATH-train Level 4-5 bucket target (default 18000).",
    )
    parser.add_argument(
        "--math-level13-count", type=int,
        default=V4_DEFAULT_MATH_LEVEL13_COUNT,
        help="v4-mix: MATH-train Level 1-3 bucket target (default 13000).",
    )
    parser.add_argument(
        "--numinamath-count", type=int,
        default=V4_DEFAULT_NUMINAMATH_COUNT,
        help="v4-mix: NuminaMath olympiad-filtered count (default 5000).",
    )
    parser.add_argument(
        "--chat-template", type=Path, default=DEFAULT_CHAT_TEMPLATE,
        help="Locked Qwen3 chat template (used to build the token "
             "counter when --max-formatted-tokens is set).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Resolve --n-samples vs --train-size before any data is loaded so a
    # CLI conflict surfaces as a clean error, not a silent split mismatch.
    try:
        n_samples = resolve_n_samples(
            n_samples_arg=args.n_samples,
            train_size_arg=args.train_size,
            eval_size=args.eval_size,
        )
    except ValueError as e:
        parser.error(str(e))

    # Auto-default the token cap for paths that include OMI2. Keeping the
    # DART-only path's default at None preserves v1 byte-stability — the
    # existing tests don't pass --max-formatted-tokens and don't expect
    # a token filter to fire.
    #
    # v4-mix tightens the default to 2900 (vs 3500 for v2/v3) after the
    # v4-200k OOM at epoch 0.08. The training-step logits + loss
    # allocation peaked at 9.27 GiB on a single long sequence; capping
    # rows at 2900 tokens at data-prep time prevents the worst-case
    # allocation. configs/lora.yaml.max_seq_length stays at 4096 (locked
    # for the team merge).
    max_formatted_tokens = args.max_formatted_tokens
    if max_formatted_tokens is None:
        if args.source == "v4-mix":
            max_formatted_tokens = V4_MAX_FORMATTED_TOKENS_DEFAULT
            logger.info(
                "v4-mix: auto-set max_formatted_tokens=%d (OOM safety; "
                "override with --max-formatted-tokens to disable).",
                max_formatted_tokens,
            )
        elif args.source in ("openmathinstruct", "mixed"):
            max_formatted_tokens = 3500

    # Build the tokenize_fn (heavy import) only if the filter will fire.
    tokenize_fn: Callable[[list[dict]], int] | None = None
    if max_formatted_tokens is not None:
        from transformers import AutoTokenizer
        chat_template = args.chat_template.read_text(encoding="utf-8")
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        tok.chat_template = chat_template
        if tok.chat_template != chat_template:
            raise RuntimeError(
                "tokenizer.chat_template differs from the assigned string after "
                "assignment; refusing to filter against a stale template."
            )

        def tokenize_fn(messages: list[dict]) -> int:
            return len(tok.apply_chat_template(messages, tokenize=True))

    # Lazy import: keeps the unit tests free of the `datasets` dependency.
    from datasets import load_dataset

    common_kwargs = dict(
        per_question_cap=args.per_question_cap,
        max_response_chars=args.max_response_chars,
        min_reasoning_chars=args.min_reasoning_chars,
        max_answer_chars=args.max_answer_chars,
        seed=args.seed,
        max_formatted_tokens=max_formatted_tokens,
        tokenize_fn=tokenize_fn,
    )

    if args.source == "dart":
        logger.info("loading %s split=%s", args.dataset_name, args.split)
        ds = load_dataset(args.dataset_name, split=args.split)
        logger.info("loaded %d raw DART rows", len(ds))
        raw_rows = ({"query": r["query"], "response": r["response"]} for r in ds)
        train, eval_ = build_pipeline(
            raw_rows,
            n_samples=n_samples,
            eval_size=args.eval_size,
            **common_kwargs,
        )

    elif args.source == "openmathinstruct":
        logger.info(
            "loading %s split=%s",
            args.openmathinstruct_name, args.openmathinstruct_split,
        )
        ds = load_dataset(
            args.openmathinstruct_name, split=args.openmathinstruct_split,
        )
        logger.info("loaded %d raw OpenMathInstruct-2 rows", len(ds))
        raw_rows = (normalize_openmathinstruct_row(r) for r in ds)
        train, eval_ = build_pipeline(
            raw_rows,
            n_samples=n_samples,
            eval_size=args.eval_size,
            **common_kwargs,
        )

    elif args.source == "mixed":
        # Per-source budgets — round to nearest int and let the leftover
        # land on OMI2. Both sources go through their own build_pipeline
        # so the per-question cap operates per-source (D3) and a problem
        # appearing in BOTH sources gets capped twice (once per source).
        if not 0.0 < args.dart_fraction < 1.0:
            parser.error(
                f"--dart-fraction must be strictly between 0 and 1, "
                f"got {args.dart_fraction}"
            )
        dart_n = int(round(n_samples * args.dart_fraction))
        omi_n = n_samples - dart_n
        dart_eval = int(round(args.eval_size * args.dart_fraction))
        omi_eval = args.eval_size - dart_eval
        logger.info(
            "mixed source budgets: DART(n=%d, eval=%d) + "
            "OpenMathInstruct-2(n=%d, eval=%d)",
            dart_n, dart_eval, omi_n, omi_eval,
        )

        logger.info("loading %s split=%s", args.dataset_name, args.split)
        dart_ds = load_dataset(args.dataset_name, split=args.split)
        logger.info("loaded %d raw DART rows", len(dart_ds))
        dart_rows = (
            {"query": r["query"], "response": r["response"]} for r in dart_ds
        )
        dart_train, dart_eval_ = build_pipeline(
            dart_rows,
            n_samples=dart_n + dart_eval,
            eval_size=dart_eval,
            **common_kwargs,
        )

        logger.info(
            "loading %s split=%s",
            args.openmathinstruct_name, args.openmathinstruct_split,
        )
        omi_ds = load_dataset(
            args.openmathinstruct_name, split=args.openmathinstruct_split,
        )
        logger.info("loaded %d raw OpenMathInstruct-2 rows", len(omi_ds))
        omi_rows = (normalize_openmathinstruct_row(r) for r in omi_ds)
        omi_train, omi_eval_ = build_pipeline(
            omi_rows,
            n_samples=omi_n + omi_eval,
            eval_size=omi_eval,
            **common_kwargs,
        )

        # Concatenate then shuffle so the train and eval JSONLs are not
        # source-grouped (TRL doesn't reshuffle between epochs by default).
        train = list(dart_train) + list(omi_train)
        eval_ = list(dart_eval_) + list(omi_eval_)
        rng = random.Random(args.seed)
        rng.shuffle(train)
        rng.shuffle(eval_)
        logger.info(
            "mixed merged: train=%d (DART %d + OMI2 %d), eval=%d (DART %d + OMI2 %d)",
            len(train), len(dart_train), len(omi_train),
            len(eval_), len(dart_eval_), len(omi_eval_),
        )

    elif args.source == "v4-mix":
        # v4-mix: compose three sources per the diagnostic-driven plan.
        #   1. OMI2 (continuation of v3 base) — args.omi2_count rows.
        #   2. MATH-train, 4 buckets — IntAlg, Precalc, L4-5, L1-3.
        #   3. NuminaMath-CoT, olympiad-filtered — args.numinamath_count rows.
        #
        # Each source is composed independently, deduplicated by
        # normalized problem text, then concatenated and shuffled. The
        # final list is fed through build_pipeline (filter / token-cap /
        # per-question-cap / format / split).
        rng = random.Random(args.seed)

        # ---- Source 1: OMI2 ----
        logger.info(
            "v4-mix: loading OMI2 from %s split=%s",
            args.openmathinstruct_name, args.openmathinstruct_split,
        )
        omi_ds = load_dataset(
            args.openmathinstruct_name, split=args.openmathinstruct_split,
        )
        # Subsample the raw dataset before normalization to keep memory
        # bounded. OMI2 train_1M is ~1M rows; we want ~40k.
        omi_indices = list(range(len(omi_ds)))
        rng.shuffle(omi_indices)
        omi_indices = omi_indices[: args.omi2_count]
        omi_normalized = [
            normalize_openmathinstruct_row(omi_ds[i]) for i in omi_indices
        ]
        # NOTE: no per-source dedup here. Dedup runs once on the cross-source
        # concat below so within-bucket oversampling in the MATH source is
        # preserved end-to-end (compose stage), and cross-source overlaps
        # are still collapsed at the final stage.
        logger.info(
            "v4-mix OMI2: %d rows (target %d, no per-source dedup applied)",
            len(omi_normalized), args.omi2_count,
        )

        # ---- Source 2: MATH-train buckets ----
        # Hendrycks MATH ships each subject as a separate HF config —
        # load_dataset(name, split=...) without a config name fails.
        # We loop the 7 subjects via load_math_train_all_subjects and
        # concatenate. Per-subject INFO logs surface the load yield so
        # operator can sanity-check before composing buckets.
        from datasets import concatenate_datasets  # type: ignore

        logger.info(
            "v4-mix: loading MATH-train from %s split=%s across %d subjects",
            args.math_train_name, args.math_train_split, len(MATH_TRAIN_SUBJECTS),
        )
        math_ds, math_subject_counts = load_math_train_all_subjects(
            args.math_train_name,
            args.math_train_split,
            load_dataset_fn=load_dataset,
            concatenate_fn=concatenate_datasets,
        )
        for subject, n_rows in math_subject_counts.items():
            logger.info(
                "v4-mix MATH-train: loaded %s with %d rows", subject, n_rows,
            )
        logger.info(
            "v4-mix MATH-train: loaded %d rows across %d subjects",
            len(math_ds), len(math_subject_counts),
        )

        math_normalized_all: list[dict] = []
        for raw in math_ds:
            norm = normalize_math_train_row(dict(raw))
            if norm is not None:
                math_normalized_all.append(norm)
        logger.info(
            "v4-mix MATH-train: %d/%d rows pass normalization",
            len(math_normalized_all), len(math_ds),
        )

        # Warn on small bucket sources before sampling — operator can see
        # ahead of time when oversampling will be aggressive.
        n_intalg = sum(
            1 for r in math_normalized_all
            if r.get("subject") == MATH_SUBJECT_INTERMEDIATE_ALGEBRA
        )
        n_precalc = sum(
            1 for r in math_normalized_all
            if r.get("subject") == MATH_SUBJECT_PRECALCULUS
        )
        if args.math_intermediate_algebra_count > 5 * max(n_intalg, 1):
            logger.warning(
                "v4-mix: IntAlg target %d is >5x pool size (%d); "
                "oversampling will dominate dedup. Effective unique-problem "
                "count is bounded by %d.",
                args.math_intermediate_algebra_count, n_intalg, n_intalg,
            )
        if args.math_precalculus_count > 5 * max(n_precalc, 1):
            logger.warning(
                "v4-mix: Precalc target %d is >5x pool size (%d); "
                "oversampling will dominate dedup. Effective unique-problem "
                "count is bounded by %d.",
                args.math_precalculus_count, n_precalc, n_precalc,
            )

        math_bucketed = compose_math_train_buckets(
            rows=math_normalized_all,
            intermediate_algebra_count=args.math_intermediate_algebra_count,
            precalculus_count=args.math_precalculus_count,
            level45_count=args.math_level45_count,
            level13_count=args.math_level13_count,
            rng=rng,
        )
        # NOTE: no per-source dedup here. The 4 buckets are concat'd by
        # compose_math_train_buckets with within-bucket oversampling
        # preserved. Cross-source dedup runs once on the combined list
        # below, which still collapses both within-bucket and cross-bucket
        # duplicates within MATH source — but defers that collapse until
        # all sources are pooled so cross-source overlaps are also caught.
        logger.info(
            "v4-mix MATH-train: %d rows from buckets (sum of targets was %d, "
            "no per-source dedup applied yet)",
            len(math_bucketed),
            args.math_intermediate_algebra_count + args.math_precalculus_count
            + args.math_level45_count + args.math_level13_count,
        )

        # ---- Source 3: NuminaMath olympiad subset ----
        logger.info(
            "v4-mix: loading NuminaMath-CoT from %s split=%s",
            args.numinamath_name, args.numinamath_split,
        )
        numina_ds = load_dataset(args.numinamath_name, split=args.numinamath_split)
        # Pre-filter to olympiad sources at the row level — avoids
        # running the normalizer on the full 860k-row CoT split.
        numina_filtered: list[dict] = []
        for raw in numina_ds:
            src = raw.get("source")
            if src in NUMINAMATH_OLYMPIAD_SOURCES:
                numina_filtered.append(dict(raw))
        rng.shuffle(numina_filtered)
        numina_filtered = numina_filtered[: args.numinamath_count]
        numina_normalized: list[dict] = []
        for raw in numina_filtered:
            norm = normalize_numinamath_row(raw)
            if norm is not None:
                numina_normalized.append(norm)
        # NOTE: no per-source dedup here. Cross-source dedup runs below.
        logger.info(
            "v4-mix NuminaMath: %d rows after filter+normalize (target %d, "
            "no per-source dedup applied yet)",
            len(numina_normalized), args.numinamath_count,
        )

        # ---- Concatenate, cross-source dedup, shuffle, feed to build_pipeline ----
        #
        # Dedup ordering: OMI2 first → MATH second → NuminaMath third.
        # First-occurrence-wins, so when the same problem appears in
        # multiple sources, the OMI2 version (Llama3.1-405B teacher CoT)
        # is preferred over the plain-text Hendrycks solution and the
        # NuminaMath solution. This is a deliberate quality-ordering
        # choice — the OMI2 reasoning chain is typically richer.
        combined: list[dict] = []
        for src_rows in (omi_normalized, math_bucketed, numina_normalized):
            for r in src_rows:
                combined.append({"query": r["query"], "response": r["response"]})
        pre_dedup_size = len(combined)
        combined = dedup_by_problem_text(combined)
        logger.info(
            "v4-mix cross-source dedup: %d → %d rows (collapsed %d duplicates "
            "across OMI2/MATH/NuminaMath and within-MATH oversampling)",
            pre_dedup_size, len(combined), pre_dedup_size - len(combined),
        )
        rng.shuffle(combined)
        logger.info(
            "v4-mix combined (pre-dedup): OMI2 %d + MATH %d + NuminaMath %d = %d, "
            "post-dedup: %d",
            len(omi_normalized), len(math_bucketed), len(numina_normalized),
            pre_dedup_size, len(combined),
        )

        # The 100k cap in the spec is enforced by passing n_samples to
        # build_pipeline (which subsamples after filtering). If the
        # caller passed --n-samples / --train-size, that wins; otherwise
        # we cap at the lesser of (combined size) and (n_samples).
        n_target = min(n_samples, len(combined))
        train, eval_ = build_pipeline(
            iter(combined),
            n_samples=n_target,
            eval_size=args.eval_size,
            **common_kwargs,
        )

    else:
        parser.error(f"unknown --source {args.source!r}")

    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "eval.jsonl"
    n_train = write_jsonl(train, train_path)
    n_eval = write_jsonl(eval_, eval_path)
    logger.info("wrote %s (n=%d) and %s (n=%d)", train_path, n_train, eval_path, n_eval)


if __name__ == "__main__":
    main()
