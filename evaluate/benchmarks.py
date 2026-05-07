"""Per-benchmark answer extraction and comparison.

Adapted from the CI's `src/benchmarks.py`, but with the dataset-loading half
removed and the public functions taking the extraction-method string directly
instead of a `BenchmarkConfig` dataclass. The actual extraction / comparison
helpers are byte-for-byte identical to the CI version.
"""

from __future__ import annotations

import json
import re
import string

from .extract_answer import extract_boxed_answer, is_equiv, normalize_final_answer


VALID_METHODS = ("boxed", "knowledge", "exact")


def extract_benchmark_answer(text: str, method: str, reference: str) -> str | None:
    """Extract a final answer according to the benchmark's evaluator."""
    if method == "boxed":
        return extract_boxed_answer(text, strip_double_curly_brace=True)
    if method == "exact":
        return text.strip()
    if method == "knowledge":
        extracted = extract_boxed_answer(text, strip_double_curly_brace=True)
        candidate = extracted if extracted is not None else text.strip()
        if _is_choice_reference(reference):
            return _extract_choice_label(candidate)
        return _clean_direct_answer(candidate)

    extracted = extract_boxed_answer(text, strip_double_curly_brace=True)
    return extracted if extracted is not None else text.strip()


def is_correct_benchmark_answer(
    extracted: str | None,
    reference: str,
    method: str,
) -> bool:
    """Compare an extracted answer to the benchmark reference."""
    if extracted is None:
        return False

    if method == "knowledge":
        if _is_choice_reference(reference):
            return extracted.upper() == reference.strip().upper()
        return _matches_direct_answer(extracted, reference)

    return is_equiv(
        normalize_final_answer(extracted),
        normalize_final_answer(reference),
    )


def _is_choice_reference(reference: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]", reference.strip().upper()))


def _extract_choice_label(text: str) -> str | None:
    candidate = text.strip().upper()
    direct = re.fullmatch(r"\(?([A-Z])\)?[.)]?", candidate)
    if direct:
        return direct.group(1)

    patterns = [
        r"(?:final\s+answer|answer|option|choice)\s*(?:is|:)?\s*\(?([A-Z])\)?",
        r"\b([A-Z])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, candidate)
        if len(matches) == 1:
            return matches[0]
    return None


def _matches_direct_answer(extracted: str, reference: str) -> bool:
    aliases = _reference_aliases(reference)
    normalized_extracted = _normalize_direct_answer(extracted)
    return any(normalized_extracted == _normalize_direct_answer(alias) for alias in aliases)


def _reference_aliases(reference: str) -> list[str]:
    try:
        parsed = json.loads(reference)
    except json.JSONDecodeError:
        return [reference]
    if isinstance(parsed, list):
        return [str(alias) for alias in parsed]
    return [str(parsed)]


def _clean_direct_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(
        r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def _normalize_direct_answer(text: str) -> str:
    text = _clean_direct_answer(text).casefold()
    text = text.strip().strip(string.punctuation + " ")
    text = re.sub(r"\s+", " ", text)
    return text
