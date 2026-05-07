"""Answer extraction and equivalence checking.

Core functions ported from OpenCompass (opencompass/datasets/math.py) with
minor refactoring to be standalone (no class dependencies, no registry).
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Boxed answer extraction
# ---------------------------------------------------------------------------

def last_boxed_only_string(string: str) -> str | None:
    """Find the last \\boxed{...} or \\fbox{...} in the string, including the
    command itself. Handles nested braces correctly."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str | None:
    """Remove the \\boxed{...} or \\fbox{...} wrapper, returning the inner content."""
    for left in ("\\boxed{", "\\fbox{"):
        if s.startswith(left) and s.endswith("}"):
            return s[len(left) : -1]
    return None


def extract_boxed_answer(
    pred_str: str, strip_double_curly_brace: bool = False
) -> str | None:
    """Extract the answer from inside the last \\boxed{} in the prediction.

    Returns None if no \\boxed{} is found.
    """
    boxed_str = last_boxed_only_string(pred_str)
    if boxed_str is None:
        return None
    answer = remove_boxed(boxed_str)
    if answer is None:
        return None
    if strip_double_curly_brace:
        match = re.match(r"^\{(.*)\}$", answer)
        if match:
            answer = match.group(1)
    return answer


# ---------------------------------------------------------------------------
# Answer normalization
# ---------------------------------------------------------------------------

def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = str(final_answer)
    SUBSTITUTIONS = [
        ("an ", ""),
        ("a ", ""),
        (".$", "$"),
        ("\\$", ""),
        ("\\ ", ""),
        (" ", ""),
        ("mbox", "text"),
        (",\\text{and}", ","),
        ("\\text{and}", ","),
        ("\\text{m}", "\\text{}"),
        ("\\le", "<"),
    ]
    REMOVED_EXPRESSIONS = [
        "square", "ways", "integers", "dollars", "mph", "inches", "ft",
        "hours", "km", "units", "\\ldots", "sue", "points", "feet", "minutes",
        "digits", "cents", "degrees", "cm", "gm", "pounds", "meters", "meals",
        "edges", "students", "childrentickets", "multiples", "\\text{s}",
        "\\text{.}", "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}",
        "\\text{}", r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!",
        "{,}", '"', "\\dots", "\n", "\r", "\f",
    ]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold, is surrounded by a box, etc.
    final_answer = re.sub(r"(\\text\{)\((.*?)\)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", r"\2", final_answer)

    assert "\n" not in final_answer
    assert "\r" not in final_answer
    assert "\f" not in final_answer

    if len(re.findall(r"finalansweris(.*)", final_answer)) > 0:
        final_answer = re.findall(r"finalansweris(.*)", final_answer)[-1]

    if len(re.findall(r"answer?is:?(.*)", final_answer)) > 0:
        final_answer = re.findall(r"answer?is:?(.*)", final_answer)[-1]

    if len(re.findall(r"oxed\{(.*?)\}", final_answer)) > 0:
        final_answer = re.findall(r"oxed\{(.*?)\}", final_answer)[-1]

    if len(re.findall(r"\$(.*?)\$", final_answer)) > 0:
        final_answer = re.findall(r"\$(.*?)\$", final_answer)[-1]

    final_answer = final_answer.strip()
    if "rac" in final_answer and "\\frac" not in final_answer:
        final_answer = final_answer.replace("rac", "\\frac")

    # Normalize shorthand TeX:
    #   \fracab -> \frac{a}{b}
    #   \sqrta  -> \sqrt{a}
    final_answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


# ---------------------------------------------------------------------------
# String stripping (v2 variant from OpenCompass MATHEvaluator)
# ---------------------------------------------------------------------------

def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        new_str += "{" + a + "}{" + b + "}" + substr[2:]
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        new_str += "{" + a + "}" + b + substr[2:]
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a_str, b_str = string.split("/")
    try:
        a = int(a_str)
        b = int(b_str)
        assert string == f"{a}/{b}"
        return f"\\frac{{{a}}}{{{b}}}"
    except (ValueError, AssertionError):
        return string


def strip_string(string: str) -> str:
    """Comprehensive string stripping for math answer comparison (v2)."""
    string = str(string).strip()
    string = string.replace("\n", "")
    string = string.rstrip(".")
    string = string.replace("\\!", "")
    string = string.replace("\\ ", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove units
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string

    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\text", "")
    string = string.replace("x\\in", "")
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")
    string = re.sub(r"\\mbox{.*?}", "", string)
    string = string.replace("'", "")
    string = string.replace('"', "")

    # j -> i
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # remove trailing zeros: 1.000 -> 1
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # get rid of "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    string = _fix_a_slash_b(string)

    return string


# ---------------------------------------------------------------------------
# Equivalence checking
# ---------------------------------------------------------------------------

def is_equiv(str1: str | None, str2: str | None) -> bool:
    """Check if two math answers are equivalent.

    Tries multiple normalization strategies:
    1. strip_string on both, compare
    2. normalize_final_answer on stripped, compare
    3. normalize_final_answer on originals, compare
    4. Direct string comparison as fallback
    """
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if ss1 == ss2:
            return True
        ss1 = normalize_final_answer(ss1)
        ss2 = normalize_final_answer(ss2)
        if ss1 == ss2:
            return True
    except Exception:
        pass

    try:
        ss1 = normalize_final_answer(str1)
        ss2 = normalize_final_answer(str2)
        if ss1 == ss2:
            return True
    except Exception:
        pass

    return str1 == str2
