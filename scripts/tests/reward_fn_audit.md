# `scripts/reward_fn.compute_reward` empirical audit

**Generated.** 2026-05-11 — hand-crafted case bank.
**Module under test.** `scripts/reward_fn.py`. Audit-driven fix landed
on 2026-05-11: `has_box` now requires non-empty stripped extracted
payload (empty-box gaming-risk mitigation). See change log below.
**Constants in play.** `CORRECTNESS_WEIGHT = 1.0`,
`FORMAT_WEIGHT = 0.05`. Reward shape:
`reward = CORRECTNESS_WEIGHT * is_equiv(extracted, gold) + FORMAT_WEIGHT * has_box`,
where `has_box` is now `bool(extracted is not None and extracted.strip())`.
Equivalence is delegated to `evaluate.extract_answer.is_equiv` (the
vendored OpenCompass copy used by the nightly CI), so any audit result
that hinges on lexical-vs-mathematical equivalence is the CI's call,
not ours.

## Summary

| Outcome     | Count |
|-------------|-------|
| PASS        | 20 |
| FAIL        | 0 |
| DOCUMENTED  | 10 |
| **Total**   | 30 |

`PASS` = expected reward matched actual.
`FAIL` = expected reward did NOT match — read the row to decide whether
intent or implementation is at fault.
`DOCUMENTED` = no hard expectation set; this row records observed
behavior so a future regression would surface.

## Cases

| id | category | description | completion | gold | expected | actual | status | intent |
|---|---|---|---|---|---|---|---|---|
| `correct_int` | correct | plain integer | `Reasoning... \\boxed{42}` | `42` | 1.0500 | 1.0500 | **PASS** | full reward |
| `correct_negative` | correct | negative integer | `x = \\boxed{-7}` | `-7` | 1.0500 | 1.0500 | **PASS** | full reward |
| `correct_decimal` | correct | decimal | `Therefore \\boxed{3.14}` | `3.14` | 1.0500 | 1.0500 | **PASS** | full reward |
| `correct_frac_matching` | correct | matching fraction | `\\boxed{\\frac{1}{2}}` | `\\frac{1}{2}` | 1.0500 | 1.0500 | **PASS** | full reward |
| `correct_text` | correct | text answer | `Color: \\boxed{red}` | `red` | 1.0500 | 1.0500 | **PASS** | full reward |
| `wrong_simple` | wrong | wrong int | `\\boxed{42}` | `7` | 0.0500 | 0.0500 | **PASS** | format-only |
| `wrong_sign` | wrong | sign flip | `\\boxed{5}` | `-5` | 0.0500 | 0.0500 | **PASS** | format-only |
| `wrong_off_by_one` | wrong | off-by-one | `\\boxed{99}` | `100` | 0.0500 | 0.0500 | **PASS** | format-only |
| `no_box_phrase` | no_box | no box, answer in prose | `The answer is 42.` | `42` | 0.0000 | 0.0000 | **PASS** | no signal |
| `no_box_label` | no_box | no box, 'Answer:' label | `Answer: 42` | `42` | 0.0000 | 0.0000 | **PASS** | no signal |
| `no_box_empty_string` | no_box | empty generation | `` | `42` | 0.0000 | 0.0000 | **PASS** | no signal |
| `multi_box_correct_last` | multi | second box correct | `First try \\boxed{42}; on review \\boxed{43}.` | `43` | 1.0500 | 1.0500 | **PASS** | last-box-wins |
| `multi_box_wrong_last` | multi | first correct, last wrong | `First \\boxed{43}, then \\boxed{42}.` | `43` | 0.0500 | 0.0500 | **PASS** | last-box-wins penalizes |
| `multi_box_three` | multi | three boxes | `\\boxed{1} \\boxed{2} \\boxed{3}` | `3` | 1.0500 | 1.0500 | **PASS** | last-box-wins |
| `frac_half_vs_decimal` | equiv_frac | 1/2 vs 0.5 | `\\boxed{\\frac{1}{2}}` | `0.5` | (measure) | 1.0500 | **DOCUMENTED** | is_equiv should bridge |
| `frac_dfrac_vs_frac` | equiv_frac | \dfrac vs \frac same value | `\\boxed{\\dfrac{1}{2}}` | `\\frac{1}{2}` | (measure) | 1.0500 | **DOCUMENTED** | expect equivalent |
| `frac_slash_vs_decimal` | equiv_frac | 1/2 (slash) vs 0.5 | `\\boxed{1/2}` | `0.5` | (measure) | 1.0500 | **DOCUMENTED** | TeX shorthand |
| `frac_dfrac_vs_decimal` | equiv_frac | \dfrac{3}{4} vs 0.75 | `\\boxed{\\dfrac{3}{4}}` | `0.75` | (measure) | 0.0500 | **DOCUMENTED** | is_equiv |
| `sqrt_unsimplified` | equiv_sqrt | 2\sqrt{3} vs \sqrt{12} | `\\boxed{2\\sqrt{3}}` | `\\sqrt{12}` | (measure) | 0.0500 | **DOCUMENTED** | math equiv but lexically different |
| `sqrt_same` | equiv_sqrt | \sqrt{2} matching | `\\boxed{\\sqrt{2}}` | `\\sqrt{2}` | 1.0500 | 1.0500 | **PASS** | trivially equal |
| `empty_box` | empty_box | \boxed{} | `\\boxed{}` | `42` | 0.0000 | 0.0000 | **PASS** | empty-box gaming-risk fix: required non-empty stripped payload |
| `empty_box_whitespace` | empty_box | \boxed{ } | `\\boxed{ }` | `42` | 0.0000 | 0.0000 | **PASS** | same fix covers whitespace-only payload |
| `truncated_partial` | truncated | no closing brace | `Reasoning... \\boxed{42 + ` | `42` | 0.0000 | 0.0000 | **PASS** | extraction fails |
| `truncated_opener` | truncated | just opener | `Leading text \\boxed{` | `42` | 0.0000 | 0.0000 | **PASS** | extraction fails |
| `inner_whitespace` | latex | padded inner whitespace | `\\boxed{ 42 }` | `42` | (measure) | 1.0500 | **DOCUMENTED** | normalize_final strips spaces |
| `latex_thin_space` | latex | \, thin space | `\\boxed{42\\,000}` | `42000` | (measure) | 0.0500 | **DOCUMENTED** | TeX thin-space |
| `latex_text_wrapper` | latex | \text{} wrapper around answer | `\\boxed{\\text{42}}` | `42` | (measure) | 1.0500 | **DOCUMENTED** | normalize peels \text{} |
| `neg_match` | sign | negative match | `\\boxed{-5}` | `-5` | 1.0500 | 1.0500 | **PASS** | exact |
| `neg_with_space` | sign | minus then space | `\\boxed{- 5}` | `-5` | (measure) | 1.0500 | **DOCUMENTED** | normalize strips whitespace? |
| `pos_explicit` | sign | explicit + sign vs none | `\\boxed{+5}` | `5` | (measure) | 0.0500 | **DOCUMENTED** | is_equiv on +5 vs 5 |


## Findings

The PASS rows confirm the reward function's headline contract: `1.05`
for a correct boxed answer, `0.05` for a wrong boxed answer, `0.0` for
no box or empty-box. The DOCUMENTED rows record where
`evaluate.extract_answer.is_equiv` draws its equivalence boundaries —
these are inherited from OpenCompass (vendored CI scoring code) and
apply identically to nightly CI scoring, so what the audit observes
here is what the CI will count.

### Boundaries `is_equiv` DOES bridge (full reward fires)

- **`\frac{1}{2}` ↔ `0.5`** and **`1/2` ↔ `0.5`** (`frac_half_vs_decimal`,
  `frac_slash_vs_decimal`): the 1/2-decimal pair is special-cased inside
  `normalize_final_answer`.
- **`\dfrac` ↔ `\frac`** (`frac_dfrac_vs_frac`): display-vs-inline
  fraction macros are aliased.
- **`\boxed{ 42 }` ↔ `42`** (`inner_whitespace`): inner padding stripped.
- **`\boxed{\text{42}}` ↔ `42`** (`latex_text_wrapper`): the
  `\text{...}` wrapper is peeled by the normalize regex.
- **`\boxed{- 5}` ↔ `-5`** (`neg_with_space`): minus-then-space collapses.

### Boundaries `is_equiv` does NOT bridge (format-only reward)

These rows scored `0.05` even though a human grader would call them
correct:

- **`\dfrac{3}{4}` vs `0.75`** (`frac_dfrac_vs_decimal`): only the
  `1/2 ↔ 0.5` pair is hard-coded in the normalize substitutions; other
  rational↔decimal pairs are NOT bridged. The CI will count
  `\boxed{\dfrac{3}{4}}` against a `0.75` gold as wrong.
- **`2\sqrt{3}` vs `\sqrt{12}`** (`sqrt_unsimplified`): no symbolic
  simplification, exactly as the proposal pinned (exact-match verifier;
  SymPy is a v2 stretch goal).
- **`42\,000` vs `42000`** (`latex_thin_space`): the LaTeX thin-space
  is dropped by `normalize_final_answer`, but the result still failed
  to match the literal digit-string. Worth a closer look if the SFT
  data contains thin-space-separated digits.
- **`+5` vs `5`** (`pos_explicit`): explicit `+` is NOT stripped. The
  model rarely emits `+5` so this is low-priority.

### RLVR-specific design considerations

- **Empty-box gaming risk — RESOLVED 2026-05-11.** Prior to the fix,
  `\boxed{}` and `\boxed{ }` both scored `0.05` because
  `extract_boxed_answer` returned the empty string (not `None`), so
  `has_box=True` fired. That gave GRPO a `+0.05/rollout` attractor
  any policy could harvest by giving up.
  Fix: `compute_reward` now requires the extracted payload to be
  non-empty after `.strip()`. New behavior is asserted by
  `test_empty_box_returns_zero_not_format_reward` and
  `test_whitespace_only_box_returns_zero_not_format_reward` in
  `scripts/tests/test_reward_fn.py`, and reflected in the
  `empty_box` / `empty_box_whitespace` rows above (now PASS at
  expected=0.0000).
- **Last-box-wins coupling.** `multi_box_wrong_last` shows that an
  intermediate correct box does not save a later wrong one. Aligned
  with CI scoring, but means a model that "thinks out loud" with
  preliminary boxed guesses must commit to the same answer at the end.
  Stage 7 prompt-curation should keep this in mind: a difficulty band
  of `[0.2, 0.8]` solve-rate is computed under the same last-box rule.

### Net read

The audit surfaced one reward-function bug (empty-box gaming risk),
which was fixed in the same audit cycle and locked in with two new
unit tests. All other PASS rows match intent; all DOCUMENTED rows
record the OpenCompass `is_equiv` boundary that the nightly CI applies
identically. Remaining DOCUMENTED items (decimal↔arbitrary-fraction,
unsimplified radicals, thin-space digits, explicit-`+` sign) are
acknowledged limitations of the exact-match verifier (with a
SymPy/hybrid verifier deferred to v2 stretch per the proposal).

## Change log

- **2026-05-11.** Empty-box gaming-risk fix landed in
  `scripts/reward_fn.py`: `has_box = bool(extracted is not None and
  extracted.strip())`. `empty_box` and `empty_box_whitespace` rows
  flipped from DOCUMENTED to PASS at 0.0000.
