"""Verify the locked chat template renders correctly on the Qwen3-1.7B tokenizer.

Runs on RCP (needs the actual tokenizer downloaded). Three tests:

  1. Generation-prompt path — what the CS-552 CI exercises.
  2. Defense in depth — `enable_thinking=False` kwarg must NOT flip thinking off.
  3. Round-trip with an assistant turn — what SFTTrainer sees during training.

Exit codes are distinct so the failure mode is unambiguous from the RCP log:

  0  all checks passed
  1  one or more chat-template assertions failed -> the Jinja is broken
     (fix it in emainelpe-shared and propagate to all four expert repos)
  2  verification infrastructure failed -> the script itself can't run
     (template file missing, tokenizer download failed, transformers absent)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[1]

EXIT_OK = 0
EXIT_TEMPLATE_BUG = 1
EXIT_INFRA_BUG = 2

_results: dict = {"passed": 0, "failed": 0, "failures": []}


def setup_error(msg: str) -> NoReturn:
    print(f"[SETUP ERROR] {msg}", file=sys.stderr)
    sys.exit(EXIT_INFRA_BUG)


def check(name: str, condition: bool, hint: str, rendered: str) -> None:
    if condition:
        _results["passed"] += 1
        print(f"[OK] {name}")
        return
    _results["failed"] += 1
    _results["failures"].append(name)
    print(
        f"[FAIL] {name}\n"
        f"  hint: {hint}\n"
        f"  --- rendered output (verbatim) ---\n"
        f"{rendered}\n"
        f"  --- end ---"
    )


def safe_render(tokenizer, label: str, conversation: list[dict], **kwargs) -> str | None:
    try:
        return tokenizer.apply_chat_template(conversation, tokenize=False, **kwargs)
    except Exception as e:
        _results["failed"] += 1
        _results["failures"].append(label)
        print(f"[FAIL] {label}: template raised {type(e).__name__}: {e}")
        return None


def test_generation_prompt(tokenizer) -> None:
    print("=== Test 1: generation prompt (add_generation_prompt=True) ===")
    rendered = safe_render(
        tokenizer,
        "Test 1 render",
        [{"role": "user", "content": "What is 2+2?"}],
        add_generation_prompt=True,
    )
    if rendered is None:
        return

    check(
        "1.A suffix is exactly '<|im_start|>assistant\\n'",
        rendered.endswith("<|im_start|>assistant\n"),
        "wrong suffix; the template's add_generation_prompt branch is broken or "
        "the locked template was not loaded",
        rendered,
    )
    check(
        "1.B no empty-think marker (thinking is ON)",
        "<think>\n\n</think>" not in rendered,
        "thinking is silently OFF: the rendered prompt contains '<think>\\n\\n</think>', "
        "which is the Qwen3 disable-thinking marker. The locked template must hardpin "
        "enable_thinking=true.",
        rendered,
    )


def test_kwarg_cant_disable_thinking(tokenizer) -> None:
    print("=== Test 2: enable_thinking=False kwarg must NOT disable thinking ===")
    rendered = safe_render(
        tokenizer,
        "Test 2 render",
        [{"role": "user", "content": "What is 2+2?"}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if rendered is None:
        return

    check(
        "2.A hardline shadows the kwarg",
        "<think>\n\n</think>" not in rendered,
        "the enable_thinking=False kwarg disabled thinking. The locked template MUST "
        "set enable_thinking=true at the top so the kwarg is shadowed; do not honor "
        "the kwarg.",
        rendered,
    )


def test_assistant_round_trip(tokenizer) -> None:
    print("=== Test 3: assistant round-trip (add_generation_prompt=False) ===")

    # The assistant branch of chat_template.jinja does NOT pass message.content
    # through verbatim. It splits on '</think>' and re-emits the two halves:
    #
    #   reasoning_part = message.content.split('</think>')[0] + '</think>\n\n'
    #   answer_part    = message.content.split('</think>')[-1].lstrip('\n')
    #
    # then concatenates them as
    #   <|im_start|>assistant\n{reasoning_part}{answer_part}<|im_end|>\n
    #
    # For our input "<think>\nAdd them.\n</think>\n\n\\boxed{4}" the splits give:
    #   reasoning_part = "<think>\nAdd them.\n" + "</think>\n\n"
    #                  = "<think>\nAdd them.\n</think>\n\n"
    #   answer_part    = "\n\n\\boxed{4}".lstrip('\n')
    #                  = "\\boxed{4}"
    #
    # which reconstitutes byte-for-byte to the original content. The match is
    # therefore COINCIDENTAL: it only works because our input is already in the
    # canonical "<think>\n{r}\n</think>\n\n{a}" shape produced by
    # data/prepare_sft.format_response(). Inputs in other shapes get normalized
    # by this split-and-re-emit, not passed through.
    convo = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>\nAdd them.\n</think>\n\n\\boxed{4}"},
    ]
    rendered = safe_render(
        tokenizer,
        "Test 3 render",
        convo,
        add_generation_prompt=False,
    )
    if rendered is None:
        return

    check(
        "3.A user turn rendered correctly",
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n" in rendered,
        "the user turn is missing or malformed",
        rendered,
    )
    check(
        "3.B assistant turn reconstituted correctly",
        "<|im_start|>assistant\n<think>\nAdd them.\n</think>\n\n\\boxed{4}<|im_end|>\n"
        in rendered,
        "the assistant turn does not match the expected reconstitution; the Jinja's "
        "split-on-'</think>' and re-emit logic may have changed",
        rendered,
    )
    check(
        "3.C no spurious generation prompt",
        not rendered.endswith("<|im_start|>assistant\n"),
        "the template appended a generation prompt despite add_generation_prompt=False",
        rendered,
    )
    check(
        "3.D <think> appears exactly once",
        rendered.count("<think>") == 1,
        "the template emitted multiple <think> blocks; check the assistant branch "
        "for double-injection between the reasoning_content path and the inline-<think> path",
        rendered,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--template",
        type=Path,
        default=REPO_ROOT / "chat_template" / "chat_template.jinja",
    )
    args = parser.parse_args()

    if not args.template.exists():
        setup_error(f"template file not found at {args.template}")
    template = args.template.read_text(encoding="utf-8")
    if not template.strip():
        setup_error(f"template file is empty: {args.template}")

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        setup_error(f"transformers not installed: {e}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        setup_error(
            f"failed to load tokenizer for {args.model}: {type(e).__name__}: {e}. "
            f"Likely causes: HF_HOME unset, no internet on RCP, or model gated."
        )

    tokenizer.chat_template = template
    if tokenizer.chat_template != template:
        setup_error(
            "tokenizer.chat_template differs from the assigned string after assignment. "
            "A future transformers version may be sanitizing the template; investigate "
            "before proceeding."
        )

    print(f"loaded model:    {args.model}")
    print(f"loaded template: {args.template}  ({len(template)} chars)")
    print()

    test_generation_prompt(tokenizer)
    print()
    test_kwarg_cant_disable_thinking(tokenizer)
    print()
    test_assistant_round_trip(tokenizer)
    print()

    total = _results["passed"] + _results["failed"]
    if _results["failed"] == 0:
        print(f"=== Summary: {total}/{total} passed ===")
        sys.exit(EXIT_OK)

    print(
        f"=== Summary: {_results['passed']}/{total} passed "
        f"({_results['failed']} failures) ==="
    )
    for name in _results["failures"]:
        print(f"  - {name}")
    print()
    print(
        "Exit code 1 means the chat template (Jinja) is the bug, not this script. "
        "Fix in emainelpe-shared and propagate."
    )
    sys.exit(EXIT_TEMPLATE_BUG)


if __name__ == "__main__":
    main()
