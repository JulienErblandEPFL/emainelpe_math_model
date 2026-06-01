"""Merge a trained LoRA adapter into ``Qwen/Qwen3-1.7B`` and write a
complete, vLLM-compatible checkpoint to disk.

This is the merge half of the former ``scripts/merge_and_push.py``. It
writes everything the course CI needs (merged safetensors, ``config.json``,
tokenizer files, locked chat template, and ``generation_config.json``)
to ``--output-dir`` and runs a vLLM smoke check that the merged model
emits a parseable ``\\boxed{...}``. It does NOT push anywhere — use
``scripts/push.py`` for that.

Pure helpers (``write_generation_config``, ``chat_templates_byte_match``,
``read_saved_chat_template``, ``run_file_preflight``) live at module
scope and are CPU-testable. Heavy pieces (``transformers``, ``peft``,
``vllm``) are deferred into ``main()``.

Usage::

    python scripts/merge.py \\
        --adapter-dir <path/to/adapter/final> \\
        --output-dir  <path/to/merged_model>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("merge")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCKED_TEMPLATE = REPO_ROOT / "chat_template" / "chat_template.jinja"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "merged_model"

BASE_MODEL_ID = "Qwen/Qwen3-1.7B"

# Qwen3 special-token IDs (from the published tokenizer config). Pinned
# here so the generation_config.json schema is identical across merges.
QWEN3_BOS_TOKEN_ID = 151643
QWEN3_PAD_TOKEN_ID = 151643
QWEN3_EOS_TOKEN_IDS = (151645, 151643)
TRANSFORMERS_VERSION = "4.51.0"

GENERATION_CONFIG_REQUIRED_KEYS = (
    "bos_token_id",
    "do_sample",
    "eos_token_id",
    "pad_token_id",
    "temperature",
    "top_k",
    "top_p",
    "transformers_version",
)


class PreflightError(RuntimeError):
    """Raised when a pre-write sanity check fails."""


# =============================================================================
# Pure helpers — CPU-testable, no torch/peft/vllm imports.
# =============================================================================

def build_generation_config(
    *, temperature: float, top_p: float, top_k: int
) -> dict:
    """Build the generation_config.json payload.

    Schema is fixed by the CI eval contract (see CLAUDE.md). Only
    sampling values come from the CLI; everything else is locked.
    """
    return {
        "bos_token_id": QWEN3_BOS_TOKEN_ID,
        "do_sample": True,
        "eos_token_id": list(QWEN3_EOS_TOKEN_IDS),
        "pad_token_id": QWEN3_PAD_TOKEN_ID,
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "transformers_version": TRANSFORMERS_VERSION,
    }


def write_generation_config(
    output_dir: Path,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_generation_config(
        temperature=temperature, top_p=top_p, top_k=top_k
    )
    target = output_dir / "generation_config.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def chat_templates_byte_match(saved: Path, locked: Path) -> bool:
    """Byte-for-byte compare two chat-template files."""
    return Path(saved).read_bytes() == Path(locked).read_bytes()


def read_saved_chat_template(output_dir: Path) -> str:
    """Read the chat template that ``tokenizer.save_pretrained`` stored.

    Newer transformers versions write it to ``chat_template.jinja`` as a
    sidecar; older ones embed it under ``tokenizer_config.json``'s
    ``chat_template`` key. Try the sidecar first, fall back to the
    config. Raise ``PreflightError`` if neither is present or the value
    is empty.
    """
    output_dir = Path(output_dir)
    sidecar = output_dir / "chat_template.jinja"
    if sidecar.is_file():
        text = sidecar.read_text(encoding="utf-8")
        if text:
            return text
    tok_config = output_dir / "tokenizer_config.json"
    if tok_config.is_file():
        try:
            data = json.loads(tok_config.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise PreflightError(
                f"tokenizer_config.json at {tok_config} is not valid JSON: {e}"
            ) from e
        ct = data.get("chat_template")
        if isinstance(ct, str) and ct:
            return ct
    raise PreflightError(
        f"No chat template found at {output_dir}: looked in "
        "chat_template.jinja and tokenizer_config.json[chat_template]."
    )


def _missing(items: Iterable[str], directory: Path) -> list[str]:
    return [name for name in items if not (directory / name).is_file()]


def run_file_preflight(output_dir: Path, locked_template: Path) -> None:
    """File-system preflight checks. Pure: no model loading.

    Raises ``PreflightError`` on the first failure with a clear message.
    """
    output_dir = Path(output_dir)
    locked_template = Path(locked_template)

    if not output_dir.is_dir():
        raise PreflightError(f"Output dir does not exist: {output_dir}")

    if not (output_dir / "config.json").is_file():
        raise PreflightError(f"Missing config.json under {output_dir}")

    safetensors = list(output_dir.glob("*.safetensors"))
    if not safetensors:
        raise PreflightError(
            f"No *.safetensors weight files found under {output_dir}"
        )

    missing_tok = _missing(
        ["tokenizer.json", "tokenizer_config.json"], output_dir
    )
    if missing_tok:
        raise PreflightError(
            f"Missing tokenizer file(s) under {output_dir}: {missing_tok}"
        )

    gen_cfg_path = output_dir / "generation_config.json"
    if not gen_cfg_path.is_file():
        raise PreflightError(
            f"Missing generation_config.json under {output_dir}"
        )
    try:
        gen_cfg = json.loads(gen_cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise PreflightError(
            f"generation_config.json is not valid JSON: {e}"
        ) from e
    missing_keys = [k for k in GENERATION_CONFIG_REQUIRED_KEYS if k not in gen_cfg]
    if missing_keys:
        raise PreflightError(
            f"generation_config.json missing required keys: {missing_keys}"
        )

    saved_template = read_saved_chat_template(output_dir)
    locked_text = locked_template.read_text(encoding="utf-8")
    if saved_template != locked_text:
        raise PreflightError(
            "Saved chat_template differs from the locked file at "
            f"{locked_template}. The Phase 3 merge requires byte-identical "
            "templates across all four experts; refusing to proceed."
        )


# =============================================================================
# CLI / main — heavy imports deferred into the body.
# =============================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge a LoRA adapter into Qwen3-1.7B and write a "
                    "complete vLLM-compatible checkpoint to disk.",
    )
    p.add_argument(
        "--adapter-dir", type=Path, required=True,
        help="Path to the trained adapter dir (must exist).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write the merged checkpoint locally. "
             f"Default: {DEFAULT_OUTPUT_DIR} (repo-relative).",
    )
    p.add_argument(
        "--temperature", type=float, default=0.4,
        help="Sampling temperature written into generation_config.json. "
             "Default 0.4: the calibrated peak from the 2026-05-11 SFT "
             "temperature sweep.",
    )
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument(
        "--locked-template", type=Path, default=DEFAULT_LOCKED_TEMPLATE,
        help="Locked chat template Jinja, used by the byte-diff preflight.",
    )
    return p.parse_args(argv)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(levelname)s: %(message)s",
        stream=sys.stdout,
    )


def merge_adapter_and_save(adapter_dir: Path, output_dir: Path) -> None:
    """Load base + adapter, merge, save merged weights and tokenizer."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading base model %s in bf16", BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    logger.info("Loading tokenizer from adapter dir %s", adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))

    logger.info("Loading LoRA adapter from %s", adapter_dir)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    logger.info("Calling merge_and_unload() — folding LoRA into base weights")
    merged = peft_model.merge_and_unload()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving merged model to %s (safe_serialization=True)", output_dir)
    merged.save_pretrained(str(output_dir), safe_serialization=True)

    logger.info("Saving tokenizer to %s", output_dir)
    tokenizer.save_pretrained(str(output_dir))


def smoke_inference_vllm(output_dir: Path, gen_cfg: dict) -> str:
    """vLLM smoke: 'What is 2+2?' → must contain <think> and \\boxed{."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    output_dir = Path(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
    if not tokenizer.chat_template:
        raise PreflightError(
            f"Tokenizer at {output_dir} has no chat_template after reload."
        )

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    logger.info("Loading merged checkpoint into vLLM for smoke inference")
    llm = LLM(
        model=str(output_dir),
        dtype="bfloat16",
        max_model_len=20480,
        gpu_memory_utilization=0.85,
    )
    sampling = SamplingParams(
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
        top_k=gen_cfg["top_k"],
        max_tokens=512,
        seed=42,
    )
    outputs = llm.generate([prompt], sampling)
    text = outputs[0].outputs[0].text

    logger.info("=== smoke output ===\n%s\n=== end ===", text)

    if "<think>" not in text:
        raise PreflightError(
            "Smoke output is missing <think>. Thinking mode looks broken."
        )
    if r"\boxed{" not in text:
        raise PreflightError(
            r"Smoke output is missing \boxed{. The model is not emitting "
            "answers in the format the CI scores against."
        )
    return text


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    args = _parse_args(argv)

    adapter_dir: Path = args.adapter_dir
    output_dir: Path = args.output_dir
    locked_template: Path = args.locked_template

    if not adapter_dir.is_dir():
        logger.error("Adapter dir does not exist: %s", adapter_dir)
        return 2
    if not locked_template.is_file():
        logger.error(
            "Locked chat template not found: %s. Cannot run byte-diff preflight.",
            locked_template,
        )
        return 2

    logger.info("adapter_dir   = %s", adapter_dir)
    logger.info("output_dir    = %s", output_dir)
    logger.info("base_model    = %s (bf16)", BASE_MODEL_ID)
    logger.info(
        "sampling      = temperature=%s top_p=%s top_k=%s",
        args.temperature, args.top_p, args.top_k,
    )

    # ---- merge + save ------------------------------------------------------
    try:
        merge_adapter_and_save(adapter_dir, output_dir)
    except Exception as e:
        logger.exception("Merge/save failed: %s", e)
        return 3

    # ---- write generation_config.json --------------------------------------
    try:
        gen_cfg_path = write_generation_config(
            output_dir,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        logger.info("Wrote %s", gen_cfg_path)
    except Exception as e:
        logger.exception("Writing generation_config.json failed: %s", e)
        return 4

    # ---- preflight (file checks + chat-template byte diff) -----------------
    try:
        run_file_preflight(output_dir, locked_template)
    except PreflightError as e:
        logger.error("Preflight failed: %s", e)
        return 5

    # ---- preflight (tokenizer round-trip) ----------------------------------
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(output_dir))
        if not tok.chat_template:
            raise PreflightError(
                f"tokenizer.chat_template is empty after reload from {output_dir}"
            )
        logger.info("Tokenizer round-trip OK; chat_template is set.")
    except PreflightError as e:
        logger.error("Preflight failed: %s", e)
        return 5
    except Exception as e:
        logger.exception("Tokenizer round-trip failed: %s", e)
        return 5

    # ---- smoke inference via vLLM -----------------------------------------
    try:
        gen_cfg = build_generation_config(
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k
        )
        smoke_inference_vllm(output_dir, gen_cfg)
    except PreflightError as e:
        logger.error("Smoke inference rejected: %s", e)
        return 6
    except Exception as e:
        logger.exception("Smoke inference failed: %s", e)
        return 6

    # ---- summary ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("merge: SUCCESS")
    print("=" * 60)
    print(f"  adapter_dir   : {adapter_dir}")
    print(f"  output_dir    : {output_dir}")
    print(f"  temperature   : {args.temperature}")
    print(f"  top_p         : {args.top_p}")
    print(f"  top_k         : {args.top_k}")
    print()
    print("Next steps:")
    print("  - Evaluate locally: python scripts/run_eval.py --model <output-dir>")
    print("  - Upload:           python scripts/push.py --model-dir <output-dir> --hf-repo <org/repo>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
