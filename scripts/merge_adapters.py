"""Weight-space linear merge of multiple LoRA adapters.

Loads N adapter directories (each containing ``adapter_config.json`` +
``adapter_model.safetensors``), verifies they share an identical LoRA
spec, computes a weighted sum of their tensors, and writes the merged
adapter to ``--output-dir`` together with the tokenizer files copied
verbatim from the first adapter.

The Phase 3 merge (``configs/lora.yaml`` locked at r=32, alpha=64, the
seven target modules) requires byte-identical LoRA spec across source
adapters; ``validate_spec_match`` enforces that. Mismatched specs fail
fast with a clear message.

CLI::

    python scripts/merge_adapters.py \\
        --adapter /path/to/adapter1 --weight 0.5 \\
        --adapter /path/to/adapter2 --weight 0.3 \\
        --adapter /path/to/adapter3 --weight 0.2 \\
        --output-dir /path/to/merged_adapter \\
        [--dare-drop-rate 0.2] [--seed 42]

Pure helpers (``validate_weights``, ``validate_spec_match``,
``load_adapter_config``) are CPU-testable. Tensor operations
(``weighted_sum_tensors``, ``dare_drop_and_rescale``,
``merge_state_dicts``, ``load_adapter_state_dict``,
``save_merged_adapter``) lazy-import ``torch`` and ``safetensors`` so
the test suite runs on a laptop without those wheels — tests gate them
with ``pytest.importorskip``.

Out of scope:
  - Rank-expanding merges (concatenating adapters into rank Nr). That
    would break the Phase 3 ``configs/lora.yaml`` contract.
  - Base-model fold-in. ``merge_and_push.py`` handles that.
  - TIES sign-resolution / trim. Pure linear + optional DARE only.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("merge_adapters")

# Scalar / list fields that must agree across all source adapters.
# target_modules is checked separately (order-insensitive set compare).
LORA_SPEC_KEYS = (
    "r",
    "lora_alpha",
    "lora_dropout",
    "bias",
    "task_type",
    "modules_to_save",
)

WEIGHT_SUM_TOLERANCE = 1e-6

# Tokenizer / chat-template sidecars copied verbatim from adapter[0].
# Only files that actually exist are copied; missing entries are skipped.
TOKENIZER_SIDECAR_FILES = (
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)


class SpecMismatchError(ValueError):
    """Two adapters disagree on a locked LoRA spec field."""


class WeightValidationError(ValueError):
    """CLI weights are malformed (count or sum)."""


# =============================================================================
# Pure helpers — no torch / no safetensors. CPU-testable.
# =============================================================================

def load_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Missing adapter_config.json under {adapter_dir}"
        )
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def validate_weights(weights: list[float], n_adapters: int) -> None:
    if len(weights) != n_adapters:
        raise WeightValidationError(
            f"Got {n_adapters} adapters but {len(weights)} weights; "
            "counts must match."
        )
    total = sum(weights)
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise WeightValidationError(
            f"Weights must sum to 1.0 (within {WEIGHT_SUM_TOLERANCE:g}); "
            f"got {weights} summing to {total!r}."
        )


def validate_spec_match(configs: list[dict[str, Any]]) -> None:
    """Verify all configs share an identical LoRA spec.

    Raises ``SpecMismatchError`` on the first disagreement found.
    ``target_modules`` is compared as a set so JSON-order drift does
    not trip the check.
    """
    if len(configs) < 2:
        return
    ref = configs[0]
    for i, cfg in enumerate(configs[1:], start=1):
        for key in LORA_SPEC_KEYS:
            if ref.get(key) != cfg.get(key):
                raise SpecMismatchError(
                    f"Adapter[0] and adapter[{i}] disagree on '{key}': "
                    f"{ref.get(key)!r} vs {cfg.get(key)!r}. The Phase 3 "
                    "merge requires identical LoRA spec across all "
                    "source adapters."
                )
        ref_tm = frozenset(ref.get("target_modules") or [])
        new_tm = frozenset(cfg.get("target_modules") or [])
        if ref_tm != new_tm:
            raise SpecMismatchError(
                f"Adapter[0] and adapter[{i}] disagree on target_modules: "
                f"{sorted(ref_tm)} vs {sorted(new_tm)}."
            )


# =============================================================================
# Tensor ops — lazy-import torch. Gated in tests on ``importorskip``.
# =============================================================================

def weighted_sum_tensors(tensors, weights):
    """Compute ``sum(w_i * t_i)`` over a list of equal-shape tensors.

    Accumulates in fp32 for numerical stability, then casts back to the
    input dtype (bf16/fp16 LoRA tensors round-trip exactly here since
    the cast is the last op).
    """
    import torch

    if len(tensors) != len(weights):
        raise ValueError(
            f"weighted_sum_tensors: {len(tensors)} tensors vs "
            f"{len(weights)} weights"
        )
    ref_shape = tensors[0].shape
    ref_dtype = tensors[0].dtype
    accum = torch.zeros(ref_shape, dtype=torch.float32)
    for t, w in zip(tensors, weights):
        if t.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch in weighted_sum_tensors: "
                f"{tuple(t.shape)} vs {tuple(ref_shape)}"
            )
        accum = accum + float(w) * t.to(torch.float32)
    return accum.to(ref_dtype)


def dare_drop_and_rescale(tensor, drop_rate: float, generator=None):
    """DARE-style drop-and-rescale.

    Zero each parameter with probability ``drop_rate``, then divide
    survivors by ``1 - drop_rate`` so the per-parameter expectation is
    preserved. Reduces interference between adapters before the
    weighted sum.

    ``drop_rate == 0.0`` short-circuits and returns the input tensor
    unchanged, so a pure-linear merge is deterministic and consumes no
    RNG state.
    """
    import torch

    if not (0.0 <= drop_rate < 1.0):
        raise ValueError(f"drop_rate must be in [0, 1); got {drop_rate}")
    if drop_rate == 0.0:
        return tensor

    keep_prob = 1.0 - drop_rate
    probs = torch.full(tensor.shape, keep_prob, dtype=torch.float32)
    mask = torch.bernoulli(probs, generator=generator).to(tensor.dtype)
    return (tensor * mask) / keep_prob


def merge_state_dicts(
    state_dicts: list[dict],
    weights: list[float],
    *,
    drop_rate: float = 0.0,
    seed: int | None = None,
) -> dict:
    """Weighted sum across N LoRA adapter state-dicts.

    All input dicts must share the same key set. When ``drop_rate > 0``,
    DARE drop-and-rescale is applied independently to each adapter's
    tensor before the weighted sum, using a torch.Generator seeded
    with ``seed`` for reproducibility.
    """
    import torch

    if not state_dicts:
        raise ValueError("merge_state_dicts: empty state_dicts list")
    if len(state_dicts) != len(weights):
        raise ValueError(
            f"merge_state_dicts: {len(state_dicts)} dicts vs "
            f"{len(weights)} weights"
        )

    ref_keys = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        if set(sd.keys()) != ref_keys:
            extra = sorted(set(sd.keys()) - ref_keys)[:5]
            missing = sorted(ref_keys - set(sd.keys()))[:5]
            raise ValueError(
                f"State-dict key mismatch between adapter[0] and "
                f"adapter[{i}]: extra={extra}... missing={missing}..."
            )

    gen = None
    if drop_rate > 0.0 and seed is not None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))

    merged: dict = {}
    for key in sorted(ref_keys):
        tensors = [sd[key] for sd in state_dicts]
        if drop_rate > 0.0:
            tensors = [
                dare_drop_and_rescale(t, drop_rate, generator=gen)
                for t in tensors
            ]
        merged[key] = weighted_sum_tensors(tensors, weights)
    return merged


# =============================================================================
# File I/O — lazy-imports safetensors.
# =============================================================================

def load_adapter_state_dict(adapter_dir: Path) -> dict:
    """Load ``adapter_model.safetensors`` into a CPU state dict."""
    from safetensors.torch import load_file

    path = Path(adapter_dir) / "adapter_model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing adapter_model.safetensors at {path}"
        )
    return load_file(str(path), device="cpu")


def save_merged_adapter(
    output_dir: Path,
    state_dict: dict,
    source_adapter: Path,
) -> None:
    """Write merged state dict + copy adapter_config + tokenizer files.

    The output directory ends up structurally identical to a single-run
    adapter directory: ``adapter_model.safetensors`` +
    ``adapter_config.json`` + tokenizer sidecars, ready to load via
    ``PeftModel.from_pretrained``.
    """
    from safetensors.torch import save_file

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_file(state_dict, str(output_dir / "adapter_model.safetensors"))

    source_adapter = Path(source_adapter)
    shutil.copy2(
        source_adapter / "adapter_config.json",
        output_dir / "adapter_config.json",
    )

    for name in TOKENIZER_SIDECAR_FILES:
        src = source_adapter / name
        if src.is_file():
            shutil.copy2(src, output_dir / name)


# =============================================================================
# CLI / main.
# =============================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Weight-space linear merge of N LoRA adapters.",
    )
    p.add_argument(
        "--adapter", action="append", type=Path, required=True,
        dest="adapters",
        help="Path to a LoRA adapter directory. Repeat once per adapter.",
    )
    p.add_argument(
        "--weight", action="append", type=float, required=True,
        dest="weights",
        help="Mixing weight for the most recent --adapter. Pair with each "
             "--adapter in order. Must sum to 1.0 (tol 1e-6).",
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where to write the merged adapter.",
    )
    p.add_argument(
        "--dare-drop-rate", type=float, default=0.0,
        help="DARE drop rate per parameter (0.0 = pure linear merge, "
             "deterministic). Typical DARE values: 0.1-0.3.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Seed for the DARE Bernoulli draws. Ignored when "
             "--dare-drop-rate=0.",
    )
    return p.parse_args(argv)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(levelname)s: %(message)s",
        stream=sys.stdout,
    )


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    args = _parse_args(argv)

    adapters: list[Path] = args.adapters
    weights: list[float] = args.weights

    try:
        validate_weights(weights, n_adapters=len(adapters))
    except WeightValidationError as e:
        logger.error("%s", e)
        return 2

    configs = []
    for d in adapters:
        try:
            configs.append(load_adapter_config(d))
        except FileNotFoundError as e:
            logger.error("%s", e)
            return 2

    try:
        validate_spec_match(configs)
    except SpecMismatchError as e:
        logger.error("LoRA spec mismatch: %s", e)
        return 3

    logger.info("Merging %d adapters:", len(adapters))
    for d, w in zip(adapters, weights):
        logger.info("  %.4f  %s", w, d)
    logger.info(
        "dare_drop_rate = %.3f (seed=%d)", args.dare_drop_rate, args.seed,
    )

    logger.info("Loading state dicts (lazy import of safetensors)")
    state_dicts = [load_adapter_state_dict(d) for d in adapters]

    logger.info(
        "Computing weighted sum across %d tensors per adapter",
        len(state_dicts[0]),
    )
    merged = merge_state_dicts(
        state_dicts,
        weights,
        drop_rate=args.dare_drop_rate,
        seed=args.seed,
    )

    logger.info("Writing merged adapter to %s", args.output_dir)
    save_merged_adapter(args.output_dir, merged, source_adapter=adapters[0])

    print("\n" + "=" * 60)
    print("merge_adapters: SUCCESS")
    print("=" * 60)
    print(f"  output_dir     : {args.output_dir}")
    print(f"  n_adapters     : {len(adapters)}")
    print(f"  weights        : {weights}")
    print(f"  dare_drop_rate : {args.dare_drop_rate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
