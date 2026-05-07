"""Tests for choose_precision.

Pure logic on (requested, cuda_available, bf16_supported) -> "bf16"|"fp16".
Probing torch.cuda happens in main(); this helper takes booleans so we can
test all branches on a CPU laptop with no CUDA installed.
"""
from __future__ import annotations

import pytest

from scripts.train_sft import choose_precision


# ---- auto branch -----------------------------------------------------------

def test_auto_picks_bf16_when_supported():
    assert choose_precision("auto", cuda_available=True, bf16_supported=True) == "bf16"


def test_auto_picks_fp16_when_bf16_not_supported():
    assert choose_precision("auto", cuda_available=True, bf16_supported=False) == "fp16"


# ---- explicit overrides ----------------------------------------------------

def test_explicit_bf16_passes_when_supported():
    assert choose_precision("bf16", cuda_available=True, bf16_supported=True) == "bf16"


def test_explicit_bf16_raises_when_unsupported():
    with pytest.raises(RuntimeError, match="bf16"):
        choose_precision("bf16", cuda_available=True, bf16_supported=False)


def test_explicit_fp16_always_returns_fp16_on_gpu():
    assert choose_precision("fp16", cuda_available=True, bf16_supported=True) == "fp16"
    assert choose_precision("fp16", cuda_available=True, bf16_supported=False) == "fp16"


# ---- no-CUDA refusal -------------------------------------------------------

def test_no_cuda_raises_runtime_error_under_auto():
    with pytest.raises(RuntimeError, match="CUDA"):
        choose_precision("auto", cuda_available=False, bf16_supported=False)


def test_no_cuda_raises_runtime_error_under_explicit():
    """Even an explicit --precision request must fail if there's no GPU.
    We refuse to run training on CPU silently."""
    with pytest.raises(RuntimeError, match="CUDA"):
        choose_precision("bf16", cuda_available=False, bf16_supported=False)
    with pytest.raises(RuntimeError, match="CUDA"):
        choose_precision("fp16", cuda_available=False, bf16_supported=False)


# ---- input validation ------------------------------------------------------

def test_unknown_precision_string_raises_value_error():
    with pytest.raises(ValueError, match="unknown precision"):
        choose_precision("int8", cuda_available=True, bf16_supported=True)
