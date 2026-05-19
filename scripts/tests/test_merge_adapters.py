"""CPU-only unit tests for scripts/merge_adapters.

Pure-Python validation (weight sum, LoRA spec match) runs everywhere.
Tensor-touching tests are gated on ``pytest.importorskip`` so the suite
runs on the user's CPU-only laptop AND on the cluster pod where torch +
safetensors are installed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.merge_adapters import (
    SpecMismatchError,
    WeightValidationError,
    merge_state_dicts,
    save_merged_adapter,
    validate_spec_match,
    validate_weights,
)


def _spec(**overrides):
    """Build a v3-shaped LoRA adapter_config dict; override specific keys."""
    base = {
        "base_model_name_or_path": "Qwen/Qwen3-1.7B",
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "modules_to_save": None,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    }
    base.update(overrides)
    return base


# -----------------------------------------------------------------------------
# Test 1 — weight validation must enforce sum-to-1.0 (and count match).
# Pure Python; no torch import.
# -----------------------------------------------------------------------------

def test_validate_weights_enforces_sum_to_one():
    # Valid: sums to exactly 1.0 within tolerance.
    validate_weights([0.5, 0.3, 0.2], n_adapters=3)
    validate_weights([1.0], n_adapters=1)

    # Reject non-unit sum.
    with pytest.raises(WeightValidationError, match="sum to 1.0"):
        validate_weights([0.5, 0.3], n_adapters=2)

    # Reject count mismatch (3 adapters, 2 weights).
    with pytest.raises(WeightValidationError, match="counts must match"):
        validate_weights([0.5, 0.5], n_adapters=3)


# -----------------------------------------------------------------------------
# Test 2 — LoRA spec validation must fail on mismatched r.
# Pure Python; no torch import.
# -----------------------------------------------------------------------------

def test_validate_spec_match_rejects_mismatched_r():
    # Identical specs accepted (and target_modules in shuffled order
    # also accepted — set compare, not list compare).
    a = _spec()
    b = _spec(target_modules=list(reversed(a["target_modules"])))
    validate_spec_match([a, b])

    # Mismatched r raises with the offending key in the message.
    with pytest.raises(SpecMismatchError, match="'r'"):
        validate_spec_match([_spec(), _spec(r=16)])


# -----------------------------------------------------------------------------
# Test 3 — weighted sum of synthetic tensors matches the hand-computed
# closed form. Confirms the arithmetic + dtype handling.
# -----------------------------------------------------------------------------

def test_weighted_sum_matches_expected():
    torch = pytest.importorskip("torch")
    from scripts.merge_adapters import weighted_sum_tensors

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    # 50/50 mix.
    result = weighted_sum_tensors([a, b], [0.5, 0.5])
    expected = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    assert torch.allclose(result, expected)

    # Asymmetric mix; 3-way.
    c = torch.full((2, 2), 10.0)
    result3 = weighted_sum_tensors([a, b, c], [0.25, 0.5, 0.25])
    expected3 = 0.25 * a + 0.5 * b + 0.25 * c
    assert torch.allclose(result3, expected3)


# -----------------------------------------------------------------------------
# Test 4 — DARE drop_rate=0.0 must be byte-equal to a pure linear merge.
# Guarantees the new --dare-drop-rate=0.0 default is exactly the old
# no-DARE behavior (no RNG state consumed, no rounding drift).
# -----------------------------------------------------------------------------

def test_dare_drop_rate_zero_equals_pure_linear():
    torch = pytest.importorskip("torch")

    sd1 = {
        "lora_A": torch.randn(4, 8),
        "lora_B": torch.randn(8, 4),
    }
    sd2 = {
        "lora_A": torch.randn(4, 8),
        "lora_B": torch.randn(8, 4),
    }

    merged = merge_state_dicts([sd1, sd2], [0.6, 0.4], drop_rate=0.0)
    expected = {k: 0.6 * sd1[k] + 0.4 * sd2[k] for k in sd1}

    for k in expected:
        assert torch.allclose(merged[k], expected[k])


# -----------------------------------------------------------------------------
# Test 5 — end-to-end output structure: merged dir contains the
# safetensors weights, adapter_config.json, chat_template, and
# tokenizer sidecars. Also confirms the safetensors round-trip
# preserves the weighted-sum semantics.
# -----------------------------------------------------------------------------

def test_merge_writes_full_output_structure(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file, save_file

    spec = _spec()
    adapter_dirs = []
    state_dicts = []
    for i in range(2):
        d = tmp_path / f"adapter_{i}"
        d.mkdir()
        (d / "adapter_config.json").write_text(json.dumps(spec))
        # Tokenizer sidecars (must propagate to the merged dir).
        (d / "tokenizer_config.json").write_text("{}")
        (d / "chat_template.jinja").write_text("dummy template")
        sd = {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight":
                torch.full((4, 8), float(i + 1)),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight":
                torch.full((8, 4), float(i + 1) * 0.1),
        }
        save_file(sd, str(d / "adapter_model.safetensors"))
        adapter_dirs.append(d)
        state_dicts.append(sd)

    merged = merge_state_dicts(state_dicts, [0.5, 0.5], drop_rate=0.0)
    output_dir = tmp_path / "merged"
    save_merged_adapter(output_dir, merged, source_adapter=adapter_dirs[0])

    # All four file types present.
    assert (output_dir / "adapter_model.safetensors").is_file()
    assert (output_dir / "adapter_config.json").is_file()
    assert (output_dir / "chat_template.jinja").is_file()
    assert (output_dir / "tokenizer_config.json").is_file()

    # adapter_config preserved (r=32 from the source).
    cfg = json.loads((output_dir / "adapter_config.json").read_text())
    assert cfg["r"] == 32
    assert cfg["lora_alpha"] == 64

    # Round-trip safetensors and confirm weighted-sum semantics.
    reloaded = load_file(
        str(output_dir / "adapter_model.safetensors"), device="cpu",
    )
    key_a = "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    expected_a = 0.5 * torch.full((4, 8), 1.0) + 0.5 * torch.full((4, 8), 2.0)
    assert torch.allclose(reloaded[key_a], expected_a)
