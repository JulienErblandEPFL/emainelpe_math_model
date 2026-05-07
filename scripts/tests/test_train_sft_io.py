"""Pure-Python tests for the I/O helpers in scripts/train_sft.

Both helpers are stdlib + pyyaml only — no torch/peft/trl required.
"""
from __future__ import annotations

from pathlib import Path

from scripts.train_sft import load_chat_template, load_lora_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LORA_YAML = REPO_ROOT / "configs" / "lora.yaml"
CHAT_TEMPLATE = REPO_ROOT / "chat_template" / "chat_template.jinja"


def test_load_lora_yaml_exposes_locked_keys():
    cfg = load_lora_yaml(LORA_YAML)
    assert cfg["base_model"] == "Qwen/Qwen3-1.7B"
    assert cfg["max_seq_length"] == 4096
    assert cfg["lora"]["r"] == 32
    assert cfg["lora"]["alpha"] == 64
    assert cfg["lora"]["dropout"] == 0.05
    assert cfg["lora"]["bias"] == "none"
    assert cfg["lora"]["task_type"] == "CAUSAL_LM"
    assert set(cfg["lora"]["target_modules"]) == {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    }


def test_load_chat_template_returns_byte_identical_string():
    raw = CHAT_TEMPLATE.read_bytes()
    loaded = load_chat_template(CHAT_TEMPLATE)
    assert loaded.encode("utf-8") == raw
