"""Checkpoint loading shared by pairwise evaluation and single-item scoring."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import torch
from transformers import AutoModel, AutoTokenizer

from physics_difficulty.models.qwen_pairwise import QwenPairwiseRater
from physics_difficulty.schema import FEATURE_VALUES, LEGACY_MERGED_FEATURE_VALUES


def checkpoint_feature_values(
    checkpoint_config: dict[str, Any], state: dict[str, Any]
) -> dict[str, list[str]]:
    configured = checkpoint_config.get("feature_values")
    if configured is not None:
        if not isinstance(configured, dict):
            raise ValueError("checkpoint feature_values must be an object")
        return {str(name): [str(value) for value in values] for name, values in configured.items()}
    auxiliary_state = state.get("auxiliary_heads") or {}
    step_weight = auxiliary_state.get("step_count.weight")
    if step_weight is not None and int(step_weight.shape[0]) == 4:
        return {name: list(values) for name, values in LEGACY_MERGED_FEATURE_VALUES.items()}
    return {name: list(values) for name, values in FEATURE_VALUES.items()}


def load_pairwise_rater(model_path: str, checkpoint_dir: str | Path, device: torch.device, bf16: bool = True) -> tuple[QwenPairwiseRater, Any]:
    checkpoint = Path(checkpoint_dir)
    for required in (checkpoint / "adapter", checkpoint / "pairwise_head.pt"):
        if not required.exists():
            raise ValueError(f"checkpoint is missing {required}")
    tokenizer_source = checkpoint / "tokenizer" if (checkpoint / "tokenizer").is_dir() else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if bf16 and device.type == "cuda" else torch.float32
    base = AutoModel.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=True,
        device_map={"": device.index or 0} if device.type == "cuda" else None,
    )
    from peft import PeftModel
    backbone = PeftModel.from_pretrained(base, checkpoint / "adapter", is_trainable=False)
    config_path = checkpoint / "pairwise_config.json"
    checkpoint_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    auxiliary_features = bool(checkpoint_config.get("auxiliary_features", False))
    state = torch.load(checkpoint / "pairwise_head.pt", map_location=device)
    feature_values = checkpoint_feature_values(checkpoint_config, state)
    model = QwenPairwiseRater(
        backbone, auxiliary_features=auxiliary_features, feature_values=feature_values
    ).to(device)
    model.norm.load_state_dict(state["norm"])
    model.score_head.load_state_dict(state["score_head"])
    if auxiliary_features:
        if "auxiliary_heads" not in state:
            raise ValueError("V2 checkpoint is missing auxiliary_heads")
        model.auxiliary_heads.load_state_dict(state["auxiliary_heads"])
    model.eval()
    return model, tokenizer
