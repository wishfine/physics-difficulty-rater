"""Checkpoint loading shared by pairwise evaluation and single-item scoring."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import torch
from transformers import AutoModel, AutoTokenizer

from physics_difficulty.models.qwen_pairwise import QwenPairwiseRater


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
    model = QwenPairwiseRater(backbone, auxiliary_features=auxiliary_features).to(device)
    state = torch.load(checkpoint / "pairwise_head.pt", map_location=device)
    model.norm.load_state_dict(state["norm"])
    model.score_head.load_state_dict(state["score_head"])
    if auxiliary_features:
        if "auxiliary_heads" not in state:
            raise ValueError("V2 checkpoint is missing auxiliary_heads")
        model.auxiliary_heads.load_state_dict(state["auxiliary_heads"])
    model.eval()
    return model, tokenizer
