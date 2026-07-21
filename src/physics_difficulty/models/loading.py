"""Load a resumable local-rater checkpoint for evaluation or inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from physics_difficulty.models.qwen_difficulty import QwenDifficultyRater


def _require_checkpoint_files(checkpoint_dir: Path) -> None:
    required = (checkpoint_dir / "adapter", checkpoint_dir / "difficulty_heads.pt")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Invalid rater checkpoint {checkpoint_dir}: missing {', '.join(missing)}")


def load_rater(model_path: str | Path, checkpoint_dir: str | Path) -> tuple[QwenDifficultyRater, Any, torch.device]:
    """Restore Qwen, LoRA adapter, and all task heads in evaluation mode.

    Training checkpoints intentionally save only PEFT adapter weights plus the
    local heads. The immutable Qwen base is always loaded from ``model_path``.
    """
    checkpoint = Path(checkpoint_dir)
    _require_checkpoint_files(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer_source = checkpoint / "tokenizer" if (checkpoint / "tokenizer").is_dir() else Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    from peft import PeftModel
    backbone = PeftModel.from_pretrained(base, checkpoint / "adapter", is_trainable=False)
    model = QwenDifficultyRater(backbone).to(device)
    head_state = torch.load(checkpoint / "difficulty_heads.pt", map_location=device)
    required_heads = ("norm", "difficulty_head", "feature_heads")
    missing_heads = [name for name in required_heads if name not in head_state]
    if missing_heads:
        raise ValueError(f"Invalid difficulty_heads.pt in {checkpoint}: missing {missing_heads}")
    model.norm.load_state_dict(head_state["norm"])
    model.difficulty_head.load_state_dict(head_state["difficulty_head"])
    model.feature_heads.load_state_dict(head_state["feature_heads"])
    model.eval()
    return model, tokenizer, device
