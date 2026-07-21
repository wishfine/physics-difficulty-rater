#!/usr/bin/env python3
"""Create an evaluation-only checkpoint before any local-rater training step."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from physics_difficulty.models.qwen_difficulty import QwenDifficultyRater


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True, help="Target checkpoint directory, for example checkpoint-initial")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    required = (output / "adapter", output / "difficulty_heads.pt")
    if any(path.exists() for path in required) and not args.force:
        raise FileExistsError(f"Initial checkpoint already exists at {output}; pass --force only when intentionally recreating it")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map={"": 0} if device.type == "cuda" else None,
    )
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    from peft import LoraConfig, get_peft_model
    targets = sorted({name.split(".")[-1] for name, module in base.named_modules() if isinstance(module, torch.nn.Linear)})
    if not targets:
        raise RuntimeError("No linear modules found for LoRA")
    backbone = get_peft_model(
        base,
        LoraConfig(r=8, lora_alpha=16, target_modules=targets, lora_dropout=0.05, bias="none", task_type="FEATURE_EXTRACTION"),
    )
    model = QwenDifficultyRater(backbone).to(device)
    output.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "tokenizer")
    torch.save(
        {"norm": model.norm.state_dict(), "difficulty_head": model.difficulty_head.state_dict(), "feature_heads": model.feature_heads.state_dict()},
        output / "difficulty_heads.pt",
    )
    (output / "initialization.json").write_text(
        json.dumps({"seed": args.seed, "model_path": str(Path(args.model_path).resolve()), "training_steps": 0, "purpose": "untrained_local_rater_baseline"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"checkpoint_dir": str(output.resolve()), "seed": args.seed, "training_steps": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
