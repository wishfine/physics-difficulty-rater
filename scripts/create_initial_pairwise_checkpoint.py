#!/usr/bin/env python3
"""Create a seed-matched untrained LoRA + scalar-head comparison baseline."""
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

from physics_difficulty.models.qwen_pairwise import QwenPairwiseRater


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModel.from_pretrained(args.model_path, dtype=dtype, trust_remote_code=True, device_map={"": 0} if device.type == "cuda" else None)
    from peft import LoraConfig, get_peft_model
    targets = sorted({name.split(".")[-1] for name, module in base.named_modules() if isinstance(module, torch.nn.Linear)})
    backbone = get_peft_model(base, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=targets, bias="none", task_type="FEATURE_EXTRACTION"))
    model = QwenPairwiseRater(backbone).to(device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "tokenizer")
    torch.save({"norm": model.norm.state_dict(), "score_head": model.score_head.state_dict()}, output / "pairwise_head.pt")
    state = {"schema_version": "pairwise_initial_checkpoint_v1", "model_path": str(Path(args.model_path).resolve()), "seed": args.seed, "training_steps": 0}
    (output / "initial_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**state, "checkpoint_dir": str(output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
