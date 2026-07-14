#!/usr/bin/env python3
"""Evaluate a saved V2 adapter and its auxiliary classification heads."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from physics_difficulty.data.dataset import DifficultyDataset
from physics_difficulty.models.qwen_difficulty import QwenDifficultyRater
from physics_difficulty.schema import DIFFICULTY_LEVELS, FEATURE_VALUES


def macro_accuracy(predictions: list[int], labels: list[int], classes: int) -> float:
    values = []
    for class_id in range(classes):
        indices = [index for index, label in enumerate(labels) if label == class_id]
        if indices:
            values.append(sum(predictions[index] == class_id for index in indices) / len(indices))
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint_dir", required=True, help="directory containing adapter/ and difficulty_heads.pt")
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer_source = checkpoint / "tokenizer" if (checkpoint / "tokenizer").is_dir() else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True)
    from peft import PeftModel
    backbone = PeftModel.from_pretrained(backbone, checkpoint / "adapter")
    model = QwenDifficultyRater(backbone).to(device)
    head_state = torch.load(checkpoint / "difficulty_heads.pt", map_location=device)
    model.norm.load_state_dict(head_state["norm"])
    model.difficulty_head.load_state_dict(head_state["difficulty_head"])
    model.feature_heads.load_state_dict(head_state["feature_heads"])
    model.eval()

    dataset = DifficultyDataset(args.eval_file, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    difficulty_predictions: list[int] = []
    difficulty_labels: list[int] = []
    feature_predictions = {name: [] for name in FEATURE_VALUES}
    feature_labels = {name: [] for name in FEATURE_VALUES}
    with torch.no_grad():
        for batch in loader:
            output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            difficulty_predictions.extend(output["difficulty_logits"].float().argmax(dim=-1).cpu().tolist())
            difficulty_labels.extend(batch["difficulty_labels"].tolist())
            for name, logits in output["feature_logits"].items():
                feature_predictions[name].extend(logits.float().argmax(dim=-1).cpu().tolist())
                feature_labels[name].extend(batch["feature_labels"][name].tolist())

    result = {
        "records": len(difficulty_labels),
        "difficulty_accuracy": sum(a == b for a, b in zip(difficulty_predictions, difficulty_labels)) / max(1, len(difficulty_labels)),
        "difficulty_macro_accuracy": macro_accuracy(difficulty_predictions, difficulty_labels, len(DIFFICULTY_LEVELS)),
        "difficulty_class_support": {level: difficulty_labels.count(index) for index, level in enumerate(DIFFICULTY_LEVELS)},
        "feature_accuracy": {
            name: sum(a == b for a, b in zip(feature_predictions[name], feature_labels[name])) / max(1, len(feature_labels[name]))
            for name in FEATURE_VALUES
        },
        "checkpoint_dir": str(checkpoint.resolve()),
    }
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
