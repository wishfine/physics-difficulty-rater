#!/usr/bin/env python3
"""Fit validation temperature scaling for five-way difficulty probabilities."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from physics_difficulty.data.dataset import DifficultyDataset
from physics_difficulty.models.loading import load_rater


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--validation_file", required=True)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    model, tokenizer, device = load_rater(args.model_path, args.checkpoint_dir)
    dataset = DifficultyDataset(args.validation_file, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            logits.append(output["difficulty_logits"].float().cpu())
            labels.append(batch["difficulty_labels"])
    logits_tensor, labels_tensor = torch.cat(logits), torch.cat(labels)
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits_tensor / log_temperature.exp().clamp_min(0.05), labels_tensor)
        loss.backward()
        return loss
    optimizer.step(closure)
    result = {"temperature": float(log_temperature.exp().detach().clamp_min(0.05)), "selection": {"validation_records": len(labels_tensor)}, "fallback_policy": {"min_max_probability": 0.6, "min_margin": 0.1, "max_entropy": 1.2, "source": "initial_validation_policy_review_required"}}
    output = Path(args.output_file) if args.output_file else Path(args.checkpoint_dir) / "calibration.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
