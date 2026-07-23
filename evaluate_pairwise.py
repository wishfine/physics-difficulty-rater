#!/usr/bin/env python3
"""Evaluate a soft Bradley-Terry checkpoint on held-out question pairs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.data.pairwise_dataset import PairwiseDifficultyDataset
from physics_difficulty.models.pairwise_loading import load_pairwise_rater
from physics_difficulty.pairwise.metrics import soft_pairwise_metrics
from physics_difficulty.schema import FEATURE_VALUES


def classification_metrics(predictions: list[int], targets: list[int], class_count: int) -> dict[str, float | int]:
    supports = [0] * class_count
    true_positive = [0] * class_count
    predicted = [0] * class_count
    correct = 0
    for prediction, target in zip(predictions, targets):
        supports[target] += 1
        predicted[prediction] += 1
        correct += int(prediction == target)
        true_positive[target] += int(prediction == target)
    recalls = [true_positive[index] / supports[index] for index in range(class_count) if supports[index]]
    f1s = []
    for index in range(class_count):
        if not supports[index]:
            continue
        precision = true_positive[index] / predicted[index] if predicted[index] else 0.0
        recall = true_positive[index] / supports[index]
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "records": len(targets),
        "accuracy": correct / max(1, len(targets)),
        "balanced_accuracy": sum(recalls) / max(1, len(recalls)),
        "macro_f1": sum(f1s) / max(1, len(f1s)),
        "observed_classes": len(recalls),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--predictions-file")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_pairwise_rater(args.model_path, checkpoint, device, args.bf16)

    auxiliary_features = bool(model.auxiliary_features)
    dataset = PairwiseDifficultyDataset(args.eval_file, tokenizer, args.max_length, require_auxiliary_features=auxiliary_features)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    probabilities: list[float] = []
    targets: list[float] = []
    prediction_rows = []
    auxiliary_predictions = {name: [] for name in FEATURE_VALUES}
    auxiliary_targets = {name: [] for name in FEATURE_VALUES}
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), int(batch["pair_count"]))
            batch_probabilities = torch.sigmoid(outputs["pair_logits"].float()).cpu().tolist()
            batch_targets = batch["soft_targets"].tolist()
            score_a = outputs["score_a"].float().cpu().tolist()
            score_b = outputs["score_b"].float().cpu().tolist()
            probabilities.extend(batch_probabilities)
            targets.extend(batch_targets)
            if auxiliary_features:
                for name in FEATURE_VALUES:
                    for side in ("a", "b"):
                        side_targets = batch[f"auxiliary_targets_{side}"][name]
                        side_predictions = outputs[f"auxiliary_logits_{side}"][name].argmax(dim=-1).cpu()
                        valid = side_targets.ge(0)
                        auxiliary_targets[name].extend(side_targets[valid].tolist())
                        auxiliary_predictions[name].extend(side_predictions[valid].tolist())
            for index, pair_id in enumerate(batch["pair_ids"]):
                prediction_rows.append({
                    "pair_id": pair_id,
                    "question_a_id": batch["question_a_ids"][index],
                    "question_b_id": batch["question_b_ids"][index],
                    "score_a": score_a[index],
                    "score_b": score_b[index],
                    "predicted_probability_a_harder": batch_probabilities[index],
                    "teacher_soft_target": batch_targets[index],
                })

    metrics = {
        "records": len(dataset),
        "pairwise": soft_pairwise_metrics(probabilities, targets),
        "checkpoint_dir": str(checkpoint.resolve()),
        "eval_file": str(Path(args.eval_file).resolve()),
    }
    if auxiliary_features:
        metrics["auxiliary"] = {
            name: classification_metrics(auxiliary_predictions[name], auxiliary_targets[name], len(values))
            for name, values in FEATURE_VALUES.items()
        }
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.predictions_file:
        predictions = Path(args.predictions_file)
        predictions.parent.mkdir(parents=True, exist_ok=True)
        with predictions.open("w", encoding="utf-8") as target:
            for row in prediction_rows:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
