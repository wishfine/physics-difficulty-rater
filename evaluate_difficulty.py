#!/usr/bin/env python3
"""Evaluate a checkpoint on a frozen labelled split; no training is performed."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from physics_difficulty.data.dataset import DifficultyDataset
from physics_difficulty.evaluation.metrics import calibration_metrics, classification_metrics
from physics_difficulty.models.loading import load_rater
from physics_difficulty.schema import DIFFICULTY_LEVELS, FEATURE_VALUES, MULTI_LABEL_FEATURES


def multi_label_metrics(predictions: list[list[int]], labels: list[list[int]], values: list[str]) -> dict:
    per_tag, f1s = {}, []
    for index, value in enumerate(values):
        tp = sum(prediction[index] and label[index] for prediction, label in zip(predictions, labels))
        fp = sum(prediction[index] and not label[index] for prediction, label in zip(predictions, labels))
        fn = sum(not prediction[index] and label[index] for prediction, label in zip(predictions, labels))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_tag[value] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(label[index] for label in labels)}
        f1s.append(f1)
    total_tp = sum(sum(prediction[index] and label[index] for prediction, label in zip(predictions, labels)) for index in range(len(values)))
    total_fp = sum(sum(prediction[index] and not label[index] for prediction, label in zip(predictions, labels)) for index in range(len(values)))
    total_fn = sum(sum(not prediction[index] and label[index] for prediction, label in zip(predictions, labels)) for index in range(len(values)))
    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    return {"exact_match_accuracy": sum(prediction == label for prediction, label in zip(predictions, labels)) / max(1, len(labels)), "micro_f1": 2 * precision * recall / max(1e-12, precision + recall), "macro_f1": sum(f1s) / len(f1s), "per_tag": per_tag}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--calibration_file")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint_dir)
    calibration_path = Path(args.calibration_file) if args.calibration_file else checkpoint / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.is_file() else {}
    temperature = float(calibration.get("temperature", 1.0))
    thresholds = calibration.get("feature_thresholds", {"problem_structure": 0.5})

    model, tokenizer, device = load_rater(args.model_path, checkpoint)
    dataset = DifficultyDataset(args.eval_file, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    predictions, labels, probabilities, metadata = [], [], [], []
    feature_predictions = {name: [] for name in FEATURE_VALUES}
    feature_labels = {name: [] for name in FEATURE_VALUES}
    with torch.no_grad():
        for batch in loader:
            output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probability = torch.softmax(output["difficulty_logits"].float() / temperature, dim=-1).cpu()
            probabilities.extend(probability.tolist())
            predictions.extend(probability.argmax(dim=-1).tolist())
            labels.extend(batch["difficulty_labels"].tolist())
            metadata.extend(batch["metadata"])
            for name, logits in output["feature_logits"].items():
                if name in MULTI_LABEL_FEATURES:
                    feature_predictions[name].extend((torch.sigmoid(logits.float()) >= float(thresholds.get(name, 0.5))).int().cpu().tolist())
                    feature_labels[name].extend(batch["feature_labels"][name].int().tolist())
                else:
                    feature_predictions[name].extend(logits.float().argmax(dim=-1).cpu().tolist())
                    feature_labels[name].extend(batch["feature_labels"][name].tolist())

    feature_metrics = {}
    for name, values in FEATURE_VALUES.items():
        if name in MULTI_LABEL_FEATURES:
            feature_metrics[name] = multi_label_metrics(feature_predictions[name], feature_labels[name], values)
        else:
            feature_metrics[name] = classification_metrics(feature_predictions[name], feature_labels[name], len(values))

    slices = {}
    for field in ("has_analysis", "has_subquestions", "input_length_bucket", "has_image_url", "image_dependency_risk", "raw_api_disagreement"):
        groups: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(metadata):
            groups[str(item.get(field, "unknown"))].append(index)
        slices[field] = {value: classification_metrics([predictions[index] for index in indices], [labels[index] for index in indices]) for value, indices in groups.items()}

    result = {
        "records": len(labels), "difficulty": classification_metrics(predictions, labels), "calibration": calibration_metrics(probabilities, labels),
        "difficulty_class_support": {level: labels.count(index) for index, level in enumerate(DIFFICULTY_LEVELS)}, "feature_metrics": feature_metrics,
        "slices": slices, "checkpoint_dir": str(checkpoint.resolve()), "temperature": temperature,
    }
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(labels), "macro_f1": result["difficulty"]["macro_f1"], "balanced_accuracy": result["difficulty"]["balanced_accuracy"], "output_file": str(output_file)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
