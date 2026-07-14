#!/usr/bin/env python3
"""Batch inference with calibrated confidence and explicit fallback reasons."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from physics_difficulty.data.dataset import DifficultyDataset
from physics_difficulty.data.formatting import canonical_sections, diagnostics, format_question
from physics_difficulty.models.loading import load_rater
from physics_difficulty.schema import DIFFICULTY_LEVELS, FEATURE_VALUES, MULTI_LABEL_FEATURES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--input_file", required=True, help="raw question JSONL")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--calibration_file")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint_dir)
    calibration_path = Path(args.calibration_file) if args.calibration_file else checkpoint / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.is_file() else {}
    temperature = float(calibration.get("temperature", 1.0))
    feature_threshold = float(calibration.get("feature_thresholds", {}).get("problem_structure", 0.5))
    fallback_policy = calibration.get("fallback_policy", {"min_max_probability": 0.6, "min_margin": 0.1, "max_entropy": 1.2})
    raw_records = [json.loads(line) for line in Path(args.input_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    items = []
    for index, record in enumerate(raw_records):
        text = format_question(record)
        items.append({"id": str(record.get("question_id", index)), "text": text, "input_sections": canonical_sections(record), "diagnostics": diagnostics(record, text)})
    temporary = Path(args.output_file).with_suffix(".input.tmp.jsonl")
    temporary.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8")
    try:
        model, tokenizer, device = load_rater(args.model_path, checkpoint)
        dataset = DifficultyDataset(str(temporary), tokenizer, args.max_length, require_labels=False)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        results = []
        with torch.no_grad():
            for batch in loader:
                output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                probabilities = torch.softmax(output["difficulty_logits"].float() / temperature, dim=-1).cpu().tolist()
                feature_logits = {name: logits.float().cpu() for name, logits in output["feature_logits"].items()}
                for row_index, probability in enumerate(probabilities):
                    ordered = sorted(probability, reverse=True)
                    maximum, margin = ordered[0], ordered[0] - ordered[1]
                    entropy = -sum(value * math.log(max(value, 1e-12)) for value in probability)
                    reasons = []
                    meta, truncation = batch["metadata"][row_index], batch["truncation"][row_index]
                    if maximum < float(fallback_policy["min_max_probability"]): reasons.append("low_max_probability")
                    if margin < float(fallback_policy["min_margin"]): reasons.append("small_top1_top2_margin")
                    if entropy > float(fallback_policy["max_entropy"]): reasons.append("high_entropy")
                    if truncation["truncated"]: reasons.append("long_input_truncated")
                    if not meta.get("has_analysis"): reasons.append("missing_analysis")
                    if meta.get("image_dependency_risk") == "high": reasons.append("image_dependency_risk")
                    feature_output = {}
                    for name, values in FEATURE_VALUES.items():
                        logits = feature_logits[name][row_index]
                        if name in MULTI_LABEL_FEATURES:
                            feature_output[name] = [value for value, score in zip(values, torch.sigmoid(logits).tolist()) if score >= feature_threshold]
                        else:
                            feature_output[name] = values[int(logits.argmax().item())]
                    prediction_id = max(range(5), key=lambda class_id: probability[class_id])
                    results.append({"id": batch["ids"][row_index], "difficulty_level": DIFFICULTY_LEVELS[prediction_id], "difficulty_probability": probability, "difficulty_score": sum(value * (index + 1) for index, value in enumerate(probability)), "calibrated_confidence": maximum, "entropy": entropy, "top1_top2_margin": margin, "auxiliary_features": feature_output, "fallback_recommendation": bool(reasons), "fallback_reasons": reasons, "recommended_route": "vision_or_human" if "image_dependency_risk" in reasons else "api" if reasons else "local_model", "truncation": truncation})
        Path(args.output_file).write_text("".join(json.dumps(result, ensure_ascii=False) + "\n" for result in results), encoding="utf-8")
        print(json.dumps({"records": len(results), "fallback_count": sum(item["fallback_recommendation"] for item in results), "output_file": args.output_file}, ensure_ascii=False))
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
