#!/usr/bin/env python3
"""Evaluate calibrated BT single-question predictions on adjudicated gold labels."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.evaluation.metrics import classification_metrics
from physics_difficulty.schema import DIFFICULTY_LEVELS


def load_unique(path: Path, id_keys: tuple[str, ...]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = next(
                (str(row[key]).strip() for key in id_keys if row.get(key) is not None and str(row[key]).strip()),
                "",
            )
            if not question_id:
                raise ValueError(f"{path}: line {line_number} lacks a question ID")
            if question_id in rows:
                raise ValueError(f"{path}: duplicate question ID {question_id}")
            rows[question_id] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-complete-coverage", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    predictions = load_unique(Path(args.predictions), ("question_id", "id"))
    gold = load_unique(Path(args.gold), ("id", "question_id"))
    common = sorted(set(predictions) & set(gold))
    missing_predictions = sorted(set(gold) - set(predictions))
    unexpected_predictions = sorted(set(predictions) - set(gold))
    if args.require_complete_coverage and (missing_predictions or unexpected_predictions):
        raise ValueError(
            f"prediction/gold coverage mismatch: missing={len(missing_predictions)}, "
            f"unexpected={len(unexpected_predictions)}"
        )
    if not common:
        raise ValueError("predictions and gold have no common question IDs")

    predicted_ids: list[int] = []
    gold_ids: list[int] = []
    acceptable_hits = 0
    confidence_slices: dict[str, list[tuple[int, int]]] = {}
    for question_id in common:
        prediction = predictions[question_id]
        target = gold[question_id]
        if not prediction.get("calibration_id") or prediction.get("difficulty_level") not in DIFFICULTY_LEVELS:
            raise ValueError(f"prediction {question_id} is not calibrated")
        predicted_id = DIFFICULTY_LEVELS.index(str(prediction["difficulty_level"]))
        gold_id = int(target.get("gold_difficulty_id"))
        predicted_ids.append(predicted_id)
        gold_ids.append(gold_id)
        acceptable = target.get("acceptable_difficulty_ids") or [gold_id]
        acceptable_hits += predicted_id in {int(value) for value in acceptable}
        confidence = str(target.get("gold_confidence") or "unspecified")
        confidence_slices.setdefault(confidence, []).append((predicted_id, gold_id))

    report = {
        "schema_version": "calibrated_pairwise_gold_evaluation_v1",
        "records": len(common),
        "gold_records": len(gold),
        "prediction_records": len(predictions),
        "missing_prediction_count": len(missing_predictions),
        "unexpected_prediction_count": len(unexpected_predictions),
        "calibration_ids": sorted({str(predictions[question_id]["calibration_id"]) for question_id in common}),
        "strict": classification_metrics(predicted_ids, gold_ids, len(DIFFICULTY_LEVELS)),
        "acceptable_level_accuracy": acceptable_hits / len(common),
        "gold_distribution": dict(Counter(DIFFICULTY_LEVELS[value] for value in gold_ids)),
        "predicted_distribution": dict(Counter(DIFFICULTY_LEVELS[value] for value in predicted_ids)),
        "gold_confidence_slices": {
            name: {
                "records": len(pairs),
                **classification_metrics(
                    [prediction for prediction, _ in pairs],
                    [target for _, target in pairs],
                    len(DIFFICULTY_LEVELS),
                ),
            }
            for name, pairs in confidence_slices.items()
        },
        "warning": "Gold is for final reporting, not checkpoint selection or threshold fitting.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
