#!/usr/bin/env python3
"""Apply four supplied score thresholds and evaluate matched review labels."""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.evaluation.metrics import classification_metrics
from physics_difficulty.schema import DIFFICULTY_LEVELS, DIFFICULTY_TO_ID


def load_unique(path: Path, id_keys: tuple[str, ...]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = next((str(row[key]).strip() for key in id_keys if str(row.get(key) or "").strip()), "")
            if not question_id:
                raise ValueError(f"{path}: line {line_number} lacks a question ID")
            if question_id in rows:
                raise ValueError(f"{path}: line {line_number} duplicates question ID {question_id}")
            rows[question_id] = row
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(thresholds) != len(DIFFICULTY_LEVELS) - 1:
        raise ValueError("exactly four comma-separated thresholds are required")
    if any(left >= right for left, right in zip(thresholds, thresholds[1:])):
        raise ValueError("thresholds must be strictly increasing")
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--thresholds", required=True, help="Four increasing raw-score thresholds: t1,t2,t3,t4")
    parser.add_argument("--review-label-field", default="model_difficulty_level")
    parser.add_argument("--predictions-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    score_rows = load_unique(Path(args.scores), ("question_id", "id"))
    review_rows = load_unique(Path(args.reviews), ("question_id", "id"))
    predictions_path = Path(args.predictions_output)
    report_path = Path(args.report_output)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    predictions: list[int] = []
    labels: list[int] = []
    verdicts: Counter[str] = Counter()
    missing_review_ids: list[str] = []
    predicted_counts: Counter[str] = Counter()
    with predictions_path.open("w", encoding="utf-8") as target:
        for question_id, score_row in score_rows.items():
            if "raw_difficulty_score" not in score_row:
                raise ValueError(f"score row {question_id} lacks raw_difficulty_score")
            raw_score = float(score_row["raw_difficulty_score"])
            predicted_id = bisect.bisect_right(thresholds, raw_score)
            predicted_level = DIFFICULTY_LEVELS[predicted_id]
            result = {
                "question_id": question_id,
                "raw_difficulty_score": raw_score,
                "difficulty_level_id": predicted_id,
                "difficulty_level": predicted_level,
                "raw_score_thresholds": thresholds,
            }
            predicted_counts[predicted_level] += 1
            review = review_rows.get(question_id)
            if review is None:
                missing_review_ids.append(question_id)
            else:
                label = str(review.get(args.review_label_field) or "").strip()
                if label not in DIFFICULTY_TO_ID:
                    raise ValueError(f"review row {question_id} has invalid {args.review_label_field}={label!r}")
                result["standard_difficulty_level"] = label
                result["standard_difficulty_level_id"] = DIFFICULTY_TO_ID[label]
                result["review_verdict"] = review.get("verdict")
                result["exact_match"] = predicted_level == label
                predictions.append(predicted_id)
                labels.append(DIFFICULTY_TO_ID[label])
                verdicts[str(review.get("verdict") or "unspecified")] += 1
            target.write(json.dumps(result, ensure_ascii=False) + "\n")

    report = {
        "schema_version": "fixed_pairwise_threshold_evaluation_v1",
        "scores_records": len(score_rows),
        "matched_review_records": len(labels),
        "missing_review_records": len(missing_review_ids),
        "missing_review_question_ids": missing_review_ids,
        "raw_score_thresholds": thresholds,
        "levels": list(DIFFICULTY_LEVELS),
        "predicted_level_counts_all_scores": dict(predicted_counts),
        "review_verdict_counts_matched": dict(verdicts),
        "metrics": classification_metrics(predictions, labels, len(DIFFICULTY_LEVELS)),
        "interpretation": "Accuracy is calculated only for score/review question-ID matches.",
        "scores": str(Path(args.scores).resolve()),
        "reviews": str(Path(args.reviews).resolve()),
        "predictions_output": str(predictions_path.resolve()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "matched_review_records": report["matched_review_records"],
        "accuracy": report["metrics"]["accuracy"],
        "report_output": str(report_path.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
