#!/usr/bin/env python3
"""Compare global scalar outputs from two checkpoints on identical questions."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.calibration import DEFAULT_DISTRIBUTION
from physics_difficulty.pairwise.inference_audit import (
    pearson,
    spearman,
    thresholds,
    top_bottom_overlap,
)


def load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        question_id = str(row["question_id"])
        if question_id in rows:
            raise ValueError(f"duplicate question id {question_id} in {path}")
        rows[question_id] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    left = load(Path(args.left))
    right = load(Path(args.right))
    if set(left) != set(right):
        raise ValueError("score runs do not contain identical question IDs")
    ids = sorted(left)
    mismatched_hashes = [qid for qid in ids if left[qid].get("text_sha256") != right[qid].get("text_sha256")]
    if mismatched_hashes:
        raise ValueError(f"text hash mismatch for {len(mismatched_hashes)} questions")
    left_scores = [float(left[qid]["raw_difficulty_score"]) for qid in ids]
    right_scores = [float(right[qid]["raw_difficulty_score"]) for qid in ids]
    left_thresholds = thresholds(left_scores, DEFAULT_DISTRIBUTION)
    right_thresholds = thresholds(right_scores, DEFAULT_DISTRIBUTION)
    report = {
        "schema_version": "single_question_checkpoint_comparison_v1",
        "records": len(ids),
        "pearson_raw_scores": pearson(left_scores, right_scores),
        "spearman_rank_correlation": spearman(left_scores, right_scores),
        "top_bottom_10pct_overlap": top_bottom_overlap(left_scores, right_scores),
        "left": {"scores": str(Path(args.left).resolve()), "mean": statistics.fmean(left_scores), "thresholds": left_thresholds},
        "right": {"scores": str(Path(args.right).resolve()), "mean": statistics.fmean(right_scores), "thresholds": right_thresholds},
    }
    left_levels = [sum(score >= boundary for boundary in left_thresholds) for score in left_scores]
    right_levels = [sum(score >= boundary for boundary in right_thresholds) for score in right_scores]
    matrix = [[0] * 5 for _ in range(5)]
    for source, target in zip(left_levels, right_levels):
        matrix[source][target] += 1
    report["own_threshold_bucket_agreement"] = {
        "agreement": sum(a == b for a, b in zip(left_levels, right_levels)) / len(ids),
        "confusion_matrix": matrix,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
