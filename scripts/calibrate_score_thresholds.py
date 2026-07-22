#!/usr/bin/env python3
"""Fit four fixed score thresholds to a declared five-level distribution."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LEVELS = ["送分题", "基础题", "中等题", "拔高题", "压轴题"]


def quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calibrate an empty score set")
    position = probability * (len(sorted_values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] * (upper - position) + sorted_values[upper] * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="JSONL from score_pairwise_questions.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--distribution", nargs=5, type=float, default=[0.20, 0.20, 0.30, 0.20, 0.10], metavar=("L1", "L2", "L3", "L4", "L5"))
    args = parser.parse_args()
    if any(value <= 0 for value in args.distribution) or not math.isclose(sum(args.distribution), 1.0, abs_tol=1e-8):
        raise ValueError("five distribution values must be positive and sum to 1")
    rows = [json.loads(line) for line in Path(args.scores).read_text(encoding="utf-8").splitlines() if line.strip()]
    identifiers = [str(row["id"]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("reference score IDs must be unique")
    scores = sorted(float(row["difficulty_score"]) for row in rows)
    if len(scores) < 100:
        raise ValueError("use at least 100 reference questions for threshold calibration")
    cumulative = []
    running = 0.0
    for proportion in args.distribution[:-1]:
        running += proportion
        cumulative.append(running)
    thresholds = [quantile(scores, probability) for probability in cumulative]
    if any(left >= right for left, right in zip(thresholds, thresholds[1:])):
        raise ValueError("score ties produced non-increasing thresholds; use more reference questions or inspect score collapse")
    result = {
        "schema_version": "pairwise_score_calibration_v1",
        "method": "fixed_reference_distribution",
        "levels": LEVELS,
        "distribution": dict(zip(LEVELS, args.distribution)),
        "quantiles": cumulative,
        "thresholds": thresholds,
        "reference_records": len(scores),
        "score_range": {"minimum": scores[0], "maximum": scores[-1]},
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "scores_file": str(Path(args.scores).resolve()),
        "warning": "These are fixed thresholds learned on the declared reference population; do not recompute them for each inference batch.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
