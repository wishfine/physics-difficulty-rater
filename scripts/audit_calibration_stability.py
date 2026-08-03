#!/usr/bin/env python3
"""Estimate threshold uncertainty and sample-size sensitivity."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.calibration import DEFAULT_DISTRIBUTION
from physics_difficulty.pairwise.inference_audit import (
    bootstrap_thresholds,
    migration_summary,
    percentile_interval,
    thresholds,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--sample-sizes", default="1000,2000,5000,10000")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.scores).read_text(encoding="utf-8").splitlines() if line.strip()]
    values = [float(row["raw_difficulty_score"]) for row in rows]
    if not values:
        raise ValueError("score file is empty")
    full = thresholds(values, DEFAULT_DISTRIBUTION)
    bootstraps = bootstrap_thresholds(values, DEFAULT_DISTRIBUTION, repetitions=args.repetitions, seed=args.seed)
    generator = random.Random(args.seed)
    sample_reports = {}
    for size in sorted({int(value) for value in args.sample_sizes.split(",") if value.strip()}):
        if size > len(values):
            continue
        indices = list(range(len(values)))
        generator.shuffle(indices)
        sampled = [values[index] for index in indices[:size]]
        candidate = thresholds(sampled, DEFAULT_DISTRIBUTION)
        sample_reports[str(size)] = {
            "thresholds": candidate,
            "absolute_threshold_delta": [abs(a - b) for a, b in zip(full, candidate)],
            "full_population_bucket_migration": migration_summary(values, full, candidate),
        }
    report = {
        "schema_version": "calibration_stability_audit_v1",
        "records": len(values),
        "target_distribution": list(DEFAULT_DISTRIBUTION),
        "full_thresholds": full,
        "bootstrap_repetitions": args.repetitions,
        "bootstrap_95pct_intervals": [percentile_interval(row[index] for row in bootstraps) for index in range(4)],
        "sample_size_sensitivity": sample_reports,
        "interpretation": "Threshold stability only; this does not measure difficulty-label accuracy.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
