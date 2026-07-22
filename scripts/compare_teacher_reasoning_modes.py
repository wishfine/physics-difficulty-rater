#!/usr/bin/env python3
"""Compare Qwen teacher reasoning modes without treating agreement as gold accuracy."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.labels import aggregate_pair_votes


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def aggregate_run(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)

    aggregated: dict[str, dict[str, Any]] = {}
    failed_pairs = 0
    for pair_id, pair_rows in grouped.items():
        try:
            aggregated[pair_id] = aggregate_pair_votes(pair_rows)
        except ValueError:
            failed_pairs += 1

    valid_rows = [row for row in rows if row.get("valid")]
    tokens = sum(int(row.get("output_token_count", 0) or 0) for row in rows)
    valid_tokens = sum(int(row.get("output_token_count", 0) or 0) for row in valid_rows)
    gaps = [float(item["position_bias_gap"]) for item in aggregated.values()]
    targets = [float(item["soft_target"]) for item in aggregated.values()]
    metrics = {
        "vote_rows": len(rows),
        "valid_votes": len(valid_rows),
        "parse_success_rate": len(valid_rows) / max(1, len(rows)),
        "output_tokens": tokens,
        "mean_output_tokens_per_valid_vote": valid_tokens / max(1, len(valid_rows)),
        "aggregated_pairs": len(aggregated),
        "unaggregated_pairs": failed_pairs,
        "mean_position_bias_gap": sum(gaps) / max(1, len(gaps)),
        "high_position_bias_rate": sum(gap > 0.30 for gap in gaps) / max(1, len(gaps)),
        "uncertain_pair_rate": sum(0.30 <= target <= 0.70 for target in targets) / max(1, len(targets)),
    }
    return metrics, aggregated


def hard_label(target: float) -> str:
    if target > 0.5:
        return "A"
    if target < 0.5:
        return "B"
    return "tie"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", nargs=3, metavar=("NAME", "MANIFEST", "RAW_VOTES"), required=True,
        help="Repeat for every mode to compare.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.run) < 2:
        raise ValueError("at least two --run entries are required")
    names = [entry[0] for entry in args.run]
    if len(names) != len(set(names)):
        raise ValueError("run names must be unique")

    run_metrics: dict[str, dict[str, Any]] = {}
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    for name, manifest_path, votes_path in args.run:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        metrics, aggregated = aggregate_run(load_jsonl(Path(votes_path)))
        generation_seconds = float(manifest.get("generation_wall_seconds", 0.0) or 0.0)
        metrics.update({
            "teacher_mode": manifest.get("teacher_mode", name),
            "generation_wall_seconds": generation_seconds,
            "valid_votes_per_second": metrics["valid_votes"] / generation_seconds if generation_seconds > 0 else None,
            "manifest": str(Path(manifest_path).resolve()),
            "raw_votes": str(Path(votes_path).resolve()),
        })
        run_metrics[name] = metrics
        predictions[name] = aggregated

    cross_mode: dict[str, dict[str, Any]] = {}
    for first, second in itertools.combinations(names, 2):
        common = sorted(set(predictions[first]) & set(predictions[second]))
        labels = [
            (hard_label(float(predictions[first][pair_id]["soft_target"])), hard_label(float(predictions[second][pair_id]["soft_target"])))
            for pair_id in common
        ]
        differences = [
            abs(float(predictions[first][pair_id]["soft_target"]) - float(predictions[second][pair_id]["soft_target"]))
            for pair_id in common
        ]
        cross_mode[f"{first}__vs__{second}"] = {
            "common_pairs": len(common),
            "hard_label_agreement": sum(a == b for a, b in labels) / max(1, len(labels)),
            "mean_absolute_soft_target_difference": sum(differences) / max(1, len(differences)),
        }

    report = {
        "schema_version": "teacher_reasoning_comparison_v1",
        "warning": "Cross-mode agreement is not human-gold accuracy and cannot select the best teacher by itself.",
        "runs": run_metrics,
        "cross_mode": cross_mode,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
