#!/usr/bin/env python3
"""Build final soft Bradley-Terry data from routed teacher votes."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.cascade import CascadeThresholds, decide_cascade_route
from physics_difficulty.pairwise.labels import aggregate_pair_votes, pair_reliability
from physics_difficulty.pairwise.metrics import graph_metrics, soft_target_distribution


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def grouped_votes(path: Path, known_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(path):
        pair_id = str(row["pair_id"])
        if pair_id not in known_ids:
            raise ValueError(f"vote file {path} references unknown pair {pair_id}")
        grouped[pair_id].append(row)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--nonthinking-votes", required=True)
    parser.add_argument("--thinking-votes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quarantine-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-position-bias-gap", type=float, default=0.25)
    parser.add_argument("--decisive-low", type=float, default=0.30)
    parser.add_argument("--decisive-high", type=float, default=0.70)
    parser.add_argument("--minimum-votes-per-direction", type=int, default=3)
    parser.add_argument("--medium-reliability-gap", type=float, default=0.15)
    parser.add_argument("--high-reliability-gap", type=float, default=0.30)
    args = parser.parse_args()

    pairs = load_jsonl(Path(args.pairs))
    pair_ids = {str(row["pair_id"]) for row in pairs}
    if len(pair_ids) != len(pairs):
        raise ValueError("pair IDs must be unique")
    nonthinking = grouped_votes(Path(args.nonthinking_votes), pair_ids)
    thinking = grouped_votes(Path(args.thinking_votes), pair_ids)
    thresholds = CascadeThresholds(
        max_position_bias_gap=args.max_position_bias_gap,
        decisive_low=args.decisive_low,
        decisive_high=args.decisive_high,
        minimum_votes_per_direction=args.minimum_votes_per_direction,
    )
    accepted, quarantined = [], []
    stats: Counter[str] = Counter()
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        route = decide_cascade_route(nonthinking.get(pair_id, []), thresholds)
        source = "nonthinking" if route["action"] == "accept_nonthinking" else "thinking_1024"
        votes = nonthinking.get(pair_id, []) if source == "nonthinking" else thinking.get(pair_id, [])
        counts = Counter(str(row.get("direction")) for row in votes if row.get("valid"))
        if min(counts["forward"], counts["backward"]) < args.minimum_votes_per_direction:
            quarantined.append({**pair, "quarantine_reason": f"insufficient_{source}_votes", "cascade_route": route})
            stats[f"insufficient_{source}_votes"] += 1
            continue
        aggregate = aggregate_pair_votes(votes)
        reliability = pair_reliability(
            float(aggregate["position_bias_gap"]),
            args.medium_reliability_gap,
            args.high_reliability_gap,
        )
        item = {
            "schema_version": "physics_pairwise_soft_v1",
            "pair_id": pair_id,
            "split": pair["split"],
            "question_a_id": str(pair["question_a_id"]),
            "question_b_id": str(pair["question_b_id"]),
            "question_a_text": pair["question_a_text"],
            "question_b_text": pair["question_b_text"],
            "soft_target": float(aggregate["soft_target"]),
            "sample_weight": float(reliability["sample_weight"]),
            "vote_stats": aggregate,
            "reliability": reliability,
            "cascade_route": route,
            "label_source": source,
            "metadata": {
                **(pair.get("metadata") or {}),
                "pair_source": pair.get("pair_source"),
                "images_uploaded": False,
                "raw_difficulty_used": False,
            },
        }
        if reliability["action"] == "quarantine":
            quarantined.append({**item, "quarantine_reason": "high_final_position_bias"})
            stats["high_final_position_bias"] += 1
        else:
            accepted.append(item)
            stats[f"accepted_{source}"] += 1
            stats[f"reliability_{reliability['status']}"] += 1

    output = Path(args.output)
    quarantine = Path(args.quarantine_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted), encoding="utf-8")
    quarantine.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in quarantined), encoding="utf-8")
    expected_nodes = {str(row["question_a_id"]) for row in pairs} | {str(row["question_b_id"]) for row in pairs}
    manifest = {
        "schema_version": "cascade_pairwise_soft_v1",
        "candidate_pairs": len(pairs),
        "accepted_pairs": len(accepted),
        "quarantined_pairs": len(quarantined),
        "stats": dict(stats),
        "soft_targets": soft_target_distribution(row["soft_target"] for row in accepted),
        "graph": graph_metrics(accepted, expected_nodes),
        "thresholds": vars(thresholds),
        "images_uploaded": False,
        "raw_difficulty_used": False,
    }
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
