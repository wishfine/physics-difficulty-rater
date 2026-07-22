#!/usr/bin/env python3
"""Aggregate raw Qwen votes into curated soft Bradley-Terry examples."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.labels import aggregate_pair_votes, pair_reliability
from physics_difficulty.pairwise.metrics import graph_metrics, soft_target_distribution


def load_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--raw-votes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quarantine-output")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--minimum-valid-votes-per-direction", type=int, default=3)
    parser.add_argument("--medium-position-gap", type=float, default=0.15)
    parser.add_argument("--high-position-gap", type=float, default=0.30)
    parser.add_argument("--prior", type=float, default=0.5)
    args = parser.parse_args()
    if args.minimum_valid_votes_per_direction < 1:
        raise ValueError("minimum valid votes must be positive")

    pairs = load_jsonl(Path(args.pairs))
    pair_by_id = {str(pair["pair_id"]): pair for pair in pairs}
    if len(pair_by_id) != len(pairs):
        raise ValueError("pair IDs must be unique")
    grouped: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    stats: Counter[str] = Counter()
    for vote in load_jsonl(Path(args.raw_votes)):
        pair_id = str(vote["pair_id"])
        if pair_id not in pair_by_id:
            stats["orphan_votes"] += 1
            continue
        grouped[pair_id].append(vote)
        stats["raw_votes"] += 1
        stats["valid_votes"] += bool(vote.get("valid"))

    accepted: list[Dict[str, Any]] = []
    quarantined: list[Dict[str, Any]] = []
    for pair_id, pair in pair_by_id.items():
        rows = grouped.get(pair_id, [])
        direction_counts = Counter(str(row.get("direction")) for row in rows if row.get("valid"))
        if direction_counts["forward"] < args.minimum_valid_votes_per_direction or direction_counts["backward"] < args.minimum_valid_votes_per_direction:
            quarantined.append(pair | {"quarantine_reason": "insufficient_bidirectional_votes", "valid_vote_counts": dict(direction_counts)})
            stats["insufficient_bidirectional_votes"] += 1
            continue
        aggregated = aggregate_pair_votes(rows, args.prior)
        reliability = pair_reliability(aggregated["position_bias_gap"], args.medium_position_gap, args.high_position_gap)
        item = {
            "schema_version": "physics_pairwise_soft_v1",
            "pair_id": pair_id,
            "split": pair["split"],
            "question_a_id": str(pair["question_a_id"]),
            "question_b_id": str(pair["question_b_id"]),
            "question_a_text": pair["question_a_text"],
            "question_b_text": pair["question_b_text"],
            "soft_target": aggregated["soft_target"],
            "sample_weight": reliability["sample_weight"],
            "vote_stats": aggregated,
            "reliability": reliability,
            "metadata": {
                **(pair.get("metadata") or {}),
                "pair_source": pair.get("pair_source"),
                "domain_relation": pair.get("domain_relation"),
                "images_uploaded": False,
                "raw_difficulty_used": False,
            },
        }
        if reliability["action"] == "quarantine":
            quarantined.append(item | {"quarantine_reason": "high_position_bias"})
            stats["high_position_bias"] += 1
        else:
            accepted.append(item)
            stats[f"reliability_{reliability['status']}"] += 1

    output = Path(args.output)
    quarantine = Path(args.quarantine_output) if args.quarantine_output else output.with_suffix(".quarantine.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as target:
        for item in accepted:
            target.write(json.dumps(item, ensure_ascii=False) + "\n")
    with quarantine.open("w", encoding="utf-8") as target:
        for item in quarantined:
            target.write(json.dumps(item, ensure_ascii=False) + "\n")

    expected_nodes = {str(pair["question_a_id"]) for pair in pairs} | {str(pair["question_b_id"]) for pair in pairs}
    manifest = {
        "schema_version": "physics_pairwise_soft_v1",
        "pairs": str(Path(args.pairs).resolve()),
        "raw_votes": str(Path(args.raw_votes).resolve()),
        "output": str(output.resolve()),
        "quarantine_output": str(quarantine.resolve()),
        "images_uploaded": False,
        "raw_difficulty_used": False,
        "candidate_pairs": len(pairs),
        "accepted_pairs": len(accepted),
        "quarantined_pairs": len(quarantined),
        "stats": dict(stats),
        "soft_targets": soft_target_distribution(item["soft_target"] for item in accepted),
        "graph": graph_metrics(accepted, expected_nodes),
        "aggregation": {
            "prior": args.prior,
            "minimum_valid_votes_per_direction": args.minimum_valid_votes_per_direction,
            "medium_position_gap": args.medium_position_gap,
            "high_position_gap": args.high_position_gap,
        },
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
