#!/usr/bin/env python3
"""Route production pairs after the fixed six-vote nonthinking screen."""
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--nonthinking-votes", required=True)
    parser.add_argument("--accepted-output", required=True)
    parser.add_argument("--escalated-output", required=True)
    parser.add_argument("--records-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-position-bias-gap", type=float, default=0.25)
    parser.add_argument("--decisive-low", type=float, default=0.30)
    parser.add_argument("--decisive-high", type=float, default=0.70)
    parser.add_argument("--minimum-votes-per-direction", type=int, default=3)
    args = parser.parse_args()

    pairs = load_jsonl(Path(args.pairs))
    by_id = {str(row["pair_id"]): row for row in pairs}
    if len(by_id) != len(pairs):
        raise ValueError("pair IDs must be unique")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vote in load_jsonl(Path(args.nonthinking_votes)):
        pair_id = str(vote["pair_id"])
        if pair_id not in by_id:
            raise ValueError(f"nonthinking vote references unknown pair {pair_id}")
        grouped[pair_id].append(vote)
    thresholds = CascadeThresholds(
        max_position_bias_gap=args.max_position_bias_gap,
        decisive_low=args.decisive_low,
        decisive_high=args.decisive_high,
        minimum_votes_per_direction=args.minimum_votes_per_direction,
    )
    accepted, escalated, records = [], [], []
    reasons: Counter[str] = Counter()
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        route = decide_cascade_route(grouped.get(pair_id, []), thresholds)
        item = {**pair, "cascade_route": route}
        records.append({"pair_id": pair_id, **route})
        reasons[str(route["reason"])] += 1
        (accepted if route["action"] == "accept_nonthinking" else escalated).append(item)
    write_jsonl(Path(args.accepted_output), accepted)
    write_jsonl(Path(args.escalated_output), escalated)
    write_jsonl(Path(args.records_output), records)
    manifest = {
        "schema_version": "cascade_production_routing_v1",
        "pairs": len(pairs),
        "accepted_nonthinking": len(accepted),
        "escalated_thinking_1024": len(escalated),
        "direct_acceptance_rate": len(accepted) / max(1, len(pairs)),
        "route_reason_counts": dict(reasons),
        "thresholds": vars(thresholds),
    }
    path = Path(args.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
