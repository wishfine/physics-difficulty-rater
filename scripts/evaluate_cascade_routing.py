#!/usr/bin/env python3
"""Evaluate a pre-registered nonthinking-to-thinking cascade route."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.cascade import CascadeThresholds, evaluate_cascade, select_blind_audit_pairs


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--nonthinking-votes", required=True)
    parser.add_argument("--thinking-votes", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--records-output", required=True)
    parser.add_argument("--accepted-pairs-output", required=True)
    parser.add_argument("--escalated-pairs-output", required=True)
    parser.add_argument("--human-audit-output", required=True)
    parser.add_argument("--human-audit-manifest", required=True)
    parser.add_argument("--human-audit-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--nonthinking-round", type=int, default=1)
    parser.add_argument("--max-position-bias-gap", type=float, default=0.25)
    parser.add_argument("--decisive-low", type=float, default=0.30)
    parser.add_argument("--decisive-high", type=float, default=0.70)
    parser.add_argument("--minimum-votes-per-direction", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs_path = Path(args.pairs)
    pairs = load_jsonl(pairs_path)
    pair_ids = {str(row["pair_id"]) for row in pairs}
    nonthinking_all = load_jsonl(Path(args.nonthinking_votes))
    nonthinking = [row for row in nonthinking_all if int(row.get("sampling_round", 1)) == args.nonthinking_round]
    thinking = load_jsonl(Path(args.thinking_votes))
    unknown = ({str(row["pair_id"]) for row in nonthinking + thinking} - pair_ids)
    if unknown:
        raise ValueError(f"vote files contain {len(unknown)} pair IDs absent from evaluation pairs")
    thresholds = CascadeThresholds(
        max_position_bias_gap=args.max_position_bias_gap,
        decisive_low=args.decisive_low,
        decisive_high=args.decisive_high,
        minimum_votes_per_direction=args.minimum_votes_per_direction,
    )
    report, records = evaluate_cascade(pairs, nonthinking, thinking, thresholds)
    by_id = {str(row["pair_id"]): row for row in pairs}
    record_by_id = {str(row["pair_id"]): row for row in records}

    def routed(action: str) -> list[dict[str, Any]]:
        return [
            {**by_id[pair_id], "cascade_route": record_by_id[pair_id]}
            for pair_id in sorted(by_id)
            if record_by_id[pair_id]["route_action"] == action
        ]

    accepted = routed("accept_nonthinking")
    escalated = routed("escalate_thinking_1024")
    audit_size = min(args.human_audit_size, len(pairs))
    blind_audit, audit_manifest = select_blind_audit_pairs(pairs, records, audit_size, args.seed)
    gate = {
        "minimum_direct_acceptance_rate": 0.40,
        "minimum_accepted_hard_agreement_with_thinking": 0.95,
        "maximum_accepted_mean_absolute_soft_target_difference": 0.08,
        "maximum_severe_disagreement_rate": 0.02,
    }
    severe_disagreement_rate = report["accepted_severe_disagreement_rate"]
    gate_checks = {
        "direct_acceptance_rate": report["direct_acceptance_rate"] >= gate["minimum_direct_acceptance_rate"],
        "accepted_hard_agreement_with_thinking": report["accepted_hard_agreement_with_thinking"] is not None
        and report["accepted_hard_agreement_with_thinking"] >= gate["minimum_accepted_hard_agreement_with_thinking"],
        "accepted_mean_absolute_soft_target_difference": report["accepted_mean_absolute_soft_target_difference"] is not None
        and report["accepted_mean_absolute_soft_target_difference"] <= gate["maximum_accepted_mean_absolute_soft_target_difference"],
        "severe_disagreement_rate": severe_disagreement_rate <= gate["maximum_severe_disagreement_rate"],
    }
    report.update({
        "pairs_file": str(pairs_path.resolve()),
        "nonthinking_votes_file": str(Path(args.nonthinking_votes).resolve()),
        "thinking_votes_file": str(Path(args.thinking_votes).resolve()),
        "nonthinking_round_used": args.nonthinking_round,
        "human_audit_size": audit_size,
        "accepted_severe_disagreement_rate_for_gate": severe_disagreement_rate,
        "acceptance_gate": gate,
        "acceptance_gate_checks": gate_checks,
        "acceptance_gate_status": "PASS" if all(gate_checks.values()) else "FAIL",
    })

    write_jsonl(Path(args.records_output), records)
    write_jsonl(Path(args.accepted_pairs_output), accepted)
    write_jsonl(Path(args.escalated_pairs_output), escalated)
    write_jsonl(Path(args.human_audit_output), blind_audit)
    audit_manifest_path = Path(args.human_audit_manifest)
    audit_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    audit_manifest_path.write_text(json.dumps(audit_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
