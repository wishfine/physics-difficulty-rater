#!/usr/bin/env python3
"""Aggregate blind step-count votes into auditable, conservative overrides."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.schema import FEATURE_VALUES


STEP_VALUES = set(FEATURE_VALUES["step_count"])


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-votes", required=True)
    parser.add_argument("--selection-audit", required=True, help="Private candidate provenance containing original_step_count")
    parser.add_argument("--results-output", required=True, help="All review decisions, including abstentions")
    parser.add_argument("--overrides-output", required=True, help="Only consensus decisions that changed the label")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--minimum-valid-votes", type=int, default=3)
    parser.add_argument("--minimum-winner-votes", type=int, default=2)
    args = parser.parse_args()
    if args.minimum_valid_votes < 1 or args.minimum_winner_votes < 1:
        raise ValueError("vote thresholds must be positive")
    if args.minimum_winner_votes > args.minimum_valid_votes:
        raise ValueError("minimum-winner-votes cannot exceed minimum-valid-votes")

    audit_rows = load_jsonl(Path(args.selection_audit))
    audit: dict[str, dict[str, Any]] = {}
    for row in audit_rows:
        question_id = str(row.get("question_id") or "").strip()
        step = row.get("original_step_count")
        if not question_id or step not in STEP_VALUES:
            raise ValueError("selection audit must contain unique question_id and valid original_step_count")
        if question_id in audit:
            raise ValueError(f"duplicate question ID in selection audit: {question_id}")
        audit[question_id] = row

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(Path(args.raw_votes)):
        question_id = str(row.get("question_id") or "").strip()
        if question_id not in audit:
            raise ValueError(f"vote references a question absent from selection audit: {question_id}")
        grouped[question_id].append(row)

    result_rows: list[dict[str, Any]] = []
    override_rows: list[dict[str, str]] = []
    transition_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    for question_id in sorted(audit):
        votes = grouped.get(question_id, [])
        valid_values = [str(row.get("parsed_step_count")) for row in votes if row.get("valid") and row.get("parsed_step_count") in STEP_VALUES]
        vote_counts = Counter(valid_values)
        winner, winner_count = (vote_counts.most_common(1)[0] if vote_counts else (None, 0))
        sufficient = len(valid_values) >= args.minimum_valid_votes and winner_count >= args.minimum_winner_votes
        original = str(audit[question_id]["original_step_count"])
        action = "apply" if sufficient else "abstain"
        result = {
            "schema_version": "step_count_blind_recheck_result_v1",
            "question_id": question_id,
            "original_step_count": original,
            "rechecked_step_count": winner if sufficient else None,
            "valid_vote_count": len(valid_values),
            "vote_counts": {value: vote_counts.get(value, 0) for value in FEATURE_VALUES["step_count"]},
            "winner_vote_count": winner_count,
            "consensus": "unanimous" if sufficient and winner_count == len(valid_values) else ("majority" if sufficient else "insufficient"),
            "action": action,
        }
        result_rows.append(result)
        action_counts[action] += 1
        if sufficient:
            transition_counts[f"{original}->{winner}"] += 1
            if winner != original:
                override_rows.append({"question_id": question_id, "step_count": str(winner)})

    outputs = [Path(args.results_output), Path(args.overrides_output), Path(args.manifest)]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.results_output).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result_rows), encoding="utf-8")
    Path(args.overrides_output).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in override_rows), encoding="utf-8")
    manifest = {
        "schema_version": "step_count_blind_recheck_aggregation_v1",
        "raw_votes": str(Path(args.raw_votes).resolve()),
        "selection_audit": str(Path(args.selection_audit).resolve()),
        "results_output": str(Path(args.results_output).resolve()),
        "overrides_output": str(Path(args.overrides_output).resolve()),
        "questions": len(audit), "vote_rows": sum(len(rows) for rows in grouped.values()),
        "minimum_valid_votes": args.minimum_valid_votes, "minimum_winner_votes": args.minimum_winner_votes,
        "action_counts": dict(action_counts), "changed_overrides": len(override_rows),
        "applied_transitions": dict(transition_counts),
        "input_sha256": {"raw_votes": sha256(Path(args.raw_votes)), "selection_audit": sha256(Path(args.selection_audit))},
        "output_sha256": {"results": sha256(Path(args.results_output)), "overrides": sha256(Path(args.overrides_output))},
        "warning": "Overrides are a targeted enrichment output, not a representative estimate of the source step_count distribution.",
    }
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
