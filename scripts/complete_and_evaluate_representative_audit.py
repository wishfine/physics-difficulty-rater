#!/usr/bin/env python3
"""Apply blind human labels, combine reused reviews, and evaluate cascade modes."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.cascade import evaluate_human_audit


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def load_labels(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    by_id = {str(row.get("pair_id")): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("label TSV pair IDs must be unique")
    for pair_id, row in by_id.items():
        if row.get("human_preference") not in {"A", "B", "tie"}:
            raise ValueError(f"invalid preference in label TSV for {pair_id}")
        if row.get("human_confidence") not in {"high", "medium", "low"}:
            raise ValueError(f"invalid confidence in label TSV for {pair_id}")
    return by_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-blind", required=True)
    parser.add_argument("--new-labels-tsv", required=True)
    parser.add_argument("--reused-labels", required=True)
    parser.add_argument("--evaluation-records", required=True)
    parser.add_argument("--new-completed-output", required=True)
    parser.add_argument("--combined-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--reviewer", default="Codex_single_review")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blind = load_jsonl(Path(args.new_blind))
    labels = load_labels(Path(args.new_labels_tsv))
    blind_ids = {str(row["pair_id"]) for row in blind}
    if blind_ids != set(labels):
        missing = sorted(blind_ids - set(labels))
        extra = sorted(set(labels) - blind_ids)
        raise ValueError(f"blind/label pair mismatch: missing={missing}, extra={extra}")

    completed = []
    for row in blind:
        pair_id = str(row["pair_id"])
        label = labels[pair_id]
        completed.append({
            **row,
            "human_preference": label["human_preference"],
            "human_confidence": label["human_confidence"],
            "human_notes": label.get("human_notes") or "",
            "human_reviewer": args.reviewer,
            "human_label_source": "new_blind_review",
        })
    reused = [
        {**row, "human_label_source": "reused_prior_review"}
        for row in load_jsonl(Path(args.reused_labels))
    ]
    combined = completed + reused
    combined_ids = [str(row["pair_id"]) for row in combined]
    if len(combined_ids) != len(set(combined_ids)):
        raise ValueError("combined human audit contains duplicate pair IDs")

    write_jsonl(Path(args.new_completed_output), completed)
    write_jsonl(Path(args.combined_output), combined)
    report = evaluate_human_audit(load_jsonl(Path(args.evaluation_records)), combined)
    report.update({
        "new_blind_reviews": len(completed),
        "reused_prior_reviews": len(reused),
        "reviewer": args.reviewer,
        "warning": "Single-review audit is diagnostic evidence, not expert adjudicated gold.",
    })
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
