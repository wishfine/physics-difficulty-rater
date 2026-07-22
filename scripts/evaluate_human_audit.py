#!/usr/bin/env python3
"""Evaluate one or more non-overlapping human-audit JSONL files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.cascade import evaluate_human_audit


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-records", required=True)
    parser.add_argument("--human-audit", action="append", required=True)
    parser.add_argument("--combined-output", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audits = [Path(value) for value in args.human_audit]
    combined = [row for path in audits for row in load_jsonl(path)]
    pair_ids = [str(row["pair_id"]) for row in combined]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("human audit inputs overlap; provide non-overlapping files")
    combined.sort(key=lambda row: str(row["pair_id"]))

    combined_path = Path(args.combined_output)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined), encoding="utf-8")
    report = evaluate_human_audit(load_jsonl(Path(args.evaluation_records)), combined)
    report.update({
        "human_audit_inputs": [str(path) for path in audits],
        "warning": "Single-review audit is diagnostic evidence, not expert adjudicated gold.",
    })
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
