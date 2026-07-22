#!/usr/bin/env python3
"""Prepare a route-stratified human audit without leaking teacher predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.cascade import select_representative_audit_pairs


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--evaluation-records", required=True)
    parser.add_argument("--prior-audit", required=True)
    parser.add_argument("--new-blind-output", required=True)
    parser.add_argument("--reused-labels-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stable-count", type=int, default=40)
    parser.add_argument("--escalated-agree-count", type=int, default=20)
    parser.add_argument("--escalated-disagree-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs_path = Path(args.pairs)
    records_path = Path(args.evaluation_records)
    prior_path = Path(args.prior_audit)
    blind_path = Path(args.new_blind_output)
    reused_path = Path(args.reused_labels_output)
    manifest_path = Path(args.manifest)

    blind, reused, selection = select_representative_audit_pairs(
        load_jsonl(pairs_path),
        load_jsonl(records_path),
        load_jsonl(prior_path),
        quotas={
            "stable_and_decisive": args.stable_count,
            "escalated_same_direction": args.escalated_agree_count,
            "escalated_teacher_disagreement": args.escalated_disagree_count,
        },
        seed=args.seed,
    )
    write_jsonl(blind_path, blind)
    write_jsonl(reused_path, reused)
    manifest = {
        **selection,
        "inputs": {
            "pairs": {"path": str(pairs_path.resolve()), "sha256": sha256(pairs_path)},
            "evaluation_records": {"path": str(records_path.resolve()), "sha256": sha256(records_path)},
            "prior_audit": {"path": str(prior_path.resolve()), "sha256": sha256(prior_path)},
        },
        "outputs": {
            "new_blind": {"path": str(blind_path.resolve()), "sha256": sha256(blind_path), "records": len(blind)},
            "reused_labels": {"path": str(reused_path.resolve()), "sha256": sha256(reused_path), "records": len(reused)},
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
