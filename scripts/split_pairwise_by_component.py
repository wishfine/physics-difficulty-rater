#!/usr/bin/env python3
"""Deterministically split pair edges without sharing question nodes across splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def write_jsonl(path: Path, rows: list[dict], split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            result = dict(row)
            result["split"] = split
            target.write(json.dumps(result, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.validation_ratio < 1:
        raise ValueError("validation-ratio must be in (0, 1)")

    input_path = Path(args.input)
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 2:
        raise ValueError("at least two pair records are required")
    dsu = DisjointSet()
    seen_pair_ids: set[str] = set()
    for row in rows:
        pair_id = str(row["pair_id"])
        if pair_id in seen_pair_ids:
            raise ValueError(f"duplicate pair_id: {pair_id}")
        seen_pair_ids.add(pair_id)
        dsu.union(str(row["question_a_id"]), str(row["question_b_id"]))

    components: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        components[dsu.find(str(row["question_a_id"]))].append(row)
    ordered = sorted(
        components.values(),
        key=lambda group: hashlib.sha256(
            f"{args.seed}\0{min(str(row['pair_id']) for row in group)}".encode("utf-8")
        ).hexdigest(),
    )
    target = round(len(rows) * args.validation_ratio)
    validation_components: list[list[dict]] = []
    validation_count = 0
    for component in ordered:
        candidate = validation_count + len(component)
        if abs(candidate - target) < abs(validation_count - target):
            validation_components.append(component)
            validation_count = candidate
    if not validation_components:
        validation_components = [min(ordered, key=len)]
    validation_pair_ids = {str(row["pair_id"]) for group in validation_components for row in group}
    validation_rows = [row for row in rows if str(row["pair_id"]) in validation_pair_ids]
    train_rows = [row for row in rows if str(row["pair_id"]) not in validation_pair_ids]
    if not train_rows or not validation_rows:
        raise ValueError("component structure cannot produce two non-empty splits")

    train_ids = {str(row[key]) for row in train_rows for key in ("question_a_id", "question_b_id")}
    validation_ids = {str(row[key]) for row in validation_rows for key in ("question_a_id", "question_b_id")}
    overlap = train_ids & validation_ids
    if overlap:
        raise RuntimeError(f"component split produced question leakage: {sorted(overlap)[:5]}")

    train_path, validation_path = Path(args.train_output), Path(args.validation_output)
    write_jsonl(train_path, train_rows, "train")
    write_jsonl(validation_path, validation_rows, "validation")
    report = {
        "schema_version": "pairwise_component_split_v1",
        "input": str(input_path.resolve()),
        "seed": args.seed,
        "requested_validation_ratio": args.validation_ratio,
        "connected_components": len(components),
        "train_pairs": len(train_rows),
        "validation_pairs": len(validation_rows),
        "actual_validation_ratio": len(validation_rows) / len(rows),
        "train_questions": len(train_ids),
        "validation_questions": len(validation_ids),
        "question_overlap": 0,
        "train_output": str(train_path.resolve()),
        "validation_output": str(validation_path.resolve()),
    }
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
