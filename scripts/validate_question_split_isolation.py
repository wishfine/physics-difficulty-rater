#!/usr/bin/env python3
"""Prove that independently prepared question files share no question IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.questions) < 2:
        parser.error("--questions must be provided at least twice")

    errors: list[str] = []
    sources: list[dict[str, Any]] = []
    id_sets: list[set[str]] = []
    paths = [Path(value) for value in args.questions]
    for path in paths:
        rows = load_jsonl(path)
        ids = [str(row.get("id")) for row in rows if row.get("id") is not None]
        duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
        missing_ids = len(rows) - len(ids)
        splits = sorted({str(row.get("split")) for row in rows})
        if missing_ids:
            errors.append(f"{path}: {missing_ids} records lack id")
        if duplicates:
            errors.append(f"{path}: duplicate question IDs: {duplicates[:10]}")
        if len(splits) != 1:
            errors.append(f"{path}: expected one split, got {splits}")
        id_sets.append(set(ids))
        sources.append({
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "records": len(rows),
            "unique_question_ids": len(set(ids)),
            "split": splits[0] if len(splits) == 1 else None,
        })

    overlaps: list[dict[str, Any]] = []
    for (left_index, left), (right_index, right) in combinations(enumerate(id_sets), 2):
        overlap = sorted(left & right)
        overlaps.append({
            "left": str(paths[left_index].resolve()),
            "right": str(paths[right_index].resolve()),
            "count": len(overlap),
            "sample_ids": overlap[:20],
        })
        if overlap:
            errors.append(
                f"{paths[left_index]} and {paths[right_index]} share {len(overlap)} question IDs"
            )

    report = {
        "schema_version": "question_split_isolation_v1",
        "status": "PASS" if not errors else "FAIL",
        "sources": sources,
        "overlaps": overlaps,
        "errors": errors,
        "error_count": len(errors),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
