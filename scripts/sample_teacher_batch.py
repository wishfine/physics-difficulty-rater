#!/usr/bin/env python3
"""Make a deterministic raw-data sample for API+V7 teacher labeling.

The source difficulty bucket is used only to balance coverage; it is never
copied into the training target.
"""
from __future__ import annotations

import argparse
import json
import random
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from physics_difficulty.data.formatting import format_question


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="raw 25k JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--per_raw_bucket", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    groups: dict[str, list[dict]] = defaultdict(list)
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        identifier = str(row.get("question_id") or row.get("parent_id") or "")
        if not identifier or identifier in seen_ids:
            continue
        digest = hashlib.sha256(format_question(row).encode("utf-8")).hexdigest()
        if digest in seen_texts:
            continue
        seen_ids.add(identifier)
        seen_texts.add(digest)
        groups[str(row.get("difficulty", "unknown"))].append(row)
    rng = random.Random(args.seed)
    selected: list[dict] = []
    for bucket in sorted(groups):
        rows = groups[bucket]
        rng.shuffle(rows)
        selected.extend(rows[:args.per_raw_bucket])
    rng.shuffle(selected)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(selected), "coverage_buckets": dict(Counter(str(row.get("difficulty", "unknown")) for row in selected)), "seed": args.seed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
