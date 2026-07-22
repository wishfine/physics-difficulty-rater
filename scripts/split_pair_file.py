#!/usr/bin/env python3
"""Split a pair JSONL into deterministic character-balanced shards."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.cascade import split_pairs_balanced


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    rows = load_jsonl(Path(args.input))
    if not rows:
        raise ValueError("cannot shard an empty pair file")
    shards, stats = split_pairs_balanced(rows, args.shards, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, shard in enumerate(shards):
        path = output_dir / f"shard-{index:03d}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in shard), encoding="utf-8")
        paths.append(str(path))
    manifest = {"schema_version": "balanced_pair_shards_v1", "input": args.input, **stats, "paths": paths}
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
