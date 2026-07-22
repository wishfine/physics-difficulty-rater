#!/usr/bin/env python3
"""Select an independent stratified cascade-validation sample and shard it."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.cascade import select_stratified_pairs, split_pairs_balanced


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--exclude-jsonl", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates_path = Path(args.candidates)
    candidates = load_jsonl(candidates_path)
    exclusion_paths = [Path(value) for value in args.exclude_jsonl]
    exclusion_rows = [row for path in exclusion_paths for row in load_jsonl(path)]
    excluded = {str(row["pair_id"]) for row in exclusion_rows if row.get("pair_id") is not None}
    excluded_questions = {
        str(row[key])
        for row in exclusion_rows
        for key in ("question_a_id", "question_b_id")
        if row.get(key) is not None
    }
    selected, selection_stats = select_stratified_pairs(
        candidates,
        args.sample_size,
        args.seed,
        excluded_pair_ids=excluded,
        excluded_question_ids=excluded_questions,
    )
    shards, shard_stats = split_pairs_balanced(selected, args.shard_count, args.seed)

    output = Path(args.output)
    shard_dir = Path(args.shard_dir)
    manifest_path = Path(args.manifest)
    write_jsonl(output, selected)
    shard_paths = []
    for index, rows in enumerate(shards):
        path = shard_dir / f"shard-{index:03d}.jsonl"
        write_jsonl(path, rows)
        shard_paths.append(path)

    manifest = {
        "schema_version": "cascade_validation_selection_v1",
        "candidates": str(candidates_path.resolve()),
        "candidates_sha256": sha256(candidates_path),
        "exclusion_sources": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in exclusion_paths
        ],
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
        "seed": args.seed,
        "sample_size": args.sample_size,
        "shard_count": args.shard_count,
        "selection": selection_stats,
        "sharding": shard_stats,
        "shards": [
            {"path": str(path.resolve()), "sha256": sha256(path), "pairs": len(rows)}
            for path, rows in zip(shard_paths, shards)
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
