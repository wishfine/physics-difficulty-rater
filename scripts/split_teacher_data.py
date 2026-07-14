#!/usr/bin/env python3
"""Group-aware, label-stratified split for curated V2 teacher data."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.train_ratio < 1 or not 0 < args.validation_ratio < 1 or args.train_ratio + args.validation_ratio >= 1:
        raise ValueError("ratios must be positive and sum to less than one")

    items = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        group_key = f"{item.get('source_dataset_id', 'unknown')}::{item.get('parent_id') or item['input_sha256']}"
        groups[group_key].append(item)
    label_groups: dict[int, list[tuple[str, list[dict]]]] = defaultdict(list)
    for key, group in groups.items():
        # Parent records should have one label. If malformed data violates this,
        # preserve it in one split and use the most frequent label for stratification.
        labels = Counter(row["teacher_difficulty_id"] for row in group)
        label_groups[labels.most_common(1)[0][0]].append((key, group))

    rng = random.Random(args.seed)
    splits = {"train": [], "validation": [], "test": []}
    for _, label_group in label_groups.items():
        rng.shuffle(label_group)
        total = len(label_group)
        train_end = round(total * args.train_ratio)
        validation_end = train_end + round(total * args.validation_ratio)
        for name, subset in (("train", label_group[:train_end]), ("validation", label_group[train_end:validation_end]), ("test", label_group[validation_end:])):
            for _, group in subset:
                splits[name].extend(group)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, rows in splits.items():
        (output / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        stats[name] = {"records": len(rows), "labels": dict(Counter(row["teacher_difficulty_level"] for row in rows))}
    all_levels = {"送分题", "基础题", "中等题", "拔高题", "压轴题"}
    warnings = [f"{name} is missing levels: {sorted(all_levels - set(info['labels']))}" for name, info in stats.items() if all_levels - set(info["labels"])]
    manifest = {"input": str(Path(args.input).resolve()), "seed": args.seed, "group_count": len(groups), "splits": stats, "warnings": warnings}
    (output / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
