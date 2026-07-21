#!/usr/bin/env python3
"""Build a difficulty-only gold JSONL from the GPT-5.6 adjudication CSV."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from physics_difficulty.data.formatting import canonical_sections, diagnostics, format_question
from physics_difficulty.schema import DIFFICULTY_TO_ID


def parse_acceptable_levels(value: str, primary: str) -> list[str]:
    levels = [part.strip() for part in str(value or "").split("|") if part.strip()]
    if primary not in levels:
        levels.insert(0, primary)
    unknown = [level for level in levels if level not in DIFFICULTY_TO_ID]
    if unknown:
        raise ValueError(f"Unknown acceptable difficulty levels: {unknown}")
    return list(dict.fromkeys(levels))


def read_ids(path: Path) -> set[str]:
    return {str(json.loads(line)["id"]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference_train_file", help="Optional train JSONL; fail if any gold IDs overlap it.")
    parser.add_argument("--allow_reference_overlap", action="store_true")
    parser.add_argument("--exclude_reference_overlap", action="store_true", help="Build a final holdout by excluding IDs present in --reference_train_file.")
    parser.add_argument("--skip_unrenderable", action="store_true", help="Exclude rows with no text input; their IDs are recorded in the summary.")
    args = parser.parse_args()

    labels_path = Path(args.labels_csv)
    with labels_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Adjudication CSV is empty")
    if args.exclude_reference_overlap and not args.reference_train_file:
        raise ValueError("--exclude_reference_overlap requires --reference_train_file")

    seen, output_rows, skipped_unrenderable = set(), [], []
    for row in rows:
        item_id = str(row.get("题目ID") or "").strip()
        level = str(row.get("修订后主标签") or "").strip()
        if not item_id or not level:
            raise ValueError("Every gold row needs 题目ID and 修订后主标签")
        if item_id in seen:
            raise ValueError(f"Duplicate gold ID: {item_id}")
        if level not in DIFFICULTY_TO_ID:
            raise ValueError(f"Unknown primary label for {item_id}: {level}")
        seen.add(item_id)
        acceptable_levels = parse_acceptable_levels(row.get("可接受等级", ""), level)
        source = {
            "stem": row.get("题干", ""),
            "options": row.get("选项", ""),
            "analysis": row.get("官方解析", ""),
            "stem_pic_url": row.get("题目图片URL", ""),
            "analysis_pic_url": row.get("解析图片URL", ""),
        }
        text = format_question(source)
        if not text.strip():
            if args.skip_unrenderable:
                skipped_unrenderable.append(item_id)
                continue
            raise ValueError(f"Gold row {item_id} has no renderable question text; provide source text or rerun with --skip_unrenderable")
        output_rows.append({
            "id": item_id,
            "source_dataset_id": "physics_adjudicated_gpt56_rereview_1066",
            "parent_id": item_id,
            "text": text,
            "input_sections": canonical_sections(source),
            "diagnostics": diagnostics(source, text),
            "gold_difficulty_level": level,
            "gold_difficulty_id": DIFFICULTY_TO_ID[level],
            "acceptable_difficulty_levels": acceptable_levels,
            "acceptable_difficulty_ids": [DIFFICULTY_TO_ID[value] for value in acceptable_levels],
            "gold_confidence": str(row.get("修订后置信度") or "").strip(),
            "adjudication_status": str(row.get("复核状态") or "").strip(),
            "label_source": "gpt56_rereview_adjudicated",
            "label_quality": {"tier": "gold", "sample_weight": 1.0},
        })

    overlap = set()
    if args.reference_train_file:
        overlap = seen & read_ids(Path(args.reference_train_file))
        if overlap and args.exclude_reference_overlap:
            output_rows = [row for row in output_rows if row["id"] not in overlap]
        elif overlap and not args.allow_reference_overlap:
            raise ValueError(f"Gold/train ID overlap: {len(overlap)} records. Refuse to create a final gold test set; remove overlap from training or pass --allow_reference_overlap for audit-only output.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
    print(json.dumps({
        "records": len(output_rows),
        "source_csv_records": len(rows),
        "output": str(output_path.resolve()),
        "primary_label_distribution": Counter(row["gold_difficulty_level"] for row in output_rows),
        "confidence_distribution": Counter(row["gold_confidence"] for row in output_rows),
        "reference_train_overlap": len(overlap),
        "excluded_reference_overlap_ids": sorted(overlap) if args.exclude_reference_overlap else [],
        "skipped_unrenderable_ids": skipped_unrenderable,
    }, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
