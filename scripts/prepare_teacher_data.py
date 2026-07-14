#!/usr/bin/env python3
"""Convert API + V7 JSONL into versioned, quarantined V2 training data."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from physics_difficulty.data.formatting import FORMATTER_VERSION, canonical_sections, diagnostics, format_question
from physics_difficulty.data.quality import score_label_quality
from physics_difficulty.schema import difficulty_id, normalize_knowledge_domains, normalize_v2_features


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _raw_api_disagreement(raw: Any, teacher_id: int) -> int | None:
    try:
        raw_id = int(raw) - 1
    except (TypeError, ValueError):
        return None
    return abs(raw_id - teacher_id) if 0 <= raw_id <= 4 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="API + postprocess JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--conflict-output", help="quarantine exact-text conflicts; default is next to output")
    parser.add_argument("--prompt-version", default="unknown")
    parser.add_argument("--postprocess-version", default="v7")
    parser.add_argument("--teacher-model", default="unknown")
    parser.add_argument("--source-dataset-id", default="unknown")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    conflict_output = Path(args.conflict_output) if args.conflict_output else output.with_suffix(".conflicts.jsonl")
    candidates: list[Dict[str, Any]] = []
    grouped: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    stats: Counter = Counter()

    with open(args.input, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            rating = record.get("difficulty_rating") or {}
            level = rating.get("difficulty_level")
            if not level:
                stats["missing_teacher_label"] += 1
                continue
            try:
                teacher_id = difficulty_id(level)
            except ValueError:
                stats["invalid_teacher_label"] += 1
                continue
            text = format_question(record)
            digest = text_hash(text)
            features = normalize_v2_features(rating.get("features"))
            quality = score_label_quality(level, features, record)
            item = {
                "id": str(record.get("question_id", line_number)),
                "parent_id": str(record.get("parent_id", record.get("question_id", line_number))),
                "source_dataset_id": str(record.get("source_dataset_id", args.source_dataset_id)),
                "text": text,
                "input_sections": canonical_sections(record),
                "input_sha256": digest,
                "raw_difficulty": record.get("difficulty"),
                "teacher_difficulty_level": level,
                "teacher_difficulty_id": teacher_id,
                "teacher_features": features,
                "feature_metadata": {"knowledge_domains": normalize_knowledge_domains(rating.get("features"))},
                "label_source": "api_v7",
                "label_schema_version": "v2",
                "prompt_version": args.prompt_version,
                "postprocess_version": args.postprocess_version,
                "teacher_model": args.teacher_model,
                "diagnostics": diagnostics(record, text),
                "label_quality": quality,
            }
            item["diagnostics"]["raw_api_disagreement"] = _raw_api_disagreement(item["raw_difficulty"], teacher_id)
            candidates.append(item)
            grouped[digest].append(item)

    accepted, quarantined = [], []
    for digest, items in grouped.items():
        labels = {item["teacher_difficulty_id"] for item in items}
        if len(labels) > 1:
            for item in items:
                item["label_quality"] = {"label_quality": "invalid", "sample_weight": 0.0, "conflicts": ["完全重复文本的教师标签冲突"], "conflict_severity": "severe", "feature_level_consistent": False, "review_action": "rejudge"}
                quarantined.append(item)
            stats["duplicate_label_conflict_groups"] += 1
            continue
        accepted.append(items[0])
        if len(items) > 1:
            stats["exact_duplicate_removed"] += len(items) - 1

    with output.open("w", encoding="utf-8") as target:
        for item in accepted:
            target.write(json.dumps(item, ensure_ascii=False) + "\n")
            stats[f"level_{item['teacher_difficulty_level']}"] += 1
            stats[f"quality_{item['label_quality']['label_quality']}"] += 1
    with conflict_output.open("w", encoding="utf-8") as target:
        for item in quarantined:
            target.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "input": str(Path(args.input).resolve()), "output": str(output.resolve()), "conflict_output": str(conflict_output.resolve()),
        "schema_version": "v2", "formatter_version": FORMATTER_VERSION, "records": len(accepted), "quarantined_records": len(quarantined),
        "provenance": {"prompt_version": args.prompt_version, "postprocess_version": args.postprocess_version, "teacher_model": args.teacher_model},
        "stats": dict(stats),
    }
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
