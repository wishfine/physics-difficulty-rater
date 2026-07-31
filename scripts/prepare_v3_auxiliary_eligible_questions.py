#!/usr/bin/env python3
"""Keep label-free V3 questions that have a complete, valid aux10 label row.

The output deliberately copies only the original question records.  Auxiliary
labels are used here for coverage accounting and eligibility; they are attached
to final teacher-labelled pairs later by ``attach_pairwise_auxiliary_features``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.data.text_only import forbidden_source_label_paths
from physics_difficulty.schema import FEATURE_VALUES


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_features(value: Any) -> bool:
    return isinstance(value, dict) and all(value.get(name) in values for name, values in FEATURE_VALUES.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare aux10-eligible label-free V3 questions")
    parser.add_argument("--questions", required=True, help="Text-only V3 question JSONL")
    parser.add_argument("--features", required=True, help="V2 curated JSONL containing teacher_features")
    parser.add_argument("--output", required=True, help="Eligible label-free V3 questions")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--excluded-output", help="ID-only reasons for questions not eligible for aux10")
    parser.add_argument("--minimum-class-support", type=int, default=100)
    args = parser.parse_args()
    if args.minimum_class_support < 1:
        raise ValueError("minimum-class-support must be positive")

    question_path = Path(args.questions)
    feature_path = Path(args.features)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    excluded_path = Path(args.excluded_output) if args.excluded_output else output_path.with_suffix(".excluded.jsonl")

    by_id: dict[str, dict[str, str]] = {}
    feature_schema_versions: set[str] = set()
    invalid_feature_ids: set[str] = set()
    for line_number, row in read_jsonl(feature_path):
        question_id = str(row.get("id") or "").strip()
        if not question_id:
            raise ValueError(f"feature row {line_number} has no id")
        if question_id in by_id or question_id in invalid_feature_ids:
            raise ValueError(f"duplicate feature id {question_id} at {feature_path}:{line_number}")
        features = row.get("teacher_features")
        if not valid_features(features):
            invalid_feature_ids.add(question_id)
            continue
        by_id[question_id] = {name: str(features[name]) for name in FEATURE_VALUES}
        if row.get("feature_schema_version"):
            feature_schema_versions.add(str(row["feature_schema_version"]))

    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    feature_counts = {name: Counter() for name in FEATURE_VALUES}
    for line_number, row in read_jsonl(question_path):
        question_id = str(row.get("id") or "").strip()
        if not question_id:
            raise ValueError(f"question row {line_number} has no id")
        if question_id in seen_ids:
            raise ValueError(f"duplicate question id {question_id} at {question_path}:{line_number}")
        seen_ids.add(question_id)
        forbidden = forbidden_source_label_paths(row)
        if forbidden:
            raise ValueError(f"question {question_id} is not label-free: {forbidden}")
        features = by_id.get(question_id)
        if features is None:
            excluded.append({
                "id": question_id,
                "split": str(row.get("split") or "unspecified"),
                "reason": "invalid_auxiliary_features" if question_id in invalid_feature_ids else "missing_auxiliary_features",
            })
            continue
        accepted.append(row)
        split_counts[str(row.get("split") or "unspecified")] += 1
        for name, value in features.items():
            feature_counts[name][value] += 1

    feature_report = {}
    warnings: list[str] = []
    for name, values in FEATURE_VALUES.items():
        classes = {value: feature_counts[name][value] for value in values}
        low_support = [value for value, count in classes.items() if count < args.minimum_class_support]
        if low_support:
            warnings.append(f"{name} has class support below {args.minimum_class_support}: {low_support}")
        feature_report[name] = {
            "observed_class_count": sum(count > 0 for count in classes.values()),
            "classes": classes,
            "low_support_classes": low_support,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    excluded_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted), encoding="utf-8")
    excluded_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in excluded), encoding="utf-8")
    manifest = {
        "schema_version": "v3_auxiliary_eligible_questions_v1",
        "questions": str(question_path.resolve()),
        "feature_source": str(feature_path.resolve()),
        "output": str(output_path.resolve()),
        "excluded_output": str(excluded_path.resolve()),
        "question_records": len(seen_ids),
        "feature_records_with_complete_aux10": len(by_id),
        "eligible_questions": len(accepted),
        "excluded_questions": len(excluded),
        "split_counts": dict(split_counts),
        "feature_schema_versions": sorted(feature_schema_versions),
        "minimum_class_support": args.minimum_class_support,
        "feature_coverage": feature_report,
        "questions_sha256": sha256_file(question_path),
        "features_sha256": sha256_file(feature_path),
        "warnings": warnings,
        "labels_copied_to_output": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "eligible_questions": len(accepted),
        "excluded_questions": len(excluded),
        "output": str(output_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "warning_count": len(warnings),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
