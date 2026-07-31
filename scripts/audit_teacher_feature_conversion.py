#!/usr/bin/env python3
"""Audit the frozen-18 -> frozen-10 teacher-data conversion.

This is deliberately a read-only audit.  The source ``difficulty`` field is
never consulted when checking the curated labels; it is only allowed to remain
as provenance metadata in the curated record.  The audit is useful after
converting a large refreshed teacher export (for example the 58k-record
export) because silent defaults in a feature normalizer must not go unnoticed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.schema import (  # noqa: E402
    FROZEN_18_FEATURE_NAMES,
    FEATURE_VALUES,
    DIFFICULTY_TO_ID,
    normalize_v2_features,
)


def rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: JSON record must be an object")
                yield line_number, value


def digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_parts(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    rating = row.get("difficulty_rating") or {}
    # Also accept a curated/frozen18 source.  This makes the audit repeatable
    # after the raw API export has been moved off the training machine.
    if rating:
        question_id = str(row.get("question_id") or row.get("id") or "").strip()
        level = rating.get("difficulty_level")
        features = rating.get("features")
    else:
        question_id = str(row.get("id") or row.get("question_id") or "").strip()
        level = row.get("teacher_difficulty_level")
        features = row.get("teacher_features_legacy18")
    if not question_id or not isinstance(features, dict):
        raise ValueError("source row is missing question ID or frozen-18 features")
    return question_id, str(level or ""), features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="API teacher export or frozen18 JSONL")
    parser.add_argument("--curated", required=True, help="Output of prepare_teacher_data.py")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    source_path, curated_path, report_path = map(Path, (args.source, args.curated, args.report))
    errors: list[str] = []
    source_by_id: dict[str, tuple[str, dict[str, Any], int]] = {}
    source_records = 0
    for line_number, row in rows(source_path):
        source_records += 1
        try:
            question_id, level, features = source_parts(row)
        except ValueError as error:
            errors.append(f"source:{line_number}: {error}")
            continue
        if question_id in source_by_id:
            errors.append(f"source:{line_number}: duplicate question ID {question_id}")
            continue
        source_by_id[question_id] = (level, features, line_number)

    curated_by_id: dict[str, dict[str, Any]] = {}
    curated_records = 0
    for line_number, row in rows(curated_path):
        curated_records += 1
        question_id = str(row.get("id") or "").strip()
        if not question_id:
            errors.append(f"curated:{line_number}: missing id")
            continue
        if question_id in curated_by_id:
            errors.append(f"curated:{line_number}: duplicate question ID {question_id}")
            continue
        curated_by_id[question_id] = row

    counts: Counter[str] = Counter()
    for question_id, (level, legacy, source_line) in source_by_id.items():
        row = curated_by_id.get(question_id)
        if row is None:
            counts["missing_from_curated"] += 1
            continue
        counts["matched"] += 1
        if level not in DIFFICULTY_TO_ID:
            errors.append(f"source:{source_line}: invalid difficulty level {level!r}")
        curated_level = row.get("teacher_difficulty_level")
        if curated_level != level:
            errors.append(f"id={question_id}: difficulty level changed ({level!r} -> {curated_level!r})")
        missing = [name for name in FROZEN_18_FEATURE_NAMES if name not in legacy]
        if missing:
            errors.append(f"id={question_id}: source missing frozen-18 fields {missing}")
            continue
        curated_legacy = row.get("teacher_features_legacy18")
        if not isinstance(curated_legacy, dict):
            errors.append(f"id={question_id}: curated frozen-18 features missing")
        elif digest(curated_legacy) != digest(legacy):
            errors.append(f"id={question_id}: frozen-18 features were changed")

        expected = normalize_v2_features(legacy)
        actual = row.get("teacher_features")
        if actual != expected:
            errors.append(f"id={question_id}: auxiliary features do not match schema conversion")
        else:
            counts["auxiliary_exact_match"] += 1
        for name, values in FEATURE_VALUES.items():
            if not isinstance(actual, dict) or actual.get(name) not in values:
                errors.append(f"id={question_id}: invalid auxiliary value {name}={actual!r}")
                break

    extra = sorted(set(curated_by_id) - set(source_by_id))
    counts["extra_curated_records"] = len(extra)
    report = {
        "schema_version": "teacher_feature_conversion_audit_v2",
        "source": str(source_path.resolve()),
        "curated": str(curated_path.resolve()),
        "source_records": source_records,
        "curated_records": curated_records,
        "source_unique_ids": len(source_by_id),
        "curated_unique_ids": len(curated_by_id),
        "counts": dict(counts),
        "frozen18_preserved": not any("frozen-18 features were changed" in item for item in errors),
        "auxiliary_exactly_derived": counts["auxiliary_exact_match"] == counts["matched"],
        "raw_difficulty_used": False,
        "ignored_source_fields": ["difficulty"],
        "errors": errors,
        "error_count": len(errors),
        "status": "PASS" if not errors and not extra else "FAIL",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
