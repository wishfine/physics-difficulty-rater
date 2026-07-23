#!/usr/bin/env python3
"""Attach frozen ten-dimensional teacher features to pairwise records by question ID."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.schema import FEATURE_TO_ID


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error


def validated_features(row: dict[str, Any], path: Path, line_number: int) -> dict[str, str]:
    raw = row.get("teacher_features")
    if not isinstance(raw, dict):
        raise ValueError(f"missing teacher_features at {path}:{line_number}")
    result: dict[str, str] = {}
    for name, value_to_id in FEATURE_TO_ID.items():
        value = raw.get(name)
        if value not in value_to_id:
            raise ValueError(f"invalid teacher feature {name}={value!r} at {path}:{line_number}")
        result[name] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--features", required=True, help="Frozen18 curated JSONL; only id, teacher_features, and label_quality.sample_weight are read")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--minimum-question-coverage", type=float, default=0.95)
    args = parser.parse_args()
    if not 0 <= args.minimum_question_coverage <= 1:
        raise ValueError("minimum-question-coverage must be in [0, 1]")

    pairs_path = Path(args.pairs)
    features_path = Path(args.features)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)

    feature_index: dict[str, tuple[dict[str, str], float]] = {}
    for line_number, row in read_jsonl(features_path):
        question_id = str(row.get("id") or "").strip()
        if not question_id:
            raise ValueError(f"missing id at {features_path}:{line_number}")
        if question_id in feature_index:
            raise ValueError(f"duplicate feature question id: {question_id}")
        quality = float((row.get("label_quality") or {}).get("sample_weight", 1.0))
        if not 0 < quality <= 1:
            raise ValueError(f"invalid feature quality for {question_id}: {quality}")
        feature_index[question_id] = (validated_features(row, features_path, line_number), quality)

    rows = [row for _, row in read_jsonl(pairs_path)]
    requested_ids = {
        str(row[side])
        for row in rows
        for side in ("question_a_id", "question_b_id")
    }
    matched_ids = requested_ids.intersection(feature_index)
    coverage = len(matched_ids) / max(1, len(requested_ids))
    if coverage < args.minimum_question_coverage:
        raise ValueError(
            f"auxiliary question coverage {coverage:.6f} is below required {args.minimum_question_coverage:.6f}; "
            f"matched={len(matched_ids)} requested={len(requested_ids)}"
        )

    missing_pair_sides = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as target:
        for row in rows:
            attached: dict[str, dict[str, str] | None] = {}
            qualities: dict[str, float] = {}
            for label, id_field in (("question_a", "question_a_id"), ("question_b", "question_b_id")):
                question_id = str(row[id_field])
                found = feature_index.get(question_id)
                if found is None:
                    attached[label] = None
                    qualities[label] = 0.0
                    missing_pair_sides += 1
                else:
                    attached[label], qualities[label] = found
            result = dict(row)
            result["auxiliary_features"] = attached
            result["auxiliary_feature_quality"] = qualities
            target.write(json.dumps(result, ensure_ascii=False) + "\n")

    report = {
        "schema_version": "pairwise_auxiliary_frozen10_v1",
        "pairs": str(pairs_path.resolve()),
        "features": str(features_path.resolve()),
        "output": str(output_path.resolve()),
        "pair_records": len(rows),
        "requested_questions": len(requested_ids),
        "matched_questions": len(matched_ids),
        "missing_questions": len(requested_ids - matched_ids),
        "missing_pair_sides": missing_pair_sides,
        "question_coverage": coverage,
        "feature_names": list(FEATURE_TO_ID),
        "join_key": "question_id == curated.id",
        "ignored_absolute_label_fields": ["difficulty", "raw_difficulty", "teacher_difficulty_id", "teacher_difficulty_level"],
        "input_sha256": {"pairs": sha256(pairs_path), "features": sha256(features_path)},
        "output_sha256": sha256(output_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
