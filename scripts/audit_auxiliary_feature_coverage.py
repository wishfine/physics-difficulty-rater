#!/usr/bin/env python3
"""Privately audit auxiliary-category coverage after label-free sampling."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.data.text_only import forbidden_source_label_paths
from physics_difficulty.pairwise.feature_coverage import (
    feature_coverage_report,
    validate_feature_map,
)


QUESTION_FORBIDDEN_FIELDS = {
    "difficulty",
    "raw_difficulty",
    "teacher_difficulty_id",
    "teacher_difficulty_level",
    "teacher_features",
    "teacher_features_legacy18",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield row


def forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    result = list(forbidden_source_label_paths(value, prefix))
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in QUESTION_FORBIDDEN_FIELDS or str(key).startswith("teacher_difficulty"):
                result.append(path)
            result.extend(forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(forbidden_paths(child, f"{prefix}[{index}]"))
    return sorted(set(result))


def load_question_ids(path: Path, expected_split: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for line_number, row in enumerate(jsonl_rows(path), 1):
        question_id = str(row.get("id") or "").strip()
        if not question_id or question_id in seen:
            raise ValueError(f"{path}:{line_number}: question IDs must be non-empty and unique")
        if str(row.get("split") or "") != expected_split:
            raise ValueError(
                f"{path}:{line_number}: expected split={expected_split!r}, got {row.get('split')!r}"
            )
        findings = forbidden_paths(row)
        if findings:
            raise ValueError(f"{path}:{line_number}: question is not label-free: {findings}")
        ids.append(question_id)
        seen.add(question_id)
    if not ids:
        raise ValueError(f"{path}: question file is empty")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Aux11 category coverage after question sampling. It reads "
            "private feature labels only after node selection and never rewrites "
            "or augments teacher-facing question data."
        )
    )
    parser.add_argument("--pool-questions", required=True)
    parser.add_argument("--selected-questions", required=True)
    parser.add_argument("--features-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-split", default="validation")
    parser.add_argument("--maximum-feature-jsd", type=float, default=0.05)
    args = parser.parse_args()
    if args.maximum_feature_jsd < 0:
        raise ValueError("--maximum-feature-jsd must be non-negative")

    pool_path = Path(args.pool_questions)
    selected_path = Path(args.selected_questions)
    pool_ids = load_question_ids(pool_path, args.expected_split)
    selected_ids = load_question_ids(selected_path, args.expected_split)
    if not set(selected_ids) <= set(pool_ids):
        raise ValueError("selected questions are not a subset of the validation pool")
    features = validate_feature_map(
        jsonl_rows(Path(args.features_file)), allowed_question_ids=pool_ids
    )
    coverage = feature_coverage_report(features, selected_ids)
    errors: list[str] = []
    if coverage["zero_covered_source_categories"]:
        errors.append(
            "selected sample drops one or more auxiliary categories present in the source pool"
        )
    observed = coverage["maximum_marginal_jensen_shannon_divergence"]
    if observed > args.maximum_feature_jsd:
        errors.append(
            f"maximum marginal JSD {observed:.6f} exceeds {args.maximum_feature_jsd:.6f}"
        )
    report = {
        "schema_version": "auxiliary_feature_coverage_audit_v1",
        "status": "PASS" if not errors else "FAIL",
        "pool_questions": str(pool_path.resolve()),
        "selected_questions": str(selected_path.resolve()),
        "features_file": str(Path(args.features_file).resolve()),
        "pool_questions_sha256": sha256_file(pool_path),
        "selected_questions_sha256": sha256_file(selected_path),
        "features_file_sha256": sha256_file(Path(args.features_file)),
        "expected_split": args.expected_split,
        "maximum_feature_jsd_allowed": args.maximum_feature_jsd,
        "coverage": coverage,
        "guardrails": {
            "features_used_after_node_selection_only": True,
            "absolute_difficulty_labels_used": False,
            "raw_difficulty_used": False,
            "features_exported_to_teacher": False,
        },
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
