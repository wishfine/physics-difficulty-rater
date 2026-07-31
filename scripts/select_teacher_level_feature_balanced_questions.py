#!/usr/bin/env python3
"""Select 10k questions by final teacher level and Aux11 coverage on CPU."""
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

from physics_difficulty.data.text_only import forbidden_source_label_paths
from physics_difficulty.pairwise.feature_coverage import (
    feature_coverage_report,
    validate_feature_map,
)
from physics_difficulty.pairwise.question_selection import (
    allocate_teacher_level_quotas,
    select_questions_by_teacher_level,
)
from physics_difficulty.schema import DIFFICULTY_LEVELS, FEATURE_VALUES


FORBIDDEN_QUESTION_KEYS = {
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
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            yield row


def load_excluded_ids(paths: list[str]) -> set[str]:
    excluded: set[str] = set()
    for value in paths:
        path = Path(value)
        for line_number, row in enumerate(jsonl_rows(path), 1):
            question_id = str(row.get("id") or row.get("question_id") or "").strip()
            if not question_id:
                raise ValueError(
                    f"{path}:{line_number}: exclusion row lacks id/question_id"
                )
            excluded.add(question_id)
    return excluded


def load_question_pool(
    path: Path,
    excluded_ids: set[str],
) -> tuple[list[str], dict[str, Counter[str]], int]:
    ids: list[str] = []
    seen: set[str] = set()
    excluded_records = 0
    diagnostics = {
        "input_length_bucket": Counter(),
        "has_analysis": Counter(),
        "has_subquestions": Counter(),
        "image_dependency_risk": Counter(),
    }
    for row in jsonl_rows(path):
        question_id = str(row.get("id") or row.get("question_id") or "").strip()
        if not question_id or question_id in seen:
            raise ValueError("question pool must have unique non-empty IDs")
        seen.add(question_id)
        if str(row.get("split")) != "train":
            raise ValueError(f"question {question_id} is not in the train split")
        forbidden = forbidden_source_label_paths(row)
        forbidden.extend(key for key in FORBIDDEN_QUESTION_KEYS if key in row)
        if forbidden:
            raise ValueError(
                f"question {question_id} contains forbidden label fields: "
                f"{sorted(set(forbidden))}"
            )
        if not str(row.get("text") or "").strip():
            raise ValueError(f"question {question_id} has empty text")
        if question_id in excluded_ids:
            excluded_records += 1
            continue
        values = row.get("diagnostics") or {}
        for name in diagnostics:
            diagnostics[name][str(values.get(name, "unknown"))] += 1
        ids.append(question_id)
    if not ids:
        raise ValueError("question pool is empty after exclusions")
    return ids, diagnostics, excluded_records


def load_teacher_data(
    path: Path,
    expected_ids: set[str],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    rows = list(jsonl_rows(path))
    features = validate_feature_map(rows, allowed_question_ids=expected_ids)
    levels: dict[str, str] = {}
    for row in rows:
        question_id = str(row.get("id") or row.get("question_id") or "").strip()
        if question_id not in expected_ids:
            continue
        if question_id in levels:
            raise ValueError(f"duplicate teacher-label ID: {question_id}")
        level = str(row.get("teacher_difficulty_level") or "").strip()
        if level not in DIFFICULTY_LEVELS:
            raise ValueError(
                f"question {question_id} has invalid final teacher level {level!r}"
            )
        levels[question_id] = level
    missing = expected_ids - set(levels)
    if missing:
        raise ValueError(
            f"teacher file is missing final difficulty levels for {len(missing)} questions"
        )
    return levels, features


def validate_category_floors(
    pool_features: dict[str, dict[str, str]],
    selected_ids: set[str],
    *,
    minimum_count: int,
    scope: str,
) -> dict[str, Any]:
    checks: dict[str, dict[str, dict[str, int | bool]]] = {}
    violations: list[str] = []
    for name, values in FEATURE_VALUES.items():
        checks[name] = {}
        for value in values:
            pool_count = sum(
                row[name] == value for row in pool_features.values()
            )
            if not pool_count:
                continue
            selected_count = sum(
                question_id in selected_ids and row[name] == value
                for question_id, row in pool_features.items()
            )
            required = min(minimum_count, pool_count)
            passed = selected_count >= required
            checks[name][value] = {
                "pool_count": pool_count,
                "required_selected_count": required,
                "selected_count": selected_count,
                "passed": passed,
            }
            if not passed:
                violations.append(
                    f"{scope}:{name}={value}: selected={selected_count}, "
                    f"required={required}"
                )
    if violations:
        raise ValueError(
            "auxiliary category coverage floor failed: " + "; ".join(violations[:20])
        )
    return {
        "scope": scope,
        "minimum_requested": minimum_count,
        "status": "PASS",
        "checks": checks,
    }


def write_selected_questions(
    source: Path,
    output: Path,
    selected_order: list[str],
) -> dict[str, Counter[str]]:
    selected_set = set(selected_order)
    rows = {
        str(row.get("id") or row.get("question_id")): row
        for row in jsonl_rows(source)
        if str(row.get("id") or row.get("question_id")) in selected_set
    }
    if set(rows) != selected_set:
        raise ValueError("selected question IDs could not be recovered from source")
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = {
        "input_length_bucket": Counter(),
        "has_analysis": Counter(),
        "has_subquestions": Counter(),
        "image_dependency_risk": Counter(),
    }
    with output.open("w", encoding="utf-8") as target:
        for question_id in selected_order:
            row = rows[question_id]
            values = row.get("diagnostics") or {}
            for name in diagnostics:
                diagnostics[name][str(values.get(name, "unknown"))] += 1
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    return diagnostics


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    known, _ = bootstrap.parse_known_args()
    defaults = (
        json.loads(Path(known.config).read_text(encoding="utf-8"))
        if known.config
        else {}
    )
    parser = argparse.ArgumentParser(description=__doc__, parents=[bootstrap])
    parser.set_defaults(**defaults)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--teacher-data", required=True)
    parser.add_argument(
        "--exclude-question-ids",
        action="append",
        default=[],
        help="Question JSONL whose id/question_id values must be excluded; repeatable",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--target-count",
        type=int,
        required="target_count" not in defaults,
    )
    parser.add_argument("--minimum-per-level", type=int, default=1_000)
    parser.add_argument("--distribution-fraction", type=float, default=0.80)
    parser.add_argument("--rare-fraction", type=float, default=0.10)
    parser.add_argument("--random-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-category-count-global", type=int, default=20)
    parser.add_argument("--minimum-category-count-per-level", type=int, default=3)
    parser.add_argument("--maximum-feature-jsd", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    questions_path = Path(args.questions)
    teacher_path = Path(args.teacher_data)
    excluded_ids = load_excluded_ids(args.exclude_question_ids)
    ids, pool_diagnostics, excluded_records = load_question_pool(
        questions_path, excluded_ids
    )
    expected_ids = set(ids)
    levels, features = load_teacher_data(teacher_path, expected_ids)
    level_counts = Counter(levels.values())
    quotas = allocate_teacher_level_quotas(
        {level: level_counts[level] for level in DIFFICULTY_LEVELS},
        target_count=args.target_count,
        minimum_per_level=args.minimum_per_level,
    )
    print(
        json.dumps(
            {
                "message": (
                    "Selecting questions from final teacher levels and Aux11 in CPU "
                    "memory; raw difficulty and old BT scores are not read."
                ),
                "pool_questions": len(ids),
                "target_questions": args.target_count,
                "teacher_level_quotas": quotas,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    selected = select_questions_by_teacher_level(
        ids,
        levels,
        features,
        target_count=args.target_count,
        minimum_per_level=args.minimum_per_level,
        distribution_fraction=args.distribution_fraction,
        rare_fraction=args.rare_fraction,
        random_fraction=args.random_fraction,
        minimum_category_count_global=args.minimum_category_count_global,
        minimum_category_count_per_level=args.minimum_category_count_per_level,
        seed=args.seed,
    )
    selected_ids = {row["question_id"] for row in selected}
    selected_order = [row["question_id"] for row in selected]
    selected_level_counts = Counter(
        row["teacher_difficulty_level"] for row in selected
    )
    if {
        level: selected_level_counts[level] for level in DIFFICULTY_LEVELS
    } != {
        level: quotas[level] for level in DIFFICULTY_LEVELS
    }:
        raise RuntimeError("selected teacher-level counts do not match allocated quotas")

    coverage = feature_coverage_report(features, selected_ids)
    if coverage["zero_covered_source_categories"]:
        raise ValueError("selection dropped one or more auxiliary categories")
    if (
        coverage["maximum_marginal_jensen_shannon_divergence"]
        > args.maximum_feature_jsd
    ):
        raise ValueError(
            "selected auxiliary-feature distribution exceeds maximum JSD: "
            f"{coverage['maximum_marginal_jensen_shannon_divergence']:.6f}"
        )
    global_floor_checks = validate_category_floors(
        features,
        selected_ids,
        minimum_count=args.minimum_category_count_global,
        scope="global",
    )

    per_level_coverage: dict[str, Any] = {}
    per_level_floor_checks: dict[str, Any] = {}
    for level in DIFFICULTY_LEVELS:
        level_features = {
            question_id: features[question_id]
            for question_id in ids
            if levels[question_id] == level
        }
        selected_in_level = {
            question_id
            for question_id in selected_ids
            if levels[question_id] == level
        }
        per_level_coverage[level] = feature_coverage_report(
            level_features, selected_in_level
        )
        per_level_floor_checks[level] = validate_category_floors(
            level_features,
            selected_in_level,
            minimum_count=args.minimum_category_count_per_level,
            scope=f"teacher_level_{level}",
        )

    selected_diagnostics = write_selected_questions(
        questions_path, Path(args.output), selected_order
    )
    selected_by_id = {row["question_id"]: row for row in selected}
    audit_output = Path(args.audit_output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(
        "".join(
            json.dumps(
                {
                    **selected_by_id[question_id],
                    "teacher_features": features[question_id],
                },
                ensure_ascii=False,
            )
            + "\n"
            for question_id in selected_order
        ),
        encoding="utf-8",
    )

    reason_counts = Counter(row["selection_reason"] for row in selected)
    report = {
        "schema_version": "teacher_level_aux11_question_selection_v1",
        "cpu_only": True,
        "loads_model": False,
        "questions": str(questions_path.resolve()),
        "question_pool_sha256": sha256_file(questions_path),
        "teacher_data": str(teacher_path.resolve()),
        "teacher_data_sha256": sha256_file(teacher_path),
        "pool_questions": len(ids),
        "source_questions": len(ids) + excluded_records,
        "excluded_question_ids_requested": len(excluded_ids),
        "excluded_questions": excluded_records,
        "excluded_question_files": [
            {
                "path": str(Path(path).resolve()),
                "sha256": sha256_file(Path(path)),
            }
            for path in args.exclude_question_ids
        ],
        "selected_questions": len(selected),
        "seed": args.seed,
        "teacher_level_policy": {
            "minimum_per_level_when_available": args.minimum_per_level,
            "remaining_quota": "proportional_to_source_teacher_level_distribution",
            "pool_counts": {
                level: level_counts[level] for level in DIFFICULTY_LEVELS
            },
            "selected_quotas": quotas,
            "selected_counts": {
                level: selected_level_counts[level]
                for level in DIFFICULTY_LEVELS
            },
        },
        "selection_fractions_after_category_floors": {
            "distribution_matched": args.distribution_fraction,
            "rare_feature_protection": args.rare_fraction,
            "random_exploration": args.random_fraction,
        },
        "selection_reason_counts": dict(reason_counts),
        "category_coverage_floors": {
            "global": args.minimum_category_count_global,
            "per_teacher_level_when_available": (
                args.minimum_category_count_per_level
            ),
        },
        "feature_coverage": coverage,
        "feature_coverage_by_teacher_level": per_level_coverage,
        "category_floor_checks": {
            "global": global_floor_checks,
            "by_teacher_level": per_level_floor_checks,
        },
        "diagnostics": {
            name: {
                "pool": dict(pool_diagnostics[name]),
                "selected": dict(selected_diagnostics[name]),
            }
            for name in pool_diagnostics
        },
        "guardrails": {
            "source_raw_difficulty_used": False,
            "final_teacher_difficulty_level_used_for_sampling_only": True,
            "old_bt_score_used_for_selection": False,
            "teacher_labels_written_to_selected_questions": False,
            "teacher_labels_written_to_teacher_pair_prompts": False,
        },
        "outputs": {
            "selected_questions": str(Path(args.output).resolve()),
            "selected_questions_sha256": sha256_file(Path(args.output)),
            "private_selection_audit": str(audit_output.resolve()),
            "private_selection_audit_sha256": sha256_file(audit_output),
        },
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_questions": len(selected),
                "teacher_level_counts": report["teacher_level_policy"][
                    "selected_counts"
                ],
                "selection_reason_counts": report["selection_reason_counts"],
                "maximum_feature_jsd": coverage[
                    "maximum_marginal_jensen_shannon_divergence"
                ],
                "output": report["outputs"]["selected_questions"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
