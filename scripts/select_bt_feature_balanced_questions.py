#!/usr/bin/env python3
"""Select a CPU-only BT-decile and auxiliary-feature balanced training set."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.data.text_only import forbidden_source_label_paths
from physics_difficulty.schema import FEATURE_VALUES
from physics_difficulty.pairwise.feature_coverage import (
    feature_coverage_report,
    validate_feature_map,
)
from physics_difficulty.pairwise.question_selection import (
    assign_bt_deciles,
    select_questions_by_bt_decile,
)


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
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            yield value


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


def question_pool(
    path: Path,
    excluded_ids: set[str] | None = None,
) -> tuple[list[str], dict[str, Counter[str]], int]:
    excluded_ids = excluded_ids or set()
    ids: list[str] = []
    seen: set[str] = set()
    excluded_records = 0
    diagnostics: dict[str, Counter[str]] = {
        "input_length_bucket": Counter(),
        "has_analysis": Counter(),
        "has_subquestions": Counter(),
        "image_dependency_risk": Counter(),
    }
    for row in jsonl_rows(path):
        question_id = str(row.get("id") or row.get("question_id") or "").strip()
        if not question_id or question_id in seen:
            raise ValueError("question pool must have unique non-empty IDs")
        if str(row.get("split")) != "train":
            raise ValueError(f"question {question_id} is not in the train split")
        forbidden = forbidden_source_label_paths(row)
        forbidden.extend(key for key in FORBIDDEN_QUESTION_KEYS if key in row)
        if forbidden:
            raise ValueError(
                f"question {question_id} contains forbidden label fields: {sorted(set(forbidden))}"
            )
        if not str(row.get("text") or "").strip():
            raise ValueError(f"question {question_id} has empty text")
        if question_id in excluded_ids:
            excluded_records += 1
            seen.add(question_id)
            continue
        values = row.get("diagnostics") or {}
        for name in diagnostics:
            diagnostics[name][str(values.get(name, "unknown"))] += 1
        ids.append(question_id)
        seen.add(question_id)
    if not ids:
        raise ValueError("question pool is empty")
    return ids, diagnostics, excluded_records


def load_scores(path: Path, expected_ids: set[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in jsonl_rows(path):
        question_id = str(row.get("question_id") or row.get("id") or "").strip()
        if not question_id or question_id in scores:
            raise ValueError("score file must have unique non-empty question IDs")
        score = float(row["raw_difficulty_score"])
        if not math.isfinite(score):
            raise ValueError(f"question {question_id} has a non-finite score")
        scores[question_id] = score
    if set(scores) != expected_ids:
        raise ValueError(
            f"score IDs do not match question pool: missing={len(expected_ids - set(scores))}, "
            f"extra={len(set(scores) - expected_ids)}"
        )
    return scores


def load_features(
    path: Path, expected_ids: set[str]
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    features = validate_feature_map(
        jsonl_rows(path), allowed_question_ids=expected_ids
    )
    levels = {}
    for row in jsonl_rows(path):
        question_id = str(row.get("id") or row.get("question_id") or "").strip()
        if question_id in expected_ids:
            levels[question_id] = str(
                row.get("teacher_difficulty_level") or "unknown"
            )
    return features, levels


def count_report(
    pool_values: dict[str, str],
    selected_ids: set[str],
) -> dict[str, dict[str, float | int]]:
    pool = Counter(pool_values.values())
    selected = Counter(
        value for question_id, value in pool_values.items() if question_id in selected_ids
    )
    return {
        value: {
            "pool_count": pool[value],
            "selected_count": selected[value],
            "pool_share": pool[value] / max(1, len(pool_values)),
            "selected_share": selected[value] / max(1, len(selected_ids)),
        }
        for value in sorted(pool)
    }


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
                features[name] == value for features in pool_features.values()
            )
            if pool_count == 0:
                continue
            selected_count = sum(
                question_id in selected_ids and features[name] == value
                for question_id, features in pool_features.items()
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
                    f"{scope}:{name}={value}: selected={selected_count}, required={required}"
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
    diagnostics: dict[str, Counter[str]] = {
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
    parser.add_argument("--features-file", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--scores-manifest", required=True)
    parser.add_argument(
        "--exclude-question-ids",
        action="append",
        default=[],
        help=(
            "Question JSONL whose id/question_id values must be excluded; repeatable. "
            "Use the same arguments when generating --scores."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--target-count",
        type=int,
        required="target_count" not in defaults,
    )
    parser.add_argument("--deciles", type=int, default=10)
    parser.add_argument("--distribution-fraction", type=float, default=0.80)
    parser.add_argument("--rare-fraction", type=float, default=0.10)
    parser.add_argument("--random-fraction", type=float, default=0.10)
    parser.add_argument("--maximum-feature-jsd", type=float, default=0.05)
    parser.add_argument("--minimum-category-count-global", type=int, default=20)
    parser.add_argument("--minimum-category-count-per-decile", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    questions_path = Path(args.questions)
    scores_path = Path(args.scores)
    scores_manifest = json.loads(
        Path(args.scores_manifest).read_text(encoding="utf-8")
    )
    question_hash = sha256_file(questions_path)
    score_hash = sha256_file(scores_path)
    if scores_manifest.get("questions_sha256") != question_hash:
        raise ValueError("score manifest was produced from a different question pool")
    if scores_manifest.get("output_sha256") != score_hash:
        raise ValueError("score file hash does not match its manifest")

    excluded_ids = load_excluded_ids(args.exclude_question_ids)
    ids, pool_diagnostics, excluded_records = question_pool(
        questions_path, excluded_ids
    )
    expected_ids = set(ids)
    scores = load_scores(scores_path, expected_ids)
    features, api_levels = load_features(Path(args.features_file), expected_ids)
    print(
        json.dumps(
            {
                "message": "Selecting questions in CPU memory; no model or GPU is loaded.",
                "pool_questions": len(ids),
                "target_questions": args.target_count,
                "bt_score_bins": args.deciles,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    selected = select_questions_by_bt_decile(
        ids,
        scores,
        features,
        target_count=args.target_count,
        deciles=args.deciles,
        distribution_fraction=args.distribution_fraction,
        rare_fraction=args.rare_fraction,
        random_fraction=args.random_fraction,
        minimum_category_count_global=args.minimum_category_count_global,
        minimum_category_count_per_decile=args.minimum_category_count_per_decile,
        seed=args.seed,
    )
    selected_ids = {row["question_id"] for row in selected}
    selected_order = [row["question_id"] for row in selected]
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

    assignments = assign_bt_deciles(
        ids, scores, deciles=args.deciles, seed=args.seed
    )
    per_decile_coverage = {}
    per_decile_floor_checks = {}
    for decile in range(1, args.deciles + 1):
        decile_features = {
            question_id: features[question_id]
            for question_id in ids
            if assignments[question_id] == decile
        }
        selected_in_decile = {
            question_id
            for question_id in selected_ids
            if assignments[question_id] == decile
        }
        per_decile_coverage[str(decile)] = feature_coverage_report(
            decile_features, selected_in_decile
        )
        per_decile_floor_checks[str(decile)] = validate_category_floors(
            decile_features,
            selected_in_decile,
            minimum_count=args.minimum_category_count_per_decile,
            scope=f"bt_decile_{decile}",
        )

    selected_diagnostics = write_selected_questions(
        questions_path, Path(args.output), selected_order
    )
    audit_rows = []
    selected_by_id = {row["question_id"]: row for row in selected}
    for question_id in selected_order:
        audit_rows.append(
            {
                **selected_by_id[question_id],
                "teacher_features": features[question_id],
                "api_teacher_difficulty_level": api_levels.get(
                    question_id, "unknown"
                ),
            }
        )
    audit_output = Path(args.audit_output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in audit_rows
        ),
        encoding="utf-8",
    )

    reason_counts = Counter(row["selection_reason"] for row in selected)
    decile_counts = Counter(str(row["bt_decile"]) for row in selected)
    report = {
        "schema_version": "bt_feature_balanced_question_selection_v2",
        "cpu_only": True,
        "loads_model": False,
        "questions": str(questions_path.resolve()),
        "question_pool_sha256": question_hash,
        "features_file": str(Path(args.features_file).resolve()),
        "scores": str(scores_path.resolve()),
        "scores_sha256": score_hash,
        "score_checkpoint_fingerprint": scores_manifest.get(
            "checkpoint_fingerprint"
        ),
        "pool_questions": len(ids),
        "source_questions": len(ids) + excluded_records,
        "excluded_questions": excluded_records,
        "excluded_question_files": [
            str(Path(path).resolve()) for path in args.exclude_question_ids
        ],
        "selected_questions": len(selected),
        "seed": args.seed,
        "selection_fractions": {
            "distribution_matched": args.distribution_fraction,
            "rare_feature_protection": args.rare_fraction,
            "random_exploration": args.random_fraction,
        },
        "category_coverage_floors": {
            "global": args.minimum_category_count_global,
            "per_bt_decile_when_available": args.minimum_category_count_per_decile,
        },
        "selection_reason_counts": dict(reason_counts),
        "bt_deciles": dict(sorted(decile_counts.items(), key=lambda item: int(item[0]))),
        "feature_coverage": coverage,
        "feature_coverage_by_bt_decile": per_decile_coverage,
        "category_floor_checks": {
            "global": global_floor_checks,
            "by_bt_decile": per_decile_floor_checks,
        },
        "api_teacher_level_audit_only": count_report(api_levels, selected_ids),
        "diagnostics": {
            name: {
                "pool": dict(pool_diagnostics[name]),
                "selected": dict(selected_diagnostics[name]),
            }
            for name in pool_diagnostics
        },
        "guardrails": {
            "raw_difficulty_used": False,
            "api_teacher_level_used_for_selection": False,
            "bt_score_used_only_for_equal_frequency_bins": True,
            "teacher_features_written_to_selected_questions": False,
        },
        "outputs": {
            "selected_questions": str(Path(args.output).resolve()),
            "private_selection_audit": str(audit_output.resolve()),
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
                "bt_deciles": report["bt_deciles"],
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
