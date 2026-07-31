#!/usr/bin/env python3
"""Audit versioned auxiliary labels attached to pairwise records.

The report deliberately distinguishes pair-side counts from unique-question
counts.  Pair-side counts over-weight questions with higher graph degree and
must not be used as a proxy for auxiliary-label coverage.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.schema import FEATURE_VALUES


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error


def quantile(values: list[int | float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * ratio)
    return ordered[index]


def distribution(counter: Counter[str], values: list[str]) -> dict[str, float]:
    total = sum(counter.values())
    return {value: counter[value] / total if total else 0.0 for value in values}


def entropy(probabilities: dict[str, float]) -> float:
    return -sum(value * math.log(value) for value in probabilities.values() if value > 0)


def jensen_shannon(left: dict[str, float], right: dict[str, float]) -> float:
    midpoint = {key: (left[key] + right[key]) / 2 for key in left}

    def kl(first: dict[str, float], second: dict[str, float]) -> float:
        return sum(value * math.log(value / second[key]) for key, value in first.items() if value > 0)

    return (kl(left, midpoint) + kl(right, midpoint)) / 2


def effective_sample_size(weights: list[float]) -> float:
    numerator = sum(weights) ** 2
    denominator = sum(weight * weight for weight in weights)
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit auxiliary feature-label coverage and imbalance")
    parser.add_argument("--pairs", required=True, help="Pair JSONL with auxiliary_features attached")
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-class-support", type=int, default=50)
    parser.add_argument("--minimum-effective-sample-size", type=float, default=30.0)
    parser.add_argument("--conflict-examples", type=int, default=20)
    args = parser.parse_args()
    if args.minimum_class_support < 1 or args.minimum_effective_sample_size <= 0:
        raise ValueError("minimum support and effective sample size must be positive")

    pair_path = Path(args.pairs)
    question_features: dict[str, dict[str, str]] = {}
    question_quality: dict[str, float] = {}
    question_degrees: Counter[str] = Counter()
    pair_side_counts = {name: Counter() for name in FEATURE_VALUES}
    pair_count = 0
    side_count = 0
    missing_sides = 0
    invalid_values: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    conflict_count = 0

    for line_number, row in read_jsonl(pair_path):
        pair_count += 1
        labels = row.get("auxiliary_features")
        qualities = row.get("auxiliary_feature_quality") or {}
        if not isinstance(labels, dict):
            raise ValueError(f"missing auxiliary_features at {pair_path}:{line_number}")
        for side, id_key in (("question_a", "question_a_id"), ("question_b", "question_b_id")):
            side_count += 1
            question_id = str(row.get(id_key) or "").strip()
            if not question_id:
                raise ValueError(f"missing {id_key} at {pair_path}:{line_number}")
            question_degrees[question_id] += 1
            feature_row = labels.get(side)
            if not isinstance(feature_row, dict):
                missing_sides += 1
                continue
            validated: dict[str, str] = {}
            for name, values in FEATURE_VALUES.items():
                value = feature_row.get(name)
                if value not in values:
                    invalid_values.append({
                        "line_number": line_number,
                        "question_id": question_id,
                        "feature": name,
                        "value": value,
                    })
                    continue
                validated[name] = value
                pair_side_counts[name][value] += 1
            if len(validated) != len(FEATURE_VALUES):
                continue
            raw_quality = qualities.get(side, 1.0)
            try:
                quality = float(raw_quality)
            except (TypeError, ValueError):
                quality = 0.0
            if not 0 < quality <= 1:
                invalid_values.append({
                    "line_number": line_number,
                    "question_id": question_id,
                    "feature": "auxiliary_feature_quality",
                    "value": raw_quality,
                })
                continue
            existing = question_features.get(question_id)
            if existing is not None and existing != validated:
                conflict_count += 1
                if len(conflicts) < args.conflict_examples:
                    conflicts.append({
                        "question_id": question_id,
                        "first": existing,
                        "conflicting": validated,
                        "line_number": line_number,
                    })
            else:
                question_features[question_id] = validated
            existing_quality = question_quality.get(question_id)
            if existing_quality is not None and not math.isclose(existing_quality, quality):
                conflict_count += 1
                if len(conflicts) < args.conflict_examples:
                    conflicts.append({
                        "question_id": question_id,
                        "first_quality": existing_quality,
                        "conflicting_quality": quality,
                        "line_number": line_number,
                    })
            else:
                question_quality[question_id] = quality

    unique_counts = {name: Counter() for name in FEATURE_VALUES}
    quality_by_class = {name: {value: [] for value in values} for name, values in FEATURE_VALUES.items()}
    for question_id, feature_row in question_features.items():
        for name, value in feature_row.items():
            unique_counts[name][value] += 1
            quality_by_class[name][value].append(question_quality[question_id])

    features: dict[str, Any] = {}
    for name, values in FEATURE_VALUES.items():
        unique_distribution = distribution(unique_counts[name], values)
        side_distribution = distribution(pair_side_counts[name], values)
        raw_entropy = entropy(unique_distribution)
        maximum_entropy = math.log(len(values))
        class_rows = {}
        for value in values:
            support = unique_counts[name][value]
            weights = quality_by_class[name][value]
            ess = effective_sample_size(weights)
            class_rows[value] = {
                "unique_question_support": support,
                "pair_side_support": pair_side_counts[name][value],
                "unique_question_proportion": unique_distribution[value],
                "pair_side_proportion": side_distribution[value],
                "quality_weighted_effective_sample_size": ess,
                "rare_by_support": support < args.minimum_class_support,
                "rare_by_effective_sample_size": ess < args.minimum_effective_sample_size,
            }
        features[name] = {
            "expected_class_count": len(values),
            "observed_class_count": sum(unique_counts[name][value] > 0 for value in values),
            "unique_question_entropy": raw_entropy,
            "unique_question_normalized_entropy": raw_entropy / maximum_entropy if maximum_entropy else 0.0,
            "effective_class_count": math.exp(raw_entropy),
            "dominant_class_proportion": max(unique_distribution.values(), default=0.0),
            "pair_side_vs_unique_js_divergence": jensen_shannon(unique_distribution, side_distribution),
            "classes": class_rows,
        }

    degree_values = list(question_degrees.values())
    quality_values = list(question_quality.values())
    coverage = len(question_features) / max(1, len(question_degrees))
    report = {
        "schema_version": "auxiliary_feature_quality_audit_v1",
        "pairs": str(pair_path.resolve()),
        "pair_records": pair_count,
        "pair_sides": side_count,
        "unique_pair_questions": len(question_degrees),
        "unique_questions_with_complete_auxiliary_features": len(question_features),
        "unique_question_feature_coverage": coverage,
        "missing_pair_sides": missing_sides,
        "invalid_value_count": len(invalid_values),
        "invalid_value_examples": invalid_values[:args.conflict_examples],
        "feature_or_quality_conflict_count": conflict_count,
        "feature_or_quality_conflict_examples": conflicts,
        "question_degree": {
            "minimum": min(degree_values) if degree_values else 0,
            "p10": quantile(degree_values, 0.1),
            "median": quantile(degree_values, 0.5),
            "mean": sum(degree_values) / max(1, len(degree_values)),
            "p90": quantile(degree_values, 0.9),
            "maximum": max(degree_values) if degree_values else 0,
        },
        "question_quality": {
            "minimum": min(quality_values) if quality_values else None,
            "median": quantile(quality_values, 0.5),
            "mean": sum(quality_values) / max(1, len(quality_values)),
            "maximum": max(quality_values) if quality_values else None,
        },
        "thresholds": {
            "minimum_class_support": args.minimum_class_support,
            "minimum_effective_sample_size": args.minimum_effective_sample_size,
        },
        "features": features,
        "interpretation": {
            "label_correctness_measured": False,
            "note": "This audit checks contract, coverage, imbalance, and graph-degree bias. Semantic label correctness requires independent relabeling or human review.",
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_path.resolve()),
        "unique_questions": len(question_features),
        "coverage": coverage,
        "invalid_value_count": len(invalid_values),
        "conflict_count": conflict_count,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
