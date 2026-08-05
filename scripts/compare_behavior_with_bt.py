#!/usr/bin/env python3
"""Compare behavioral difficulty with offline BT scores and teacher pair labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.behavior_accuracy import (
    behavior_pair_probability,
    distribution_summary,
    entropy_confidence,
    harmonic_mean,
    pearson_correlation,
    spearman_correlation,
)
from physics_difficulty.pairwise.metrics import soft_pairwise_metrics


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(value)
    return rows


def index_unique(rows: list[dict[str, Any]], field: str, source: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for line_number, row in enumerate(rows, 1):
        key = str(row.get(field) or "").strip()
        if not key:
            raise ValueError(f"{source}:{line_number}: missing {field}")
        if key in output:
            raise ValueError(f"{source}:{line_number}: duplicate {field}={key}")
        output[key] = row
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kendall_tau(left: list[float], right: list[float]) -> dict[str, Any]:
    try:
        from scipy.stats import kendalltau

        result = kendalltau(left, right, variant="b")
        return {"value": float(result.statistic), "p_value": float(result.pvalue), "status": "OK"}
    except (ImportError, ValueError) as exc:
        return {"value": None, "p_value": None, "status": f"UNAVAILABLE: {exc}"}


def pair_source(row: dict[str, Any]) -> str:
    return str(row.get("pair_source") or (row.get("metadata") or {}).get("pair_source") or "unknown")


def route_reason(row: dict[str, Any]) -> str:
    route = row.get("cascade_route") or {}
    return str(route.get("reason") or route.get("route_reason") or "unknown")


def summarize_pair_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"records": 0}
    predictions = [float(row["behavior_probability_a_harder"]) for row in rows]
    targets = [float(row["teacher_soft_target"]) for row in rows]
    weights = [float(row["teacher_sample_weight"]) for row in rows]
    evidence_weights = [
        float(row["teacher_sample_weight"]) * float(row["behavior_evidence_weight"])
        for row in rows
    ]
    hard = [
        row for row in rows
        if abs(float(row["teacher_soft_target"]) - 0.5) > 1e-12
        and abs(float(row["behavior_probability_a_harder"]) - 0.5) > 1e-12
    ]
    agreements = [
        (float(row["teacher_soft_target"]) > 0.5)
        == (float(row["behavior_probability_a_harder"]) > 0.5)
        for row in hard
    ]
    return {
        "records": len(rows),
        "teacher_weighted_metrics": soft_pairwise_metrics(predictions, targets, weights),
        "teacher_and_behavior_evidence_weighted_metrics": soft_pairwise_metrics(
            predictions, targets, evidence_weights
        ),
        "hard_direction_comparable_pairs": len(hard),
        "hard_direction_agreement": sum(agreements) / len(agreements) if agreements else None,
        "mean_absolute_soft_target_difference": sum(
            abs(prediction - target) for prediction, target in zip(predictions, targets)
        ) / len(rows),
        "behavior_probability_distribution": distribution_summary(predictions),
        "behavior_confidence_distribution": distribution_summary(
            row["behavior_entropy_confidence"] for row in rows
        ),
        "behavior_effective_answered_count_distribution": distribution_summary(
            row["behavior_effective_answered_count"] for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior-scores", required=True)
    parser.add_argument("--bt-scores", required=True)
    parser.add_argument("--teacher-pairs", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--question-evidence-output", required=True)
    parser.add_argument("--pair-evidence-output", required=True)
    parser.add_argument("--conflicts-output", required=True)
    parser.add_argument("--high-confidence-low", type=float, default=0.20)
    parser.add_argument("--high-confidence-high", type=float, default=0.80)
    parser.add_argument(
        "--behavior-full-weight-answer-count",
        type=float,
        default=200.0,
        help="Harmonic pair response count at which behavior evidence gets weight 1",
    )
    args = parser.parse_args()
    if not 0 <= args.high_confidence_low < 0.5 < args.high_confidence_high <= 1:
        raise ValueError("invalid high-confidence thresholds")

    behavior_path = Path(args.behavior_scores)
    bt_path = Path(args.bt_scores)
    pairs_path = Path(args.teacher_pairs)
    behavior = index_unique(load_jsonl(behavior_path), "question_id", behavior_path)
    bt = index_unique(load_jsonl(bt_path), "question_id", bt_path)
    pairs = load_jsonl(pairs_path)

    overlap_ids = sorted(set(behavior) & set(bt))
    question_evidence = []
    for question_id in overlap_ids:
        behavior_row = behavior[question_id]
        bt_row = bt[question_id]
        question_evidence.append(
            {
                "question_id": question_id,
                "bt_score": float(bt_row["bt_score"]),
                "bt_rank": bt_row.get("rank"),
                "bt_degree": bt_row.get("degree"),
                "bt_score_standard_error": bt_row.get("score_standard_error"),
                "behavior_difficulty_score": float(behavior_row["behavior_difficulty_score"]),
                "behavior_difficulty_lower_95": behavior_row["behavior_difficulty_lower_95"],
                "behavior_difficulty_upper_95": behavior_row["behavior_difficulty_upper_95"],
                "answered_count": behavior_row["answered_count"],
                "reported_percent_correct": behavior_row["reported_percent_correct"],
                "structure_type": behavior_row.get("structure_type", "unknown"),
            }
        )
    bt_values = [row["bt_score"] for row in question_evidence]
    behavior_values = [row["behavior_difficulty_score"] for row in question_evidence]

    pair_evidence = []
    missing_endpoint_counts = Counter()
    for row in pairs:
        question_a_id = str(row["question_a_id"])
        question_b_id = str(row["question_b_id"])
        if question_a_id not in behavior:
            missing_endpoint_counts["question_a_missing_behavior"] += 1
        if question_b_id not in behavior:
            missing_endpoint_counts["question_b_missing_behavior"] += 1
        if question_a_id not in behavior or question_b_id not in behavior:
            continue
        a = behavior[question_a_id]
        b = behavior[question_b_id]
        probability = behavior_pair_probability(a, b)
        effective_count = harmonic_mean(float(a["answered_count"]), float(b["answered_count"]))
        evidence_weight = min(1.0, effective_count / args.behavior_full_weight_answer_count)
        target = float(row["soft_target"])
        item = {
            "schema_version": "behavior_teacher_pair_evidence_v1",
            "pair_id": str(row["pair_id"]),
            "question_a_id": question_a_id,
            "question_b_id": question_b_id,
            "teacher_soft_target": target,
            "teacher_sample_weight": float(row.get("sample_weight", 1.0)),
            "teacher_label_source": str(row.get("label_source") or "unknown"),
            "pair_source": pair_source(row),
            "cascade_route_reason": route_reason(row),
            "behavior_probability_a_harder": probability,
            "behavior_entropy_confidence": entropy_confidence(probability),
            "behavior_effective_answered_count": effective_count,
            "behavior_evidence_weight": evidence_weight,
            "behavior_soft_target_absolute_difference": abs(target - probability),
            "behavior_direction_agrees": (
                None
                if abs(target - 0.5) <= 1e-12 or abs(probability - 0.5) <= 1e-12
                else (target > 0.5) == (probability > 0.5)
            ),
            "question_a_behavior_difficulty": a["behavior_difficulty_score"],
            "question_b_behavior_difficulty": b["behavior_difficulty_score"],
            "question_a_answered_count": a["answered_count"],
            "question_b_answered_count": b["answered_count"],
            "question_a_percent_correct": a["reported_percent_correct"],
            "question_b_percent_correct": b["reported_percent_correct"],
        }
        pair_evidence.append(item)

    low = args.high_confidence_low
    high = args.high_confidence_high
    high_confidence_comparable = [
        row for row in pair_evidence
        if (row["teacher_soft_target"] <= low or row["teacher_soft_target"] >= high)
        and (
            row["behavior_probability_a_harder"] <= low
            or row["behavior_probability_a_harder"] >= high
        )
    ]
    conflicts = [
        row for row in high_confidence_comparable
        if (
            row["teacher_soft_target"] >= high
            and row["behavior_probability_a_harder"] <= low
        )
        or (
            row["teacher_soft_target"] <= low
            and row["behavior_probability_a_harder"] >= high
        )
    ]
    conflicts.sort(
        key=lambda row: (
            -row["behavior_soft_target_absolute_difference"],
            row["pair_id"],
        )
    )

    slice_fields = ("pair_source", "teacher_label_source", "cascade_route_reason")
    slices = {}
    for field in slice_fields:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pair_evidence:
            grouped[str(row[field])].append(row)
        slices[field] = {
            key: summarize_pair_evidence(value)
            for key, value in sorted(grouped.items())
        }

    report = {
        "schema_version": "behavior_bt_external_consistency_audit_v1",
        "status": "PASS" if overlap_ids and pair_evidence else "INSUFFICIENT_OVERLAP",
        "inputs": {
            "behavior_scores": str(behavior_path.resolve()),
            "behavior_scores_sha256": sha256_file(behavior_path),
            "bt_scores": str(bt_path.resolve()),
            "bt_scores_sha256": sha256_file(bt_path),
            "teacher_pairs": str(pairs_path.resolve()),
            "teacher_pairs_sha256": sha256_file(pairs_path),
        },
        "data_contract": {
            "source_difficulty_used": False,
            "behavior_score_direction": "higher means harder",
            "bt_score_direction": "higher means harder",
            "behavior_pair_probability": "normal approximation to independent Beta posteriors",
        },
        "coverage": {
            "behavior_questions": len(behavior),
            "bt_questions": len(bt),
            "question_overlap": len(overlap_ids),
            "bt_question_behavior_coverage": len(overlap_ids) / max(1, len(bt)),
            "teacher_pairs": len(pairs),
            "teacher_pairs_with_both_behavior_endpoints": len(pair_evidence),
            "teacher_pair_behavior_coverage": len(pair_evidence) / max(1, len(pairs)),
            "missing_endpoint_counts": dict(missing_endpoint_counts),
        },
        "question_level_consistency": {
            "records": len(question_evidence),
            "pearson": pearson_correlation(bt_values, behavior_values),
            "spearman": spearman_correlation(bt_values, behavior_values),
            "kendall_tau_b": kendall_tau(bt_values, behavior_values),
            "bt_score_distribution": distribution_summary(bt_values),
            "behavior_difficulty_distribution": distribution_summary(behavior_values),
        },
        "pair_level_consistency": summarize_pair_evidence(pair_evidence),
        "high_confidence_audit": {
            "teacher_low": low,
            "teacher_high": high,
            "behavior_low": low,
            "behavior_high": high,
            "comparable_pairs": len(high_confidence_comparable),
            "agreement_pairs": len(high_confidence_comparable) - len(conflicts),
            "agreement_rate": (
                (len(high_confidence_comparable) - len(conflicts))
                / len(high_confidence_comparable)
                if high_confidence_comparable
                else None
            ),
            "severe_conflicts": len(conflicts),
            "severe_conflict_rate": (
                len(conflicts) / len(high_confidence_comparable)
                if high_confidence_comparable
                else None
            ),
        },
        "slices": slices,
        "outputs": {
            "question_evidence": str(Path(args.question_evidence_output).resolve()),
            "pair_evidence": str(Path(args.pair_evidence_output).resolve()),
            "severe_conflicts": str(Path(args.conflicts_output).resolve()),
        },
        "interpretation": (
            "This is an external consistency audit. Disagreement is a review signal, not proof that either "
            "the teacher label or behavior statistic is wrong, because response accuracy also reflects population, "
            "exposure, guessing, and item-format effects."
        ),
    }
    write_jsonl(Path(args.question_evidence_output), question_evidence)
    write_jsonl(Path(args.pair_evidence_output), pair_evidence)
    write_jsonl(Path(args.conflicts_output), conflicts)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
