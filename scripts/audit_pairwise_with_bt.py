#!/usr/bin/env python3
"""Audit labeled pair data with an offline scalar Bradley--Terry model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.metrics import graph_metrics
from physics_difficulty.pairwise.offline_bt import (
    bootstrap_rank_stability,
    cross_validate_bradley_terry,
    distribution_summary,
    fit_bradley_terry,
    residual_rows,
)


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Final labeled pair JSONL")
    parser.add_argument("--report", required=True)
    parser.add_argument("--scores-output", required=True)
    parser.add_argument("--residuals-output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-runs", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=600)
    parser.add_argument("--bootstrap-max-iterations", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--severe-residual-threshold", type=float, default=0.5)
    parser.add_argument("--maximum-severe-residual-rate", type=float, default=0.05)
    parser.add_argument("--minimum-bootstrap-spearman", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = load_jsonl(input_path)
    question_ids = {
        str(row[key])
        for row in rows
        for key in ("question_a_id", "question_b_id")
    }
    print(
        json.dumps(
            {
                "message": "Offline BT audit measures whether labeled pairs support a stable global scalar ranking.",
                "records": len(rows),
                "questions": len(question_ids),
                "folds": args.folds,
                "bootstrap_runs": args.bootstrap_runs,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    full_fit = fit_bradley_terry(
        rows,
        max_iterations=args.max_iterations,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    cross_validation = cross_validate_bradley_terry(
        rows,
        folds=args.folds,
        max_iterations=args.max_iterations,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    stability = bootstrap_rank_stability(
        rows,
        full_fit["scores"],
        runs=args.bootstrap_runs,
        max_iterations=args.bootstrap_max_iterations,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    residuals = residual_rows(rows, full_fit["scores"])
    residual_values = [row["absolute_residual"] for row in residuals]
    uncertainty_values = list(full_fit["standard_errors"].values())
    severe_count = sum(
        value >= args.severe_residual_threshold for value in residual_values
    )
    severe_rate = severe_count / max(1, len(residual_values))
    heldout = cross_validation["heldout_metrics"]
    baseline = cross_validation["constant_baseline_metrics"]
    quality_checks = {
        "graph_connected": graph_metrics(rows, question_ids)["connected_components"]
        == 1,
        "heldout_log_loss_beats_constant": heldout["soft_pairwise_log_loss"]
        < baseline["soft_pairwise_log_loss"],
        "heldout_brier_beats_constant": heldout["brier_score"]
        < baseline["brier_score"],
        "severe_residual_rate": severe_rate
        <= args.maximum_severe_residual_rate,
        "bootstrap_rank_stability": args.bootstrap_runs == 0
        or (
            stability["mean_spearman"] is not None
            and stability["mean_spearman"] >= args.minimum_bootstrap_spearman
        ),
    }

    score_rows = [
        {
            "question_id": question_id,
            "bt_score": score,
            "score_standard_error": full_fit["standard_errors"][question_id],
            "confidence_lower_95": score
            - 1.96 * full_fit["standard_errors"][question_id],
            "confidence_upper_95": score
            + 1.96 * full_fit["standard_errors"][question_id],
        }
        for question_id, score in sorted(
            full_fit["scores"].items(), key=lambda item: item[1]
        )
    ]
    report = {
        "schema_version": "offline_bt_pair_audit_v1",
        "status": "PASS" if all(quality_checks.values()) else "REVIEW",
        "input": str(input_path.resolve()),
        "records": len(rows),
        "questions": len(question_ids),
        "uses_question_text": False,
        "uses_auxiliary_features": False,
        "model": {
            "type": "scalar_soft_label_bradley_terry",
            "score_constraint": full_fit["score_constraint"],
            "l2": args.l2,
            "learning_rate": args.learning_rate,
            "max_iterations": args.max_iterations,
            "seed": args.seed,
        },
        "graph": graph_metrics(rows, question_ids),
        "full_fit": {
            "metrics": full_fit["metrics"],
            "iterations": full_fit["iterations"],
            "converged": full_fit["converged"],
        },
        "cross_validation": cross_validation,
        "residuals": {
            "distribution": distribution_summary(residual_values),
            "severe_threshold": args.severe_residual_threshold,
            "severe_count": severe_count,
            "severe_rate": severe_rate,
        },
        "score_uncertainty": {
            "method": full_fit["uncertainty_method"],
            "distribution": distribution_summary(uncertainty_values),
        },
        "bootstrap_rank_stability": stability,
        "quality_gate": {
            "maximum_severe_residual_rate": args.maximum_severe_residual_rate,
            "minimum_bootstrap_spearman": args.minimum_bootstrap_spearman,
        },
        "quality_gate_checks": quality_checks,
        "interpretation": {
            "primary_metric": "cross_validation.heldout_metrics.soft_pairwise_log_loss",
            "constant_baseline": "cross_validation.constant_baseline_metrics",
            "scope": "Internal global consistency of labeled pairs, not educational correctness or unseen-question generalization.",
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(Path(args.scores_output), score_rows)
    write_jsonl(Path(args.residuals_output), residuals)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
