#!/usr/bin/env python3
"""Clean aggregated answer behavior and derive uncertainty-aware difficulty scores."""
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

from physics_difficulty.pairwise.behavior_accuracy import (
    distribution_summary,
    row_fingerprint,
    score_behavior_row,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw behavior JSONL")
    parser.add_argument("--scores-output", required=True)
    parser.add_argument("--quarantine-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--prior-alpha", type=float, default=1.0)
    parser.add_argument("--prior-beta", type=float, default=1.0)
    parser.add_argument("--minimum-answered-count", type=int, default=20)
    parser.add_argument(
        "--reject-continuous-rate-pseudocount",
        action="store_true",
        help="Quarantine rates that cannot be exactly reconstructed as an integer count.",
    )
    parser.add_argument(
        "--maximum-invalid-rate",
        type=float,
        default=0.01,
        help="Report WARN when invalid records exceed this fraction of source rows.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    scores_by_id: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, str] = {}
    blocked_ids: set[str] = set()
    quarantine: list[dict[str, Any]] = []
    stats = Counter()
    recovery_statuses = Counter()
    structure_types = Counter()

    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            stats["source_records"] += 1
            raw: Any = None
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("JSONL row must be an object")
                if "difficulty" in raw:
                    stats["rows_containing_forbidden_difficulty"] += 1
                score = score_behavior_row(
                    raw,
                    prior_alpha=args.prior_alpha,
                    prior_beta=args.prior_beta,
                )
                if score["answered_count"] <= args.minimum_answered_count:
                    raise ValueError(
                        f"answered_count must be > {args.minimum_answered_count}"
                    )
                if (
                    args.reject_continuous_rate_pseudocount
                    and score["behavior_evidence_type"]
                    == "continuous_rate_pseudocount"
                ):
                    raise ValueError(
                        "percent_correct cannot be reconciled with an integer count"
                    )
                question_id = score["question_id"]
                fingerprint = row_fingerprint(raw)
                if question_id in blocked_ids:
                    stats["additional_rows_for_conflicting_ids"] += 1
                    quarantine.append(
                        {
                            "line_number": line_number,
                            "question_id": question_id,
                            "reason": "question_id_previously_blocked_by_conflicting_duplicate",
                        }
                    )
                    continue
                if question_id in scores_by_id:
                    if fingerprints[question_id] == fingerprint:
                        stats["exact_duplicate_rows"] += 1
                        continue
                    stats["conflicting_duplicate_ids"] += 1
                    previous = scores_by_id.pop(question_id)
                    fingerprints.pop(question_id)
                    blocked_ids.add(question_id)
                    quarantine.extend(
                        [
                            {
                                "question_id": question_id,
                                "reason": "conflicting_duplicate_question_id_previous_row",
                                "previous_answered_count": previous["answered_count"],
                                "previous_reported_percent_correct": previous[
                                    "reported_percent_correct"
                                ],
                            },
                            {
                                "line_number": line_number,
                                "question_id": question_id,
                                "reason": "conflicting_duplicate_question_id_current_row",
                            },
                        ]
                    )
                    continue
                scores_by_id[question_id] = score
                fingerprints[question_id] = fingerprint
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                stats["invalid_rows"] += 1
                quarantine.append(
                    {
                        "line_number": line_number,
                        "question_id": (
                            str(raw.get("question_id") or "")
                            if isinstance(raw, dict)
                            else ""
                        ),
                        "reason": "invalid_behavior_record",
                        "error": str(exc),
                    }
                )

    scores = sorted(scores_by_id.values(), key=lambda row: row["question_id"])
    for row in scores:
        recovery_statuses[row["recovery_status"]] += 1
        structure_types[row["structure_type"]] += 1
        if row["parent_id"] != row["question_id"]:
            stats["parent_question_id_mismatch"] += 1

    scores_path = Path(args.scores_output)
    quarantine_path = Path(args.quarantine_output)
    report_path = Path(args.report)
    write_jsonl(scores_path, scores)
    write_jsonl(quarantine_path, quarantine)
    report = {
        "schema_version": "behavior_accuracy_audit_v2",
        "status": (
            "PASS"
            if scores
            and not stats["conflicting_duplicate_ids"]
            and stats["invalid_rows"] / max(1, stats["source_records"])
            <= args.maximum_invalid_rate
            else "WARN"
        ),
        "input": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "data_contract": {
            "used_fields": [
                "question_id",
                "parent_id",
                "stem",
                "structure_type",
                "answered_count",
                "percent_correct",
                "sub_questions",
            ],
            "forbidden_source_fields": ["difficulty"],
            "source_difficulty_used": False,
            "difficulty_direction": "larger behavior_difficulty_score means harder",
        },
        "configuration": {
            "minimum_answered_count_exclusive": args.minimum_answered_count,
            "beta_prior_alpha": args.prior_alpha,
            "beta_prior_beta": args.prior_beta,
            "continuous_rate_pseudocount_accepted": (
                not args.reject_continuous_rate_pseudocount
            ),
            "maximum_invalid_rate": args.maximum_invalid_rate,
        },
        "records": {
            **dict(stats),
            "accepted_unique_questions": len(scores),
            "quarantine_rows": len(quarantine),
        },
        "recovery_status_counts": dict(recovery_statuses),
        "structure_type_counts": dict(structure_types),
        "distributions": {
            "answered_count": distribution_summary(row["answered_count"] for row in scores),
            "reported_percent_correct": distribution_summary(
                row["reported_percent_correct"] for row in scores
            ),
            "behavior_difficulty_score": distribution_summary(
                row["behavior_difficulty_score"] for row in scores
            ),
            "difficulty_interval_width_95": distribution_summary(
                row["behavior_difficulty_upper_95"] - row["behavior_difficulty_lower_95"]
                for row in scores
            ),
        },
        "outputs": {
            "scores": str(scores_path.resolve()),
            "scores_sha256": sha256_file(scores_path),
            "quarantine": str(quarantine_path.resolve()),
            "quarantine_sha256": sha256_file(quarantine_path),
        },
        "interpretation": (
            "This artifact is behavioral evidence derived from response counts, not a gold difficulty label. "
            "Rates without an exact integer-count reconstruction are retained as lower-quality continuous "
            "pseudo-count evidence. It must be used for external consistency audit before any training fusion."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
