#!/usr/bin/env python3
"""Select diverse, label-blind-to-the-reviewer candidates for step-count recheck.

Existing auxiliary values are used only to retrieve likely false negatives and
to document the sampling design.  The reviewer-facing output contains the
original label-free V3 question row only; the original step label and reasons
go to a separate audit file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.data.text_only import forbidden_source_label_paths
from physics_difficulty.schema import FEATURE_VALUES


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_rank(seed: int, question_id: str, stratum: str) -> str:
    return hashlib.sha256(f"{seed}\0{stratum}\0{question_id}".encode("utf-8")).hexdigest()


def complete_features(row: dict[str, Any]) -> dict[str, str] | None:
    values = row.get("teacher_features")
    if not isinstance(values, dict):
        return None
    if any(values.get(name) not in options for name, options in FEATURE_VALUES.items()):
        return None
    return {name: str(values[name]) for name in FEATURE_VALUES}


def candidate_reasons(question: dict[str, Any], features: dict[str, str]) -> list[str]:
    diagnostics = question.get("diagnostics") or {}
    reasons: list[str] = []
    if features["step_count"] == "9步以上":
        reasons.append("existing_9plus_control")
    if features["step_count"] == "6-8步":
        reasons.append("current_6_8")
    if features["subquestion_dependency"] == "多问且层层递进":
        reasons.append("progressive_subquestions")
    if (
        features["reasoning_chain"] == "逆向推理或临界分析"
        or features["calculation_complexity"] == "复杂方程或范围计算"
        or features["knowledge_count"] == "4个及以上"
        or features["state_count"] == "连续变化或临界状态"
        or features["constraint_count"] == "多约束"
        or features["variable_relation"] == "多变量耦合关系"
    ):
        reasons.append("high_reasoning_complexity")
    if (
        diagnostics.get("input_length_bucket") == "long"
        or int(diagnostics.get("subquestion_count") or 0) >= 4
    ):
        reasons.append("long_or_many_subquestions")
    return reasons


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    known, _ = bootstrap.parse_known_args()
    defaults = json.loads(Path(known.config).read_text(encoding="utf-8")) if known.config else {}
    parser = argparse.ArgumentParser(parents=[bootstrap], description="Build step_count targeted recheck candidates")
    parser.add_argument("--questions", required=True, help="Auxiliary-eligible, label-free train question JSONL")
    parser.add_argument("--features", required=True, help="V2 curated JSONL used only for candidate retrieval")
    parser.add_argument("--output", required=True, help="Reviewer-facing label-free question JSONL")
    parser.add_argument("--audit-output", required=True, help="Selection provenance; never pass this to the reviewer")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target-questions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratum-quotas", type=json.loads, default={})
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    if args.target_questions < 1:
        raise ValueError("target-questions must be positive")
    if not isinstance(args.stratum_quotas, dict):
        raise ValueError("stratum-quotas must be a JSON object")

    questions = load_jsonl(Path(args.questions))
    by_id: dict[str, dict[str, str]] = {}
    for row in load_jsonl(Path(args.features)):
        question_id = str(row.get("id") or "").strip()
        if not question_id:
            raise ValueError("feature row lacks id")
        if question_id in by_id:
            raise ValueError(f"duplicate feature id {question_id}")
        features = complete_features(row)
        if features is not None:
            by_id[question_id] = features

    records: list[tuple[dict[str, Any], dict[str, str], list[str]]] = []
    seen: set[str] = set()
    for row in questions:
        question_id = str(row.get("id") or "").strip()
        if not question_id or question_id in seen:
            raise ValueError("questions must have unique non-empty id values")
        seen.add(question_id)
        forbidden = forbidden_source_label_paths(row)
        if forbidden:
            raise ValueError(f"question {question_id} is not label-free: {forbidden}")
        features = by_id.get(question_id)
        if features is None:
            raise ValueError(f"question {question_id} has no complete auxiliary feature row")
        records.append((row, features, candidate_reasons(row, features)))
    if args.target_questions > len(records):
        raise ValueError("target-questions exceeds input questions")

    quotas = {str(name): int(value) for name, value in args.stratum_quotas.items()}
    supported_strata = {
        "existing_9plus_control", "current_6_8", "progressive_subquestions",
        "high_reasoning_complexity", "long_or_many_subquestions", "random_control",
    }
    unknown = sorted(set(quotas) - supported_strata)
    if unknown or any(value < 0 for value in quotas.values()):
        raise ValueError(f"invalid stratum quotas: {unknown}")
    if sum(quotas.values()) > args.target_questions:
        raise ValueError(
            f"sum of stratum quotas ({sum(quotas.values())}) exceeds "
            f"target-questions ({args.target_questions}); reduce quotas or increase the candidate budget"
        )

    selected: dict[str, tuple[dict[str, Any], dict[str, str], list[str]]] = {}
    selected_by_reason: dict[str, set[str]] = defaultdict(set)

    def select_from(candidates: list[tuple[dict[str, Any], dict[str, str], list[str]]], reason: str, quota: int) -> int:
        newly_selected = 0
        for row, features, reasons in sorted(candidates, key=lambda value: stable_rank(args.seed, str(value[0]["id"]), reason)):
            if len(selected) >= args.target_questions or len(selected_by_reason[reason]) >= quota:
                break
            question_id = str(row["id"])
            if question_id in selected:
                continue
            selected[question_id] = (row, features, reasons)
            selected_by_reason[reason].add(question_id)
            newly_selected += 1
        return newly_selected

    stratum_pools = {
        reason: [record for record in records if reason in record[2]]
        for reason in (
            "existing_9plus_control", "current_6_8", "progressive_subquestions",
            "high_reasoning_complexity", "long_or_many_subquestions",
        )
    }
    # A control must not be retrieved by any high-risk rule, otherwise it
    # cannot measure the false-negative rate of the targeted retrieval.
    stratum_pools["random_control"] = [record for record in records if not record[2]]
    quota_report = {}
    for reason in (
        "existing_9plus_control", "current_6_8", "progressive_subquestions",
        "high_reasoning_complexity", "long_or_many_subquestions",
    ):
        quota = quotas.get(reason, 0)
        newly_selected = select_from(stratum_pools[reason], reason, quota)
        quota_report[reason] = {
            "requested": quota,
            "available": len(stratum_pools[reason]),
            "newly_selected": newly_selected,
            "shortfall": max(0, quota - newly_selected),
        }
    random_quota = quotas.get("random_control", 0)
    random_selected = select_from(stratum_pools["random_control"], "random_control", random_quota)
    quota_report["random_control"] = {
        "requested": random_quota,
        "available": len(stratum_pools["random_control"]),
        "newly_selected": random_selected,
        "shortfall": max(0, random_quota - random_selected),
    }
    # Quotas can overlap.  Fill every remaining slot from a label-independent
    # stable hash so the final sample stays reproducible and representative.
    for record in sorted(records, key=lambda value: stable_rank(args.seed, str(value[0]["id"]), "representative_fill")):
        if len(selected) >= args.target_questions:
            break
        selected.setdefault(str(record[0]["id"]), record)

    ordered = sorted(selected.values(), key=lambda value: stable_rank(args.seed, str(value[0]["id"]), "output"))
    if len(ordered) != args.target_questions:
        raise RuntimeError("failed to produce requested number of unique candidates")
    output_path, audit_path, manifest_path = Path(args.output), Path(args.audit_output), Path(args.manifest)
    for path in (output_path, audit_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row, _, _ in ordered), encoding="utf-8")
    audit_rows = []
    for row, features, reasons in ordered:
        question_id = str(row["id"])
        audit_rows.append({
            "question_id": question_id,
            "selection_reasons": reasons + (["random_control"] if question_id in selected_by_reason["random_control"] else []),
            "original_step_count": features["step_count"],
            "retrieval_features": {
                name: features[name]
                for name in ("calculation_complexity", "reasoning_chain", "knowledge_count", "subquestion_dependency", "state_count", "constraint_count", "variable_relation")
            },
        })
    audit_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_rows), encoding="utf-8")
    original_counts = Counter(features["step_count"] for _, features, _ in ordered)
    selection_reason_counts = Counter(reason for row in audit_rows for reason in row["selection_reasons"])
    manifest = {
        "schema_version": "step_count_enrichment_candidates_v1",
        "questions": str(Path(args.questions).resolve()),
        "features": str(Path(args.features).resolve()),
        "output": str(output_path.resolve()),
        "audit_output": str(audit_path.resolve()),
        "seed": args.seed,
        "target_questions": args.target_questions,
        "stratum_quotas": quotas,
        "stratum_quota_report": quota_report,
        "selected_questions": len(ordered),
        "selection_reason_counts": dict(selection_reason_counts),
        "original_step_count_distribution": dict(original_counts),
        "reviewer_input_is_label_free": True,
        "raw_difficulty_used": False,
        "teacher_difficulty_used": False,
        "warnings": [
            "The audit file contains old auxiliary labels and must never be passed to the recheck model.",
            "Candidate retrieval is not a replacement for independent step_count adjudication.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_questions": len(ordered),
        "original_step_count_distribution": dict(original_counts),
        "selection_reason_counts": dict(selection_reason_counts),
        "output": str(output_path.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
