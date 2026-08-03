#!/usr/bin/env python3
"""Evaluate exported auxiliary predictions against compatible teacher features."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.evaluation.metrics import classification_metrics
from physics_difficulty.schema import merge_information_processing


def normalize_step(value: Any, allowed: list[str]) -> str:
    value = str(value or "").strip()
    if value in allowed:
        return value
    if "9步以上" in allowed and value in {"9-12步", "12步以上"}:
        return "9步以上"
    if "6步以上" in allowed and value in {"6-8步", "9-12步", "12步以上", "9步以上"}:
        return "6步以上"
    return value


def gold_features(row: dict[str, Any], vocabularies: dict[str, list[str]]) -> dict[str, str]:
    current = row.get("teacher_features") or {}
    legacy = row.get("teacher_features_legacy18") or {}
    result = {}
    for name, allowed in vocabularies.items():
        if name == "information_processing":
            value = current.get(name)
            if value not in allowed:
                value = merge_information_processing(
                    legacy.get("graph_table_requirement", current.get("graph_table_requirement")),
                    legacy.get("experiment_requirement", current.get("experiment_requirement")),
                )
        else:
            value = current.get(name, legacy.get(name))
        if name == "step_count":
            value = normalize_step(legacy.get(name, value), allowed)
        result[name] = str(value or "").strip()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--scores-manifest")
    parser.add_argument("--teacher-features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label-provenance", required=True)
    args = parser.parse_args()
    scores_path = Path(args.scores)
    manifest_path = Path(args.scores_manifest) if args.scores_manifest else scores_path.with_name(
        f"{scores_path.name[:-6] if scores_path.name.endswith('.jsonl') else scores_path.name}.manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vocabularies = manifest.get("auxiliary_feature_values")
    if not manifest.get("auxiliary_exported") or not isinstance(vocabularies, dict):
        raise ValueError("score run did not export auxiliary predictions")
    teacher = {}
    with Path(args.teacher_features).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row.get("id") or row.get("question_id") or "").strip()
            if question_id:
                teacher[question_id] = gold_features(row, vocabularies)
    predictions = {name: [] for name in vocabularies}
    labels = {name: [] for name in vocabularies}
    missing = Counter()
    scored_records = 0
    with scores_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            scored_records += 1
            question_id = str(row["question_id"])
            if question_id not in teacher:
                missing["teacher_record"] += 1
                continue
            for name, allowed in vocabularies.items():
                gold = teacher[question_id].get(name)
                predicted = (row.get("auxiliary_predictions") or {}).get(name, {}).get("label")
                if gold not in allowed or predicted not in allowed:
                    missing[f"invalid_{name}"] += 1
                    continue
                labels[name].append(allowed.index(gold))
                predictions[name].append(allowed.index(predicted))
    metrics = {
        name: {
            "records": len(labels[name]),
            **classification_metrics(predictions[name], labels[name], len(values)),
        }
        for name, values in vocabularies.items()
        if labels[name]
    }
    report = {
        "schema_version": "single_question_auxiliary_evaluation_v1",
        "scored_records": scored_records,
        "matched_teacher_records": scored_records - missing["teacher_record"],
        "label_provenance": args.label_provenance,
        "metric_semantics": (
            "same-pipeline held-out accuracy" if args.label_provenance.startswith("v3_")
            else "cross-pipeline agreement; not human-ground-truth accuracy"
        ),
        "missing": dict(missing),
        "features": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
