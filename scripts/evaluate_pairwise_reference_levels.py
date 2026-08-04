#!/usr/bin/env python3
"""Fit reference-set score thresholds and report five-level agreement."""
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

from physics_difficulty.evaluation.metrics import classification_metrics
from physics_difficulty.pairwise.calibration import DEFAULT_DISTRIBUTION, apply_calibration, build_calibration, validate_distribution
from physics_difficulty.schema import DIFFICULTY_LEVELS, DIFFICULTY_TO_ID


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_manifest_path(scores: Path) -> Path:
    stem = scores.name[:-6] if scores.name.endswith(".jsonl") else scores.name
    return scores.with_name(f"{stem}.manifest.json")


def load_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row.get("question_id") or row.get("id") or "").strip()
            if not question_id or question_id in scores:
                raise ValueError(f"{path}: line {line_number} has a missing or duplicate question ID")
            try:
                scores[question_id] = float(row["raw_difficulty_score"])
            except KeyError as error:
                raise ValueError(f"{path}: line {line_number} lacks raw_difficulty_score") from error
    if not scores:
        raise ValueError("score file is empty")
    return scores


def difficulty_id(row: dict[str, Any], field: str) -> int:
    value = row.get(field)
    if field == "teacher_difficulty_level" or isinstance(value, str) and value in DIFFICULTY_TO_ID:
        if str(value) not in DIFFICULTY_TO_ID:
            raise ValueError(f"invalid difficulty level {value!r}")
        return DIFFICULTY_TO_ID[str(value)]
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid difficulty ID {value!r}") from error
    if result not in range(len(DIFFICULTY_LEVELS)):
        raise ValueError(f"difficulty ID must be in [0, {len(DIFFICULTY_LEVELS) - 1}], got {result}")
    return result


def load_labels(path: Path, field: str) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row.get("id") or row.get("question_id") or "").strip()
            if not question_id:
                continue
            if question_id in labels:
                raise ValueError(f"{path}: line {line_number} duplicates question {question_id}")
            if field not in row:
                continue
            labels[question_id] = difficulty_id(row, field)
    if not labels:
        raise ValueError(f"no usable {field} values found in {path}")
    return labels


def parse_distribution(value: str) -> tuple[float, ...]:
    return validate_distribution([float(item.strip()) for item in value.split(",") if item.strip()])


def target_distribution(labels: list[int]) -> tuple[float, ...]:
    counts = Counter(labels)
    total = len(labels)
    distribution = tuple(counts[index] / total for index in range(len(DIFFICULTY_LEVELS)))
    if any(value == 0 for value in distribution):
        raise ValueError("natural reference distribution has an empty difficulty level")
    return validate_distribution(distribution)


def evaluate_distribution(
    name: str,
    distribution: tuple[float, ...],
    scores: list[float],
    labels: list[int],
    checkpoint_fingerprint: str,
    reference: dict[str, Any],
    calibration_path: Path,
    version_prefix: str,
) -> dict[str, Any]:
    calibration = build_calibration(
        scores,
        calibration_version=f"{version_prefix}_{name}",
        checkpoint_fingerprint=checkpoint_fingerprint,
        distribution=distribution,
        reference=reference,
    )
    calibration_path.write_text(json.dumps(calibration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    predictions = [
        int(apply_calibration(score, calibration, calibration_already_validated=True)["difficulty_level_id"])
        for score in scores
    ]
    return {
        "target_distribution": list(distribution),
        "target_counts": [value * len(labels) for value in distribution],
        "raw_score_thresholds": calibration["raw_score_thresholds"],
        "predicted_level_counts": [predictions.count(index) for index in range(len(DIFFICULTY_LEVELS))],
        "metrics": classification_metrics(predictions, labels, len(DIFFICULTY_LEVELS)),
        "calibration": str(calibration_path.resolve()),
        "calibration_id": calibration["calibration_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--scores-manifest")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--label-field", default="teacher_difficulty_id")
    parser.add_argument("--fixed-distribution", default=",".join(str(value) for value in DEFAULT_DISTRIBUTION))
    parser.add_argument("--calibration-output-dir", required=True)
    parser.add_argument("--calibration-version-prefix", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    scores_path = Path(args.scores)
    manifest_path = Path(args.scores_manifest) if args.scores_manifest else default_manifest_path(scores_path)
    labels_path = Path(args.labels)
    output_path = Path(args.output)
    calibration_dir = Path(args.calibration_output_dir)
    score_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if score_manifest.get("output_sha256") != sha256_file(scores_path):
        raise ValueError("score file hash does not match its manifest")
    fingerprint = str(score_manifest.get("checkpoint_fingerprint") or "")
    if not fingerprint:
        raise ValueError("score manifest lacks checkpoint_fingerprint")

    score_by_id = load_scores(scores_path)
    label_by_id = load_labels(labels_path, args.label_field)
    missing = sorted(set(score_by_id) - set(label_by_id))
    if missing:
        raise ValueError(f"missing standard difficulty labels for {len(missing)} scored questions; examples: {missing[:10]}")
    ids = sorted(score_by_id)
    scores = [score_by_id[question_id] for question_id in ids]
    labels = [label_by_id[question_id] for question_id in ids]
    natural = target_distribution(labels)
    fixed = parse_distribution(args.fixed_distribution)
    calibration_dir.mkdir(parents=True, exist_ok=True)
    reference = {
        "scores_file": str(scores_path.resolve()),
        "scores_sha256": sha256_file(scores_path),
        "scores_manifest": str(manifest_path.resolve()),
        "scores_manifest_sha256": sha256_file(manifest_path),
        "labels_file": str(labels_path.resolve()),
        "labels_sha256": sha256_file(labels_path),
        "label_field": args.label_field,
        "evaluation_scope": "same_reference_set; not an independent generalization estimate",
    }
    report = {
        "schema_version": "pairwise_reference_level_evaluation_v1",
        "records": len(ids),
        "levels": list(DIFFICULTY_LEVELS),
        "standard_level_counts": [labels.count(index) for index in range(len(DIFFICULTY_LEVELS))],
        "checkpoint_fingerprint": fingerprint,
        "reference": reference,
        "interpretation": (
            "Both threshold sets and agreement metrics use this same reference set. "
            "They measure reference-set calibration agreement, not independent accuracy."
        ),
        "natural_distribution": evaluate_distribution(
            "natural", natural, scores, labels, fingerprint, reference,
            calibration_dir / "natural_distribution.calibration.json", args.calibration_version_prefix,
        ),
        "fixed_distribution": evaluate_distribution(
            "fixed", fixed, scores, labels, fingerprint, reference,
            calibration_dir / "fixed_distribution.calibration.json", args.calibration_version_prefix,
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "records": report["records"],
        "natural_accuracy": report["natural_distribution"]["metrics"]["accuracy"],
        "fixed_accuracy": report["fixed_distribution"]["metrics"]["accuracy"],
        "output": str(output_path.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
