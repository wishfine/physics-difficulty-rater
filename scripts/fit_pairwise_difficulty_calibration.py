#!/usr/bin/env python3
"""Fit a frozen empirical-CDF and five-level map from reference raw scores."""
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

from physics_difficulty.pairwise.calibration import DEFAULT_DISTRIBUTION, build_calibration


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_manifest_path(scores: Path) -> Path:
    name = scores.name[:-6] if scores.name.endswith(".jsonl") else scores.name
    return scores.with_name(f"{name}.manifest.json")


def load_scores(path: Path) -> tuple[list[dict[str, Any]], list[float]]:
    rows: list[dict[str, Any]] = []
    values: list[float] = []
    ids: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row.get("question_id") or row.get("id") or "").strip()
            if not question_id:
                raise ValueError(f"line {line_number}: score row lacks question_id")
            if question_id in ids:
                raise ValueError(f"line {line_number}: duplicate question_id {question_id}")
            if "raw_difficulty_score" not in row:
                raise ValueError(f"line {line_number}: score row lacks raw_difficulty_score")
            ids.add(question_id)
            rows.append(row)
            values.append(float(row["raw_difficulty_score"]))
    if not rows:
        raise ValueError("score file is empty")
    return rows, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--scores-manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-version", required=True)
    parser.add_argument(
        "--distribution",
        default=",".join(str(value) for value in DEFAULT_DISTRIBUTION),
        help="Five comma-separated proportions from 送分题 through 压轴题",
    )
    parser.add_argument("--minimum-records", type=int, default=1000)
    parser.add_argument("--reference-note", default="")
    parser.add_argument("--allow-evaluation-split", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.minimum_records <= 0:
        raise ValueError("minimum-records must be positive")

    scores_path = Path(args.scores)
    manifest_path = Path(args.scores_manifest) if args.scores_manifest else default_manifest_path(scores_path)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise ValueError(f"calibration already exists: {output_path}; pass --overwrite to replace it")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_scores_sha256 = sha256_file(scores_path)
    if manifest.get("output_sha256") != actual_scores_sha256:
        raise ValueError("score file hash does not match its manifest")
    rows, scores = load_scores(scores_path)
    if len(rows) < args.minimum_records:
        raise ValueError(f"reference pool has {len(rows)} records, below minimum {args.minimum_records}")
    splits = Counter(str(row.get("split") or "unspecified") for row in rows)
    forbidden = {"validation", "test"} & set(splits)
    if forbidden and not args.allow_evaluation_split:
        raise ValueError(
            f"refusing to calibrate on evaluation splits {sorted(forbidden)}; "
            "use a train/production reference pool"
        )
    distribution = [float(value.strip()) for value in args.distribution.split(",") if value.strip()]
    print(json.dumps({
        "message": "Fitting frozen score thresholds; results show the fixed reference-population mapping, not model accuracy.",
        "reference_records": len(rows),
        "splits": dict(splits),
    }, ensure_ascii=False), flush=True)
    calibration = build_calibration(
        scores,
        calibration_version=args.calibration_version,
        checkpoint_fingerprint=str(manifest.get("checkpoint_fingerprint") or ""),
        distribution=distribution,
        reference={
            "scores_file": str(scores_path.resolve()),
            "scores_sha256": actual_scores_sha256,
            "scores_manifest": str(manifest_path.resolve()),
            "scores_manifest_sha256": sha256_file(manifest_path),
            "source_questions": manifest.get("questions"),
            "source_questions_sha256": manifest.get("questions_sha256"),
            "source_splits": dict(splits),
            "excluded_question_count": manifest.get("excluded_question_count", 0),
            "note": args.reference_note,
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(calibration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps({
        "calibration_version": calibration["calibration_version"],
        "calibration_id": calibration["calibration_id"],
        "records": calibration["reference_record_count"],
        "raw_score_thresholds": calibration["raw_score_thresholds"],
        "target_distribution": calibration["target_distribution"],
        "output": str(output_path.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
