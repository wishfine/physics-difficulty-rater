#!/usr/bin/env python3
"""Apply one frozen pairwise-score calibration to independently scored questions."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.calibration import apply_calibration, validate_calibration


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_manifest_path(scores: Path) -> Path:
    name = scores.name[:-6] if scores.name.endswith(".jsonl") else scores.name
    return scores.with_name(f"{name}.manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--scores-manifest")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    scores_path = Path(args.scores)
    scores_manifest_path = (
        Path(args.scores_manifest) if args.scores_manifest else default_manifest_path(scores_path)
    )
    calibration_path = Path(args.calibration)
    score_manifest = json.loads(scores_manifest_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    validate_calibration(calibration)
    if score_manifest.get("output_sha256") != sha256_file(scores_path):
        raise ValueError("score file hash does not match its manifest")
    if score_manifest.get("checkpoint_fingerprint") != calibration.get("checkpoint_fingerprint"):
        raise ValueError("scores and calibration were produced by different checkpoints")

    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    records = 0
    seen_ids: set[str] = set()
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    print(json.dumps({
        "message": "Applying frozen thresholds; results show fixed-reference difficulty levels without batch re-binning.",
        "calibration_version": calibration["calibration_version"],
    }, ensure_ascii=False), flush=True)
    with scores_path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row.get("question_id") or row.get("id") or "").strip()
            if not question_id:
                raise ValueError(f"line {line_number}: score row lacks question_id")
            if question_id in seen_ids:
                raise ValueError(f"line {line_number}: duplicate question_id {question_id}")
            seen_ids.add(question_id)
            calibrated = apply_calibration(
                float(row["raw_difficulty_score"]),
                calibration,
                calibration_already_validated=True,
            )
            result = {
                "question_id": question_id,
                "split": row.get("split"),
                "text_sha256": row.get("text_sha256"),
                **calibrated,
                "checkpoint_fingerprint": calibration["checkpoint_fingerprint"],
            }
            counts[result["difficulty_level"]] += 1
            records += 1
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
    temporary.replace(output_path)
    report = {
        "schema_version": "pairwise_difficulty_predictions_v1",
        "records": records,
        "scores": str(scores_path.resolve()),
        "scores_sha256": sha256_file(scores_path),
        "calibration": str(calibration_path.resolve()),
        "calibration_sha256": sha256_file(calibration_path),
        "calibration_version": calibration["calibration_version"],
        "calibration_id": calibration["calibration_id"],
        "checkpoint_fingerprint": calibration["checkpoint_fingerprint"],
        "difficulty_level_counts": dict(counts),
        "output": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "thresholds_recomputed": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
