#!/usr/bin/env python3
"""Validate that an API label export satisfies the V2 teacher-label contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from physics_difficulty.schema import DIFFICULTY_TO_ID, FEATURE_VALUES, PROBLEM_STRUCTURE_TAGS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--allow-legacy-problem-structure", action="store_true")
    args = parser.parse_args()
    errors = []
    checked = 0
    for line_number, line in enumerate(Path(args.input).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        checked += 1
        row = json.loads(line)
        rating = row.get("difficulty_rating") or {}
        if rating.get("difficulty_level") not in DIFFICULTY_TO_ID:
            errors.append(f"line {line_number}: invalid difficulty_level")
        features = rating.get("features") or {}
        structure = features.get("problem_structure")
        if isinstance(structure, list):
            if not structure or any(value not in PROBLEM_STRUCTURE_TAGS for value in structure):
                errors.append(f"line {line_number}: invalid multi-label problem_structure")
        elif not args.allow_legacy_problem_structure:
            errors.append(f"line {line_number}: problem_structure must be a V2 label list")
        for name in FEATURE_VALUES:
            if name == "problem_structure":
                continue
            if name == "information_processing" and args.allow_legacy_problem_structure:
                if "graph_table_requirement" in features and "experiment_requirement" in features:
                    continue
            if name not in features:
                errors.append(f"line {line_number}: missing feature {name}")
    if errors:
        raise SystemExit("Teacher-label contract failed:\n" + "\n".join(errors[:50]))
    print(json.dumps({"checked_records": checked, "contract": "teacher_label_v2", "status": "PASS"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
