#!/usr/bin/env python3
"""Validate exports from the frozen physics 18-feature teacher pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from physics_difficulty.schema import DIFFICULTY_TO_ID, FROZEN_18_FEATURE_NAMES, PROBLEM_STRUCTURE_VALUES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
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
        if features.get("problem_structure") not in PROBLEM_STRUCTURE_VALUES:
            errors.append(f"line {line_number}: invalid frozen problem_structure")
        for name in FROZEN_18_FEATURE_NAMES:
            if name not in features:
                errors.append(f"line {line_number}: missing frozen feature {name}")
    if errors:
        raise SystemExit("Teacher-label contract failed:\n" + "\n".join(errors[:50]))
    print(json.dumps({"checked_records": checked, "contract": "teacher_label_v2_frozen18", "status": "PASS"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
