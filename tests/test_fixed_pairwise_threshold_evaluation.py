from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FixedPairwiseThresholdEvaluationTests(unittest.TestCase):
    def test_writes_levels_and_matched_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores = root / "scores.jsonl"
            reviews = root / "reviews.jsonl"
            predictions = root / "predictions.jsonl"
            report = root / "report.json"
            scores.write_text(
                "".join(json.dumps(row) + "\n" for row in [
                    {"question_id": "a", "raw_difficulty_score": 0.1},
                    {"question_id": "b", "raw_difficulty_score": 0.5},
                    {"question_id": "c", "raw_difficulty_score": 1.5},
                ]),
                encoding="utf-8",
            )
            reviews.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [
                    {"question_id": "a", "model_difficulty_level": "送分题", "verdict": "correct"},
                    {"question_id": "b", "model_difficulty_level": "基础题", "verdict": "correct"},
                ]),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable, str(ROOT / "scripts" / "evaluate_fixed_pairwise_thresholds.py"),
                    "--scores", str(scores), "--reviews", str(reviews),
                    "--thresholds", "0.2,0.6,1.0,1.4",
                    "--predictions-output", str(predictions), "--report-output", str(report),
                ],
                check=True, capture_output=True, text=True,
            )
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(result["matched_review_records"], 2)
            self.assertEqual(result["missing_review_records"], 1)
            self.assertEqual(result["metrics"]["accuracy"], 1.0)
            rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[2]["difficulty_level_id"], 4)
            self.assertNotIn("standard_difficulty_level", rows[2])


if __name__ == "__main__":
    unittest.main()
