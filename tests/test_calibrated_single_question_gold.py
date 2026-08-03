from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CalibratedSingleQuestionGoldTests(unittest.TestCase):
    def test_reports_strict_and_acceptable_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            gold = root / "gold.jsonl"
            output = root / "report.json"
            prediction_rows = [
                {"question_id": "a", "difficulty_level": "基础题", "calibration_id": "c1"},
                {"question_id": "b", "difficulty_level": "中等题", "calibration_id": "c1"},
            ]
            gold_rows = [
                {"id": "a", "gold_difficulty_id": 1, "acceptable_difficulty_ids": [1]},
                {"id": "b", "gold_difficulty_id": 3, "acceptable_difficulty_ids": [2, 3]},
            ]
            predictions.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows), encoding="utf-8")
            gold.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in gold_rows), encoding="utf-8")
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "evaluate_calibrated_single_question_gold.py"),
                 "--predictions", str(predictions), "--gold", str(gold), "--output", str(output)],
                check=True, capture_output=True, text=True,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["strict"]["accuracy"], 0.5)
            self.assertEqual(report["acceptable_level_accuracy"], 1.0)
            self.assertEqual(report["calibration_ids"], ["c1"])


if __name__ == "__main__":
    unittest.main()
