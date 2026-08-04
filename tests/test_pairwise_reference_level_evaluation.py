from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PairwiseReferenceLevelEvaluationTests(unittest.TestCase):
    def test_reports_natural_and_fixed_threshold_agreement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores = root / "scores.jsonl"
            manifest = root / "scores.manifest.json"
            labels = root / "labels.jsonl"
            output = root / "report.json"
            score_rows = [
                {"question_id": f"q{index}", "raw_difficulty_score": float(index)}
                for index in range(10)
            ]
            label_ids = [0, 0, 0, 1, 1, 1, 1, 2, 3, 4]
            scores.write_text("".join(json.dumps(row) + "\n" for row in score_rows), encoding="utf-8")
            labels.write_text(
                "".join(json.dumps({"id": f"q{index}", "teacher_difficulty_id": label}) + "\n"
                        for index, label in enumerate(label_ids)),
                encoding="utf-8",
            )
            manifest.write_text(json.dumps({"output_sha256": sha256(scores), "checkpoint_fingerprint": "unit-checkpoint"}), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable, str(ROOT / "scripts" / "evaluate_pairwise_reference_levels.py"),
                    "--scores", str(scores), "--scores-manifest", str(manifest),
                    "--labels", str(labels), "--calibration-output-dir", str(root / "calibrations"),
                    "--calibration-version-prefix", "unit", "--output", str(output),
                ],
                check=True, capture_output=True, text=True,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            for actual, expected in zip(
                report["natural_distribution"]["raw_score_thresholds"], [2.7, 6.3, 7.2, 8.1]
            ):
                self.assertAlmostEqual(actual, expected)
            for actual, expected in zip(
                report["fixed_distribution"]["raw_score_thresholds"], [1.8, 3.6, 6.3, 8.1]
            ):
                self.assertAlmostEqual(actual, expected)
            self.assertEqual(report["natural_distribution"]["metrics"]["accuracy"], 1.0)
            self.assertEqual(report["fixed_distribution"]["metrics"]["accuracy"], 0.5)
            self.assertTrue((root / "calibrations" / "natural_distribution.calibration.json").is_file())
            self.assertTrue((root / "calibrations" / "fixed_distribution.calibration.json").is_file())


if __name__ == "__main__":
    unittest.main()
