import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.calibration import (
    apply_calibration,
    build_calibration,
    validate_calibration,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_scores(path: Path, scores, split: str = "train") -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "question_id": f"q{index}",
                    "split": split,
                    "text_sha256": f"text-{index}",
                    "raw_difficulty_score": float(score),
                },
                ensure_ascii=False,
            )
            + "\n"
            for index, score in enumerate(scores)
        ),
        encoding="utf-8",
    )


def write_score_manifest(path: Path, scores: Path, fingerprint: str = "checkpoint-fingerprint") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "pairwise_single_question_scores_v1",
                "records": sum(1 for line in scores.read_text(encoding="utf-8").splitlines() if line),
                "questions": "/fixed/reference/questions.jsonl",
                "questions_sha256": "questions-hash",
                "checkpoint_fingerprint": fingerprint,
                "output_sha256": sha256_file(scores),
                "excluded_question_count": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


class PairwiseCalibrationTests(unittest.TestCase):
    def test_empirical_percentile_and_five_level_boundaries(self):
        calibration = build_calibration(
            range(10),
            calibration_version="unit-v1",
            checkpoint_fingerprint="checkpoint",
            reference={"scores_sha256": "scores"},
        )
        self.assertEqual(calibration["raw_score_thresholds"], [1.8, 3.6, 6.3, 8.1])
        self.assertEqual(apply_calibration(-1, calibration)["difficulty_score"], 0.0)
        mapped = apply_calibration(2, calibration)
        self.assertEqual(mapped["difficulty_percentile"], 0.3)
        self.assertEqual(mapped["difficulty_level"], "基础题")
        self.assertEqual(apply_calibration(9, calibration)["difficulty_level"], "压轴题")

    def test_calibration_detects_content_tampering(self):
        calibration = build_calibration(
            range(20),
            calibration_version="unit-v1",
            checkpoint_fingerprint="checkpoint",
            reference={"scores_sha256": "scores"},
        )
        validate_calibration(calibration)
        calibration["raw_score_thresholds"][0] = -999
        with self.assertRaisesRegex(ValueError, "calibration_id"):
            validate_calibration(calibration)

    def test_fit_and_predict_cli_freeze_expected_distribution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_scores = root / "reference.jsonl"
            reference_manifest = root / "reference.manifest.json"
            calibration = root / "calibration.json"
            predictions = root / "predictions.jsonl"
            prediction_manifest = root / "predictions.manifest.json"
            write_scores(reference_scores, range(100))
            write_score_manifest(reference_manifest, reference_scores)

            fit = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "fit_pairwise_difficulty_calibration.py"),
                    "--scores",
                    str(reference_scores),
                    "--scores-manifest",
                    str(reference_manifest),
                    "--output",
                    str(calibration),
                    "--calibration-version",
                    "physics-reference-unit-v1",
                    "--minimum-records",
                    "100",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(fit.returncode, 0, fit.stderr)

            predict = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "predict_pairwise_difficulty.py"),
                    "--scores",
                    str(reference_scores),
                    "--scores-manifest",
                    str(reference_manifest),
                    "--calibration",
                    str(calibration),
                    "--output",
                    str(predictions),
                    "--manifest",
                    str(prediction_manifest),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(predict.returncode, 0, predict.stderr)
            rows = [
                json.loads(line)
                for line in predictions.read_text(encoding="utf-8").splitlines()
                if line
            ]
            counts = Counter(row["difficulty_level"] for row in rows)
            self.assertEqual(
                counts,
                {
                    "送分题": 20,
                    "基础题": 20,
                    "中等题": 30,
                    "拔高题": 20,
                    "压轴题": 10,
                },
            )
            report = json.loads(prediction_manifest.read_text(encoding="utf-8"))
            self.assertFalse(report["thresholds_recomputed"])

    def test_fit_refuses_validation_or_test_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores = root / "validation.jsonl"
            manifest = root / "validation.manifest.json"
            write_scores(scores, range(10), split="validation")
            write_score_manifest(manifest, scores)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "fit_pairwise_difficulty_calibration.py"),
                    "--scores",
                    str(scores),
                    "--scores-manifest",
                    str(manifest),
                    "--output",
                    str(root / "calibration.json"),
                    "--calibration-version",
                    "invalid",
                    "--minimum-records",
                    "10",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to calibrate on evaluation splits", result.stderr)


if __name__ == "__main__":
    unittest.main()
