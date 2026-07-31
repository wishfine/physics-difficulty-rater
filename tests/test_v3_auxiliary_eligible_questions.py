import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.schema import FEATURE_VALUES


def feature_row(question_id: str, **overrides):
    features = {name: values[0] for name, values in FEATURE_VALUES.items()}
    features.update(overrides)
    return {"id": question_id, "teacher_features": features, "feature_schema_version": "test_aux10"}


class AuxiliaryEligibleQuestionTests(unittest.TestCase):
    def test_keeps_only_questions_with_complete_aux10_without_copying_labels(self):
        questions = [
            {"id": "q1", "split": "train", "text": "题一"},
            {"id": "q2", "split": "train", "text": "题二"},
            {"id": "q3", "split": "train", "text": "题三"},
        ]
        invalid = feature_row("q3")
        invalid["teacher_features"]["step_count"] = "错误值"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            question_path = directory / "questions.jsonl"
            feature_path = directory / "features.jsonl"
            output = directory / "eligible.jsonl"
            manifest = directory / "eligible.manifest.json"
            question_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in questions), encoding="utf-8")
            feature_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [feature_row("q1"), invalid]), encoding="utf-8")
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "prepare_v3_auxiliary_eligible_questions.py"),
                "--questions", str(question_path), "--features", str(feature_path),
                "--output", str(output), "--manifest", str(manifest),
            ], check=True, capture_output=True, text=True)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows, [questions[0]])
            self.assertNotIn("teacher_features", json.dumps(rows, ensure_ascii=False))
            report = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(report["eligible_questions"], 1)
            self.assertEqual(report["excluded_questions"], 2)
            self.assertEqual(report["feature_coverage"]["step_count"]["classes"]["1-2步"], 1)


if __name__ == "__main__":
    unittest.main()
