from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from physics_difficulty.schema import FEATURE_VALUES


ROOT = Path(__file__).resolve().parents[1]


class ValidationQuestionSelectionTests(unittest.TestCase):
    def _question(self, question_id: str, text: str) -> dict[str, object]:
        return {
            "id": question_id,
            "split": "validation",
            "text": text,
            "diagnostics": {
                "input_length_bucket": "short",
                "has_analysis": True,
                "has_subquestions": False,
                "image_dependency_risk": "medium",
            },
        }

    def test_selects_deterministic_label_free_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "validation.jsonl"
            output = directory / "questions.jsonl"
            manifest = directory / "manifest.json"
            rows = [self._question(f"q{index}", f"题目 {index}") for index in range(8)]
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(ROOT / "scripts" / "select_validation_questions.py"),
                "--questions", str(source),
                "--output", str(output),
                "--manifest", str(manifest),
                "--records", "4",
                "--seed", "7",
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            first = output.read_text(encoding="utf-8")
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(first, output.read_text(encoding="utf-8"))
            report = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(report["selected_questions"], 4)
            self.assertFalse(report["guardrails"]["absolute_difficulty_labels_used"])
            self.assertFalse(report["guardrails"]["auxiliary_features_used_for_node_selection"])

    def test_rejects_source_label_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "validation.jsonl"
            source.write_text(
                json.dumps({**self._question("q1", "题目 1"), "difficulty": "1"}) + "\n"
                + json.dumps(self._question("q2", "题目 2")) + "\n",
                encoding="utf-8",
            )
            result = subprocess.run([
                sys.executable,
                str(ROOT / "scripts" / "select_validation_questions.py"),
                "--questions", str(source),
                "--output", str(directory / "out.jsonl"),
                "--manifest", str(directory / "manifest.json"),
                "--records", "2",
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("forbidden label fields", result.stderr)

    def test_split_isolation_rejects_normalized_text_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            train = directory / "train.jsonl"
            validation = directory / "validation.jsonl"
            report = directory / "report.json"
            train.write_text(
                json.dumps({**self._question("train", "重复 题目"), "split": "train"}) + "\n",
                encoding="utf-8",
            )
            validation.write_text(
                json.dumps(self._question("validation", "重复　题目")) + "\n",
                encoding="utf-8",
            )
            result = subprocess.run([
                sys.executable,
                str(ROOT / "scripts" / "validate_question_split_isolation.py"),
                "--questions", str(train),
                "--questions", str(validation),
                "--output", str(report),
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["overlaps"][0]["normalized_text_count"], 1)

    def test_private_auxiliary_coverage_audit_does_not_require_labels_in_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pool = directory / "pool.jsonl"
            selected = directory / "selected.jsonl"
            features = directory / "features.jsonl"
            output = directory / "audit.json"
            rows = [self._question("q1", "题目 1"), self._question("q2", "题目 2")]
            pool.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            selected.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            teacher_features = {name: values[0] for name, values in FEATURE_VALUES.items()}
            features.write_text(
                "".join(
                    json.dumps({"id": row["id"], "teacher_features": teacher_features}, ensure_ascii=False) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            subprocess.run([
                sys.executable,
                str(ROOT / "scripts" / "audit_auxiliary_feature_coverage.py"),
                "--pool-questions", str(pool),
                "--selected-questions", str(selected),
                "--features-file", str(features),
                "--output", str(output),
            ], check=True, capture_output=True, text=True)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "PASS")
            self.assertTrue(report["guardrails"]["features_used_after_node_selection_only"])
            self.assertFalse(report["guardrails"]["absolute_difficulty_labels_used"])


if __name__ == "__main__":
    unittest.main()
