import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.schema import FEATURE_VALUES


def feature(question_id, step="1-2步", **overrides):
    values = {name: options[0] for name, options in FEATURE_VALUES.items()}
    values["step_count"] = step
    values.update(overrides)
    return {"id": question_id, "teacher_features": values}


class StepCountEnrichmentCandidateTests(unittest.TestCase):
    def test_candidates_are_reproducible_and_reviewer_output_has_no_labels(self):
        questions = [
            {"id": f"q{i}", "split": "train", "text": f"题{i}", "diagnostics": {"input_length_bucket": "long" if i == 4 else "short", "subquestion_count": 4 if i == 5 else 0}}
            for i in range(6)
        ]
        feature_rows = [
            feature("q0", "9步以上"),
            feature("q1", "6-8步"),
            feature("q2", "6-8步", subquestion_dependency="多问且层层递进"),
            feature("q3", "3-5步", reasoning_chain="逆向推理或临界分析"),
            feature("q4"), feature("q5"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            questions_path, features_path = directory / "questions.jsonl", directory / "features.jsonl"
            questions_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in questions), encoding="utf-8")
            features_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in feature_rows), encoding="utf-8")
            outputs = []
            for index in range(2):
                output, audit, manifest = directory / f"out{index}.jsonl", directory / f"audit{index}.jsonl", directory / f"manifest{index}.json"
                subprocess.run([
                    sys.executable, str(ROOT / "scripts" / "build_step_count_enrichment_candidates.py"),
                    "--questions", str(questions_path), "--features", str(features_path),
                    "--output", str(output), "--audit-output", str(audit), "--manifest", str(manifest),
                    "--target-questions", "4", "--seed", "42",
                    "--stratum-quotas", json.dumps({"existing_9plus_control": 1, "current_6_8": 1, "progressive_subquestions": 1, "high_reasoning_complexity": 1, "long_or_many_subquestions": 0, "random_control": 0}),
                ], check=True, capture_output=True, text=True)
                outputs.append(output.read_text(encoding="utf-8"))
                reviewer_rows = [json.loads(line) for line in outputs[-1].splitlines()]
                self.assertNotIn("teacher_features", json.dumps(reviewer_rows, ensure_ascii=False))
                self.assertEqual(len(reviewer_rows), 4)
            self.assertEqual(outputs[0], outputs[1])

    def test_reports_quota_shortfalls_and_uses_clean_random_controls(self):
        questions = [
            {"id": "q9", "split": "train", "text": "九步题", "diagnostics": {}},
            {"id": "q6", "split": "train", "text": "六步题", "diagnostics": {}},
            {"id": "q0", "split": "train", "text": "随机题零", "diagnostics": {}},
            {"id": "q1", "split": "train", "text": "随机题一", "diagnostics": {}},
        ]
        feature_rows = [feature("q9", "9步以上"), feature("q6", "6-8步"), feature("q0"), feature("q1")]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            questions_path, features_path = directory / "questions.jsonl", directory / "features.jsonl"
            output, audit, manifest = directory / "out.jsonl", directory / "audit.jsonl", directory / "manifest.json"
            questions_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in questions), encoding="utf-8")
            features_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in feature_rows), encoding="utf-8")
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "build_step_count_enrichment_candidates.py"),
                "--questions", str(questions_path), "--features", str(features_path),
                "--output", str(output), "--audit-output", str(audit), "--manifest", str(manifest),
                "--target-questions", "4", "--seed", "42",
                "--stratum-quotas", json.dumps({
                    "existing_9plus_control": 3, "current_6_8": 1,
                    "progressive_subquestions": 0, "high_reasoning_complexity": 0,
                    "long_or_many_subquestions": 0, "random_control": 2,
                }),
            ], check=True, capture_output=True, text=True)
            report = json.loads(manifest.read_text(encoding="utf-8"))
            control = report["stratum_quota_report"]["existing_9plus_control"]
            self.assertEqual(control, {"requested": 3, "available": 1, "newly_selected": 1, "shortfall": 2})
            audit_rows = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
            random_controls = [row for row in audit_rows if "random_control" in row["selection_reasons"]]
            self.assertEqual({row["question_id"] for row in random_controls}, {"q0", "q1"})
            self.assertTrue(all(row["selection_reasons"] == ["random_control"] for row in random_controls))


if __name__ == "__main__":
    unittest.main()
