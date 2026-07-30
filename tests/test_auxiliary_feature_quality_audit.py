import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.schema import FEATURE_VALUES


def features(index: int) -> dict[str, str]:
    return {name: values[index % len(values)] for name, values in FEATURE_VALUES.items()}


class AuxiliaryFeatureQualityAuditTests(unittest.TestCase):
    def test_audit_uses_unique_question_counts_and_detects_degree_bias(self):
        rows = [
            {
                "pair_id": "p1", "question_a_id": "q1", "question_b_id": "q2",
                "auxiliary_features": {"question_a": features(0), "question_b": features(1)},
                "auxiliary_feature_quality": {"question_a": 1.0, "question_b": 0.5},
            },
            {
                "pair_id": "p2", "question_a_id": "q1", "question_b_id": "q3",
                "auxiliary_features": {"question_a": features(0), "question_b": features(1)},
                "auxiliary_feature_quality": {"question_a": 1.0, "question_b": 0.5},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pairs = directory / "pairs.jsonl"
            output = directory / "audit.json"
            pairs.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "audit_auxiliary_feature_quality.py"),
                "--pairs", str(pairs), "--output", str(output), "--minimum-class-support", "2",
            ], check=True, capture_output=True, text=True)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["unique_pair_questions"], 3)
            self.assertEqual(report["unique_questions_with_complete_auxiliary_features"], 3)
            self.assertEqual(report["question_degree"]["minimum"], 1)
            self.assertEqual(report["question_degree"]["maximum"], 2)
            structure = report["features"]["problem_structure"]
            self.assertEqual(structure["classes"]["概念判断"]["unique_question_support"], 1)
            self.assertEqual(structure["classes"]["概念判断"]["pair_side_support"], 2)
            self.assertGreater(structure["pair_side_vs_unique_js_divergence"], 0.0)


if __name__ == "__main__":
    unittest.main()
