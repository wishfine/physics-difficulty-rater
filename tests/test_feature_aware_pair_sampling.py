import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.feature_coverage import (
    feature_coverage_report,
    select_feature_balanced_ids,
)
from physics_difficulty.schema import FEATURE_VALUES


def feature_row(question_id: str, problem_structure: str) -> dict:
    features = {name: values[0] for name, values in FEATURE_VALUES.items()}
    features["problem_structure"] = problem_structure
    return {"id": question_id, "teacher_features": features}


class FeatureAwarePairSamplingTests(unittest.TestCase):
    def test_balanced_selection_keeps_rare_auxiliary_category(self):
        feature_rows = [
            feature_row(f"q{index}", "概念判断" if index < 11 else "实验探究")
            for index in range(12)
        ]
        features = {row["id"]: row["teacher_features"] for row in feature_rows}
        selected = select_feature_balanced_ids(
            [row["id"] for row in feature_rows],
            features,
            target_count=6,
            seed=42,
        )
        self.assertIn("q11", selected)
        report = feature_coverage_report(features, selected)
        self.assertEqual(
            report["features"]["problem_structure"]["实验探究"]["selected_count"],
            1,
        )
        self.assertEqual(report["zero_covered_source_categories"], 0)

    def test_feature_aware_builder_reports_node_and_edge_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            questions_path = directory / "questions.jsonl"
            features_path = directory / "features.jsonl"
            output_path = directory / "pairs.jsonl"
            selected_path = directory / "selected.jsonl"
            manifest_path = directory / "manifest.json"
            question_rows = []
            feature_rows = []
            structures = ["概念判断", "直接计算", "实验探究"]
            for index in range(12):
                question_rows.append(
                    {
                        "id": f"q{index}",
                        "split": "train",
                        "text": f"【题干】物理题目 {index}，求物理量。",
                        "diagnostics": {
                            "input_length_bucket": "short",
                            "subquestion_count": 0,
                            "has_analysis": True,
                            "has_options": False,
                            "image_dependency_risk": "medium",
                            "has_image": False,
                        },
                    }
                )
                feature_rows.append(
                    feature_row(f"q{index}", structures[index % len(structures)])
                )
            questions_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in question_rows),
                encoding="utf-8",
            )
            features_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in feature_rows),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_raw_v3_pair_candidates.py"),
                    "--questions",
                    str(questions_path),
                    "--features-file",
                    str(features_path),
                    "--output",
                    str(output_path),
                    "--selected-questions-output",
                    str(selected_path),
                    "--manifest",
                    str(manifest_path),
                    "--max-questions",
                    "9",
                    "--target-pairs",
                    "18",
                    "--minimum-degree",
                    "2",
                    "--maximum-degree",
                    "6",
                    "--seed",
                    "42",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["auxiliary_features_used_for_sampling"])
            self.assertEqual(
                manifest["feature_coverage"]["zero_covered_source_categories"], 0
            )
            self.assertIn("feature_near", manifest["source_target_weights"])
            self.assertIn("feature_contrast", manifest["source_target_weights"])
            pair = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("feature_hamming_distance", pair["metadata"])
            selected_text = selected_path.read_text(encoding="utf-8")
            self.assertNotIn("teacher_features", selected_text)
            self.assertNotIn("teacher_difficulty", selected_text)


if __name__ == "__main__":
    unittest.main()
