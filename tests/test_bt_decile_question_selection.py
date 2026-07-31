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

from physics_difficulty.pairwise.question_selection import select_questions_by_bt_decile
from physics_difficulty.schema import FEATURE_VALUES


def features_for(index: int) -> dict[str, str]:
    values = {name: categories[0] for name, categories in FEATURE_VALUES.items()}
    # One rare structure in every score decile.
    if index % 10 == 9:
        values["problem_structure"] = "实验探究"
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class BtDecileQuestionSelectionTests(unittest.TestCase):
    def test_selection_has_exact_decile_and_reason_quotas(self):
        ids = [f"q{index:03d}" for index in range(100)]
        scores = {question_id: float(index) for index, question_id in enumerate(ids)}
        features = {
            question_id: features_for(index)
            for index, question_id in enumerate(ids)
        }
        result = select_questions_by_bt_decile(
            ids,
            scores,
            features,
            target_count=50,
            deciles=10,
            distribution_fraction=0.6,
            rare_fraction=0.2,
            random_fraction=0.2,
            seed=42,
        )
        self.assertEqual(len(result), 50)
        self.assertEqual(len({row["question_id"] for row in result}), 50)
        self.assertEqual(Counter(row["bt_decile"] for row in result), Counter({index: 5 for index in range(1, 11)}))
        self.assertEqual(
            Counter(row["selection_reason"] for row in result),
            Counter({"distribution_matched": 30, "rare_feature_protection": 10, "random_exploration": 10}),
        )
        for decile in range(1, 11):
            selected_ids = {
                row["question_id"] for row in result if row["bt_decile"] == decile
            }
            self.assertTrue(
                any(features[question_id]["problem_structure"] == "实验探究" for question_id in selected_ids)
            )

    def test_cli_is_cpu_only_and_keeps_teacher_fields_out_of_selected_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            questions_path = directory / "questions.jsonl"
            features_path = directory / "features.jsonl"
            scores_path = directory / "scores.jsonl"
            scores_manifest_path = directory / "scores.manifest.json"
            selected_path = directory / "selected.jsonl"
            audit_path = directory / "selection_audit.jsonl"
            manifest_path = directory / "selection.manifest.json"
            question_rows = [
                {
                    "id": f"q{index:03d}",
                    "split": "train",
                    "text": f"【题干】物理题 {index}",
                    "diagnostics": {"input_length_bucket": "short"},
                }
                for index in range(100)
            ]
            feature_rows = [
                {
                    "id": f"q{index:03d}",
                    "teacher_difficulty_level": ["送分题", "基础题", "中等题", "拔高题", "压轴题"][index % 5],
                    "teacher_features": features_for(index),
                }
                for index in range(100)
            ]
            score_rows = [
                {
                    "question_id": f"q{index:03d}",
                    "split": "train",
                    "raw_difficulty_score": float(index),
                }
                for index in range(100)
            ]
            questions_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in question_rows),
                encoding="utf-8",
            )
            features_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in feature_rows),
                encoding="utf-8",
            )
            scores_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in score_rows),
                encoding="utf-8",
            )
            scores_manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "pairwise_single_question_scores_v1",
                        "records": 100,
                        "questions": str(questions_path.resolve()),
                        "questions_sha256": sha256_file(questions_path),
                        "output": str(scores_path.resolve()),
                        "output_sha256": sha256_file(scores_path),
                        "checkpoint_fingerprint": "test-checkpoint",
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "select_bt_feature_balanced_questions.py"),
                    "--questions",
                    str(questions_path),
                    "--features-file",
                    str(features_path),
                    "--scores",
                    str(scores_path),
                    "--scores-manifest",
                    str(scores_manifest_path),
                    "--output",
                    str(selected_path),
                    "--audit-output",
                    str(audit_path),
                    "--manifest",
                    str(manifest_path),
                    "--target-count",
                    "50",
                    "--distribution-fraction",
                    "0.6",
                    "--rare-fraction",
                    "0.2",
                    "--random-fraction",
                    "0.2",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["cpu_only"])
            self.assertEqual(manifest["selected_questions"], 50)
            self.assertEqual(manifest["bt_deciles"], {str(index): 5 for index in range(1, 11)})
            self.assertEqual(len(selected_path.read_text(encoding="utf-8").splitlines()), 50)
            selected_text = selected_path.read_text(encoding="utf-8")
            self.assertNotIn("teacher_features", selected_text)
            self.assertNotIn("teacher_difficulty_level", selected_text)
            self.assertEqual(len(audit_path.read_text(encoding="utf-8").splitlines()), 50)


if __name__ == "__main__":
    unittest.main()
