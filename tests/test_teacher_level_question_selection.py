import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.question_selection import (
    allocate_teacher_level_quotas,
    select_questions_by_teacher_level,
)
from physics_difficulty.schema import DIFFICULTY_LEVELS, FEATURE_VALUES


def features_for(index: int) -> dict[str, str]:
    features = {
        name: values[index % len(values)]
        for name, values in FEATURE_VALUES.items()
    }
    return features


class TeacherLevelQuestionSelectionTests(unittest.TestCase):
    def test_quota_allocation_uses_floor_then_proportional_remainder(self):
        counts = {
            "送分题": 100,
            "基础题": 200,
            "中等题": 300,
            "拔高题": 400,
            "压轴题": 500,
        }
        quotas = allocate_teacher_level_quotas(
            counts,
            target_count=700,
            minimum_per_level=100,
        )
        self.assertEqual(
            quotas,
            {
                "送分题": 100,
                "基础题": 129,
                "中等题": 143,
                "拔高题": 157,
                "压轴题": 171,
            },
        )
        self.assertEqual(sum(quotas.values()), 700)

    def test_selection_matches_level_quotas_and_category_floors(self):
        ids = [f"q{index:04d}" for index in range(500)]
        levels = {
            question_id: DIFFICULTY_LEVELS[index // 100]
            for index, question_id in enumerate(ids)
        }
        features = {
            question_id: features_for(index)
            for index, question_id in enumerate(ids)
        }
        selected = select_questions_by_teacher_level(
            ids,
            levels,
            features,
            target_count=250,
            minimum_per_level=25,
            minimum_category_count_global=5,
            minimum_category_count_per_level=1,
            seed=42,
        )
        self.assertEqual(len(selected), 250)
        self.assertEqual(len({row["question_id"] for row in selected}), 250)
        self.assertEqual(
            Counter(row["teacher_difficulty_level"] for row in selected),
            Counter({level: 50 for level in DIFFICULTY_LEVELS}),
        )
        self.assertIn("category_floor", {row["selection_reason"] for row in selected})

    def test_cli_ignores_raw_difficulty_and_keeps_private_labels_out_of_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            questions_path = directory / "questions.jsonl"
            teacher_path = directory / "teacher.jsonl"
            exclusions_path = directory / "excluded.jsonl"
            output_path = directory / "selected.jsonl"
            audit_path = directory / "selected.private.jsonl"
            manifest_path = directory / "selection.manifest.json"
            question_rows = [
                {
                    "id": f"q{index:03d}",
                    "split": "train",
                    "text": f"【题干】物理题 {index}",
                    "diagnostics": {"input_length_bucket": "short"},
                }
                for index in range(105)
            ]
            teacher_rows = [
                {
                    "id": row["id"],
                    "difficulty": 999,
                    "raw_difficulty": "无意义",
                    "teacher_difficulty_level": DIFFICULTY_LEVELS[index % 5],
                    "teacher_features": features_for(index),
                }
                for index, row in enumerate(question_rows)
            ]
            excluded = {"q000", "q001", "q002", "q003", "q004"}
            questions_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in question_rows),
                encoding="utf-8",
            )
            teacher_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in teacher_rows),
                encoding="utf-8",
            )
            exclusions_path.write_text(
                "".join(
                    json.dumps({"id": question_id}) + "\n"
                    for question_id in sorted(excluded)
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(
                        ROOT
                        / "scripts"
                        / "select_teacher_level_feature_balanced_questions.py"
                    ),
                    "--questions",
                    str(questions_path),
                    "--teacher-data",
                    str(teacher_path),
                    "--exclude-question-ids",
                    str(exclusions_path),
                    "--output",
                    str(output_path),
                    "--audit-output",
                    str(audit_path),
                    "--manifest",
                    str(manifest_path),
                    "--target-count",
                    "50",
                    "--minimum-per-level",
                    "5",
                    "--minimum-category-count-global",
                    "0",
                    "--minimum-category-count-per-level",
                    "0",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            output_text = output_path.read_text(encoding="utf-8")
            self.assertNotIn("difficulty", output_text)
            self.assertNotIn("teacher_features", output_text)
            selected_ids = {
                json.loads(line)["id"] for line in output_text.splitlines()
            }
            self.assertFalse(selected_ids & excluded)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["excluded_questions"], 5)
            self.assertFalse(manifest["guardrails"]["source_raw_difficulty_used"])
            self.assertFalse(
                manifest["guardrails"]["old_bt_score_used_for_selection"]
            )
            self.assertEqual(
                sum(
                    manifest["teacher_level_policy"]["selected_counts"].values()
                ),
                50,
            )


if __name__ == "__main__":
    unittest.main()
