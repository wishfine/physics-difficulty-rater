import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_row(question_id="q1"):
    return {
        "question_id": question_id,
        "difficulty": 99,
        "stem": "题干",
        "difficulty_rating": {
            "difficulty_level": "中等题",
            "features": {
                "step_count": "9-12步",
                "formula_count": "2-3个",
                "calculation_complexity": "多公式联立",
                "reasoning_chain": "多层因果推理",
                "problem_structure": "直接计算",
                "additional_structure": "无",
                "information_carrier": "文字",
                "reality_question": "否",
                "subquestion_dependency": "无多问",
                "knowledge_count": "2-3个",
                "knowledge_diff": "中等",
                "cross_module": "否",
                "state_count": "双状态",
                "constraint_count": "单一约束",
                "variable_relation": "简单正反比",
                "experiment_requirement": "无",
                "graph_table_requirement": "无",
                "error_risk": "中",
            },
        },
    }


class TeacherFeatureConversionAuditTests(unittest.TestCase):
    def test_audit_checks_exact_auxiliary_derivation_and_ignores_raw_difficulty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            curated = root / "curated.jsonl"
            report = root / "report.json"
            source.write_text(json.dumps(source_row(), ensure_ascii=False) + "\n", encoding="utf-8")
            curated.write_text(
                json.dumps(
                    {
                        "id": "q1",
                        "teacher_difficulty_level": "中等题",
                        "teacher_features_legacy18": source_row()["difficulty_rating"]["features"],
                        "teacher_features": {
                            "problem_structure": "直接计算",
                            "step_count": "6步以上",
                            "calculation_complexity": "多公式联立",
                            "reasoning_chain": "多层因果推理",
                            "knowledge_count": "2-3个",
                            "subquestion_dependency": "无多问",
                            "state_count": "双状态",
                            "constraint_count": "单一约束",
                            "variable_relation": "简单正反比",
                            "graph_table_requirement": "无",
                            "experiment_requirement": "无",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_teacher_feature_conversion.py"),
                    "--source", str(source), "--curated", str(curated), "--report", str(report),
                ],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result_json = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(result_json["status"], "PASS")
            self.assertFalse(result_json["raw_difficulty_used"])
            self.assertEqual(result_json["counts"]["auxiliary_exact_match"], 1)
            self.assertTrue(result_json["auxiliary_exactly_derived"])


if __name__ == "__main__":
    unittest.main()
