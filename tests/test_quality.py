from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physics_difficulty.data.quality import feature_conflicts


class QualityTests(unittest.TestCase):
    def test_six_plus_steps_remains_a_high_complexity_conflict_for_giveaway_items(self):
        conflicts = feature_conflicts(
            "送分题",
            {
                "step_count": "6步以上",
                "constraint_count": "无约束",
                "variable_relation": "无变量关系",
                "calculation_complexity": "简单笔算",
                "reasoning_chain": "简单因果推理",
                "knowledge_count": "2-3个",
            },
        )
        self.assertIn(("severe", "送分题与高复杂度特征冲突"), conflicts)


if __name__ == "__main__":
    unittest.main()
