from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from physics_difficulty.schema import FEATURE_VALUES, normalize_v2_features


class SchemaTests(unittest.TestCase):
    def test_schema_has_exactly_ten_features(self):
        self.assertEqual(len(FEATURE_VALUES), 10)

    def test_legacy_steps_are_merged(self):
        self.assertEqual(normalize_v2_features({"step_count": "12步以上"})["step_count"], "9步以上")

    def test_information_processing_merge(self):
        features = normalize_v2_features({"graph_table_requirement": "直接读数", "experiment_requirement": "控制变量或故障分析"})
        self.assertEqual(features["information_processing"], "图表与实验混合处理")


if __name__ == "__main__":
    unittest.main()
