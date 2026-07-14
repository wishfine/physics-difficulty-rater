from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from physics_difficulty.schema import FEATURE_VALUES, normalize_knowledge_domains, normalize_v2_features


class SchemaTests(unittest.TestCase):
    def test_schema_has_exactly_ten_features(self):
        self.assertEqual(len(FEATURE_VALUES), 10)

    def test_problem_structure_is_a_multi_label_list(self):
        result = normalize_v2_features({"problem_structure": ["实验探究", "图像表格分析", "实验探究"]})
        self.assertEqual(result["problem_structure"], ["实验探究", "图像表格分析"])

    def test_legacy_domain_structure_becomes_single_module_tag(self):
        self.assertEqual(normalize_v2_features({"problem_structure": "电路综合"})["problem_structure"], ["单模块综合"])
        self.assertEqual(normalize_knowledge_domains({"problem_structure": "电路综合"}), ["电路"])

    def test_legacy_steps_are_merged(self):
        self.assertEqual(normalize_v2_features({"step_count": "12步以上"})["step_count"], "9步以上")

    def test_information_processing_merge(self):
        features = normalize_v2_features({"graph_table_requirement": "直接读数", "experiment_requirement": "控制变量或故障分析"})
        self.assertEqual(features["information_processing"], "图表与实验混合处理")


if __name__ == "__main__":
    unittest.main()
