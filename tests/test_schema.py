from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from physics_difficulty.schema import FEATURE_VALUES, normalize_knowledge_domains, normalize_v2_features
from physics_difficulty.data.formatting import format_question
from physics_difficulty.data.truncation import render_with_token_budget


class SchemaTests(unittest.TestCase):
    def test_schema_has_exactly_ten_features(self):
        self.assertEqual(len(FEATURE_VALUES), 10)

    def test_problem_structure_preserves_frozen_single_label(self):
        result = normalize_v2_features({"problem_structure": "电路综合"})
        self.assertEqual(result["problem_structure"], "电路综合")

    def test_frozen_domain_structure_and_metadata_are_preserved(self):
        self.assertEqual(normalize_v2_features({"problem_structure": "电路综合"})["problem_structure"], "电路综合")
        self.assertEqual(normalize_knowledge_domains({"problem_structure": "电路综合"}), ["电路"])

    def test_legacy_steps_are_merged(self):
        self.assertEqual(normalize_v2_features({"step_count": "12步以上"})["step_count"], "9步以上")

    def test_information_processing_merge(self):
        features = normalize_v2_features({"graph_table_requirement": "直接读数", "experiment_requirement": "控制变量或故障分析"})
        self.assertEqual(features["information_processing"], "图表与实验混合处理")

    def test_formatter_matches_api_subquestion_convention(self):
        rendered = format_question({"stem": "主题", "sub_questions": [{"question_id": "2", "stem": "后"}, {"question_id": "1", "stem": "前", "analysis": "解析"}]})
        self.assertIn("【小题】\n\n  小题1:\n\n    题干: 前", rendered)
        self.assertLess(rendered.index("前"), rendered.index("后"))

    def test_structured_truncation_keeps_late_subquestion_title(self):
        class CharacterTokenizer:
            def encode(self, text, add_special_tokens=False): return list(text)
            def decode(self, tokens, skip_special_tokens=True): return "".join(tokens)
        sections = [
            {"kind": "parent_题干", "title": "【题干】", "text": "主" * 30, "required": True},
            {"kind": "subquestions_header", "title": "【小题】", "text": "", "required": True},
            {"kind": "subquestion_header", "title": "  小题1:", "text": "", "required": True},
            {"kind": "child_题干", "title": "    题干:", "text": "子" * 30, "required": True},
        ]
        rendered, info = render_with_token_budget(sections, CharacterTokenizer(), 50)
        self.assertTrue(info["truncated"])
        self.assertIn("小题1", rendered)


if __name__ == "__main__":
    unittest.main()
