import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.data.text_only import forbidden_source_label_paths, leakage_findings, normalize_text_only, question_group_identifier, question_identifier
from physics_difficulty.pairwise.labels import aggregate_pair_votes, pair_reliability, smoothed_probability
from physics_difficulty.pairwise.metrics import graph_metrics, soft_pairwise_metrics


class PairwiseTests(unittest.TestCase):
    def test_text_only_removes_image_payloads(self):
        text = "题干 ![图](https://example/a.png) <img src='x'> 【图片】 结尾"
        normalized = normalize_text_only(text)
        self.assertEqual(normalized, "题干 结尾")
        self.assertNotIn("http", normalized)

    def test_forbidden_historical_label_is_found_recursively(self):
        row = {"metadata": {"source": {"difficulty": 4}}, "soft_target": 0.5}
        self.assertEqual(forbidden_source_label_paths(row), ["metadata.source.difficulty"])

    def test_physics_phrase_about_ease_is_not_label_leakage(self):
        self.assertEqual(leakage_findings("导体对电流阻碍作用的大小表示导电的难易程度。"), [])
        findings = leakage_findings("该试题难易程度为中等。")
        self.assertEqual([item["type"] for item in findings], ["difficulty_assessment"])

    def test_existing_question_id_is_preserved_and_never_invented(self):
        self.assertEqual(question_identifier({"id": 123}), "123")
        self.assertEqual(question_group_identifier({"parent_id": 99}, "123"), "99")
        self.assertEqual(question_group_identifier({}, "123"), "123")
        with self.assertRaises(ValueError):
            question_identifier({"stem": "没有ID"})

    def test_bidirectional_votes_use_real_question_identity(self):
        def vote(direction, winner):
            return {
                "pair_id": "p1", "question_a_id": "qa", "question_b_id": "qb",
                "direction": direction, "winner_question_id": winner, "valid": True,
            }

        rows = [vote("forward", "qa") for _ in range(3)] + [vote("backward", "qa") for _ in range(2)] + [vote("backward", "qb")]
        result = aggregate_pair_votes(rows)
        self.assertGreater(result["soft_target"], 0.5)
        self.assertEqual(result["valid_votes"], 6)

    def test_invalid_winner_is_rejected(self):
        rows = [
            {"pair_id": "p", "question_a_id": "a", "question_b_id": "b", "direction": "forward", "winner_question_id": "x", "valid": True},
            {"pair_id": "p", "question_a_id": "a", "question_b_id": "b", "direction": "backward", "winner_question_id": "a", "valid": True},
        ]
        with self.assertRaises(ValueError):
            aggregate_pair_votes(rows)

    def test_smoothing_reliability_and_metrics(self):
        self.assertAlmostEqual(smoothed_probability(3, 3), 0.875)
        self.assertEqual(pair_reliability(0.2)["sample_weight"], 0.5)
        metrics = soft_pairwise_metrics([0.9, 0.1, 0.5], [0.8, 0.2, 0.5])
        self.assertEqual(metrics["pairwise_accuracy"], 1.0)
        self.assertEqual(metrics["non_tied_pair_count"], 2)

    def test_graph_metrics_include_isolated_nodes(self):
        pairs = [{"question_a_id": "a", "question_b_id": "b"}]
        metrics = graph_metrics(pairs, ["a", "b", "c"])
        self.assertEqual(metrics["connected_components"], 2)
        self.assertAlmostEqual(metrics["node_coverage"], 2 / 3)

    def test_preparation_drops_raw_difficulty_instead_of_using_it(self):
        source = {
            "id": "q1",
            "difficulty": 5,
            "raw_difficulty": 4,
            "stem": "小球做匀速直线运动。",
            "analysis": "速度保持不变。",
            "text": "【题干】\n小球做匀速直线运动。\n\n【解析】\n速度保持不变。",
            "input_sections": [
                {"kind": "parent_题干", "title": "【题干】", "text": "小球做匀速直线运动。", "required": True},
                {"kind": "parent_解析", "title": "【解析】", "text": "速度保持不变。", "required": False},
            ],
            "input_sha256": hashlib.sha256("【题干】\n小球做匀速直线运动。\n\n【解析】\n速度保持不变。".encode("utf-8")).hexdigest(),
            "parent_id": "q1",
            "source_dataset_id": "physics",
            "teacher_difficulty_id": 1,
            "teacher_difficulty_level": "基础题",
            "teacher_features": {key: "x" for key in (
                "calculation_complexity", "constraint_count", "information_processing",
                "knowledge_count", "problem_structure", "reasoning_chain", "state_count",
                "step_count", "subquestion_dependency", "variable_relation",
            )},
            "teacher_features_legacy18": {key: "x" for key in (
                "additional_structure", "calculation_complexity", "constraint_count",
                "cross_module", "error_risk", "experiment_requirement", "formula_count",
                "graph_table_requirement", "information_carrier", "knowledge_count",
                "knowledge_diff", "problem_structure", "reality_question", "reasoning_chain",
                "state_count", "step_count", "subquestion_dependency", "variable_relation",
            )},
            "feature_metadata": {"knowledge_domains": ["力学"], "difficulty": 5},
            "diagnostics": {"has_image_url": True, "raw_difficulty": 4},
            "feature_schema_version": "v2_frozen18",
            "label_schema_version": "v2_frozen18",
            "label_source": "api_v7_frozen18",
            "prompt_version": "frozen_physics_prompt",
            "postprocess_version": "v7",
            "teacher_model": "current_physics_api",
            "label_quality": {"sample_weight": 0.7},
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "input.jsonl"
            output_path = directory / "questions.jsonl"
            input_path.write_text(json.dumps(source, ensure_ascii=False) + "\n", encoding="utf-8")
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "prepare_pairwise_questions.py"),
                "--input", str(input_path), "--output", str(output_path),
                "--manifest", str(directory / "manifest.json"), "--split", "pilot",
            ], check=True, capture_output=True, text=True)
            prepared = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(forbidden_source_label_paths(prepared), [])
            self.assertEqual(prepared["teacher_difficulty_id"], 1)
            self.assertNotIn("teacher_difficulty_level", prepared)
            self.assertEqual(prepared["feature_metadata"], {"knowledge_domains": ["力学"]})
            self.assertTrue(prepared["diagnostics"]["has_image"])
            self.assertFalse(prepared["diagnostics"]["images_uploaded"])

    def test_prepared_contract_rejects_raw_unprocessed_record(self):
        source = {"id": "q1", "stem": "尚未处理的题目", "difficulty": 3}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "raw.jsonl"
            input_path.write_text(json.dumps(source, ensure_ascii=False) + "\n", encoding="utf-8")
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts" / "prepare_pairwise_questions.py"),
                "--input", str(input_path), "--output", str(directory / "out.jsonl"),
                "--manifest", str(directory / "manifest.json"), "--split", "train",
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("frozen18 input is missing fields", result.stderr)

    def test_candidate_vote_aggregate_validate_smoke_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            questions = directory / "questions.jsonl"
            question_rows = [
                {
                    "id": f"q{index}", "split": "pilot", "text": f"【题干】\n物理题 {index}",
                    "teacher_difficulty_id": index % 3,
                    "feature_metadata": {"knowledge_domains": ["力学"]},
                    "teacher_features": {"problem_structure": "直接计算"},
                    "diagnostics": {"has_image": False, "images_uploaded": False},
                }
                for index in range(6)
            ]
            questions.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in question_rows), encoding="utf-8")
            candidates = directory / "candidates.jsonl"
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "build_pair_candidates.py"),
                "--questions", str(questions), "--output", str(candidates),
                "--manifest", str(directory / "candidate_manifest.json"),
                "--selected-questions-output", str(directory / "selected.jsonl"),
                "--target-pairs", "6", "--minimum-degree", "2", "--maximum-degree", "4",
            ], check=True, capture_output=True, text=True)
            pair_rows = [json.loads(line) for line in candidates.read_text(encoding="utf-8").splitlines()]
            votes = []
            for pair in pair_rows:
                winner = max(pair["question_a_id"], pair["question_b_id"])
                for direction in ("forward", "backward"):
                    for sample_index in range(3):
                        votes.append({
                            "pair_id": pair["pair_id"], "question_a_id": pair["question_a_id"],
                            "question_b_id": pair["question_b_id"], "direction": direction,
                            "winner_question_id": winner, "sample_index": sample_index,
                            "raw_output": "A", "valid": True,
                        })
            raw_votes = directory / "votes.jsonl"
            raw_votes.write_text("".join(json.dumps(row) + "\n" for row in votes), encoding="utf-8")
            curated = directory / "pairs.jsonl"
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "aggregate_pairwise_votes.py"),
                "--pairs", str(candidates), "--raw-votes", str(raw_votes),
                "--output", str(curated), "--manifest", str(directory / "pair_manifest.json"),
            ], check=True, capture_output=True, text=True)
            validation = subprocess.run([
                sys.executable, str(ROOT / "scripts" / "validate_pairwise_data.py"),
                "--input", str(curated), "--questions", str(directory / "selected.jsonl"),
            ], check=True, capture_output=True, text=True)
            report = json.loads(validation.stdout)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["records"], 6)
            self.assertEqual(report["graph"]["node_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
