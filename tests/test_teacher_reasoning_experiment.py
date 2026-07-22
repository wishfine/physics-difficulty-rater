import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_teacher_module():
    path = ROOT / "scripts" / "run_local_pairwise_teacher.py"
    spec = importlib.util.spec_from_file_location("run_local_pairwise_teacher", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TeacherReasoningExperimentTests(unittest.TestCase):
    def test_vllm_environment_disables_flashinfer_sampler_by_default(self):
        teacher = load_teacher_module()
        previous = os.environ.pop("VLLM_USE_FLASHINFER_SAMPLER", None)
        try:
            teacher.configure_vllm_environment()
            self.assertEqual(os.environ["VLLM_USE_FLASHINFER_SAMPLER"], "0")
        finally:
            if previous is None:
                os.environ.pop("VLLM_USE_FLASHINFER_SAMPLER", None)
            else:
                os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = previous

    def test_final_vote_parser_accepts_reasoning_but_rejects_truncated_reasoning(self):
        teacher = load_teacher_module()
        self.assertEqual(teacher.parse_vote("<think>先比较建模过程</think>\nA"), "A")
        self.assertEqual(teacher.parse_vote("分析过程很长\n最终答案：B"), "B")
        self.assertIsNone(teacher.parse_vote("<think>还没有得出结论"))
        self.assertIsNone(teacher.parse_vote("A 或 B"))

    def test_three_reasoning_configs_are_explicit_and_comparable(self):
        expected = {
            "qwen3_32b_pairwise_teacher_nonthinking.json": (False, 4, 0.7, 0.8, 8),
            "qwen3_32b_pairwise_teacher_thinking_512.json": (True, 512, 0.6, 0.95, 4),
            "qwen3_32b_pairwise_teacher_thinking_1024.json": (True, 1024, 0.6, 0.95, 4),
        }
        for name, (thinking, budget, temperature, top_p, batch_size) in expected.items():
            config = json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
            self.assertEqual(config["enable_thinking"], thinking)
            self.assertEqual(config["max_new_tokens"], budget)
            self.assertEqual(config["temperature"], temperature)
            self.assertEqual(config["top_p"], top_p)
            self.assertEqual(config["top_k"], 20)
            self.assertEqual(config["min_p"], 0.0)
            self.assertEqual(config["batch_size"], batch_size)
            self.assertEqual(config["gpu_memory_utilization"], 0.82)
            self.assertEqual(config["max_num_batched_tokens"], 4096)
            self.assertEqual(config["max_num_seqs"], 32)
            self.assertEqual(config["tensor_parallel_size"], 2)
            self.assertEqual(config["initial_samples_per_direction"], 3)
            self.assertEqual(config["uncertain_samples_per_direction"], 5)
            self.assertEqual(config["maximum_samples_per_direction"], 10)

    def test_json_config_overrides_code_defaults(self):
        teacher = load_teacher_module()
        config = ROOT / "configs" / "qwen3_32b_pairwise_teacher_thinking_512.json"
        args = teacher.parse_args([
            "--config", str(config),
            "--pairs", "pairs.jsonl",
            "--raw-votes-output", "votes.jsonl",
            "--manifest", "manifest.json",
            "--model-path", "model",
        ])
        self.assertTrue(args.enable_thinking)
        self.assertEqual(args.mode_name, "thinking_512")
        self.assertEqual(args.temperature, 0.6)
        self.assertEqual(args.top_p, 0.95)
        self.assertEqual(args.max_new_tokens, 512)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.gpu_memory_utilization, 0.82)

    def test_vote_summary_tracks_parse_rate_tokens_and_throughput(self):
        teacher = load_teacher_module()
        rows = [
            {"valid": True, "output_token_count": 2},
            {"valid": True, "output_token_count": 6},
            {"valid": False, "output_token_count": 10},
        ]
        summary = teacher.summarize_vote_rows(rows, generation_seconds=4.0)
        self.assertEqual(summary["total_vote_rows"], 3)
        self.assertEqual(summary["valid_votes"], 2)
        self.assertAlmostEqual(summary["parse_success_rate"], 2 / 3)
        self.assertEqual(summary["output_tokens"], 18)
        self.assertEqual(summary["valid_output_tokens"], 8)
        self.assertEqual(summary["mean_output_tokens_per_valid_vote"], 4.0)
        self.assertEqual(summary["valid_votes_per_second"], 0.5)

    def test_comparison_report_compares_common_pair_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            runs = []
            for name, target, seconds, tokens in (
                ("nonthinking", 0.8, 2.0, 12),
                ("thinking_512", 0.7, 4.0, 24),
            ):
                votes = directory / f"{name}.jsonl"
                vote_rows = []
                for direction in ("forward", "backward"):
                    for index in range(3):
                        winner = "qa" if index < 2 else "qb"
                        vote_rows.append({
                            "pair_id": "p1", "question_a_id": "qa", "question_b_id": "qb",
                            "direction": direction, "winner_question_id": winner,
                            "valid": True, "output_token_count": tokens // 6,
                        })
                votes.write_text("".join(json.dumps(row) + "\n" for row in vote_rows), encoding="utf-8")
                manifest = directory / f"{name}.manifest.json"
                manifest.write_text(json.dumps({
                    "teacher_mode": name,
                    "generation_wall_seconds": seconds,
                    "expected_soft_target_for_test": target,
                }), encoding="utf-8")
                runs.extend(["--run", name, str(manifest), str(votes)])

            output = directory / "comparison.json"
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts" / "compare_teacher_reasoning_modes.py"),
                *runs, "--output", str(output),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["runs"]["nonthinking"]["aggregated_pairs"], 1)
            self.assertEqual(report["cross_mode"]["nonthinking__vs__thinking_512"]["common_pairs"], 1)
            self.assertEqual(report["cross_mode"]["nonthinking__vs__thinking_512"]["hard_label_agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()
