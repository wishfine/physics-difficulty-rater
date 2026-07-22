import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def question(index: int):
    mechanics = index % 2 == 0
    topic = "小车在水平面上运动，分析速度和摩擦力" if mechanics else "电阻接入电路，分析电流和电功率"
    length_bucket = ("short", "medium", "long")[index % 3]
    subquestions = index % 4
    return {
        "id": f"q{index:03d}",
        "parent_id": f"q{index:03d}",
        "question_group_id": f"q{index:03d}",
        "split": "train",
        "text": f"【题干】\n{topic}，题号 {index}。\n\n【解析】\n根据物理规律逐步求解。",
        "text_sha256": f"hash-{index}",
        "normalized_text_sha256": f"normalized-{index}",
        "text_schema_version": "v3_text_only_raw25k_v1",
        "diagnostics": {
            "has_analysis": True,
            "has_options": index % 3 == 0,
            "has_subquestions": bool(subquestions),
            "subquestion_count": subquestions,
            "has_image": True,
            "has_image_url": True,
            "has_image_reference_marker": index % 2 == 0,
            "image_dependency_risk": "high" if index % 2 == 0 else "medium",
            "images_uploaded": False,
            "char_length": 100 + index,
            "input_length_bucket": length_bucket,
        },
        "source_provenance": {"source_file": "raw.jsonl", "source_line": index + 1},
    }


class RawV3PairCandidateTests(unittest.TestCase):
    def run_builder(self, rows, directory: Path, suffix: str = "one"):
        questions = directory / "questions.jsonl"
        questions.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        output = directory / f"{suffix}.pairs.jsonl"
        selected = directory / f"{suffix}.questions.jsonl"
        manifest = directory / f"{suffix}.manifest.json"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_raw_v3_pair_candidates.py"),
                "--questions",
                str(questions),
                "--output",
                str(output),
                "--selected-questions-output",
                str(selected),
                "--manifest",
                str(manifest),
                "--max-questions",
                "40",
                "--target-pairs",
                "80",
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
        return result, output, selected, manifest

    def test_unlabelled_graph_is_deterministic_connected_and_degree_bounded(self):
        rows = [question(index) for index in range(60)]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            first, first_output, _, first_manifest = self.run_builder(rows, directory, "one")
            self.assertEqual(first.returncode, 0, first.stderr)
            second, second_output, _, _ = self.run_builder(rows, directory, "two")
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(first_output.read_bytes(), second_output.read_bytes())

            pairs = [json.loads(line) for line in first_output.read_text(encoding="utf-8").splitlines() if line]
            manifest = json.loads(first_manifest.read_text(encoding="utf-8"))
            self.assertEqual(len(pairs), 80)
            self.assertEqual(len({tuple(sorted((row["question_a_id"], row["question_b_id"]))) for row in pairs}), 80)
            self.assertTrue(all(row["question_a_id"] != row["question_b_id"] for row in pairs))
            self.assertTrue(all(row["pair_source"] in {
                "lexical_near", "structure_matched", "random_global", "graph_bridge", "low_degree_repair"
            } for row in pairs))
            serialized = json.dumps(pairs, ensure_ascii=False)
            self.assertNotIn("difficulty", serialized)
            self.assertNotIn("teacher_", serialized)
            self.assertEqual(manifest["graph"]["node_coverage"], 1.0)
            self.assertEqual(manifest["graph"]["connected_components"], 1)
            self.assertGreaterEqual(manifest["graph"]["degree"]["minimum"], 2)
            self.assertLessEqual(manifest["graph"]["degree"]["maximum"], 6)
            self.assertEqual(manifest["raw_difficulty_used"], False)
            lexical_scores = [row["metadata"].get("lexical_jaccard", -1.0) for row in pairs if row["pair_source"] == "lexical_near"]
            random_scores = [row["metadata"].get("lexical_jaccard", -1.0) for row in pairs if row["pair_source"] == "random_global"]
            self.assertTrue(lexical_scores and random_scores)
            self.assertGreaterEqual(min(lexical_scores + random_scores), 0.0)
            self.assertGreater(sum(lexical_scores) / len(lexical_scores), sum(random_scores) / len(random_scores))
            self.assertGreater(
                manifest.get("mean_lexical_jaccard_by_source", {}).get("lexical_near", -1.0),
                manifest.get("mean_lexical_jaccard_by_source", {}).get("random_global", -1.0),
            )

    def test_builder_rejects_any_historical_label_field(self):
        rows = [question(index) for index in range(4)]
        rows[0]["difficulty"] = 3
        with tempfile.TemporaryDirectory() as tmp:
            result, _, _, _ = self.run_builder(rows, Path(tmp))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("forbidden historical label", result.stderr)

    def test_lexical_near_does_not_silently_degrade_to_random_when_lsh_is_sparse(self):
        rows = [question(index) for index in range(60)]
        for index, row in enumerate(rows):
            digest = hashlib.sha256(f"unique-{index}".encode()).hexdigest()
            row["text"] = f"【题干】\n{digest}\n\n【解析】\n{digest[::-1]}"
        with tempfile.TemporaryDirectory() as tmp:
            result, _, _, manifest_path = self.run_builder(rows, Path(tmp))
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertGreater(manifest["source_counts"].get("lexical_near", 0), 0)


if __name__ == "__main__":
    unittest.main()
