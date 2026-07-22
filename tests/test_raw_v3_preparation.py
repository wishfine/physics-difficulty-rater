import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.data.text_only import forbidden_source_label_paths


def raw_record(question_id: str, stem: str, **overrides):
    record = {
        "parent_id": question_id,
        "question_id": question_id,
        "stem": stem,
        "options": "A. 是  B. 否",
        "analysis": "根据物理规律判断。",
        "sub_questions": [],
        "stem_pic_url": "https://example.test/stem.png",
        "analysis_pic_url": "https://example.test/analysis.png",
        "difficulty": 5,
    }
    record.update(overrides)
    return record


class RawV3PreparationTests(unittest.TestCase):
    def run_preparation(self, records, directory: Path, suffix: str = "one"):
        source = directory / "raw.jsonl"
        source.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
            encoding="utf-8",
        )
        output_dir = directory / suffix
        manifest = directory / f"{suffix}.manifest.json"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "prepare_raw_v3_questions.py"),
                "--input",
                str(source),
                "--output-dir",
                str(output_dir),
                "--manifest",
                str(manifest),
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return output_dir, json.loads(manifest.read_text(encoding="utf-8"))

    def test_raw_source_is_cleaned_deduplicated_and_split_without_labels(self):
        records = [raw_record(f"q{index}", f"物理题 {index}") for index in range(30)]
        records.extend(
            [
                raw_record("duplicate", "  物理题 0  "),
                raw_record("empty", "【题干】", options="", analysis="", stem_pic_url="", analysis_pic_url=""),
                raw_record("leak", "该试题难易程度为中等。"),
                raw_record("physics", "导电的难易程度与电阻有关。"),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir, manifest = self.run_preparation(records, Path(tmp))
            accepted = []
            for split in ("train", "validation", "test"):
                rows = [json.loads(line) for line in (output_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]
                self.assertTrue(all(row["split"] == split for row in rows))
                accepted.extend(rows)

            accepted_ids = {row["id"] for row in accepted}
            self.assertIn("physics", accepted_ids)
            self.assertNotIn("duplicate", accepted_ids)
            self.assertNotIn("empty", accepted_ids)
            self.assertNotIn("leak", accepted_ids)
            self.assertEqual(len(accepted), 31)
            self.assertTrue(all(forbidden_source_label_paths(row) == [] for row in accepted))
            self.assertTrue(all("teacher_difficulty_id" not in row for row in accepted))
            self.assertTrue(all("https://" not in row["text"] for row in accepted))

            quarantine = [json.loads(line) for line in (output_dir / "quarantine.jsonl").read_text(encoding="utf-8").splitlines() if line]
            reasons = {row["id"]: row["reason"] for row in quarantine}
            self.assertEqual(reasons["duplicate"], "duplicate_normalized_text")
            self.assertEqual(reasons["empty"], "semantically_empty")
            self.assertEqual(reasons["leak"], "label_leakage")
            self.assertEqual(manifest["forbidden_source_fields"], ["difficulty"])
            self.assertFalse(manifest["raw_difficulty_used"])
            self.assertEqual(manifest["stats"]["source_records"], 34)
            self.assertEqual(manifest["stats"]["accepted"], 31)

    def test_split_membership_is_reproducible(self):
        records = [raw_record(f"q{index}", f"题目 {index}") for index in range(100)]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            output_one, _ = self.run_preparation(records, directory, "one")
            output_two, _ = self.run_preparation(records, directory, "two")

            def membership(output_dir):
                result = {}
                for split in ("train", "validation", "test"):
                    for line in (output_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
                        if line:
                            result[json.loads(line)["id"]] = split
                return result

            self.assertEqual(membership(output_one), membership(output_two))


if __name__ == "__main__":
    unittest.main()
