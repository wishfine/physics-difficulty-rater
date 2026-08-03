from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CalibrationReferenceSelectionTests(unittest.TestCase):
    def test_stable_label_free_selection_with_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "questions.jsonl"
            excluded = root / "excluded.jsonl"
            rows = [{"id": str(index), "text": f"【题干】物理题{index}"} for index in range(20)]
            source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            excluded.write_text(json.dumps({"question_a_id": "3", "question_a_text": "【题干】物理题3"}, ensure_ascii=False) + "\n", encoding="utf-8")
            output = root / "selected.jsonl"
            smoke = root / "smoke.jsonl"
            manifest = root / "manifest.json"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "select_calibration_reference_questions.py"),
                "--input", str(source), "--output", str(output), "--smoke-output", str(smoke),
                "--manifest", str(manifest), "--records", "10", "--smoke-records", "4",
                "--exclude", str(excluded),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            selected = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            smoke_rows = [json.loads(line) for line in smoke.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(selected), 10)
            self.assertEqual(smoke_rows, selected[:4])
            self.assertNotIn("3", {row["id"] for row in selected})
            report = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertFalse(report["labels_used_for_proportional_stratification"])
            self.assertFalse(report["labels_exported"])
            self.assertEqual(report["distribution_claim"], "source_distribution_only; natural-business status requires external provenance")

    def test_preserves_declared_business_difficulty_proportions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "questions.jsonl"
            labels = root / "labels.jsonl"
            questions = [{"id": str(index), "text": f"【题干】业务题{index}"} for index in range(100)]
            teacher = [
                {"id": str(index), "teacher_difficulty_level": "送分题" if index < 60 else "基础题"}
                for index in range(100)
            ]
            source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in questions), encoding="utf-8")
            labels.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in teacher), encoding="utf-8")
            output = root / "selected.jsonl"
            smoke = root / "smoke.jsonl"
            manifest = root / "manifest.json"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "select_calibration_reference_questions.py"),
                 "--input", str(source), "--output", str(output), "--smoke-output", str(smoke),
                 "--manifest", str(manifest), "--records", "50", "--smoke-records", "10",
                 "--stratification-labels", str(labels), "--business-natural-distribution"],
                check=True, capture_output=True, text=True,
            )
            report = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(report["distribution_claim"], "business_natural_distribution")
            self.assertEqual(report["stratification"]["reference_counts"], {"送分题": 30, "基础题": 20})
            self.assertEqual(report["stratification"]["smoke_counts"], {"送分题": 6, "基础题": 4})
            selected_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all("teacher_difficulty_level" not in row for row in selected_rows))


if __name__ == "__main__":
    unittest.main()
