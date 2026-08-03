import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_step_count_recheck.py"


def load_rechecker():
    spec = importlib.util.spec_from_file_location("run_step_count_recheck", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class StepCountRecheckTests(unittest.TestCase):
    def test_parser_accepts_only_the_three_step_count_values(self):
        rechecker = load_rechecker()
        self.assertIn("只能从以下三档中选一档", rechecker.USER_TEMPLATE)
        self.assertIn("- 6步以上", rechecker.USER_TEMPLATE)
        self.assertNotIn("9步以上", rechecker.USER_TEMPLATE)
        self.assertEqual(rechecker.parse_step_count('{"step_count":"6步以上"}'), "6步以上")
        self.assertEqual(rechecker.parse_step_count('分析完成。\n{"step_count": "3-5步"}'), "3-5步")
        self.assertIsNone(rechecker.parse_step_count('{"step_count":"9步以上"}'))
        self.assertIsNone(rechecker.parse_step_count('{"step_count":"12步以上"}'))
        self.assertIsNone(rechecker.parse_step_count('{"step_count":"10步"}'))
        self.assertIsNone(rechecker.parse_step_count('我认为是 6-8 步'))

    def test_dry_run_refuses_label_bearing_reviewer_input(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "input.jsonl"
            source.write_text(json.dumps({
                "id": "q1", "text": "【题干】示例", "teacher_features": {"step_count": "6步以上"},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            result = __import__("subprocess").run([
                sys.executable, str(SCRIPT), "--input", str(source),
                "--raw-votes-output", str(directory / "votes.jsonl"),
                "--manifest", str(directory / "manifest.json"),
                "--model-path", "/model", "--dry-run",
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not label-free", result.stderr)


if __name__ == "__main__":
    unittest.main()
