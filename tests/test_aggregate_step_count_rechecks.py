import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aggregate_step_count_rechecks.py"


class AggregateStepCountRecheckTests(unittest.TestCase):
    def test_only_consensus_rechecked_values_become_overrides(self):
        audit_rows = [
            {"question_id": "q1", "original_step_count": "6-8步"},
            {"question_id": "q2", "original_step_count": "3-5步"},
        ]
        votes = [
            *[{"question_id": "q1", "valid": True, "parsed_step_count": value} for value in ("9步以上", "9步以上", "6-8步")],
            *[{"question_id": "q2", "valid": True, "parsed_step_count": value} for value in ("6-8步", "3-5步", "9步以上")],
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            audit, raw_votes = directory / "audit.jsonl", directory / "votes.jsonl"
            results, overrides, manifest = directory / "results.jsonl", directory / "overrides.jsonl", directory / "manifest.json"
            audit.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_rows), encoding="utf-8")
            raw_votes.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in votes), encoding="utf-8")
            subprocess.run([
                sys.executable, str(SCRIPT), "--raw-votes", str(raw_votes), "--selection-audit", str(audit),
                "--results-output", str(results), "--overrides-output", str(overrides), "--manifest", str(manifest),
                "--minimum-valid-votes", "3", "--minimum-winner-votes", "2",
            ], check=True, capture_output=True, text=True)
            rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["rechecked_step_count"], "9步以上")
            self.assertEqual(rows[0]["action"], "apply")
            self.assertEqual(rows[1]["action"], "abstain")
            override_rows = [json.loads(line) for line in overrides.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(override_rows, [{"question_id": "q1", "step_count": "9步以上"}])


if __name__ == "__main__":
    unittest.main()
