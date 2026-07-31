import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_v3_pair_construction.py"


def question(question_id, *, length="short", subquestions=False, image_risk="low"):
    return {
        "id": question_id, "split": "train", "text": f"题目 {question_id}",
        "diagnostics": {
            "input_length_bucket": length, "has_subquestions": subquestions,
            "subquestion_count": 2 if subquestions else 0, "has_analysis": True,
            "has_options": False, "image_dependency_risk": image_risk,
        },
    }


class AuditV3PairConstructionTests(unittest.TestCase):
    def test_audit_reports_question_coverage_and_pair_graph_gate(self):
        pool = [question("q1"), question("q2", length="medium"), question("q3", subquestions=True), question("q4", image_risk="high")]
        selected = pool[:3]
        pairs = [
            {"pair_id": "p1", "question_a_id": "q1", "question_b_id": "q2", "pair_source": "lexical_near", "metadata": {"lexical_jaccard": 0.1}},
            {"pair_id": "p2", "question_a_id": "q2", "question_b_id": "q3", "pair_source": "graph_bridge", "metadata": {"lexical_jaccard": 0.0}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            paths = {name: directory / f"{name}.jsonl" for name in ("pool", "selected", "pairs")}
            for name, rows in (("pool", pool), ("selected", selected), ("pairs", pairs)):
                paths[name].write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            report = directory / "report.json"
            subprocess.run([
                sys.executable, str(SCRIPT), "--pool-questions", str(paths["pool"]),
                "--selected-questions", str(paths["selected"]), "--pairs", str(paths["pairs"]),
                "--output", str(report), "--minimum-degree", "1", "--maximum-degree", "2",
                "--minimum-selected-per-stratum", "1", "--minimum-pool-per-stratum", "1",
            ], check=True, capture_output=True, text=True)
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "WARN")
            self.assertEqual(result["question_selection"]["pool_questions"], 4)
            self.assertEqual(result["question_selection"]["selected_questions"], 3)
            self.assertEqual(result["pair_construction"]["graph"]["connected_components"], 1)
            self.assertEqual(result["pair_construction"]["integrity"]["unknown_question_endpoints"], 0)


if __name__ == "__main__":
    unittest.main()
