import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.offline_bt import (
    connectivity_preserving_folds,
    cross_validate_bradley_terry,
    fit_bradley_terry,
)


def synthetic_pairs() -> list[dict]:
    scores = {"q0": -1.5, "q1": -0.5, "q2": 0.4, "q3": 1.6}
    rows = []
    index = 0
    for left_index, left in enumerate(scores):
        for right in list(scores)[left_index + 1 :]:
            probability = 1.0 / (1.0 + math.exp(-(scores[left] - scores[right])))
            rows.append(
                {
                    "pair_id": f"p{index}",
                    "question_a_id": left,
                    "question_b_id": right,
                    "soft_target": probability,
                    "sample_weight": 1.0,
                }
            )
            index += 1
    return rows


class OfflineBradleyTerryAuditTests(unittest.TestCase):
    def test_fit_recovers_global_score_order(self):
        result = fit_bradley_terry(synthetic_pairs(), max_iterations=1500, learning_rate=0.1, seed=7)
        scores = result["scores"]
        self.assertLess(scores["q0"], scores["q1"])
        self.assertLess(scores["q1"], scores["q2"])
        self.assertLess(scores["q2"], scores["q3"])
        self.assertLess(result["final_log_loss"], 0.55)

    def test_connectivity_preserving_folds_keep_backbone_in_every_fit(self):
        rows = synthetic_pairs()
        assignment = connectivity_preserving_folds(rows, folds=3, seed=42)
        self.assertEqual(set(assignment), {row["pair_id"] for row in rows})
        self.assertGreater(sum(value == -1 for value in assignment.values()), 0)
        for fold in range(3):
            fit_rows = [row for row in rows if assignment[row["pair_id"]] != fold]
            nodes = {row[key] for row in fit_rows for key in ("question_a_id", "question_b_id")}
            adjacency = {node: set() for node in nodes}
            for row in fit_rows:
                adjacency[row["question_a_id"]].add(row["question_b_id"])
                adjacency[row["question_b_id"]].add(row["question_a_id"])
            seen = {next(iter(nodes))}
            frontier = list(seen)
            while frontier:
                node = frontier.pop()
                for neighbor in adjacency[node] - seen:
                    seen.add(neighbor)
                    frontier.append(neighbor)
            self.assertEqual(seen, nodes)

    def test_cross_validation_beats_constant_probability_baseline(self):
        report = cross_validate_bradley_terry(
            synthetic_pairs(),
            folds=3,
            max_iterations=1000,
            learning_rate=0.1,
            seed=11,
        )
        self.assertEqual(report["completed_folds"], 3)
        self.assertLess(
            report["heldout_metrics"]["soft_pairwise_log_loss"],
            report["constant_baseline_metrics"]["soft_pairwise_log_loss"],
        )

    def test_cli_writes_report_scores_and_residuals(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "pairs.jsonl"
            report_path = directory / "report.json"
            scores_path = directory / "scores.jsonl"
            residuals_path = directory / "residuals.jsonl"
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in synthetic_pairs()),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_pairwise_with_bt.py"),
                    "--input",
                    str(input_path),
                    "--report",
                    str(report_path),
                    "--scores-output",
                    str(scores_path),
                    "--residuals-output",
                    str(residuals_path),
                    "--folds",
                    "3",
                    "--bootstrap-runs",
                    "3",
                    "--max-iterations",
                    "800",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], "offline_bt_pair_audit_v1")
            self.assertEqual(report["status"], "PASS")
            self.assertTrue(
                report["quality_gate_checks"]["heldout_log_loss_beats_constant"]
            )
            self.assertEqual(report["records"], 6)
            self.assertEqual(report["questions"], 4)
            self.assertEqual(len(scores_path.read_text(encoding="utf-8").splitlines()), 4)
            self.assertEqual(len(residuals_path.read_text(encoding="utf-8").splitlines()), 6)


if __name__ == "__main__":
    unittest.main()
