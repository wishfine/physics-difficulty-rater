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
    connectivity_preserving_bootstrap_sample,
    connectivity_preserving_folds,
    cross_validate_bradley_terry,
    fit_bradley_terry,
    graph_connectivity_risks,
    pair_graph_integrity,
    run_negative_controls,
    summarize_residual_slices,
)
from physics_difficulty.pairwise.metrics import soft_pairwise_metrics


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
    def test_weighted_metrics_do_not_treat_low_weight_error_equally(self):
        predictions = [0.9, 0.9]
        targets = [1.0, 0.0]
        unweighted = soft_pairwise_metrics(predictions, targets)
        weighted = soft_pairwise_metrics(predictions, targets, weights=[1.0, 0.01])
        self.assertLess(
            weighted["soft_pairwise_log_loss"],
            unweighted["soft_pairwise_log_loss"],
        )
        self.assertLess(weighted["brier_score"], unweighted["brier_score"])
        self.assertEqual(weighted["weight_sum"], 1.01)

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

    def test_connectivity_preserving_bootstrap_keeps_every_node_connected(self):
        rows = synthetic_pairs()
        for seed in range(10):
            sampled = connectivity_preserving_bootstrap_sample(rows, seed=seed)
            metrics = pair_graph_integrity(
                sampled,
                expected_question_ids={"q0", "q1", "q2", "q3"},
            )
            self.assertEqual(metrics["connected_components"], 1)
            self.assertEqual(metrics["missing_expected_nodes"], 0)
            self.assertEqual(len(sampled), len(rows))

    def test_bootstrap_uses_the_same_fixed_backbone_across_runs(self):
        rows = synthetic_pairs()
        first = connectivity_preserving_bootstrap_sample(
            rows, seed=10, backbone_seed=42
        )
        second = connectivity_preserving_bootstrap_sample(
            rows, seed=11, backbone_seed=42
        )
        first_backbone = {
            row["bootstrap_source_pair_id"]
            for row in first
            if row["bootstrap_is_backbone"]
        }
        second_backbone = {
            row["bootstrap_source_pair_id"]
            for row in second
            if row["bootstrap_is_backbone"]
        }
        self.assertEqual(first_backbone, second_backbone)
        self.assertEqual(len(first_backbone), 3)

    def test_bootstrap_reports_per_question_score_and_rank_stability(self):
        from physics_difficulty.pairwise.offline_bt import bootstrap_rank_stability

        rows = synthetic_pairs()
        fitted = fit_bradley_terry(rows, max_iterations=800, learning_rate=0.1)
        stability = bootstrap_rank_stability(
            rows,
            fitted["scores"],
            runs=5,
            max_iterations=500,
            learning_rate=0.1,
            seed=23,
        )
        node = stability["node_stability"]["q0"]
        self.assertIn("score_ci95_low", node)
        self.assertIn("score_ci95_high", node)
        self.assertIn("rank_standard_deviation", node)
        self.assertIn("bottom_10_percent_frequency", node)

    def test_integrity_detects_reverse_duplicate_and_missing_expected_node(self):
        rows = synthetic_pairs()
        rows.append(
            {
                **rows[0],
                "pair_id": "reverse-copy",
                "question_a_id": rows[0]["question_b_id"],
                "question_b_id": rows[0]["question_a_id"],
                "soft_target": 1.0 - rows[0]["soft_target"],
            }
        )
        integrity = pair_graph_integrity(
            rows,
            expected_question_ids={"q0", "q1", "q2", "q3", "q-missing"},
        )
        self.assertEqual(integrity["duplicate_undirected_edges"], 1)
        self.assertEqual(integrity["missing_expected_nodes"], 1)
        self.assertEqual(integrity["node_coverage"], 0.8)

    def test_graph_risk_audit_finds_bridge_and_articulation_node(self):
        rows = synthetic_pairs()[:3]
        rows = [
            {
                "pair_id": "p0",
                "question_a_id": "q0",
                "question_b_id": "q1",
                "soft_target": 0.5,
                "sample_weight": 1.0,
            },
            {
                "pair_id": "p1",
                "question_a_id": "q1",
                "question_b_id": "q2",
                "soft_target": 0.5,
                "sample_weight": 1.0,
            },
            {
                "pair_id": "p2",
                "question_a_id": "q2",
                "question_b_id": "q0",
                "soft_target": 0.5,
                "sample_weight": 1.0,
            },
            {
                "pair_id": "p3",
                "question_a_id": "q2",
                "question_b_id": "q3",
                "soft_target": 0.5,
                "sample_weight": 1.0,
            },
        ]
        risks = graph_connectivity_risks(rows)
        self.assertEqual(risks["bridge_edge_count"], 1)
        self.assertEqual(risks["bridge_edges"], [["q2", "q3"]])
        self.assertEqual(risks["articulation_nodes"], ["q2"])

    def test_residual_slices_report_each_pair_source(self):
        rows = synthetic_pairs()
        for index, row in enumerate(rows):
            row["pair_source"] = "near" if index % 2 == 0 else "bridge"
            row["label_source"] = "nonthinking" if index < 3 else "thinking_1024"
            row["cascade_route"] = {"reason": "stable" if index < 3 else "uncertain"}
            row["reliability"] = {"status": "high" if index < 3 else "medium"}
            row["vote_stats"] = {"position_bias_gap": 0.0 if index < 3 else 0.4}
            row["metadata"] = {"feature_hamming_distance": index}
        fitted = fit_bradley_terry(rows, max_iterations=800, learning_rate=0.1)
        from physics_difficulty.pairwise.offline_bt import residual_rows

        residuals = residual_rows(rows, fitted["scores"])
        slices = summarize_residual_slices(residuals, severe_threshold=0.5)
        self.assertEqual(set(slices["pair_source"]), {"near", "bridge"})
        self.assertEqual(
            sum(item["records"] for item in slices["pair_source"].values()),
            len(rows),
        )
        self.assertEqual(set(slices["route_reason"]), {"stable", "uncertain"})
        self.assertIn("feature_distance_bucket", slices)

    def test_cross_validation_beats_constant_probability_baseline(self):
        rows = synthetic_pairs()
        for index, row in enumerate(rows):
            row["pair_source"] = "near" if index % 2 == 0 else "bridge"
            row["label_source"] = "nonthinking" if index < 3 else "thinking_1024"
            row["cascade_route"] = {"reason": "stable" if index < 3 else "uncertain"}
            row["reliability"] = {"status": "high" if index < 3 else "medium"}
            row["vote_stats"] = {"position_bias_gap": 0.0 if index < 3 else 0.4}
            row["metadata"] = {"feature_hamming_distance": index}
        report = cross_validate_bradley_terry(
            rows,
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
        self.assertIn("heldout_weighted_metrics", report)
        self.assertIn("constant_baseline_weighted_metrics", report)
        self.assertEqual(
            set(report["heldout_slice_metrics"]["pair_source"]),
            {"near", "bridge"},
        )
        self.assertEqual(
            set(report["heldout_slice_metrics"]["route_reason"]),
            {"stable", "uncertain"},
        )
        self.assertEqual(
            set(report["heldout_slice_metrics"]["position_bias_bucket"]),
            {"none_or_low", "high"},
        )

    def test_negative_controls_preserve_records_and_report_degradation(self):
        controls = run_negative_controls(
            synthetic_pairs(),
            folds=3,
            max_iterations=500,
            learning_rate=0.1,
            l2=1e-4,
            seed=19,
        )
        self.assertEqual(
            set(controls),
            {"shuffled_soft_targets", "flipped_direction_10_percent"},
        )
        for report in controls.values():
            self.assertEqual(report["records"], 6)
            self.assertEqual(report["cross_validation"]["heldout_pairs"], 3)

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
                    "--negative-controls",
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
            score = json.loads(scores_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("degree", score)
            self.assertIn("weighted_information", score)
            self.assertIn("rank", score)
            self.assertIn("residual_slices", report)
            self.assertEqual(
                set(report["negative_controls"]),
                {"shuffled_soft_targets", "flipped_direction_10_percent"},
            )
            self.assertTrue(
                report["quality_gate_checks"][
                    "heldout_weighted_log_loss_beats_constant"
                ]
            )

    def test_cli_expected_questions_turns_missing_node_into_review(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "pairs.jsonl"
            questions_path = directory / "questions.jsonl"
            report_path = directory / "report.json"
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in synthetic_pairs()),
                encoding="utf-8",
            )
            questions_path.write_text(
                "".join(
                    json.dumps({"question_id": question_id}) + "\n"
                    for question_id in ["q0", "q1", "q2", "q3", "q-missing"]
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_pairwise_with_bt.py"),
                    "--input",
                    str(input_path),
                    "--questions",
                    str(questions_path),
                    "--report",
                    str(report_path),
                    "--scores-output",
                    str(directory / "scores.jsonl"),
                    "--residuals-output",
                    str(directory / "residuals.jsonl"),
                    "--folds",
                    "3",
                    "--bootstrap-runs",
                    "0",
                    "--max-iterations",
                    "800",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "REVIEW")
            self.assertEqual(report["graph_integrity"]["missing_expected_nodes"], 1)
            self.assertFalse(report["quality_gate_checks"]["node_coverage"])

    def test_cli_writes_preflight_error_report_for_duplicate_edge(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rows = synthetic_pairs()
            rows.append(
                {
                    **rows[0],
                    "pair_id": "reverse-copy",
                    "question_a_id": rows[0]["question_b_id"],
                    "question_b_id": rows[0]["question_a_id"],
                    "soft_target": 1.0 - rows[0]["soft_target"],
                }
            )
            input_path = directory / "pairs.jsonl"
            report_path = directory / "report.json"
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
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
                    str(directory / "scores.jsonl"),
                    "--residuals-output",
                    str(directory / "residuals.jsonl"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ERROR")
            self.assertIn("duplicate_undirected_edges", report["fatal_integrity_errors"])


if __name__ == "__main__":
    unittest.main()
