import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.cascade import (
    CascadeThresholds,
    decide_cascade_route,
    evaluate_cascade,
    evaluate_human_audit,
    merge_vote_rows,
    select_blind_audit_pairs,
    select_representative_audit_pairs,
    select_stratified_pairs,
    split_pairs_balanced,
)


def pair(index: int, source: str = "random_global", left_bucket: str = "short", right_bucket: str = "short", length: int = 20):
    return {
        "pair_id": f"p{index:03d}",
        "question_a_id": f"a{index}",
        "question_b_id": f"b{index}",
        "question_a_text": "甲" * length,
        "question_b_text": "乙" * (length // 2),
        "pair_source": source,
        "metadata": {"length_bucket_a": left_bucket, "length_bucket_b": right_bucket},
    }


def vote(pair_row, direction: str, sample_index: int, winner: str, *, valid: bool = True, round_index: int = 1):
    return {
        "pair_id": pair_row["pair_id"],
        "question_a_id": pair_row["question_a_id"],
        "question_b_id": pair_row["question_b_id"],
        "direction": direction,
        "sample_index": sample_index,
        "sampling_round": round_index,
        "winner_question_id": winner if valid else None,
        "valid": valid,
        "output_token_count": 2,
        "teacher": {"mode": "test"},
    }


def unanimous_votes(pair_row, winner_key: str):
    winner = pair_row[winner_key]
    return [
        vote(pair_row, direction, index, winner)
        for direction in ("forward", "backward")
        for index in range(3)
    ]


class CascadeTeacherTests(unittest.TestCase):
    def test_stratified_selection_is_deterministic_excludes_prior_pairs_and_covers_strata(self):
        rows = [pair(i, "random_global" if i < 8 else "lexical_near", "short", "short" if i % 2 else "medium") for i in range(16)]
        first, first_stats = select_stratified_pairs(rows, sample_size=8, seed=42, excluded_pair_ids={"p000", "p009"})
        second, second_stats = select_stratified_pairs(list(reversed(rows)), sample_size=8, seed=42, excluded_pair_ids={"p000", "p009"})
        self.assertEqual([row["pair_id"] for row in first], [row["pair_id"] for row in second])
        self.assertEqual(first_stats, second_stats)
        self.assertEqual(len(first), 8)
        self.assertNotIn("p000", {row["pair_id"] for row in first})
        self.assertNotIn("p009", {row["pair_id"] for row in first})
        self.assertGreaterEqual(len(first_stats["selected_by_stratum"]), 2)

    def test_stratified_selection_excludes_every_pair_touching_a_prior_question(self):
        rows = [pair(i) for i in range(4)]
        selected, stats = select_stratified_pairs(
            rows,
            sample_size=3,
            seed=42,
            excluded_question_ids={"a1"},
        )
        self.assertNotIn("p001", {row["pair_id"] for row in selected})
        self.assertEqual(stats["excluded_by_question_id"], 1)

    def test_balanced_split_preserves_every_pair_once_and_balances_text_load(self):
        rows = [pair(i, length=(i + 1) * 20) for i in range(10)]
        shards, stats = split_pairs_balanced(rows, shard_count=2, seed=7)
        flattened = [row["pair_id"] for shard in shards for row in shard]
        self.assertEqual(sorted(flattened), sorted(row["pair_id"] for row in rows))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertLessEqual(abs(stats["shards"][0]["text_characters"] - stats["shards"][1]["text_characters"]), max(len(row["question_a_text"]) + len(row["question_b_text"]) for row in rows))

    def test_route_accepts_decisive_order_stable_votes_and_escalates_position_bias(self):
        row = pair(1)
        thresholds = CascadeThresholds(max_position_bias_gap=0.25, decisive_low=0.30, decisive_high=0.70, minimum_votes_per_direction=3)
        stable = decide_cascade_route(unanimous_votes(row, "question_a_id"), thresholds)
        self.assertEqual(stable["action"], "accept_nonthinking")
        self.assertAlmostEqual(stable["soft_target"], 0.875)

        biased = [
            *[vote(row, "forward", i, row["question_a_id"]) for i in range(3)],
            *[vote(row, "backward", i, row["question_b_id"]) for i in range(3)],
        ]
        decision = decide_cascade_route(biased, thresholds)
        self.assertEqual(decision["action"], "escalate_thinking_1024")
        self.assertEqual(decision["reason"], "position_sensitive_and_uncertain")

    def test_route_escalates_when_a_direction_lacks_three_valid_votes(self):
        row = pair(2)
        rows = unanimous_votes(row, "question_a_id")[:-1]
        decision = decide_cascade_route(rows, CascadeThresholds())
        self.assertEqual(decision["action"], "escalate_thinking_1024")
        self.assertEqual(decision["reason"], "insufficient_valid_votes")

    def test_evaluation_reports_direct_coverage_agreement_and_severe_disagreement(self):
        accepted = pair(1)
        escalated = pair(2)
        nonthinking = unanimous_votes(accepted, "question_a_id") + [
            *[vote(escalated, "forward", i, escalated["question_a_id"]) for i in range(3)],
            *[vote(escalated, "backward", i, escalated["question_b_id"]) for i in range(3)],
        ]
        thinking = unanimous_votes(accepted, "question_a_id") + unanimous_votes(escalated, "question_b_id")
        report, records = evaluate_cascade([accepted, escalated], nonthinking, thinking, CascadeThresholds())
        self.assertEqual(report["direct_accept_count"], 1)
        self.assertEqual(report["escalated_count"], 1)
        self.assertEqual(report["accepted_hard_agreement_with_thinking"], 1.0)
        self.assertEqual(report["severe_disagreement_count"], 0)
        self.assertEqual(report["accepted_severe_disagreement_count"], 0)
        self.assertEqual({record["pair_id"] for record in records}, {"p001", "p002"})

    def test_escalated_opposite_direction_is_not_counted_as_accepted_routing_failure(self):
        row = pair(5)
        nonthinking = [
            *[vote(row, "forward", i, row["question_a_id"]) for i in range(3)],
            vote(row, "backward", 0, row["question_a_id"]),
            *[vote(row, "backward", i + 1, row["question_b_id"]) for i in range(2)],
        ]
        thinking = unanimous_votes(row, "question_b_id")
        report, records = evaluate_cascade([row], nonthinking, thinking, CascadeThresholds())
        self.assertEqual(records[0]["route_action"], "escalate_thinking_1024")
        self.assertTrue(records[0]["severe_disagreement"])
        self.assertEqual(report["severe_disagreement_count"], 1)
        self.assertEqual(report["accepted_severe_disagreement_count"], 0)

    def test_merge_rejects_duplicate_vote_identity(self):
        row = pair(1)
        rows = unanimous_votes(row, "question_a_id")
        with self.assertRaisesRegex(ValueError, "duplicate vote identity"):
            merge_vote_rows([rows, [rows[0]]])

    def test_blind_audit_contains_no_teacher_predictions(self):
        rows = [pair(i) for i in range(4)]
        evaluation = [
            {"pair_id": "p000", "hard_disagreement": True, "nonthinking_position_bias_gap": 0.0, "thinking_position_bias_gap": 0.0, "thinking_soft_target": 0.9},
            {"pair_id": "p001", "hard_disagreement": False, "nonthinking_position_bias_gap": 0.8, "thinking_position_bias_gap": 0.0, "thinking_soft_target": 0.9},
            {"pair_id": "p002", "hard_disagreement": False, "nonthinking_position_bias_gap": 0.0, "thinking_position_bias_gap": 0.0, "thinking_soft_target": 0.5},
            {"pair_id": "p003", "hard_disagreement": False, "nonthinking_position_bias_gap": 0.0, "thinking_position_bias_gap": 0.0, "thinking_soft_target": 0.9},
        ]
        selected, manifest = select_blind_audit_pairs(rows, evaluation, sample_size=4, seed=42)
        self.assertEqual(len(selected), 4)
        self.assertEqual(manifest["selected"], 4)
        for item in selected:
            self.assertEqual(item["human_preference"], None)
            serialized = json.dumps(item, ensure_ascii=False)
            self.assertNotIn("soft_target", serialized)
            self.assertNotIn("position_bias", serialized)
            self.assertNotIn("hard_disagreement", serialized)

    def test_representative_audit_prefers_new_pairs_then_reuses_prior_labels_by_stratum(self):
        rows = [pair(i) for i in range(9)]
        evaluation = []
        for index in range(9):
            if index < 4:
                route_action, hard_disagreement = "accept_nonthinking", False
            elif index < 7:
                route_action, hard_disagreement = "escalate_thinking_1024", False
            else:
                route_action, hard_disagreement = "escalate_thinking_1024", True
            evaluation.append({
                "pair_id": f"p{index:03d}",
                "route_action": route_action,
                "hard_disagreement": hard_disagreement,
                "nonthinking_soft_target": 0.9,
                "thinking_soft_target": 0.1,
            })
        prior = [
            {"pair_id": "p001", "human_preference": "A", "human_confidence": "high", "human_notes": "old stable"},
            {"pair_id": "p005", "human_preference": "B", "human_confidence": "medium", "human_notes": "old agree"},
            {"pair_id": "p007", "human_preference": "tie", "human_confidence": "low", "human_notes": "old conflict"},
        ]

        blind, reused, manifest = select_representative_audit_pairs(
            rows,
            evaluation,
            prior,
            quotas={
                "stable_and_decisive": 3,
                "escalated_same_direction": 3,
                "escalated_teacher_disagreement": 2,
            },
            seed=42,
        )
        self.assertEqual(len(blind), 6)
        self.assertEqual(len(reused), 2)
        self.assertEqual(manifest["selected_total"], 8)
        self.assertEqual(manifest["new_total"], 6)
        self.assertEqual(manifest["reused_total"], 2)
        self.assertEqual(manifest["strata"]["stable_and_decisive"]["reused"], 0)
        self.assertEqual(manifest["strata"]["escalated_same_direction"]["reused"], 1)
        self.assertEqual(manifest["strata"]["escalated_teacher_disagreement"]["reused"], 1)
        selected_ids = {row["pair_id"] for row in blind + reused}
        self.assertEqual(len(selected_ids), 8)
        for item in blind:
            serialized = json.dumps(item, ensure_ascii=False)
            self.assertIsNone(item["human_preference"])
            self.assertNotIn("soft_target", serialized)
            self.assertNotIn("route_action", serialized)
            self.assertNotIn("hard_disagreement", serialized)

        blind_again, reused_again, manifest_again = select_representative_audit_pairs(
            list(reversed(rows)),
            list(reversed(evaluation)),
            list(reversed(prior)),
            quotas={
                "stable_and_decisive": 3,
                "escalated_same_direction": 3,
                "escalated_teacher_disagreement": 2,
            },
            seed=42,
        )
        self.assertEqual(blind, blind_again)
        self.assertEqual(reused, reused_again)
        self.assertEqual(manifest, manifest_again)

    def test_human_audit_evaluation_excludes_human_ties_from_directional_accuracy(self):
        evaluation = [
            {
                "pair_id": "p000", "route_action": "accept_nonthinking", "hard_disagreement": False,
                "nonthinking_soft_target": 0.9, "thinking_soft_target": 0.8, "final_soft_target": 0.9,
            },
            {
                "pair_id": "p001", "route_action": "escalate_thinking_1024", "hard_disagreement": True,
                "nonthinking_soft_target": 0.8, "thinking_soft_target": 0.2, "final_soft_target": 0.2,
            },
            {
                "pair_id": "p002", "route_action": "escalate_thinking_1024", "hard_disagreement": False,
                "nonthinking_soft_target": 0.5, "thinking_soft_target": 0.2, "final_soft_target": 0.2,
            },
        ]
        human = [
            {"pair_id": "p000", "human_preference": "A", "human_confidence": "high"},
            {"pair_id": "p001", "human_preference": "B", "human_confidence": "medium"},
            {"pair_id": "p002", "human_preference": "tie", "human_confidence": "low"},
        ]
        report = evaluate_human_audit(evaluation, human)
        self.assertEqual(report["human_records"], 3)
        self.assertEqual(report["human_non_tie_records"], 2)
        self.assertEqual(report["overall"]["nonthinking"]["correct"], 1)
        self.assertEqual(report["overall"]["nonthinking"]["directional_accuracy"], 0.5)
        self.assertEqual(report["overall"]["thinking_1024"]["directional_accuracy"], 1.0)
        self.assertEqual(report["overall"]["cascade_final"]["directional_accuracy"], 1.0)
        self.assertEqual(report["strata"]["stable_and_decisive"]["human_records"], 1)
        self.assertEqual(report["strata"]["escalated_teacher_disagreement"]["human_records"], 1)

    def test_human_audit_reports_stratum_coverage_and_population_weighted_estimate(self):
        evaluation = [
            {"pair_id": "p000", "route_action": "accept_nonthinking", "hard_disagreement": False,
             "nonthinking_soft_target": 0.9, "thinking_soft_target": 0.9, "final_soft_target": 0.9},
            {"pair_id": "p001", "route_action": "accept_nonthinking", "hard_disagreement": False,
             "nonthinking_soft_target": 0.9, "thinking_soft_target": 0.9, "final_soft_target": 0.9},
            {"pair_id": "p002", "route_action": "escalate_thinking_1024", "hard_disagreement": False,
             "nonthinking_soft_target": 0.2, "thinking_soft_target": 0.2, "final_soft_target": 0.2},
            {"pair_id": "p003", "route_action": "escalate_thinking_1024", "hard_disagreement": True,
             "nonthinking_soft_target": 0.9, "thinking_soft_target": 0.2, "final_soft_target": 0.2},
        ]
        human = [
            {"pair_id": "p000", "human_preference": "A", "human_confidence": "high"},
            {"pair_id": "p002", "human_preference": "B", "human_confidence": "high"},
            {"pair_id": "p003", "human_preference": "B", "human_confidence": "high"},
        ]
        report = evaluate_human_audit(evaluation, human)
        self.assertEqual(report["population_stratum_counts"]["stable_and_decisive"], 2)
        self.assertEqual(report["strata"]["stable_and_decisive"]["audit_coverage"], 0.5)
        self.assertEqual(report["population_weighted_directional_accuracy"]["nonthinking"], 0.75)
        self.assertEqual(report["population_weighted_directional_accuracy"]["cascade_final"], 1.0)

    def test_cascade_command_line_pipeline(self):
        rows = [pair(i, "random_global" if i % 2 else "lexical_near", "short", "medium" if i % 3 else "short") for i in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            candidates = directory / "candidates.jsonl"
            candidates.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            excluded = directory / "excluded.jsonl"
            excluded.write_text(
                json.dumps({"pair_id": "p000", "question_a_id": "unused-a", "question_b_id": "unused-b"}) + "\n"
                + json.dumps({"pair_id": "old-pair", "question_a_id": "a1", "question_b_id": "unused-c"}) + "\n",
                encoding="utf-8",
            )
            selected = directory / "selected.jsonl"
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "prepare_cascade_validation_pairs.py"),
                "--candidates", str(candidates), "--exclude-jsonl", str(excluded),
                "--output", str(selected), "--shard-dir", str(directory / "shards"),
                "--manifest", str(directory / "selection.manifest.json"),
                "--sample-size", "6", "--shard-count", "2", "--seed", "42",
            ], check=True, capture_output=True, text=True)
            selected_rows = [json.loads(line) for line in selected.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(selected_rows), 6)
            self.assertNotIn("p000", {row["pair_id"] for row in selected_rows})
            self.assertNotIn("p001", {row["pair_id"] for row in selected_rows})
            self.assertEqual(sum(1 for path in (directory / "shards").glob("shard-*.jsonl")), 2)

            nonthinking_rows = []
            thinking_shards = [[], []]
            for index, row in enumerate(selected_rows):
                nonthinking_rows.extend(unanimous_votes(row, "question_a_id"))
                thinking_shards[index % 2].extend(unanimous_votes(row, "question_a_id"))
            nonthinking_path = directory / "nonthinking.jsonl"
            nonthinking_path.write_text("".join(json.dumps(row) + "\n" for row in nonthinking_rows), encoding="utf-8")
            thinking_paths = []
            for index, shard_rows in enumerate(thinking_shards):
                path = directory / f"thinking-{index}.jsonl"
                path.write_text("".join(json.dumps(row) + "\n" for row in shard_rows), encoding="utf-8")
                thinking_paths.append(path)
            merged = directory / "thinking.merged.jsonl"
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "merge_teacher_vote_shards.py"),
                "--input", str(thinking_paths[0]), "--input", str(thinking_paths[1]),
                "--output", str(merged), "--manifest", str(directory / "merge.manifest.json"),
            ], check=True, capture_output=True, text=True)

            report = directory / "report.json"
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "evaluate_cascade_routing.py"),
                "--pairs", str(selected), "--nonthinking-votes", str(nonthinking_path),
                "--thinking-votes", str(merged), "--report", str(report),
                "--records-output", str(directory / "records.jsonl"),
                "--accepted-pairs-output", str(directory / "accepted.jsonl"),
                "--escalated-pairs-output", str(directory / "escalated.jsonl"),
                "--human-audit-output", str(directory / "audit.jsonl"),
                "--human-audit-manifest", str(directory / "audit.manifest.json"),
                "--human-audit-size", "4",
            ], check=True, capture_output=True, text=True)
            metrics = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(metrics["direct_accept_count"], 6)
            self.assertEqual(metrics["accepted_hard_agreement_with_thinking"], 1.0)
            self.assertEqual(metrics["acceptance_gate_status"], "PASS")
            audit_text = (directory / "audit.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("soft_target", audit_text)

    def test_server_cascade_wrapper_has_safe_usage_boundary(self):
        result = subprocess.run(
            ["bash", str(ROOT / "scripts" / "server_run_cascade_validation.sh")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("MODEL_PATH", result.stderr)
        self.assertIn("GPU_PAIR_1", result.stderr)


if __name__ == "__main__":
    unittest.main()
