import json
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.behavior_accuracy import (
    behavior_pair_probability,
    beta_posterior_summary,
    recover_correct_count,
    score_behavior_row,
    spearman_correlation,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class BehaviorAccuracyUnitTests(unittest.TestCase):
    def test_recovers_integer_count_from_two_decimal_percentage(self):
        result = recover_correct_count(148, Decimal("81.08"))
        self.assertEqual(result["correct_count"], 120)
        self.assertEqual(result["recovery_status"], "rounded_exact_unique")
        self.assertEqual(result["matching_integer_count_candidate_count"], 1)
        self.assertLess(result["absolute_percent_reconstruction_error"], 0.002)

    def test_higher_correct_rate_produces_lower_difficulty(self):
        easy = beta_posterior_summary(100, 90)
        hard = beta_posterior_summary(100, 20)
        self.assertLess(easy["behavior_difficulty_score"], hard["behavior_difficulty_score"])

    def test_unrecoverable_rate_is_retained_as_continuous_pseudocount(self):
        result = score_behavior_row(
            {
                "question_id": "partial-credit",
                "stem": "可部分得分题",
                "answered_count": "21",
                "percent_correct": "50.00",
                "sub_questions": [],
            }
        )
        self.assertEqual(result["behavior_evidence_type"], "continuous_rate_pseudocount")
        self.assertEqual(result["correct_count"], None)
        self.assertEqual(result["effective_correct_count"], 10.5)
        self.assertEqual(result["behavior_evidence_quality"], 0.5)

    def test_pair_probability_follows_harder_direction(self):
        hard = score_behavior_row(
            {
                "question_id": "hard",
                "stem": "困难题",
                "answered_count": "100",
                "percent_correct": "20.00",
                "sub_questions": [],
            }
        )
        easy = score_behavior_row(
            {
                "question_id": "easy",
                "stem": "简单题",
                "answered_count": "100",
                "percent_correct": "90.00",
                "sub_questions": [],
            }
        )
        self.assertGreater(behavior_pair_probability(hard, easy), 0.99)
        self.assertLess(behavior_pair_probability(easy, hard), 0.01)

    def test_spearman_handles_rank_order_and_ties(self):
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [30, 20, 10]), -1.0)


class BehaviorAccuracyCliTests(unittest.TestCase):
    def test_clean_and_compare_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.jsonl"
            base_rows = [
                {
                    "parent_id": "q_easy",
                    "question_id": "q_easy",
                    "stem": "直接判断",
                    "options": "A. 是\nB. 否",
                    "analysis": "答案 A",
                    "structure_type": "danxuan",
                    "answered_count": "148",
                    "percent_correct": "81.08",
                    "difficulty": "2",
                    "sub_questions": [],
                },
                {
                    "parent_id": "q_hard",
                    "question_id": "q_hard",
                    "stem": "综合计算",
                    "structure_type": "jisuan",
                    "answered_count": "100",
                    "percent_correct": "20.00",
                    "difficulty": "5",
                    "sub_questions": [],
                },
            ]
            conflicting = {
                "question_id": "q_conflict",
                "stem": "题一",
                "answered_count": "50",
                "percent_correct": "40.00",
                "difficulty": "1",
                "sub_questions": [],
            }
            write_jsonl(
                raw,
                base_rows
                + [base_rows[0], conflicting, {**conflicting, "percent_correct": "60.00"}]
                + [
                    {
                        "question_id": "too_few",
                        "stem": "样本不足",
                        "answered_count": "20",
                        "percent_correct": "50.00",
                        "sub_questions": [],
                    }
                ],
            )
            scores = root / "scores.jsonl"
            quarantine = root / "quarantine.jsonl"
            clean_report = root / "clean_report.json"
            clean = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/audit_behavior_accuracy.py"),
                    "--input",
                    str(raw),
                    "--scores-output",
                    str(scores),
                    "--quarantine-output",
                    str(quarantine),
                    "--report",
                    str(clean_report),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(clean.returncode, 0, clean.stderr)
            clean_result = json.loads(clean_report.read_text(encoding="utf-8"))
            self.assertEqual(clean_result["records"]["accepted_unique_questions"], 2)
            self.assertEqual(clean_result["records"]["exact_duplicate_rows"], 1)
            self.assertEqual(clean_result["records"]["conflicting_duplicate_ids"], 1)
            self.assertFalse(clean_result["data_contract"]["source_difficulty_used"])

            bt = root / "bt.jsonl"
            pairs = root / "pairs.jsonl"
            write_jsonl(
                bt,
                [
                    {"question_id": "q_easy", "bt_score": -1.0, "rank": 1, "degree": 4},
                    {"question_id": "q_hard", "bt_score": 1.0, "rank": 2, "degree": 4},
                ],
            )
            write_jsonl(
                pairs,
                [
                    {
                        "pair_id": "agree",
                        "question_a_id": "q_hard",
                        "question_b_id": "q_easy",
                        "soft_target": 0.9,
                        "sample_weight": 1.0,
                        "label_source": "thinking_1024",
                        "metadata": {"pair_source": "feature_contrast"},
                    },
                    {
                        "pair_id": "conflict",
                        "question_a_id": "q_easy",
                        "question_b_id": "q_hard",
                        "soft_target": 0.9,
                        "sample_weight": 0.8,
                        "label_source": "nonthinking",
                        "metadata": {"pair_source": "feature_near"},
                    },
                ],
            )
            report = root / "comparison.json"
            question_evidence = root / "questions.jsonl"
            pair_evidence = root / "pair_evidence.jsonl"
            conflicts = root / "conflicts.jsonl"
            compare = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/compare_behavior_with_bt.py"),
                    "--behavior-scores",
                    str(scores),
                    "--bt-scores",
                    str(bt),
                    "--teacher-pairs",
                    str(pairs),
                    "--report",
                    str(report),
                    "--question-evidence-output",
                    str(question_evidence),
                    "--pair-evidence-output",
                    str(pair_evidence),
                    "--conflicts-output",
                    str(conflicts),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(compare.returncode, 0, compare.stderr)
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(result["coverage"]["question_overlap"], 2)
            self.assertEqual(result["coverage"]["teacher_pairs_with_both_behavior_endpoints"], 2)
            self.assertAlmostEqual(result["question_level_consistency"]["spearman"], 1.0)
            self.assertEqual(result["high_confidence_audit"]["severe_conflicts"], 1)
            conflict_rows = [
                json.loads(line)
                for line in conflicts.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(conflict_rows[0]["pair_id"], "conflict")


if __name__ == "__main__":
    unittest.main()
