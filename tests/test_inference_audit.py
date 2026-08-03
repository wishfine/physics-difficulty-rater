from __future__ import annotations

import unittest

from physics_difficulty.pairwise.inference_audit import (
    bootstrap_thresholds,
    migration_summary,
    pearson,
    spearman,
    thresholds,
    top_bottom_overlap,
)


class InferenceAuditTests(unittest.TestCase):
    def test_correlations_and_overlap(self):
        left = [1.0, 2.0, 3.0, 4.0]
        right = [10.0, 20.0, 30.0, 40.0]
        self.assertAlmostEqual(pearson(left, right), 1.0)
        self.assertAlmostEqual(spearman(left, right), 1.0)
        overlap = top_bottom_overlap(left, right, 0.25)
        self.assertEqual(overlap["top_overlap"], 1.0)
        self.assertEqual(overlap["bottom_overlap"], 1.0)

    def test_thresholds_and_migration(self):
        values = list(range(10))
        boundaries = thresholds(values, (0.2, 0.2, 0.3, 0.2, 0.1))
        self.assertEqual(boundaries, [1.8, 3.6, 6.3, 8.1])
        summary = migration_summary(values, boundaries, boundaries)
        self.assertEqual(summary["agreement"], 1.0)
        self.assertEqual(summary["changed_records"], 0)

    def test_bootstrap_is_repeatable(self):
        values = list(range(20))
        first = bootstrap_thresholds(values, (0.2, 0.2, 0.3, 0.2, 0.1), repetitions=5, seed=7)
        second = bootstrap_thresholds(values, (0.2, 0.2, 0.3, 0.2, 0.1), repetitions=5, seed=7)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
