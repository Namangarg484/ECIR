"""Small deterministic unit tests for the ECIR add-on experiment helpers."""
import unittest

import numpy as np

from experiments.ecir.kappa_diagnostics import rankdata, spearman
from experiments.ecir.query_direction_sensitivity import hierarchical_interval
from experiments.ecir.session_knn import (SessionIndex,
                                           reconstruct_training_sessions)


class SessionKNNTests(unittest.TestCase):
    def test_reconstructs_longest_training_prefix(self):
        rows = [
            {"group": "b", "context": [8], "target": 9},
            {"group": "a", "context": [1], "target": 2},
            {"group": "a", "context": [1, 2], "target": 3},
        ]
        self.assertEqual(reconstruct_training_sessions(rows),
                         [("a", (1, 2, 3)), ("b", (8, 9))])

    def test_vector_neighbors_and_prefix_scores(self):
        index = SessionIndex([(0, 1, 2), (0, 3), (4, 5)], 6, 0.0)
        neighbors = index.neighbors([0, 1], decay=1.0, limit=2)
        self.assertEqual([session for session, _ in neighbors], [0, 1])
        scores = index.score_checkpoints([0, 1], 1.0, [1, 2])
        self.assertGreater(scores[1][2], 0.0)
        self.assertEqual(scores[1][3], 0.0)
        self.assertGreater(scores[2][3], 0.0)

    def test_active_group_is_not_its_own_neighbor(self):
        index = SessionIndex([(0, 1), (0, 2)], 3, 0.0,
                             session_groups=["active", "other"])
        neighbors = index.neighbors([0, 1], 1.0, 2,
                                    excluded_group="active")
        self.assertEqual([session for session, _ in neighbors], [1])

    def test_recent_duplicate_wins(self):
        weights = SessionIndex.current_weights([1, 2, 1], 0.5)
        self.assertEqual(weights, {1: 1.0, 2: 0.5})


class DiagnosticMathTests(unittest.TestCase):
    def test_average_tie_ranks(self):
        np.testing.assert_allclose(rankdata([10, 20, 20, 30]),
                                   [1.0, 2.5, 2.5, 4.0])

    def test_spearman_extremes_and_constant(self):
        self.assertAlmostEqual(spearman([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [6, 4, 2]), -1.0)
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))

    def test_hierarchical_interval_constant_difference(self):
        differences = np.full((3, 4, 5), 0.25)
        mean, low, high = hierarchical_interval(
            differences, ["a", "a", "b", "c", "c"], 100, 2026)
        self.assertAlmostEqual(mean, 0.25)
        self.assertAlmostEqual(low, 0.25)
        self.assertAlmostEqual(high, 0.25)

    def test_hierarchical_interval_checks_query_alignment(self):
        with self.assertRaises(ValueError):
            hierarchical_interval(np.zeros((2, 2, 3)), ["a", "b"], 100, 1)


if __name__ == "__main__":
    unittest.main()
