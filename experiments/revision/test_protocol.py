"""Offline unit tests; invoke this module explicitly, not repository-wide discovery."""
import unittest
from collections import Counter

import numpy as np

from .prepare import make_example
from .protocol import ItemBM25, check_splits, metrics_from_scores, stable_seed
from .run import negatives


class ProtocolTests(unittest.TestCase):
    def test_exclusion_and_ties(self):
        result = metrics_from_scores([1., .5, .5, -.2], 2, [0])
        self.assertEqual(result["rank"], 2)
        self.assertAlmostEqual(result["NDCG@10"], 1 / np.log2(3))

    def test_excluded_minus_one_cannot_reenter(self):
        self.assertEqual(metrics_from_scores([-1., -1.], 1, [0])["rank"], 1)

    def test_repeat_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics_from_scores([.1, .2], 1, [1])

    def test_mrr_cutoff(self):
        result = metrics_from_scores(np.arange(60), 0, [])
        self.assertEqual(result["rank"], 60)
        self.assertEqual(result["MRR@50"], 0.)

    def test_no_target_shift(self):
        counts = Counter()
        self.assertIsNone(make_example("u", ["a", "b", "missing"], 2, {"a", "b"}, counts))
        self.assertEqual(counts["missing_target_embedding"], 1)

    def test_missing_context_does_not_shorten_query(self):
        self.assertIsNone(make_example("u", ["missing", "b"], 1, {"b"}, Counter()))

    def test_leave_two_out_prefixes(self):
        seq = list("abcde")
        catalog = set(seq)
        train = [make_example("u", seq, i, catalog, Counter()) for i in range(1, len(seq) - 2)]
        validation = [make_example("u", seq, len(seq) - 2, catalog, Counter())]
        test = [make_example("u", seq, len(seq) - 1, catalog, Counter())]
        check_splits(dict(train=train, validation=validation, test=test))
        self.assertFalse(set("de") & {i for r in train for i in r["context"] + [r["target"]]})
        self.assertEqual(test[0]["context"], list("abcd"))

    def test_event_overlap_fails(self):
        row = make_example("u", ["a", "b"], 1, {"a", "b"}, Counter())
        with self.assertRaises(ValueError):
            check_splits(dict(train=[row], validation=[row], test=[]))

    def test_future_training_prefix_fails(self):
        seq = list("abcd")
        train = make_example("u", seq, 3, set(seq), Counter())
        validation = make_example("u", seq, 2, set(seq), Counter())
        with self.assertRaises(ValueError):
            check_splits(dict(train=[train], validation=[validation], test=[]))

    def test_bm25_empty_fallback(self):
        np.testing.assert_allclose(ItemBM25(["", ""]).weights([0, 1]), [.5, .5])

    def test_negative_exclusions(self):
        row = {"excluded": [0, 1], "target": 2, "event": "u"}
        self.assertEqual(set(negatives(np.arange(4), row, 64, np.random.default_rng(7))), {3})

    def test_stable_seed(self):
        self.assertEqual(stable_seed(42, "test", "u"), stable_seed(42, "test", "u"))
        self.assertNotEqual(stable_seed(42, "test", "u"), stable_seed(42, "test", "v"))


if __name__ == "__main__":
    unittest.main()
