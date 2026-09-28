"""Small offline checks for ECIR-only statistics; no training or downloads."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from experiments.ecir.baselines import training_statistics
from experiments.ecir.report import record_for
from src.models.sasrec import SASRecDualEncoder


class ECIRTests(unittest.TestCase):
    def test_sasrec_avoids_mps_unsupported_nested_tensor_path(self):
        model = SASRecDualEncoder(
            input_dim=4, hidden_dim=4, num_layers=1, num_heads=1,
            max_seq_len=3, dropout=0.)
        self.assertFalse(model.transformer.enable_nested_tensor)
        output = model(torch.randn(2, 3, 4), torch.tensor([3, 2]))
        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertTrue(torch.isfinite(output).all())

    def test_statistics_reconstruct_each_training_sequence_once(self):
        data = SimpleNamespace(
            ids=list(range(4)),
            splits={
                "train": [
                    {"group": "a", "context": [0], "target": 1},
                    {"group": "a", "context": [0, 1], "target": 2},
                    {"group": "b", "context": [1], "target": 3},
                ],
                "validation": [
                    {"group": "a", "context": [0, 1, 2], "target": 3},
                ],
                "test": [
                    {"group": "b", "context": [1, 3], "target": 0},
                ],
            })
        popularity, transitions = training_statistics(data)
        np.testing.assert_array_equal(popularity, [1, 2, 1, 1])
        self.assertEqual(transitions[0][1], 1)
        self.assertEqual(transitions[1][2], 1)
        self.assertEqual(transitions[1][3], 1)
        self.assertNotIn(0, transitions[3])

    def test_training_and_inference_dispersion_remain_separate(self):
        scalar = np.asarray([[[[0.]], [[.2]]],
                             [[[.4]], [[.6]]]])
        values = np.repeat(scalar, 4, axis=3)
        record = record_for("synthetic", "full", values)
        self.assertAlmostEqual(record["NDCG@10 mean"], .3)
        self.assertAlmostEqual(record["NDCG@10 train_sd"], np.sqrt(.08))
        self.assertAlmostEqual(record["NDCG@10 inference_sd"], np.sqrt(.02))
        self.assertEqual(record["training_seeds"], 2)
        self.assertEqual(record["inference_seeds"], 2)


if __name__ == "__main__":
    unittest.main()
