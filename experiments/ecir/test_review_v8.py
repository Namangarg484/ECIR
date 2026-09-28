"""Small synthetic checks; run explicitly, never as part of fitting."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from experiments.ecir.matched_radius_selection import choose_fixed
from experiments.ecir.probe_redundancy import (
    COUNTS, evaluate_cell, geometry_metrics, overlap_metrics, top_items,
)
from experiments.ecir.review_common import cell, prepare_output


class MatchedSelectionTests(unittest.TestCase):
    def test_tie_uses_declared_grid_order(self):
        candidates = [{"kappa": 10., "mean_validation_ndcg10": .2},
                      {"kappa": 500., "mean_validation_ndcg10": .2}]
        self.assertEqual(choose_fixed(candidates)["kappa"], 10.)

    def test_selection_uses_validation_only(self):
        candidates = [{"kappa": 10., "mean_validation_ndcg10": .1, "test": .9},
                      {"kappa": 500., "mean_validation_ndcg10": .2, "test": 0.}]
        self.assertEqual(choose_fixed(candidates)["kappa"], 500.)

    def test_reject_nonfinite_selection(self):
        with self.assertRaises(ValueError):
            choose_fixed([{"kappa": 10., "mean_validation_ndcg10": float("nan")}])


class ProbeTests(unittest.TestCase):
    def test_topk_ties_and_history_exclusions(self):
        scores = np.array([.4, .7, .7, .7, .2], dtype=np.float32)
        self.assertEqual(top_items(scores, [1], 2).tolist(), [2, 3])
        self.assertEqual(top_items(scores, [1], 10).tolist(), [2, 3, 0, 4])

    def test_partial_topk_matches_full_lexsort(self):
        rng = np.random.default_rng(3)
        scores = rng.integers(0, 6, 100).astype(np.float32)
        excluded = [1, 4, 6, 12]
        legal = np.array([i for i in range(100) if i not in excluded])
        expected = legal[np.lexsort((legal, -scores[legal]))][:10]
        np.testing.assert_array_equal(top_items(scores, excluded, 10), expected)

    def test_union_coverage_is_not_merged_recall(self):
        stats = overlap_metrics([[1, 2], [2, 3], [3, 4]], 4)
        self.assertEqual(stats["unique_candidates"], 4)
        self.assertEqual(stats["new_candidates_last_probe"], 1)
        self.assertEqual(stats["additional_candidates_vs_first"], 2)
        self.assertEqual(stats["target_gain_last_probe"], 1.)
        self.assertAlmostEqual(stats["pairwise_topk_jaccard"], 2 / 9)

    def test_identical_probes(self):
        stats = overlap_metrics([[1, 2], [1, 2]], 2)
        self.assertEqual(stats["pairwise_topk_jaccard"], 1.)
        self.assertEqual(stats["target_gain_vs_first"], 0.)
        self.assertEqual(stats["new_candidates_last_probe"], 0)
        geometry = geometry_metrics(np.array([[1., 0.], [1., 0.]]))
        self.assertAlmostEqual(geometry["pairwise_angle_degrees"], 0.)

    def test_single_probe_pairwise_statistics_undefined(self):
        self.assertIsNone(overlap_metrics([[1]], 1)["pairwise_topk_jaccard"])
        self.assertIsNone(geometry_metrics(np.array([[1., 0.]]))["pairwise_cosine"])

    def test_orthogonal_geometry(self):
        geometry = geometry_metrics(np.eye(2))
        self.assertAlmostEqual(geometry["pairwise_angle_degrees"], 90.)

    def test_inference_cell_reproduces_rank_and_redundancy(self):
        data = SimpleNamespace(
            emb=torch.tensor([[1., 0.], [.5, .5], [0., 1.], [1., 0.]]),
            splits={"test": [{"event": "e", "group": "g", "target": 0, "excluded": [3]}]})
        def identical_directions(data, rows, method, model, params, config, seed, split):
            return torch.tensor([1., 0.]).repeat(len(rows), config["samples"], 1)
        with patch("experiments.ecir.probe_redundancy.directions", identical_directions), \
                patch("experiments.ecir.probe_redundancy.seed_all"):
            records = evaluate_cell(data, Mock(), "full", {}, {"batch_size": 2},
                                    3101, {s: 1. for s in COUNTS})
        self.assertEqual(len(records), 10)
        last = next(r for r in records if r["samples"] == 20 and r["cutoff"] == 10)
        self.assertEqual(last["merged_ndcg10"], 1.)
        self.assertEqual(last["unique_candidates"], 3.)
        self.assertEqual(last["pairwise_topk_jaccard"], 1.)
        self.assertEqual(last["new_candidates_last_probe"], 0.)


class ResumeTests(unittest.TestCase):
    def test_verified_resume_and_changed_spec_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            prepare_output(folder, {"version": 1})
            self.assertEqual(cell(folder, "one", lambda: {"value": 1}), {"value": 1})
            def unexpected_compute():
                self.fail("A completed cell must not rerun")
            self.assertEqual(cell(folder, "one", unexpected_compute), {"value": 1})
            with self.assertRaises(ValueError):
                prepare_output(folder, {"version": 2})

    def test_tampered_cache_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            prepare_output(folder, {})
            cell(folder, "one", lambda: {"value": 1})
            path = folder / "cells/one.json"
            saved = json.loads(path.read_text())
            saved["payload"]["value"] = 2
            path.write_text(json.dumps(saved))
            with self.assertRaises(ValueError):
                cell(folder, "one", lambda: {})


if __name__ == "__main__":
    unittest.main()
