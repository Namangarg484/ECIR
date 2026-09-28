"""Small protocol checks for the new read-only reviewer diagnostics."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from experiments.ecir.target_support import summarise_method, support_partition
from experiments.ecir.validation_probe_redundancy import COUNTS, evaluate_cell
from experiments.revision.protocol import METRICS


class SupportTests(unittest.TestCase):
    def test_context_item_counts_as_warm_without_held_out_leakage(self):
        train = [{"context": ["seen-in-context"], "target": "seen-as-target"}]
        test = [
            {"event": "a", "target": "seen-in-context", "excluded": []},
            {"event": "b", "target": "unseen", "excluded": []},
        ]
        support, observed = support_partition(train, test)
        self.assertEqual(support, {"a": True, "b": False})
        self.assertNotIn("unseen", observed)

    def test_saved_rankings_partition_without_changing_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rows = [
                {"event": "a", "group": "g1", **{key: 1. for key in METRICS}},
                {"event": "b", "group": "g2", **{key: 0. for key in METRICS}},
            ]
            path = folder / "popularity.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            (folder / "summary.json").write_text(json.dumps(
                [{"method": "popularity", "file": path.name}]))
            result = summarise_method(
                folder, {"files": {path.name: "verified"}}, "popularity",
                {"a": True, "b": False}, {"a": "g1", "b": "g2"})
            self.assertEqual([row["events"] for row in result], [1, 1, 2])
            self.assertEqual([row["NDCG@10"] for row in result], [1., 0., .5])


class ValidationProbeTests(unittest.TestCase):
    def test_validation_directions_and_nested_counts(self):
        data = SimpleNamespace(
            emb=torch.tensor([[1., 0.], [0., 1.], [.5, .5]]),
            splits={"validation": [{"event": "v", "group": "g", "target": 0,
                                     "excluded": [2]}]})
        def directions_for_validation(data, rows, method, model, params,
                                      config, seed, split):
            self.assertEqual(split, "validation")
            return torch.tensor([1., 0.]).repeat(len(rows), config["samples"], 1)
        with patch("experiments.ecir.validation_probe_redundancy.directions",
                   directions_for_validation), \
                patch("experiments.ecir.validation_probe_redundancy.seed_all"):
            rows = evaluate_cell(data, Mock(), "full", {}, {"batch_size": 2}, 3101)
        self.assertEqual([row["samples"] for row in rows], list(COUNTS))
        self.assertTrue(all(row["merged_ndcg10"] == 1. for row in rows))
        self.assertIsNone(rows[0]["pairwise_topk_jaccard"])
        self.assertEqual(rows[-1]["pairwise_topk_jaccard"], 1.)


if __name__ == "__main__":
    unittest.main()
