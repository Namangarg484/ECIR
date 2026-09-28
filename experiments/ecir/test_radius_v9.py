"""Synthetic CPU tests for selection, epoch recovery, optimizer state, and pairing."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from experiments.ecir import frozen_radius as frozen
from experiments.ecir import radius_v9 as experiment
from experiments.ecir import radius_v9_storage as storage
from experiments.revision.protocol import METRICS, aggregate, digest
from src.models.vce_model import VCEModel


class SelectionTests(unittest.TestCase):
    def test_global_and_per_center_can_differ(self):
        selected = experiment.choose_fixed([10., 500.], [
            [[.9, .1], [.9, .1]], [[.2, 1.], [.2, 1.]]])
        self.assertEqual(selected["per_center_kappas"], [10., 500.])
        self.assertEqual(selected["global_kappa"], 10.)  # Exact global tie: first.

    def test_mean_of_all_noise_seeds_not_first(self):
        matrix = np.zeros((1, 10, 2))
        matrix[0, 0] = [1., 0.]
        matrix[0, 1:, 1] = .5
        self.assertEqual(experiment.choose_fixed([10., 500.], matrix)["global_kappa"], 500.)

    def test_bad_grid_shape_and_nonfinite_scores(self):
        with self.assertRaises(ValueError):
            experiment.choose_fixed([10., 500.], [[.1, .2]])
        with self.assertRaises(ValueError):
            experiment.choose_fixed([10.], [[[float("nan")]]])

    def test_epoch_ties_keep_earlier_checkpoint(self):
        self.assertEqual(experiment.update_best(.5, 3, .5, 1), (.5, 1))


class StorageTests(unittest.TestCase):
    def test_state_roundtrip_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "state"
            storage.state_save(folder, {"head": torch.tensor([1., 2.]), "epoch": 1})
            torch.testing.assert_close(storage.state_load(folder)["head"], torch.tensor([1., 2.]))
            with self.assertRaises(ValueError):
                storage.state_save(folder, {})
            (folder / "state.pt").write_bytes(b"damaged")
            with self.assertRaises(ValueError):
                storage.state_load(folder)

    def test_cell_resume_and_spec_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            storage.prepare(folder, {"grid": [10., 500.]})
            storage.cell(folder / "cell.json", lambda: {"score": .2})
            self.assertEqual(storage.cell(folder / "cell.json", lambda: self.fail("reran")), {"score": .2})
            with self.assertRaises(ValueError):
                storage.prepare(folder, {"grid": [10.]})

    def test_text_resume_and_manifest_exclude_partial_files(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            storage.prepare(folder, {"version": 9})
            storage.text_write(folder / "report.csv", "a,b\n1,2\n")
            storage.text_write(folder / "report.csv", "a,b\n1,2\n")
            (folder / ".partial-test").write_text("incomplete")
            storage.finish(folder, {"version": 9})
            self.assertTrue(storage.complete(folder, {"version": 9}))
            files = json.loads((folder / "manifest.json").read_text())["files"]
            self.assertNotIn(".partial-test", files)


class ToyData:
    def __init__(self):
        generator = torch.Generator().manual_seed(5)
        self.emb = F.normalize(torch.randn(4, 8, generator=generator), dim=-1)
        rows = [{"event": "a", "group": "a", "context": [0], "target": 2, "excluded": [0]},
                {"event": "b", "group": "b", "context": [1], "target": 3, "excluded": [1]}]
        self.splits = {"train": rows, "validation": rows}

    def batch(self, rows, max_context):
        contexts = [r["context"] for r in rows]
        anchors = self.emb[torch.tensor(contexts)]
        return contexts, anchors, torch.zeros((len(rows), 1), dtype=torch.bool), anchors[:, 0]


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.data = ToyData()
        self.revision = {"batch_size": 2, "max_context": 10, "negative_count": 2,
                         "samples": 5, "temperature": .07, "weight_decay": .01}

    def test_optimizer_resume_matches_uninterrupted_cpu_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "center.pt"
            torch.manual_seed(7)
            torch.save(VCEModel(embed_dim=8).state_dict(), checkpoint)
            first = frozen.load_model(self.data, checkpoint, "cpu", "smooth")
            optimizer = torch.optim.AdamW(list(first.radius_parameters()), lr=.0001, weight_decay=.01)
            center_digest = first.center_digest()
            experiment.train_epoch(self.data, first, optimizer, self.revision, 42, 1)
            saved = {**frozen.head_checkpoint(first, checkpoint, "smooth"),
                     "optimizer_state": optimizer.state_dict(), "training_seed": 42, "epoch": 1}
            storage.state_save(root / "epoch", saved)
            second = frozen.load_model(self.data, checkpoint, "cpu", "smooth")
            second_optimizer = torch.optim.AdamW(list(second.radius_parameters()), lr=.0001, weight_decay=.01)
            experiment.restore(second, second_optimizer, storage.state_load(root / "epoch"),
                               checkpoint, "smooth", 42, 1)
            experiment.train_epoch(self.data, first, optimizer, self.revision, 42, 2)
            experiment.train_epoch(self.data, second, second_optimizer, self.revision, 42, 2)
            for a, b in zip(first.radius_parameters(), second.radius_parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(first.center_digest(), center_digest)
            self.assertEqual(second.center_digest(), center_digest)

    def test_ten_seed_epoch_selection_early_stopping_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "center.pt"
            torch.save(VCEModel(embed_dim=8).state_dict(), checkpoint)
            args = SimpleNamespace(out=root / "run", device="cpu")
            config = {"epochs": 5, "patience": 1, "radius_head_learning_rate": .0001,
                      "selection_inference_seeds": list(range(3101, 3111))}
            state = {"epoch": 0}
            def fake_train(data, model, optimizer, revision, seed, epoch):
                state["epoch"] = epoch
                return {"training_loss": .1}
            def fake_validation(data, split, hard, smooth, kappas, revision, noise_seed, **kwargs):
                self.assertEqual(split, "validation")
                epoch = state["epoch"]
                value = (.9 if noise_seed == 3101 else .1) if epoch == 1 else (.3 if epoch == 2 else .2)
                return {"adaptive_smooth": [{metric: value for metric in METRICS}]}
            with patch.object(experiment, "train_epoch", side_effect=fake_train) as train, \
                    patch.object(frozen, "evaluate_expanded_suite", side_effect=fake_validation) as evaluate, \
                    patch.object(frozen, "kappa_statistics", return_value={"count": 2}):
                selected = experiment.fit_head(args, self.data, checkpoint, config, self.revision, 42, "smooth")
                self.assertEqual(selected["best_epoch"], 2)
                self.assertEqual(selected["epochs_completed"], 3)
                self.assertEqual(train.call_count, 3)
                self.assertEqual(evaluate.call_count, 30)
            with patch.object(experiment, "train_epoch", side_effect=AssertionError("retrained")), \
                    patch.object(frozen, "evaluate_expanded_suite", side_effect=AssertionError("reevaluated")):
                self.assertEqual(experiment.fit_head(args, self.data, checkpoint, config, self.revision, 42, "smooth"),
                                 selected)


class PairedReportTests(unittest.TestCase):
    def test_method_support_and_center_replication(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            rows = [{"event": "a", "group": "g", **{m: 1. for m in METRICS}}]
            summary, files = [], {}
            for seed in (42, 123):
                for noise in (None, 3101, 3102):
                    methods = ["no_expansion"] if noise is None else list(experiment.METHODS[1:])
                    name = f"{seed}-{noise}.json"
                    storage.cell(folder / name, lambda: {method: rows for method in methods})
                    files[name] = digest(folder / name)
                    for method in methods:
                        summary.append({"method": method, "train_seed": seed, "inference_seed": noise,
                                        "file": name, "n": 1, **aggregate(rows)})
            values, support = experiment.arrays_from_results(folder, {"files": files}, summary,
                                                             [42, 123], [3101, 3102])
            self.assertEqual(values["no_expansion"].shape, (2, 2, 1, 4))
            self.assertEqual(support, [("a", "g")])
            with self.assertRaises(ValueError):
                experiment.arrays_from_results(folder, {"files": files}, summary + [summary[0]],
                                               [42, 123], [3101, 3102])


if __name__ == "__main__":
    unittest.main()
