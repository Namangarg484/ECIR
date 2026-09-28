"""Optional synthetic fit/test integration check. No downloads or external inputs."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .protocol import METHODS, digest, dump, verify_manifest, write_jsonl
from .run import fit, test


class RunnerTests(unittest.TestCase):
    def test_synthetic_six_method_handoff(self):
        with tempfile.TemporaryDirectory(prefix="vce-revision-test-") as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            ids = [str(i) for i in range(8)]
            dump(data / "catalog.json", {"ids": ids, "texts": ["item " + i for i in ids]})
            np.save(data / "embeddings.npy", np.eye(8, dtype=np.float32), allow_pickle=False)
            def row(group, context, target):
                return {"event": json.dumps([group, len(context)]), "group": group,
                        "context": context, "excluded": context, "target": target}
            write_jsonl(data / "train.jsonl", [row("a", ["0"], "1"), row("b", ["2"], "3"), row("c", ["4"], "5")])
            write_jsonl(data / "validation.jsonl", [row("v", ["0", "2"], "4")])
            write_jsonl(data / "test.jsonl", [row("t", ["1", "3"], "5")])
            names = ["catalog.json", "embeddings.npy", "train.jsonl", "validation.jsonl", "test.jsonl"]
            dump(data / "manifest.json", {"dataset": "synthetic", "files": {n: digest(data / n) for n in names}})
            config = root / "config.json"
            dump(config, {"seeds": [42], "selection_seed": 42, "epochs": 1,
                          "batch_size": 2, "learning_rates": [.0001], "fixed_kappas": [50.],
                          "weight_decay": .01, "patience": 1, "max_context": 2,
                          "negative_count": 2, "samples": 2, "temperature": .07})
            fit(SimpleNamespace(data=data, config=config, out=root / "run", device="cpu"))
            self.assertFalse((root / "results").exists())
            test(SimpleNamespace(data=data, run=root / "run", out=root / "results", device="cpu"))
            verify_manifest(root / "results")
            summary = json.loads((root / "results/summary.json").read_text())
            self.assertEqual({r["method"] for r in summary}, set(METHODS))
            self.assertTrue(all(r["n"] == 1 for r in summary))


if __name__ == "__main__":
    unittest.main()
