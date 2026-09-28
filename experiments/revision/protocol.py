"""Shared, single-target, full-catalog evaluation and auditable file utilities."""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

METHODS = ("uniform", "weighted_prf", "learned_centroid", "weighted_fixed",
           "learned_fixed", "full")
LABELS = {
    "uniform": "Uniform centroid (reference)", "weighted_prf": "Weighted PRF",
    "learned_centroid": "Learned centroid, no sampling",
    "weighted_fixed": "Weighted PRF, fixed spread",
    "learned_fixed": "Learned centroid, fixed spread", "full": "Full method",
}
METRICS = ("NDCG@10", "Recall@10", "Recall@50", "MRR@50")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts):
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8], 16)


def dump(path, value):
    with open(path, "x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + "\n")


def verify_manifest(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"Artifact changed: {folder / name}")
    return manifest


def metrics_from_scores(scores, target, excluded):
    """Exact rank; tied scores break by ascending canonical catalog index."""
    scores = np.asarray(scores, dtype=np.float32).copy()
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Expected finite one-dimensional full-catalog scores")
    excluded = np.asarray(sorted(set(excluded)), dtype=np.int64)
    if target in excluded:
        raise ValueError("Repeated target is ineligible under unseen-item protocol")
    scores[excluded] = -np.inf
    indices = np.arange(len(scores))
    rank = 1 + int(np.sum((scores > scores[target]) |
                         ((scores == scores[target]) & (indices < target))))
    return {"rank": rank, "NDCG@10": 1 / math.log2(rank + 1) if rank <= 10 else 0.,
            "Recall@10": float(rank <= 10), "Recall@50": float(rank <= 50),
            "MRR@50": 1 / rank if rank <= 50 else 0.}


def aggregate(rows):
    if not rows:
        raise ValueError("Empty evaluation support")
    return {key: float(np.mean([r[key] for r in rows])) for key in METRICS}


class ItemBM25:
    """Positive-IDF BM25, k1=1.5, b=.75; weights only observed context items.

    Query is concatenated metadata of all context items (including each item's
    own text), preserving repeated query terms. No dialogue or target text.
    Corpus statistics use frozen catalog metadata, not interaction labels.
    """
    def __init__(self, texts):
        self.docs = [Counter(re.findall(r"\w+", t.lower())) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = max(float(np.mean(self.lengths)), 1.)
        df = Counter(t for d in self.docs for t in d)
        self.idf = {t: math.log(1 + (len(texts) - n + .5) / (n + .5))
                    for t, n in df.items()}

    def weights(self, context):
        query = sum((self.docs[i] for i in context), Counter())
        values = []
        for i in context:
            norm = 1.5 * (.25 + .75 * self.lengths[i] / self.avg)
            values.append(sum(qf * self.idf[t] * tf * 2.5 / (tf + norm)
                              for t, tf in self.docs[i].items()
                              if (qf := query[t])))
        weights = np.asarray(values, dtype=np.float32)
        return weights / weights.sum() if weights.sum() > 0 else np.ones(len(context), dtype=np.float32) / len(context)


def check_splits(splits):
    """Event-level separation, not globally disjoint item identities."""
    event_sets = []
    positions = {}
    for name in ("train", "validation", "test"):
        rows = splits[name]
        events = [r["event"] for r in rows]
        if len(events) != len(set(events)):
            raise ValueError(f"Duplicate prediction event in {name}")
        for r in rows:
            if not r["context"] or r["target"] in r["excluded"]:
                raise ValueError(f"Ineligible example: {r['event']}")
            if not set(r["context"]).issubset(r["excluded"]):
                raise ValueError("Context must be excluded at retrieval")
            group, position = json.loads(r["event"])
            if group != r["group"] or position != len(r["context"]):
                raise ValueError("Event position must match its full, unfiltered prefix")
            positions.setdefault(group, {}).setdefault(name, []).append(position)
        event_sets.append(set(events))
    if any(event_sets[i] & event_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Prediction-event leakage across splits")
    for group, parts in positions.items():
        for earlier, later in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if earlier in parts and later in parts and max(parts[earlier]) >= min(parts[later]):
                raise ValueError(f"Held-out event could enter earlier-split context: {group}")
