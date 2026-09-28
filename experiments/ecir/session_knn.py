"""Validation-selected recency-weighted vector session-kNN baseline.

This is a true session-neighbor method: it retrieves historical training
sessions that overlap the active session, then scores items occurring in the
nearest sessions.  It never uses item embeddings, dialogue, or held-out
labels.  The implementation is a fully specified V-SKNN-style variant rather
than the first-order transition baseline used in ``baselines.py``.
"""
import argparse
import heapq
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.revision.protocol import (METRICS, aggregate, digest, dump,
                                           metrics_from_scores, read_jsonl,
                                           verify_manifest, write_jsonl)
from experiments.revision.run import configure_runtime, environment


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root / "experiments/revision/protocol.py",
             root / "experiments/revision/run.py"]
    return {str(path.relative_to(root)): digest(path) for path in paths}


def validate_config(config):
    section = config.get("session_knn", {})
    neighbors = section.get("neighbor_counts", [])
    decays = section.get("recency_decays", [])
    if (not neighbors or any(not isinstance(value, int) or value <= 0
                             for value in neighbors) or
            len(set(neighbors)) != len(neighbors)):
        raise ValueError("neighbor_counts must contain unique positive integers")
    if (not decays or any(not 0 < float(value) <= 1 for value in decays) or
            len(set(map(float, decays))) != len(decays)):
        raise ValueError("recency_decays must contain unique values in (0, 1]")
    if not isinstance(section.get("max_context"), int) or section["max_context"] <= 0:
        raise ValueError("max_context must be a positive integer")
    if not 0 <= float(section.get("popularity_backoff", -1)) < 1e-6:
        raise ValueError("popularity_backoff must be nonnegative and below 1e-6")
    return section


class SessionData:
    """Prepared split loader that deliberately does not load item embeddings."""
    def __init__(self, folder, split_names):
        self.folder = Path(folder)
        self.manifest = verify_manifest(self.folder)
        catalog = json.loads((self.folder / "catalog.json").read_text())
        self.ids = catalog["ids"]
        self.index = {item: position for position, item in enumerate(self.ids)}
        self.splits = {}
        for split in split_names:
            self.splits[split] = [
                dict(row,
                     context=[self.index[item] for item in row["context"]],
                     target=self.index[row["target"]],
                     excluded=[self.index[item] for item in row["excluded"]])
                for row in read_jsonl(self.folder / f"{split}.jsonl")
            ]


def reconstruct_training_sessions(rows):
    """Recover each group's longest observed training prefix, including target."""
    longest = {}
    for row in rows:
        sequence = tuple(row["context"] + [row["target"]])
        previous = longest.get(row["group"])
        if previous is None or len(sequence) > len(previous):
            longest[row["group"]] = sequence
        elif len(sequence) == len(previous) and sequence != previous:
            raise ValueError(f"Conflicting training prefixes for group {row['group']}")
    if not longest:
        raise ValueError("Cannot build session-kNN from an empty training split")
    return [(group, longest[group]) for group in sorted(longest)]


class SessionIndex:
    """Inverted historical-session index and deterministic V-SKNN scorer."""
    def __init__(self, sessions, catalog_size, popularity_backoff,
                 session_groups=None):
        self.sessions = [tuple(session) for session in sessions]
        self.session_groups = (list(session_groups) if session_groups is not None
                               else [None] * len(self.sessions))
        if len(self.session_groups) != len(self.sessions):
            raise ValueError("Every historical session must have one group ID")
        self.positions = []
        self.item_arrays = []
        self.age_arrays = []
        postings = defaultdict(list)
        counts = Counter()
        for session_id, session in enumerate(self.sessions):
            latest = {}
            for position, item in enumerate(session):
                latest[item] = position
                counts[item] += 1
            self.positions.append(latest)
            items = np.asarray(sorted(latest), dtype=np.int64)
            self.item_arrays.append(items)
            self.age_arrays.append(np.asarray(
                [len(session) - 1 - latest[item] for item in items],
                dtype=np.int64))
            for item in items:
                postings[item].append(session_id)
        self.postings = dict(postings)
        popularity = np.zeros(catalog_size, dtype=np.float64)
        for item, count in counts.items():
            popularity[item] = count
        if popularity.max() > 0:
            popularity /= popularity.max()
        self.backoff = float(popularity_backoff) * popularity

    @staticmethod
    def current_weights(context, decay):
        """Latest occurrence wins; recent active-session items have larger weight."""
        latest = {}
        length = len(context)
        for position, item in enumerate(context):
            latest[item] = float(decay) ** (length - 1 - position)
        return latest

    def neighbors(self, context, decay, limit, excluded_group=None):
        weights = self.current_weights(context, decay)
        query_norm = math.sqrt(sum(value * value for value in weights.values()))
        candidates = set()
        for item in weights:
            candidates.update(self.postings.get(item, ()))
        similarities = []
        for session_id in candidates:
            if (excluded_group is not None and
                    self.session_groups[session_id] == excluded_group):
                continue
            positions = self.positions[session_id]
            numerator = sum(weight for item, weight in weights.items()
                            if item in positions)
            denominator = query_norm * math.sqrt(len(positions))
            if numerator > 0 and denominator > 0:
                similarities.append((session_id, numerator / denominator))
        # The secondary session-id key makes all similarity ties reproducible.
        return heapq.nsmallest(
            min(limit, len(similarities)), similarities,
            key=lambda pair: (-pair[1], pair[0]))

    def score_checkpoints(self, context, decay, neighbor_counts,
                          excluded_group=None):
        requested = sorted(set(neighbor_counts))
        neighbors = self.neighbors(context, decay, requested[-1],
                                   excluded_group)
        scores = self.backoff.copy()
        outputs = {}
        next_index = 0
        for rank, (session_id, similarity) in enumerate(neighbors, start=1):
            items = self.item_arrays[session_id]
            scores[items] += similarity * np.power(
                float(decay), self.age_arrays[session_id])
            while next_index < len(requested) and requested[next_index] == rank:
                outputs[requested[next_index]] = scores.copy()
                next_index += 1
        # If fewer than K neighbors exist, larger K values share the final scores.
        while next_index < len(requested):
            outputs[requested[next_index]] = scores.copy()
            next_index += 1
        return outputs


def evaluate_grid(data, split, index, section):
    neighbors = section["neighbor_counts"]
    collected = {(int(k), float(decay)): [] for decay in section["recency_decays"]
                 for k in neighbors}
    rows = data.splits[split]
    for position, row in enumerate(rows, start=1):
        context = row["context"][-section["max_context"]:]
        for decay in section["recency_decays"]:
            scores = index.score_checkpoints(context, float(decay), neighbors,
                                              row["group"])
            for k in neighbors:
                collected[(int(k), float(decay))].append({
                    "event": row["event"], "group": row["group"],
                    **metrics_from_scores(scores[int(k)], row["target"],
                                          row["excluded"]),
                })
        if position % 250 == 0 or position == len(rows):
            print(f"V-SKNN {split} grid: {position}/{len(rows)} events",
                  flush=True)
    return collected


def evaluate_selected(data, split, index, section, selection):
    result = []
    k = int(selection["neighbor_count"])
    decay = float(selection["recency_decay"])
    rows = data.splits[split]
    for position, row in enumerate(rows, start=1):
        context = row["context"][-section["max_context"]:]
        scores = index.score_checkpoints(context, decay, [k], row["group"])[k]
        result.append({"event": row["event"], "group": row["group"],
                       **metrics_from_scores(scores, row["target"],
                                             row["excluded"])})
        if position % 250 == 0 or position == len(rows):
            print(f"V-SKNN {split}: {position}/{len(rows)} events", flush=True)
    return result


def build_index(data, section):
    grouped = reconstruct_training_sessions(data.splits["train"])
    groups = [group for group, _ in grouped]
    sessions = [session for _, session in grouped]
    return SessionIndex(sessions, len(data.ids), section["popularity_backoff"],
                        groups)


def fit(args):
    config = json.loads(args.config.read_text())
    section = validate_config(config)
    data = SessionData(args.data, ("train", "validation"))
    index = build_index(data, section)
    grid = evaluate_grid(data, "validation", index, section)
    candidates = []
    chosen = None
    # Config order is the declared deterministic tie-break order.
    for decay in section["recency_decays"]:
        for k in section["neighbor_counts"]:
            metrics = aggregate(grid[(int(k), float(decay))])
            candidate = {"neighbor_count": int(k),
                         "recency_decay": float(decay),
                         "validation": metrics}
            candidates.append(candidate)
            if chosen is None or metrics["NDCG@10"] > chosen["validation"]["NDCG@10"]:
                chosen = candidate
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out / "config.json", config)
    dump(args.out / "selection.json", {
        "method": "vsknn", "selection_metric": "validation NDCG@10",
        "selected": chosen, "candidates": candidates,
        "historical_sessions": len(index.sessions),
    })
    files = ["config.json", "selection.json"]
    dump(args.out / "manifest.json", {
        "schema": 1, "experiment": "ecir-session-knn",
        "dataset": data.manifest["dataset"], "config": config,
        "data_manifest_sha256": digest(args.data / "manifest.json"),
        "source_sha256": source_hashes(), "environment": environment(),
        "device": "cpu", "selection_metric": "validation NDCG@10; never test",
        "files": {name: digest(args.out / name) for name in files},
        "command": sys.argv,
    })
    print(f"Frozen V-SKNN selection: {args.out / 'selection.json'}. Test was not read.")


def test(args):
    manifest = verify_manifest(args.run)
    if manifest.get("experiment") != "ecir-session-knn":
        raise ValueError("Run folder is not an ECIR session-kNN fit")
    if source_hashes() != manifest["source_sha256"]:
        raise ValueError("Session-kNN source changed since validation selection")
    if digest(args.data / "manifest.json") != manifest["data_manifest_sha256"]:
        raise ValueError("Prepared data changed since validation selection")
    runtime = environment()
    for key in ("determinism", "mps_cpu_fallback"):
        expected = manifest["environment"].get(key, runtime[key])
        if runtime[key] != expected:
            raise ValueError(f"Session-kNN test runtime {key} differs from fit")
    data = SessionData(args.data, ("train", "test"))
    section = validate_config(manifest["config"])
    selection = json.loads((args.run / "selection.json").read_text())["selected"]
    index = build_index(data, section)
    rows = evaluate_selected(data, "test", index, section, selection)
    args.out.mkdir(parents=True, exist_ok=False)
    name = "vsknn.jsonl"
    write_jsonl(args.out / name, rows)
    summary = [{"dataset": data.manifest["dataset"], "method": "vsknn",
                "seed": None, "n": len(rows), "file": name,
                **aggregate(rows)}]
    dump(args.out / "summary.json", summary)
    files = [name, "summary.json"]
    dump(args.out / "manifest.json", {
        "schema": 1, "experiment": "ecir-session-knn",
        "dataset": data.manifest["dataset"], "config": manifest["config"],
        "selection": selection,
        "data_manifest_sha256": digest(args.data / "manifest.json"),
        "run_manifest_sha256": digest(args.run / "manifest.json"),
        "source_sha256": source_hashes(), "environment": environment(),
        "device": "cpu", "files": {name: digest(args.out / name) for name in files},
        "command": sys.argv,
    })
    print(f"Frozen V-SKNN test results saved: {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("fit", "test"):
        part = subparsers.add_parser(command)
        part.add_argument("--data", type=Path, required=True)
        part.add_argument("--out", type=Path, required=True)
        part.add_argument("--cpu-threads", type=int,
                          default=min(4, os.cpu_count() or 1))
        part.add_argument("--interop-threads", type=int, default=1)
        part.add_argument("--determinism", choices=("strict", "warn"),
                          default="strict")
        part.set_defaults(device="cpu")
        if command == "fit":
            part.add_argument("--config", type=Path,
                              default=Path("experiments/ecir/additional_config.json"))
        else:
            part.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; choose a new versioned path")
    try:
        configure_runtime(args)
        (fit if args.command == "fit" else test)(args)
    except (KeyError, ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
