"""Inference-free warm/cold target audit of the retained ECIR test rankings.

An item is warm if it occurs anywhere in the prepared training interactions,
including an observed context item. No validation or test interaction is used
to define support. This script never selects a method or changes a ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from experiments.revision.protocol import METRICS, digest, dump, read_jsonl, verify_manifest

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("talkplay", "ml1m", "lastfm", "amazon_music")
SOURCES = {
    "primary": ("uniform", "weighted_prf", "learned_centroid"),
    "baseline": ("popularity", "last_item", "transition_knn", "content_sasrec"),
    "stochastic": ("weighted_fixed", "learned_fixed", "full"),
    "session": ("vsknn",),
}


def support_partition(train, evaluation):
    """Return event -> warm flag, using training context and targets only."""
    observed = {item for row in train for item in (*row["context"], row["target"])}
    if not observed:
        raise ValueError("No observed training items")
    result = {}
    for row in evaluation:
        event = row["event"]
        if event in result:
            raise ValueError(f"Duplicate prediction event: {event}")
        if row["target"] in row["excluded"]:
            raise ValueError(f"Target excluded from evaluation: {event}")
        result[event] = row["target"] in observed
    if not result:
        raise ValueError("No prediction events")
    return result, observed


def verify_inputs(root, dataset):
    data = root / "artifacts/revision/v1/data" / dataset
    locations = {
        "primary": root / "artifacts/revision/mps-v1/results" / dataset,
        "baseline": root / "artifacts/ecir/mps-v2/baseline-results" / dataset,
        "stochastic": root / "artifacts/ecir/mps-v2/stochastic-results" / dataset,
        "session": root / "artifacts/ecir/additional-v1/session-knn-results" / dataset,
    }
    data_manifest = verify_manifest(data)
    if data_manifest["dataset"] != dataset:
        raise ValueError(f"Data manifest dataset mismatch: {dataset}")
    manifests = {name: verify_manifest(folder) for name, folder in locations.items()}
    for name, manifest in manifests.items():
        if manifest.get("dataset") != dataset:
            raise ValueError(f"Result manifest dataset mismatch: {dataset}/{name}")
        linked = manifest.get("data_manifest_sha256")
        if linked is not None and linked != digest(data / "manifest.json"):
            raise ValueError(f"Result/data provenance mismatch: {dataset}/{name}")
    return data, locations, data_manifest, manifests


def summarise_method(folder, manifest, method, support, groups):
    summary_path = folder / "summary.json"
    if summary_path.exists():
        selected = [row["file"] for row in json.loads(summary_path.read_text())
                    if row["method"] == method]
    elif method == "vsknn":
        selected = ["vsknn.jsonl"]
    else:
        raise ValueError(f"Missing result summary: {folder}")
    if not selected or len(selected) != len(set(selected)):
        raise ValueError(f"No unique result files for {method} in {folder}")
    expected = set(support)
    totals = {"warm": defaultdict(float), "cold": defaultdict(float)}
    counts = {"warm": 0, "cold": 0}
    for name in selected:
        if name not in manifest["files"]:
            raise ValueError(f"Unverified result file: {folder / name}")
        rows = read_jsonl(folder / name)
        if len(rows) != len(expected) or {row["event"] for row in rows} != expected:
            raise ValueError(f"Different or repeated event support: {folder / name}")
        for row in rows:
            event = row["event"]
            if row["group"] != groups[event]:
                raise ValueError(f"Event group differs: {folder / name}/{event}")
            partition = "warm" if support[event] else "cold"
            counts[partition] += 1
            for metric in METRICS:
                value = float(row[metric])
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite metric: {folder / name}/{event}")
                totals[partition][metric] += value
    results = []
    for partition in ("warm", "cold", "all"):
        count = counts[partition] if partition != "all" else sum(counts.values())
        if partition == "all":
            sums = {metric: totals["warm"][metric] + totals["cold"][metric]
                    for metric in METRICS}
        else:
            sums = totals[partition]
        results.append({"method": method, "partition": partition,
                        "events": count // len(selected), "files": len(selected),
                        **{metric: sums[metric] / count if count else None
                           for metric in METRICS}})
    return results


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(root, out):
    if out.exists():
        raise ValueError(f"Create-only output already exists: {out}")
    main_report = root / "paper/generated/ecir-mps-v2"
    session_report = root / "paper/generated/ecir-additional-v1"
    verify_manifest(main_report)
    verify_manifest(session_report)
    with (main_report / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        published = list(csv.DictReader(handle))
    with (session_report / "session_knn.csv").open(
            encoding="utf-8", newline="") as handle:
        published_sessions = list(csv.DictReader(handle))
    coverage, metrics, provenance = [], [], {}
    for dataset in DATASETS:
        data, locations, data_manifest, manifests = verify_inputs(root, dataset)
        train = read_jsonl(data / "train.jsonl")
        catalog = json.loads((data / "catalog.json").read_text())["ids"]
        catalog_set = set(catalog)
        if len(catalog) != len(catalog_set):
            raise ValueError(f"Duplicate catalog IDs: {dataset}")
        splits = {split: read_jsonl(data / f"{split}.jsonl")
                  for split in ("validation", "test")}
        observed = {item for row in train for item in (*row["context"], row["target"])}
        if not observed <= catalog_set:
            raise ValueError(f"Training item outside catalog: {dataset}")
        for split, events in splits.items():
            support, _ = support_partition(train, events)
            if any(row["target"] not in catalog_set for row in events):
                raise ValueError(f"Held-out target outside catalog: {dataset}/{split}")
            warm = sum(support.values())
            coverage.append({"dataset": dataset, "split": split,
                             "events": len(events), "warm_events": warm,
                             "cold_events": len(events) - warm,
                             "warm_fraction": warm / len(events),
                             "cold_fraction": 1 - warm / len(events),
                             "training_items": len(observed),
                             "catalog_items": len(catalog)})
            if split != "test":
                continue
            groups = {row["event"]: row["group"] for row in events}
            for source, methods in SOURCES.items():
                for method in methods:
                    method_rows = summarise_method(locations[source], manifests[source],
                                                   method, support, groups)
                    overall = next(row for row in method_rows
                                   if row["partition"] == "all")
                    if method == "vsknn":
                        matches = [row for row in published_sessions
                                   if row["dataset"] == dataset]
                        expected = {metric: float(matches[0][f"V-SKNN {metric}"])
                                    for metric in METRICS} if len(matches) == 1 else None
                    else:
                        matches = [row for row in published if
                                   row["dataset"] == dataset and row["method"] == method]
                        expected = {metric: float(matches[0][f"{metric} mean"])
                                    for metric in METRICS} if len(matches) == 1 else None
                    if expected is None or any(
                            not math.isclose(overall[metric], expected[metric],
                                             rel_tol=0, abs_tol=1e-10)
                            for metric in METRICS):
                        raise ValueError(f"Conditional analysis does not reproduce "
                                         f"published overall metrics: {dataset}/{method}")
                    for row in method_rows:
                        metrics.append({"dataset": dataset, **row})
        provenance[dataset] = {
            "data_manifest_sha256": digest(data / "manifest.json"),
            "result_manifest_sha256": {
                name: digest(folder / "manifest.json")
                for name, folder in locations.items()},
            "catalog_size": data_manifest["catalog_size"],
        }
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "target_coverage.csv", coverage)
    write_csv(out / "warm_cold_metrics.csv", metrics)
    (out / "REPORT.md").write_text(
        "# Training-item support and conditional test effectiveness\n\n"
        "Warm means the target occurs in the prepared training context or target "
        "of at least one event; cold means it never occurs in training interactions. "
        "Validation/test items do not define support. The catalog and eligible "
        "candidate set are unchanged in all metrics.\n\n"
        "`target_coverage.csv` reports event-weighted coverage on both held-out "
        "splits. `warm_cold_metrics.csv` reports the original saved full-catalog "
        "rankings, partitioned by test-target support. Each method's repeated "
        "files are averaged equally; every file must have identical events. "
        "Empty partitions have blank metric cells. This is a descriptive "
        "post-hoc analysis; no model is selected or evaluated again.\n",
        encoding="utf-8")
    dump(out / "manifest.json", {
        "schema": 1, "experiment": "ecir-target-support", "datasets": list(DATASETS),
        "source_sha256": {
            "experiments/ecir/target_support.py": digest(Path(__file__)),
            "experiments/revision/protocol.py": digest(
                ROOT / "experiments/revision/protocol.py")},
        "inputs": provenance,
        "main_report_manifest_sha256": digest(main_report / "manifest.json"),
        "session_report_manifest_sha256": digest(session_report / "manifest.json"),
        "files": {name: digest(out / name) for name in
                  ("target_coverage.csv", "warm_cold_metrics.csv", "REPORT.md")},
    })
    print(f"Wrote {out / 'REPORT.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
