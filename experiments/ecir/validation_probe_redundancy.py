"""Validation-split probe redundancy with the original frozen checkpoints.

Descriptive, post-hoc analysis only. Uses the original validation events and
event-specific inference directions; neither test events nor test directions
enter this computation. No model is fitted or selected.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.ecir import query_direction_sensitivity as sweep
from experiments.ecir.probe_redundancy import (
    geometry_metrics, overlap_metrics, top_items,
)
from experiments.ecir.review_common import (
    analysis_sources, cell, check_sources, finish, payload_hash,
    prepare_output, write_or_check,
)
from experiments.revision.protocol import digest, metrics_from_scores, verify_manifest
from experiments.revision.run import Data, configure_runtime, directions, seed_all

COUNTS = (1, 5, 20)
CUTOFF = 10
FIELDS = ("pairwise_topk_jaccard", "unique_candidates", "target_union_coverage",
          "merged_ndcg10", "merged_recall", "pairwise_angle_degrees")


@torch.no_grad()
def evaluate_cell(data, model, method, params, revision, inference_seed):
    seed_all(inference_seed)
    model.eval()
    rows = data.splits["validation"]
    totals = {count: {key: 0. for key in FIELDS} for count in COUNTS}
    for start in range(0, len(rows), revision["batch_size"]):
        batch = rows[start:start + revision["batch_size"]]
        queries = directions(data, batch, method, model, params,
                             dict(revision, samples=max(COUNTS)),
                             inference_seed, "validation")
        if start == 0:
            for samples in COUNTS[:-1]:
                reference = directions(data, batch, method, model, params,
                                       dict(revision, samples=samples),
                                       inference_seed, "validation")
                torch.testing.assert_close(queries[:, :samples], reference,
                                           rtol=1e-5, atol=1e-6)
        vectors = queries.cpu().numpy()
        top_lists = [[] for _ in batch]
        merged = None
        for index in range(max(COUNTS)):
            scores = (queries[:, index] @ data.emb.T).cpu().numpy()
            merged = scores.copy() if merged is None else np.maximum(merged, scores)
            for event_index, row in enumerate(batch):
                top_lists[event_index].append(
                    top_items(scores[event_index], row["excluded"], CUTOFF))
            count = index + 1
            if count not in COUNTS:
                continue
            for event_index, row in enumerate(batch):
                overlap = overlap_metrics(top_lists[event_index], row["target"])
                geometry = geometry_metrics(vectors[event_index, :count])
                ranking = metrics_from_scores(merged[event_index], row["target"],
                                              row["excluded"])
                values = {**overlap, **geometry,
                          "merged_ndcg10": ranking["NDCG@10"],
                          "merged_recall": ranking["Recall@10"]}
                for key in FIELDS:
                    if values[key] is not None:
                        totals[count][key] += values[key]
    return [{"samples": count, "queries": len(rows),
             **{key: (None if count == 1 and key in
                      {"pairwise_topk_jaccard", "pairwise_angle_degrees"}
                      else totals[count][key] / len(rows)) for key in FIELDS}}
            for count in COUNTS]


def summarize(cells):
    output = []
    for method in sweep.METHODS:
        for samples in COUNTS:
            group = [row for row in cells if row["method"] == method and
                     row["samples"] == samples]
            if len(group) != 50 or {row["queries"] for row in group} != {1520}:
                raise ValueError(f"Expected 5 centers x 10 seeds x 1520 validation events: "
                                 f"{method}/S={samples}")
            output.append({"method": method, "samples": samples,
                           "queries_per_cell": group[0]["queries"],
                           "cells": len(group),
                           **{key: (None if group[0][key] is None else
                                    float(np.mean([row[key] for row in group])))
                              for key in FIELDS}})
    return output


def run(args):
    parent = sweep.verify_run_inputs(args)
    reference = verify_manifest(args.reference_sweep)
    if reference.get("experiment") != "ecir-query-direction-sensitivity":
        raise ValueError("Expected the original probe-count sweep")
    if reference["stochastic_run_manifest_sha256"] != digest(
            args.stochastic_run / "manifest.json"):
        raise ValueError("Reference sweep uses different checkpoints")
    if reference["data_manifest_sha256"] != parent["data_manifest_sha256"]:
        raise ValueError("Reference sweep uses different prepared data")
    sweep.validate_sweep_config(reference["sweep_config"], parent)
    seeds = reference["sweep_config"]["training_seeds"]
    noises = reference["sweep_config"]["inference_seeds"]
    if len(seeds) != 5 or len(noises) != 10:
        raise ValueError("Expected the original five training and ten inference seeds")
    data = Data(args.data, args.device, ["validation"])
    if len(data.splits["validation"]) != 1520:
        raise ValueError("TalkPlay validation support changed")
    spec = {
        "schema": 1, "experiment": "ecir-validation-probe-redundancy",
        "source_sha256": analysis_sources(__file__),
        "frozen_source_sha256": {
            **sweep.source_hashes(),
            "experiments/ecir/probe_redundancy.py": digest(
                Path(__file__).with_name("probe_redundancy.py"))},
        "data_manifest_sha256": digest(args.data / "manifest.json"),
        "stochastic_run_manifest_sha256": digest(
            args.stochastic_run / "manifest.json"),
        "reference_sweep_manifest_sha256": digest(
            args.reference_sweep / "manifest.json"),
        "validation_support_sha256": payload_hash(
            [(row["event"], row["group"]) for row in data.splits["validation"]]),
        "sample_counts": list(COUNTS), "cutoff": CUTOFF,
        "training_seeds": seeds, "inference_seeds": noises,
        "split": "validation", "analysis": "post-hoc; no fitting or selection",
    }
    prepare_output(args.out, spec)
    if (args.out / "manifest.json").exists():
        verify_manifest(args.out)
        print("Validation probe diagnostics already complete and verified.")
        return
    selection = json.loads((args.stochastic_run / "selection.json").read_text())
    all_cells = []
    for method in sweep.METHODS:
        chosen = selection["methods"][method]
        for seed in seeds:
            model = sweep.load_model(data,
                                     args.stochastic_run /
                                     chosen["checkpoints"][str(seed)], args.device)
            for inference_seed in noises:
                record = cell(
                    args.out, f"{method}-train-{seed}-noise-{inference_seed}",
                    lambda: evaluate_cell(data, model, method, chosen["params"],
                                          parent["revision_config"], inference_seed))
                all_cells.extend({"method": method, "train_seed": seed,
                                  "inference_seed": inference_seed, **row}
                                 for row in record)
            del model
    write_or_check(args.out / "aggregate.json", summarize(all_cells))
    finish(args.out, spec)
    print(f"Saved validation diagnostics: {args.out}")


def report(args):
    manifest = verify_manifest(args.results)
    if manifest.get("experiment") != "ecir-validation-probe-redundancy":
        raise ValueError("Wrong validation diagnostic results")
    check_sources(manifest["source_sha256"])
    check_sources(manifest["frozen_source_sha256"])
    test_manifest = verify_manifest(args.test_report)
    if test_manifest.get("experiment") != "ecir-probe-redundancy-report":
        raise ValueError("Wrong test diagnostic report")
    if test_manifest["result_manifest_sha256"] != digest(
            args.test_results / "manifest.json"):
        raise ValueError("Test report provenance changed")
    with (args.test_report / "probe_redundancy.csv").open(
            encoding="utf-8", newline="") as handle:
        test_rows = list(csv.DictReader(handle))
    validation = json.loads((args.results / "aggregate.json").read_text())
    if args.out.exists():
        raise ValueError(f"Create-only report already exists: {args.out}")
    combined = []
    for row in validation:
        match = [item for item in test_rows if item["method"] == row["method"]
                 and int(item["samples"]) == row["samples"]
                 and int(item["cutoff"]) == CUTOFF]
        if len(match) != 1:
            raise ValueError("Missing corresponding frozen test diagnostic")
        test = match[0]
        combined.append({"method": row["method"], "samples": row["samples"],
                         "cutoff": CUTOFF, "validation_events": row["queries_per_cell"],
                         "test_events": int(test["queries_per_cell"]),
                         "validation_jaccard": row["pairwise_topk_jaccard"],
                         "test_jaccard": (float(test["pairwise_topk_jaccard"])
                                          if test["pairwise_topk_jaccard"] else None),
                         "validation_unique": row["unique_candidates"],
                         "test_unique": float(test["unique_candidates"]),
                         "validation_union_hit": row["target_union_coverage"],
                         "test_union_hit": float(test["target_union_coverage"]),
                         "validation_merged_ndcg10": row["merged_ndcg10"],
                         "test_merged_ndcg10": float(test["merged_ndcg10"])})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(combined[0]))
    writer.writeheader()
    writer.writerows(combined)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "validation_vs_test.csv").write_text(buffer.getvalue(), encoding="utf-8")
    (args.out / "REPORT.md").write_text(
        "# Probe redundancy on validation and test\n\n"
        "The validation side uses existing selected checkpoints with the same "
        "five training and ten inference seeds as the earlier test diagnostic. "
        "Directions are generated for validation events using the validation "
        "split key. Results are descriptive and post-hoc; they are not used "
        "to select any checkpoint, radius, or probe count.\n\n"
        "Each Jaccard is the mean pairwise overlap of per-probe top-10 sets. "
        "Unique candidates are their union size; merged NDCG uses the original "
        "max-score ranker. The validation and test columns have different "
        "events, so similarity is a replication check, not a paired effect.\n",
        encoding="utf-8")
    finish(args.out, {
        "schema": 1, "experiment": "ecir-validation-probe-redundancy-report",
        "result_manifest_sha256": digest(args.results / "manifest.json"),
        "test_report_manifest_sha256": digest(args.test_report / "manifest.json"),
        "source_sha256": analysis_sources(__file__)})
    print(f"Saved validation/test comparison: {args.out}")


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    part = subparsers.add_parser("run")
    for name in ("data", "original-run", "stochastic-run", "reference-sweep", "out"):
        part.add_argument("--" + name, type=Path, required=True)
    part.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    part.add_argument("--cpu-threads", type=int, default=4)
    part.add_argument("--interop-threads", type=int, default=1)
    part.add_argument("--determinism", choices=("strict", "warn"), default="warn")
    part = subparsers.add_parser("report")
    for name in ("results", "test-results", "test-report", "out"):
        part.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        configure_runtime(args)
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
