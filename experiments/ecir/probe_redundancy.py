"""Inference-only TalkPlay probe overlap, coverage, and angular diagnostics.

Uses the same frozen checkpoints, evaluator, exclusions, and nested noise as
the existing probe-count sweep. No fitting, radius selection, or S selection.
"""
import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.ecir import query_direction_sensitivity as sweep
from experiments.ecir.frozen_radius import csv_file
from experiments.ecir.review_common import (
    analysis_sources, cell, check_sources, finish, payload_hash, prepare_output, write_or_check,
)
from experiments.revision.protocol import digest, metrics_from_scores, verify_manifest
from experiments.revision.run import Data, configure_runtime, directions, seed_all

COUNTS = (1, 2, 5, 10, 20)
CUTOFFS = (10, 50)


def top_items(scores, excluded, k):
    """Exact top-k, using canonical ascending-index ties (never torch.topk ties)."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or not np.isfinite(scores).all() or k < 1:
        raise ValueError("Expected finite catalog scores and positive cutoff")
    legal = np.ones(len(scores), dtype=bool)
    legal[list(excluded)] = False
    indices = np.flatnonzero(legal)
    count = min(k, len(indices))
    if not count:
        raise ValueError("No eligible candidates")
    values = scores[indices]
    threshold = np.partition(values, len(values) - count)[len(values) - count]
    greater = indices[values > threshold]
    ties = indices[values == threshold][:count - len(greater)]
    candidates = np.concatenate((greater, ties))
    return candidates[np.lexsort((candidates, -scores[candidates]))]


def overlap_metrics(top_lists, target):
    sets = [set(items) for items in top_lists]
    union = set().union(*sets)
    before = set().union(*sets[:-1]) if len(sets) > 1 else set()
    overlaps = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]
    return {"unique_candidates": len(union),
            "new_candidates_last_probe": len(union - before),
            "additional_candidates_vs_first": len(union - sets[0]),
            "pairwise_topk_jaccard": float(np.mean(overlaps)) if overlaps else None,
            "target_union_coverage": float(target in union),
            "target_gain_vs_first": float(target in union and target not in sets[0]),
            "target_gain_last_probe": float(target in union and target not in before)}


def geometry_metrics(vectors):
    if len(vectors) < 2:
        return {"pairwise_cosine": None, "pairwise_angle_degrees": None}
    vectors = np.asarray(vectors, dtype=np.float64)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    cosines = np.clip((vectors @ vectors.T)[np.triu_indices(len(vectors), 1)], -1., 1.)
    return {"pairwise_cosine": float(cosines.mean()),
            "pairwise_angle_degrees": float(np.degrees(np.arccos(cosines)).mean())}


@torch.no_grad()
def evaluate_cell(data, model, method, params, revision, inference_seed, expected):
    seed_all(inference_seed)
    model.eval()
    totals = {(s, k): {} for s in COUNTS for k in CUTOFFS}
    rows = data.splits["test"]
    for start in range(0, len(rows), revision["batch_size"]):
        batch = rows[start:start + revision["batch_size"]]
        queries = directions(data, batch, method, model, params,
                             dict(revision, samples=max(COUNTS)), inference_seed, "test")
        # Confirm nested draws against the existing evaluator on the first batch.
        if start == 0:
            for samples in COUNTS[:-1]:
                reference = directions(data, batch, method, model, params,
                                       dict(revision, samples=samples), inference_seed, "test")
                torch.testing.assert_close(queries[:, :samples], reference, rtol=1e-5, atol=1e-6)
        vectors = queries.cpu().numpy()
        lists = [[] for _ in batch]
        merged = None
        for index in range(max(COUNTS)):
            scores = (queries[:, index] @ data.emb.T).cpu().numpy()
            merged = scores.copy() if merged is None else np.maximum(merged, scores)
            for j, row in enumerate(batch):
                if row["target"] in row["excluded"]:
                    raise ValueError("Repeated target violates evaluation protocol")
                lists[j].append(top_items(scores[j], row["excluded"], max(CUTOFFS)))
            samples = index + 1
            if samples not in COUNTS:
                continue
            for j, row in enumerate(batch):
                ranking = metrics_from_scores(merged[j], row["target"], row["excluded"])
                geometry = geometry_metrics(vectors[j, :samples])
                for k in CUTOFFS:
                    record = {**overlap_metrics([items[:k] for items in lists[j]], row["target"]),
                              **geometry, "merged_recall": ranking[f"Recall@{k}"],
                              "merged_ndcg10": ranking["NDCG@10"]}
                    for key, value in record.items():
                        if value is not None:
                            totals[samples, k][key] = totals[samples, k].get(key, 0.) + value
    output = []
    for (samples, k), values in totals.items():
        means = {key: value / len(rows) for key, value in values.items()}
        # A mechanism diagnostic must reproduce the ranking result it explains.
        if not np.isclose(means["merged_ndcg10"], expected[samples], rtol=0, atol=1e-10):
            raise ValueError(f"S={samples} diagnostic ranking differs from frozen sweep: "
                             f"{means['merged_ndcg10']} != {expected[samples]}. "
                             "Do not use these diagnostics until the mismatch is resolved.")
        output.append({"samples": samples, "cutoff": k, "queries": len(rows), **means})
    return output


def summarize(cells):
    output = []
    for method in sweep.METHODS:
        for samples in COUNTS:
            for k in CUTOFFS:
                group = [row for row in cells if row["method"] == method and
                         row["samples"] == samples and row["cutoff"] == k]
                if not group:
                    raise ValueError("Missing diagnostic cells")
                keys = set(group[0]) - {"method", "train_seed", "inference_seed",
                                        "samples", "cutoff", "queries"}
                record = {"method": method, "samples": samples, "cutoff": k,
                          "queries_per_cell": group[0]["queries"], "cells": len(group)}
                for key in sorted(keys):
                    record[key] = float(np.mean([row[key] for row in group]))
                # Empty pairwise values at S=1 are undefined, not zero diversity.
                for key in ("pairwise_topk_jaccard", "pairwise_cosine", "pairwise_angle_degrees"):
                    record.setdefault(key, None)
                output.append(record)
    return output


def run(args):
    parent = sweep.verify_run_inputs(args)
    reference = verify_manifest(args.reference_sweep)
    if reference.get("experiment") != "ecir-query-direction-sensitivity":
        raise ValueError("Expected the saved probe-count sweep")
    if reference["stochastic_run_manifest_sha256"] != digest(args.stochastic_run / "manifest.json"):
        raise ValueError("Sweep and diagnostic checkpoints differ")
    for key in ("data_manifest_sha256", "original_run_manifest_sha256"):
        if reference[key] != parent[key]:
            raise ValueError(f"Sweep provenance mismatch: {key}")
    if reference["source_sha256"] != sweep.source_hashes():
        raise ValueError("Frozen sensitivity source changed")
    section = reference["sweep_config"]
    sweep.validate_sweep_config(section, parent)
    if section["sample_counts"] != list(COUNTS):
        raise ValueError("Unexpected sample counts")
    summary = json.loads((args.reference_sweep / "summary.json").read_text())
    expected = {(r["method"], r["train_seed"], r["inference_seed"], r["samples"]):
                r["NDCG@10"] for r in summary}
    if len(expected) != len(summary):
        raise ValueError("Duplicate reference cells")
    data = Data(args.data, args.device, ["test"])
    spec = {"experiment": "ecir-probe-redundancy", "schema": 1,
            "source_sha256": analysis_sources(__file__),
            "frozen_source_sha256": {**sweep.source_hashes(),
                "experiments/ecir/frozen_radius.py": digest(Path(__file__).with_name("frozen_radius.py"))},
            "stochastic_run_manifest_sha256": digest(args.stochastic_run / "manifest.json"),
            "reference_sweep_manifest_sha256": digest(args.reference_sweep / "manifest.json"),
            "data_manifest_sha256": digest(args.data / "manifest.json"),
            "sample_counts": list(COUNTS), "cutoffs": list(CUTOFFS),
            "support_sha256": payload_hash([(r["event"], r["group"]) for r in data.splits["test"]]),
            "analysis": "post-hoc diagnostics; no selection or fitting",
            "device": args.device, "environment": sweep.environment()}
    prepare_output(args.out, spec)
    if (args.out / "manifest.json").exists():
        verify_manifest(args.out)
        print("Probe diagnostics already complete and verified.")
        return
    selection = json.loads((args.stochastic_run / "selection.json").read_text())
    all_cells = []
    for method in sweep.METHODS:
        chosen = selection["methods"][method]
        for seed in section["training_seeds"]:
            model = sweep.load_model(data, args.stochastic_run / chosen["checkpoints"][str(seed)],
                                     args.device)
            for noise_seed in section["inference_seeds"]:
                expected_means = {s: expected[method, seed, noise_seed, s] for s in COUNTS}
                values = cell(args.out, f"{method}-train-{seed}-noise-{noise_seed}",
                              lambda: evaluate_cell(data, model, method, chosen["params"],
                                                    parent["revision_config"], noise_seed, expected_means))
                all_cells.extend({"method": method, "train_seed": seed,
                                  "inference_seed": noise_seed, **row} for row in values)
            del model
    aggregates = summarize(all_cells)
    write_or_check(args.out / "aggregate.json", aggregates)
    finish(args.out, spec)
    print(f"Probe diagnostics complete: {args.out}")


def report(args):
    manifest = verify_manifest(args.results)
    if manifest.get("experiment") != "ecir-probe-redundancy":
        raise ValueError("Expected probe-redundancy results")
    check_sources(manifest["source_sha256"])
    check_sources(manifest["frozen_source_sha256"])
    report_spec = {"experiment": "ecir-probe-redundancy-report",
                   "result_manifest_sha256": digest(args.results / "manifest.json"),
                   "source_sha256": analysis_sources(__file__)}
    if (args.out / "manifest.json").exists():
        previous = verify_manifest(args.out)
        if any(previous.get(key) != value for key, value in report_spec.items()):
            raise ValueError("Completed report inputs changed; choose a new output")
        print(f"Probe report already complete and verified: {args.out}")
        return
    if args.out.exists():
        raise ValueError("Incomplete report output: move it aside before rerunning")
    rows = json.loads((args.results / "aggregate.json").read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    # Identical column order across records, including undefined S=1 fields.
    fields = list(rows[-1])
    csv_file(args.out / "probe_redundancy.csv", [{key: r.get(key) for key in fields} for r in rows])
    (args.out / "REPORT.md").write_text(
        "# Probe redundancy diagnostics\n\n"
        "CSV means average all test events, five training seeds, and ten inference seeds.\n"
        "Top-k ties use ascending catalog index after full-history exclusions.\n"
        "Pairwise Jaccard averages all probe pairs, not only adjacent probes.\n"
        "Unique candidates and target union coverage refer to the union of per-probe top-k sets;\n"
        "this union can contain S*k items and is not fixed-budget Recall@k.\n"
        "merged_recall and merged_ndcg10 instead use the original max-score ranker.\n"
        "target_gain_last_probe compares S with S-1; target_gain_vs_first compares S with 1.\n"
        "Angles/cosines measure query-vector geometry, not retrieved-item geometry.\n"
        "S=1 pairwise statistics are undefined. These are descriptive post-hoc diagnostics,\n"
        "not significance tests, causal explanations, or hyperparameter-selection evidence.\n"
        "Every seed/count NDCG was checked against the retained sensitivity sweep.\n")
    with (args.out / "REPORT.md").open("a") as handle:
        handle.write("\n## Top-10 diagnostics\n\n"
                     "| Method | S | Jaccard | Unique candidates | Union hit | Merged Recall@10 | Angle (deg) |\n"
                     "|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            if row["cutoff"] != 10:
                continue
            jaccard = row["pairwise_topk_jaccard"]
            angle = row["pairwise_angle_degrees"]
            handle.write(f"| {row['method']} | {row['samples']} | "
                         f"{jaccard if jaccard is not None else 'undefined'} | "
                         f"{row['unique_candidates']:.3f} | {row['target_union_coverage']:.6f} | "
                         f"{row['merged_recall']:.6f} | {angle if angle is not None else 'undefined'} |\n")
    finish(args.out, report_spec)
    print(f"Report saved: {args.out}")


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    part = sub.add_parser("run")
    for flag in ("data", "original-run", "stochastic-run", "reference-sweep", "out"):
        part.add_argument("--" + flag, type=Path, required=True)
    part.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    part.add_argument("--cpu-threads", type=int, default=4)
    part.add_argument("--interop-threads", type=int, default=1)
    part.add_argument("--determinism", choices=("strict", "warn"), default="warn")
    part = sub.add_parser("report")
    for flag in ("results", "out"):
        part.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        configure_runtime(args)
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
