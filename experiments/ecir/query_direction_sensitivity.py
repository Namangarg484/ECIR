"""Frozen-checkpoint TalkPlay sensitivity to the number of query directions.

This is a post-hoc sensitivity analysis, not a model-selection stage. It evaluates
the already selected learned-fixed and adaptive ACE checkpoints at the predeclared
S values. Random draws are nested: a run with S directions uses the first S draws
from the same event/seed-specific stream used by every larger value of S.
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.ecir.stochastic_eval import (
    load_model,
    source_hashes as stochastic_source_hashes,
)
from experiments.revision.protocol import (
    METRICS,
    aggregate,
    digest,
    dump,
    read_jsonl,
    stable_seed,
    verify_manifest,
    write_jsonl,
)
from experiments.revision.run import (
    Data,
    configure_runtime,
    environment,
    evaluate,
    seed_all,
    sources as revision_sources,
)

METHODS = ("learned_fixed", "full")
LABELS = {
    "learned_fixed": "Learned center + fixed radius",
    "full": "ACE (adaptive radius)",
}
RECOMMENDED_COUNTS = [1, 2, 5, 10, 20]
LATEX_ROW_END = r"\\"


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root / "experiments/revision/run.py",
             root / "experiments/revision/protocol.py",
             root / "experiments/ecir/stochastic_eval.py",
             root / "src/models/vce_model.py"]
    return {str(path.relative_to(root)): digest(path) for path in paths}


def validate_sweep_config(section, stochastic_manifest):
    if section.get("dataset") != "talkplay":
        raise ValueError("Direction sensitivity is predeclared for TalkPlay only")
    if section.get("split") != "test":
        raise ValueError("Direction sensitivity must use the frozen test split")
    if section.get("methods") != list(METHODS):
        raise ValueError(f"direction_sensitivity.methods must equal {list(METHODS)}")
    if section.get("sample_counts") != RECOMMENDED_COUNTS:
        raise ValueError(
            f"direction_sensitivity.sample_counts must equal {RECOMMENDED_COUNTS}")

    revision = stochastic_manifest["revision_config"]
    extension = stochastic_manifest["extension_config"]
    if section.get("training_seeds") != revision["seeds"]:
        raise ValueError("Sensitivity training seeds must match the frozen checkpoints")
    if section.get("inference_seeds") != extension["inference_seeds"]:
        raise ValueError("Sensitivity inference seeds must match the ECIR evaluation")
    if revision["samples"] not in section["sample_counts"]:
        raise ValueError("The original query-direction count must occur in the sweep")


def verify_run_inputs(args):
    stochastic_manifest = verify_manifest(args.stochastic_run)
    original_manifest = verify_manifest(args.original_run)
    if stochastic_manifest.get("experiment") != "ecir-stochastic-evaluation":
        raise ValueError("Expected an ECIR stochastic-run artifact")
    if stochastic_manifest.get("dataset") != "talkplay":
        raise ValueError("Direction sensitivity is restricted to TalkPlay")
    if original_manifest.get("dataset") != "talkplay":
        raise ValueError("Original run is not the TalkPlay run")
    if args.device != stochastic_manifest["device"]:
        raise ValueError("Sensitivity device must match the stochastic fit device")
    if digest(args.data / "manifest.json") != stochastic_manifest["data_manifest_sha256"]:
        raise ValueError("Prepared data differs from the frozen stochastic run")
    if digest(args.original_run / "manifest.json") != stochastic_manifest[
            "original_run_manifest_sha256"]:
        raise ValueError("Original run differs from the stochastic run's parent")
    if revision_sources() != original_manifest["source_sha256"]:
        raise ValueError("Frozen revision source no longer matches the original fit")
    if stochastic_source_hashes() != stochastic_manifest["source_sha256"]:
        raise ValueError("Frozen stochastic-evaluation source has changed")
    runtime = environment()
    for key in ("determinism", "mps_cpu_fallback"):
        expected = stochastic_manifest["environment"].get(key, runtime[key])
        if runtime[key] != expected:
            raise ValueError(f"Sensitivity runtime {key} differs from frozen fit")
    return stochastic_manifest


def run_sweep(args):
    stochastic_manifest = verify_run_inputs(args)
    config = json.loads(args.config.read_text())
    section = config.get("direction_sensitivity", {})
    validate_sweep_config(section, stochastic_manifest)
    selection = json.loads((args.stochastic_run / "selection.json").read_text())
    revision_config = stochastic_manifest["revision_config"]
    data = Data(args.data, args.device, [section["split"]])

    args.out.mkdir(parents=True, exist_ok=False)
    summary, files = [], []
    total = (len(METHODS) * len(section["training_seeds"]) *
             len(section["inference_seeds"]) * len(section["sample_counts"]))
    completed = 0
    for method in METHODS:
        chosen = selection["methods"][method]
        for training_seed in section["training_seeds"]:
            checkpoint = args.stochastic_run / chosen["checkpoints"][str(training_seed)]
            model = load_model(data, checkpoint, args.device)
            for inference_seed in section["inference_seeds"]:
                for samples in section["sample_counts"]:
                    seed_all(inference_seed)
                    evaluation_config = dict(revision_config, samples=samples)
                    rows = evaluate(data, section["split"], method, model,
                                    chosen["params"], evaluation_config,
                                    inference_seed)
                    name = (f"{method}.samples-{samples}.train-{training_seed}."
                            f"inference-{inference_seed}.jsonl")
                    write_jsonl(args.out / name, rows)
                    files.append(name)
                    summary.append({
                        "dataset": "talkplay", "split": section["split"],
                        "method": method, "samples": samples,
                        "train_seed": training_seed,
                        "inference_seed": inference_seed,
                        "n": len(rows), "file": name, **aggregate(rows),
                    })
                    completed += 1
                    print(f"direction sensitivity {completed}/{total}: "
                          f"{method} train={training_seed} "
                          f"inference={inference_seed} S={samples}", flush=True)
            del model

    dump(args.out / "summary.json", summary)
    files.append("summary.json")
    dump(args.out / "manifest.json", {
        "schema": 1,
        "experiment": "ecir-query-direction-sensitivity",
        "dataset": "talkplay",
        "analysis_status": "post-hoc fixed sweep; no model selection",
        "sweep_config": section,
        "revision_config": revision_config,
        "data_manifest_sha256": digest(args.data / "manifest.json"),
        "original_run_manifest_sha256": digest(args.original_run / "manifest.json"),
        "stochastic_run_manifest_sha256": digest(
            args.stochastic_run / "manifest.json"),
        "source_sha256": source_hashes(),
        "environment": environment(),
        "device": args.device,
        "files": {name: digest(args.out / name) for name in files},
        "command": sys.argv,
    })
    print(f"Direction-sensitivity results saved: {args.out}")
    print("No checkpoints were trained or selected by this analysis.")


def load_entry(folder, manifest, entry, reference=None):
    if entry["file"] not in manifest["files"]:
        raise ValueError(f"Unverified result file: {entry['file']}")
    rows = sorted(read_jsonl(folder / entry["file"]), key=lambda row: row["event"])
    support = [(row["event"], row["group"]) for row in rows]
    if len(support) != len(set(support)):
        raise ValueError(f"Duplicate query in {entry['file']}")
    if reference is not None and support != reference:
        raise ValueError("All sensitivity cells must share identical ordered support")
    recomputed = aggregate(rows)
    if entry["n"] != len(rows) or any(
            abs(recomputed[metric] - entry[metric]) > 1e-10
            for metric in METRICS):
        raise ValueError(f"Summary mismatch for {entry['file']}")
    values = np.asarray([[row[metric] for metric in METRICS] for row in rows],
                        dtype=np.float64)
    return support, values


def summarize_values(method, samples, values):
    """Summarize [training seed, inference seed, query, metric] values."""
    record = {"method": method, "samples": samples,
              "queries": values.shape[2],
              "training_seeds": values.shape[0],
              "inference_seeds": values.shape[1]}
    train_scores = values.mean(axis=(1, 2))
    inference_scores = values.mean(axis=2)
    for index, metric in enumerate(METRICS):
        record[f"{metric} mean"] = float(values[..., index].mean())
        record[f"{metric} train_sd"] = float(
            train_scores[:, index].std(ddof=1))
        within_variances = inference_scores[:, :, index].var(axis=1, ddof=1)
        record[f"{metric} inference_sd"] = float(
            math.sqrt(within_variances.mean()))
    return record


def hierarchical_interval(differences, groups, repeats, seed):
    """Resample training seeds, inference seeds, and session/user clusters."""
    if differences.ndim != 3:
        raise ValueError("Expected [training seed, inference seed, query] differences")
    names = sorted(set(groups))
    if not names or differences.shape[2] != len(groups):
        raise ValueError("Groups must align with the query dimension")
    indices = {name: [] for name in names}
    for query_index, group in enumerate(groups):
        indices[group].append(query_index)
    totals = np.stack([
        differences[:, :, indices[name]].sum(axis=2) for name in names
    ], axis=2)
    counts = np.asarray([len(indices[name]) for name in names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=np.float64)
    train_count, inference_count = differences.shape[:2]
    for index in range(repeats):
        train_draw = rng.integers(0, train_count, train_count)
        inference_draw = rng.integers(0, inference_count, inference_count)
        group_draw = rng.integers(0, len(names), len(names))
        selected = totals[np.ix_(train_draw, inference_draw, group_draw)]
        denominator = (train_count * inference_count * counts[group_draw].sum())
        samples[index] = selected.sum() / denominator
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(differences.mean()), float(low), float(high)


def csv_file(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_report(args):
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 hierarchical bootstrap replicates")
    manifest = verify_manifest(args.results)
    if manifest.get("experiment") != "ecir-query-direction-sensitivity":
        raise ValueError("Expected query-direction-sensitivity results")
    if source_hashes() != manifest["source_sha256"]:
        raise ValueError("Sensitivity source changed since result generation")
    section = manifest["sweep_config"]
    validate_sweep_config(section, {
        "revision_config": manifest["revision_config"],
        "extension_config": {"inference_seeds": section["inference_seeds"]},
    })
    summary = json.loads((args.results / "summary.json").read_text())
    expected = {
        (method, samples, training_seed, inference_seed)
        for method in METHODS
        for samples in section["sample_counts"]
        for training_seed in section["training_seeds"]
        for inference_seed in section["inference_seeds"]
    }
    actual = {(row["method"], row["samples"], row["train_seed"],
               row["inference_seed"]) for row in summary}
    if len(summary) != len(expected) or actual != expected:
        raise ValueError("Missing, duplicate, or unexpected sensitivity result cells")

    entries = {(row["method"], row["samples"], row["train_seed"],
                row["inference_seed"]): row for row in summary}
    reference, values_by_cell = None, {}
    for method in METHODS:
        for samples in section["sample_counts"]:
            by_training_seed = []
            for training_seed in section["training_seeds"]:
                by_inference_seed = []
                for inference_seed in section["inference_seeds"]:
                    entry = entries[(method, samples, training_seed,
                                     inference_seed)]
                    support, values = load_entry(args.results, manifest, entry,
                                                 reference)
                    if reference is None:
                        reference = support
                    by_inference_seed.append(values)
                by_training_seed.append(np.stack(by_inference_seed))
            values_by_cell[(method, samples)] = np.stack(by_training_seed)

    records = [
        summarize_values(method, samples, values_by_cell[(method, samples)])
        for samples in section["sample_counts"] for method in METHODS
    ]
    seed_rows = []
    original_samples = manifest["revision_config"]["samples"]
    fixed = values_by_cell[("learned_fixed", original_samples)][..., 0]
    adaptive = values_by_cell[("full", original_samples)][..., 0]
    differences = adaptive - fixed
    for index, training_seed in enumerate(section["training_seeds"]):
        seed_rows.append({
            "training_seed": training_seed,
            "ACE NDCG@10": float(adaptive[index].mean()),
            "learned-fixed NDCG@10": float(fixed[index].mean()),
            "ACE - learned-fixed": float(differences[index].mean()),
        })
    mean, low, high = hierarchical_interval(
        differences, [group for _, group in reference], args.bootstrap,
        stable_seed("direction-sensitivity-hierarchical", original_samples))
    comparison = [{
        "comparison": "full - learned_fixed",
        "samples": original_samples,
        "mean_difference": mean,
        "hierarchical_95_low": low,
        "hierarchical_95_high": high,
        "positive_training_seeds": sum(
            row["ACE - learned-fixed"] > 0 for row in seed_rows),
        "training_seeds": len(seed_rows),
        "inference_seeds": len(section["inference_seeds"]),
        "clusters": len(set(group for _, group in reference)),
        "replicates": args.bootstrap,
    }]

    args.out.mkdir(parents=True, exist_ok=False)
    csv_file(args.out / "direction_sensitivity.csv", records)
    csv_file(args.out / "seed_differences_s5.csv", seed_rows)
    csv_file(args.out / "seed_aware_comparison_s5.csv", comparison)

    tex = [
        r"\begin{table}[t]", r"\centering\small",
        r"\begin{tabular}{rccc}", r"\toprule",
        r"Directions $S$ & Learned+fixed & ACE & $\Delta$" + LATEX_ROW_END,
        r"\midrule",
    ]
    for samples in section["sample_counts"]:
        fixed_record = next(row for row in records
                            if row["method"] == "learned_fixed" and
                            row["samples"] == samples)
        full_record = next(row for row in records
                           if row["method"] == "full" and
                           row["samples"] == samples)
        delta = (full_record["NDCG@10 mean"] -
                 fixed_record["NDCG@10 mean"])
        tex.append(" & ".join([
            str(samples), f"{fixed_record['NDCG@10 mean']:.5f}",
            f"{full_record['NDCG@10 mean']:.5f}", f"{delta:+.5f}",
        ]) + LATEX_ROW_END)
    tex.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{TalkPlay full-catalog NDCG@10 sensitivity to the number "
        r"of query directions. Values average five training seeds and ten "
        r"paired inference seeds. The sweep uses frozen checkpoints and does "
        r"not select $S$ on the test set.}",
        r"\label{tab:direction-sensitivity}", r"\end{table}",
    ])
    (args.out / "direction_sensitivity.tex").write_text(
        "\n".join(tex) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(5.6, 3.6))
    for method in METHODS:
        method_rows = [row for row in records if row["method"] == method]
        axis.errorbar(
            [row["samples"] for row in method_rows],
            [row["NDCG@10 mean"] for row in method_rows],
            yerr=[row["NDCG@10 train_sd"] for row in method_rows],
            marker="o", capsize=3, label=LABELS[method])
    axis.set_xscale("log", base=2)
    axis.set_xticks(section["sample_counts"],
                    labels=[str(value) for value in section["sample_counts"]])
    axis.set_xlabel("Number of query directions (S)")
    axis.set_ylabel("NDCG@10")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.out / "direction_sensitivity.pdf", bbox_inches="tight")
    figure.savefig(args.out / "direction_sensitivity.png", dpi=200,
                   bbox_inches="tight")
    plt.close(figure)

    report = [
        "# TalkPlay query-direction sensitivity", "",
        "This is a post-hoc, inference-only sensitivity analysis using frozen",
        "learned-fixed and ACE checkpoints. The test results are not used to",
        "select or retrain a model.", "",
        "## Seed-aware S=5 comparison", "",
        f"ACE - learned fixed: {mean:+.6f}, hierarchical 95% interval ",
        f"[{low:+.6f}, {high:+.6f}]. The difference is positive for ",
        f"{comparison[0]['positive_training_seeds']}/{len(seed_rows)} training seeds.",
        "The interval jointly resamples training seeds, paired inference seeds,",
        "and session/user clusters; it is exploratory and not multiplicity-corrected.",
        "", "## Generated files", "",
        "- `direction_sensitivity.csv`: all S-by-method metric summaries",
        "- `seed_differences_s5.csv`: per-training-seed paired differences",
        "- `seed_aware_comparison_s5.csv`: hierarchical interval",
        "- `direction_sensitivity.tex`: compact paper table",
        "- `direction_sensitivity.pdf` and `.png`: sensitivity plot",
    ]
    (args.out / "REPORT.md").write_text("\n".join(report) + "\n",
                                        encoding="utf-8")
    files = sorted(path for path in args.out.iterdir() if path.is_file())
    dump(args.out / "manifest.json", {
        "schema": 1,
        "experiment": "ecir-query-direction-sensitivity-report",
        "result_manifest_sha256": digest(args.results / "manifest.json"),
        "report_source_sha256": digest(Path(__file__)),
        "bootstrap_replicates": args.bootstrap,
        "files": {path.name: digest(path) for path in files},
    })
    print(f"Direction-sensitivity report saved: {args.out}")


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--data", type=Path, required=True)
    run_parser.add_argument("--original-run", type=Path, required=True)
    run_parser.add_argument("--stochastic-run", type=Path, required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--config", type=Path,
                            default=Path("experiments/ecir/additional_config.json"))
    run_parser.add_argument("--device", choices=["cpu", "cuda", "mps"],
                            default="cpu")
    run_parser.add_argument("--cpu-threads", type=int,
                            default=min(4, os.cpu_count() or 1))
    run_parser.add_argument("--interop-threads", type=int, default=1)
    run_parser.add_argument("--determinism", choices=["strict", "warn"],
                            default="strict")

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--results", type=Path, required=True)
    report_parser.add_argument("--out", type=Path, required=True)
    report_parser.add_argument("--bootstrap", type=int, default=10000)

    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; choose a new versioned path")
    try:
        if args.command == "run":
            configure_runtime(args)
            run_sweep(args)
        else:
            make_report(args)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
