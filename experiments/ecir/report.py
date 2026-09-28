"""Build ECIR tables, figures, and paired intervals from verified result files."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from experiments.revision.protocol import (METRICS, aggregate, digest, dump,
                                           read_jsonl, stable_seed,
                                           verify_manifest)

DATASETS = ("talkplay", "ml1m", "lastfm", "amazon_music")
METHODS = ("popularity", "last_item", "transition_knn", "content_sasrec",
           "uniform", "weighted_prf", "learned_centroid", "weighted_fixed",
           "learned_fixed", "full")
LABELS = {
    "popularity": "Popularity",
    "last_item": "Last-item cosine",
    "transition_knn": "Transition-kNN",
    "content_sasrec": "Content-SASRec",
    "uniform": "Uniform centroid",
    "weighted_prf": "Weighted PRF",
    "learned_centroid": "Learned centroid",
    "weighted_fixed": "Weighted PRF + fixed spread",
    "learned_fixed": "Learned centroid + fixed spread",
    "full": "Full (adaptive spread)",
}
TRAINED = {"content_sasrec", "learned_centroid", "learned_fixed", "full"}
STOCHASTIC = {"weighted_fixed", "learned_fixed", "full"}
LATEX_ROW_END = " " + "\\" * 2
CONTRASTS = (
    ("content_sasrec", "uniform"),
    ("learned_centroid", "weighted_prf"),
    ("weighted_fixed", "weighted_prf"),
    ("learned_fixed", "learned_centroid"),
    ("full", "learned_fixed"),
    ("full", "learned_centroid"),
    ("full", "weighted_prf"),
    ("full", "content_sasrec"),
    ("full", "popularity"),
    ("full", "last_item"),
    ("full", "transition_knn"),
)


def csv_file(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def index_folders(folders, experiment=None):
    indexed = {}
    for folder in folders:
        manifest = verify_manifest(folder)
        dataset = manifest.get("dataset")
        if dataset not in DATASETS or dataset in indexed:
            raise ValueError(f"Unexpected or duplicate dataset: {dataset}")
        if experiment is not None and manifest.get("experiment") != experiment:
            raise ValueError(f"Wrong experiment in {folder}")
        indexed[dataset] = (folder, manifest)
    if set(indexed) != set(DATASETS):
        raise ValueError("Exactly one result folder for every retained dataset is required")
    return indexed


def require_same(indexed, fields, label):
    signatures = set()
    for _, manifest in indexed.values():
        values = [manifest[field] for field in fields]
        signatures.add(json.dumps(values, sort_keys=True))
    if len(signatures) != 1:
        raise ValueError(f"{label} results do not share {', '.join(fields)}")


def verify_sources(root, manifest):
    for relative, expected in manifest["source_sha256"].items():
        path = root / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Current source differs from evaluated source: {relative}")


def checked_rows(folder, manifest, entry, reference):
    if entry["file"] not in manifest["files"]:
        raise ValueError(f"Unverified per-query file: {entry['file']}")
    rows = sorted(read_jsonl(folder / entry["file"]), key=lambda row: row["event"])
    events = [(row["event"], row["group"]) for row in rows]
    if len(events) != len(set(events)):
        raise ValueError(f"Duplicate evaluation event in {entry['file']}")
    if reference is not None and events != reference:
        raise ValueError("All methods must use identical ordered test support")
    recomputed = aggregate(rows)
    if entry["n"] != len(rows) or any(
            abs(recomputed[metric] - entry[metric]) > 1e-10 for metric in METRICS):
        raise ValueError(f"Summary mismatch for {entry['file']}")
    values = np.asarray([[row[metric] for metric in METRICS] for row in rows],
                        dtype=np.float64)
    return rows, events, values


def entries_for(summary, method):
    return [entry for entry in summary if entry["method"] == method]


def paired_interval(differences, groups, repeats, seed):
    """Cluster bootstrap on per-query differences after averaging repetitions."""
    names = sorted(set(groups))
    by_group = {name: [] for name in names}
    for difference, group in zip(differences, groups):
        by_group[group].append(difference)
    totals = np.asarray([sum(by_group[name]) for name in names])
    counts = np.asarray([len(by_group[name]) for name in names])
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats)
    for index in range(repeats):
        selected = rng.integers(0, len(names), len(names))
        samples[index] = totals[selected].sum() / counts[selected].sum()
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(np.mean(differences)), float(low), float(high)


def record_for(dataset, method, values):
    """Values have axes: training seed, inference seed, query, metric."""
    record = {
        "dataset": dataset,
        "method": method,
        "queries": values.shape[2],
        "training_seeds": values.shape[0] if method in TRAINED else 0,
        "inference_seeds": values.shape[1] if method in STOCHASTIC else 0,
    }
    train_scores = values.mean(axis=(1, 2))
    inference_scores = values.mean(axis=2)
    for index, metric in enumerate(METRICS):
        record[f"{metric} mean"] = float(values[..., index].mean())
        record[f"{metric} train_sd"] = (
            float(train_scores[:, index].std(ddof=1)) if values.shape[0] > 1 else 0.)
        if values.shape[1] > 1:
            within_variances = inference_scores[:, :, index].var(axis=1, ddof=1)
            record[f"{metric} inference_sd"] = float(math.sqrt(within_variances.mean()))
        else:
            record[f"{metric} inference_sd"] = 0.
    return record


def tex_escape(value):
    return value.replace("_", r"\_")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-results", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-results", type=Path, nargs="+", required=True)
    parser.add_argument("--stochastic-results", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.out.exists() or args.bootstrap < 100:
        parser.error("Use a new output path and at least 100 bootstrap replicates")

    originals = index_folders(args.original_results)
    baselines = index_folders(args.baseline_results,
                              "ecir-representative-baselines")
    stochastic = index_folders(args.stochastic_results,
                               "ecir-stochastic-evaluation")
    require_same(originals, (("config"), ("source_sha256")), "Original")
    require_same(baselines, (("config"), ("source_sha256")), "Baseline")
    require_same(stochastic, (("extension_config"), ("revision_config"),
                              ("source_sha256")), "Stochastic")
    runtime_signatures = set()
    for collection in (originals, baselines, stochastic):
        for _, manifest in collection.values():
            environment = manifest["environment"]
            runtime_signatures.add((
                manifest["device"], environment["determinism"],
                environment["mps_cpu_fallback"]))
    if len(runtime_signatures) != 1:
        raise ValueError("All result families must share device and determinism policy")
    root = Path(__file__).resolve().parents[2]
    all_records, comparisons, provenance = [], [], {}

    for dataset in DATASETS:
        original_folder, original_manifest = originals[dataset]
        baseline_folder, baseline_manifest = baselines[dataset]
        stochastic_folder, stochastic_manifest = stochastic[dataset]
        for folder in (original_folder, baseline_folder, stochastic_folder):
            provenance[str(folder / "manifest.json")] = digest(folder / "manifest.json")
        for manifest in (original_manifest, baseline_manifest, stochastic_manifest):
            verify_sources(root, manifest)
        data_hashes = {original_manifest["data_manifest_sha256"],
                       baseline_manifest["data_manifest_sha256"],
                       stochastic_manifest["data_manifest_sha256"]}
        if len(data_hashes) != 1:
            raise ValueError(f"Prepared data differs across {dataset} experiments")
        if stochastic_manifest["original_run_manifest_sha256"] != original_manifest["run_manifest_sha256"]:
            raise ValueError(f"Stochastic extension does not use {dataset}'s original run")

        original_summary = json.loads((original_folder / "summary.json").read_text())
        baseline_summary = json.loads((baseline_folder / "summary.json").read_text())
        stochastic_summary = json.loads((stochastic_folder / "summary.json").read_text())
        seeds = original_manifest["config"]["seeds"]
        if baseline_manifest["config"]["seeds"] != seeds:
            raise ValueError(f"Baseline training seeds differ for {dataset}")
        inference_seeds = stochastic_manifest["extension_config"]["inference_seeds"]
        method_values, reference, groups = {}, None, None

        for method in ("uniform", "weighted_prf", "learned_centroid"):
            entries = entries_for(original_summary, method)
            if [entry["seed"] for entry in entries] != seeds:
                raise ValueError(f"Incomplete original repetitions for {dataset}/{method}")
            repetitions = []
            for entry in entries:
                rows, events, values = checked_rows(
                    original_folder, original_manifest, entry, reference)
                if reference is None:
                    reference = events
                    groups = [row["group"] for row in rows]
                repetitions.append(values)
            stacked = np.stack(repetitions)[:, None, :, :]
            if method != "learned_centroid":
                if not np.array_equal(stacked, np.repeat(stacked[:1], len(seeds), axis=0)):
                    raise ValueError(f"Deterministic method unexpectedly varies: {method}")
                stacked = stacked[:1]
            method_values[method] = stacked

        for method in ("popularity", "last_item", "transition_knn", "content_sasrec"):
            entries = entries_for(baseline_summary, method)
            expected = seeds if method == "content_sasrec" else [None]
            if [entry["seed"] for entry in entries] != expected:
                raise ValueError(f"Incomplete baseline repetitions for {dataset}/{method}")
            repetitions = []
            for entry in entries:
                _, _, values = checked_rows(
                    baseline_folder, baseline_manifest, entry, reference)
                repetitions.append(values)
            method_values[method] = np.stack(repetitions)[:, None, :, :]

        for method in ("weighted_fixed", "learned_fixed", "full"):
            entries = entries_for(stochastic_summary, method)
            train_seeds = [None] if method == "weighted_fixed" else seeds
            expected = [(train_seed, inference_seed) for train_seed in train_seeds
                        for inference_seed in inference_seeds]
            actual = [(entry["train_seed"], entry["inference_seed"])
                      for entry in entries]
            if actual != expected:
                raise ValueError(f"Incomplete stochastic repetitions for {dataset}/{method}")
            by_train = []
            for train_seed in train_seeds:
                by_inference = []
                for inference_seed in inference_seeds:
                    entry = next(row for row in entries
                                 if row["train_seed"] == train_seed and
                                 row["inference_seed"] == inference_seed)
                    _, _, values = checked_rows(
                        stochastic_folder, stochastic_manifest, entry, reference)
                    by_inference.append(values)
                by_train.append(np.stack(by_inference))
            method_values[method] = np.stack(by_train)

        for method in METHODS:
            values = method_values[method]
            all_records.append(record_for(dataset, method, values))

        query_means = {method: values.mean(axis=(0, 1))
                       for method, values in method_values.items()}
        for left, right in CONTRASTS:
            differences = query_means[left][:, 0] - query_means[right][:, 0]
            mean, low, high = paired_interval(
                differences, groups, args.bootstrap,
                stable_seed("ecir-bootstrap", dataset, left, right))
            comparisons.append({
                "dataset": dataset, "comparison": f"{left} - {right}",
                "metric": "NDCG@10", "mean_difference": mean,
                "bootstrap_95_low": low, "bootstrap_95_high": high,
                "clusters": len(set(groups)), "replicates": args.bootstrap,
            })

    args.out.mkdir(parents=True)
    all_records.sort(key=lambda row: (DATASETS.index(row["dataset"]),
                                     METHODS.index(row["method"])))
    csv_file(args.out / "metrics.csv", all_records)
    csv_file(args.out / "paired_comparisons.csv", comparisons)

    markdown = [
        "# ECIR experiment results", "",
        "All values were regenerated from verified, common-support per-query files.",
        "Each cell is mean ± training-seed SD / inference-seed SD. Training SD is",
        "computed after averaging inference repetitions; inference SD is the pooled",
        "within-training-seed SD. Zeros denote deterministic axes, not certainty.", "",
        "| Dataset | Method | NDCG@10 | Recall@10 | Recall@50 | MRR@50 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in all_records:
        cells = [f"{row[metric + ' mean']:.5f} ± "
                 f"{row[metric + ' train_sd']:.5f} / "
                 f"{row[metric + ' inference_sd']:.5f}" for metric in METRICS]
        markdown.append("| " + " | ".join(
            [row["dataset"], LABELS[row["method"]], *cells]) + " |")
    markdown.extend([
        "", "Paired intervals in `paired_comparisons.csv` resample session/user",
        "clusters after averaging all repetitions per query. They are exploratory,",
        "condition on these trained models/catalogs, and are not multiplicity-corrected.",
    ])
    with (args.out / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(markdown) + "\n")

    result_tex = []
    for dataset in DATASETS:
        rows = [row for row in all_records if row["dataset"] == dataset]
        maxima = {metric: max(row[metric + " mean"] for row in rows)
                  for metric in METRICS}
        result_tex.extend([
            r"\begin{table}[t]", r"\centering\scriptsize",
            r"\begin{tabular}{lrrrr}", r"\toprule",
            "Method & NDCG@10 & R@10 & R@50 & MRR@50" + LATEX_ROW_END,
            r"\midrule",
        ])
        for row in rows:
            cells = []
            for metric in METRICS:
                value = row[metric + " mean"]
                text = f"{value:.5f}"
                cells.append(r"\textbf{" + text + "}" if value == maxima[metric]
                             else text)
            result_tex.append(" & ".join([LABELS[row["method"]], *cells]) +
                              LATEX_ROW_END)
        result_tex.extend([
            r"\bottomrule", r"\end{tabular}",
            r"\caption{Full-catalog results on " + tex_escape(dataset) +
            r". Means average all declared repetitions; dispersion is reported separately.}",
            r"\label{tab:ecir-results-" + dataset.replace("_", "-") + r"}",
            r"\end{table}", "",
        ])
    with (args.out / "results.tex").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(result_tex) + "\n")

    uncertainty_tex = [
        r"\begin{table*}[t]", r"\centering\scriptsize",
        r"\begin{tabular}{llrrr}", r"\toprule",
        "Dataset & Method & NDCG@10 & Train SD & Inference SD" + LATEX_ROW_END,
        r"\midrule",
    ]
    for row in all_records:
        if row["method"] not in STOCHASTIC:
            continue
        uncertainty_tex.append(" & ".join([
            tex_escape(row["dataset"]), LABELS[row["method"]],
            f"{row['NDCG@10 mean']:.5f}", f"{row['NDCG@10 train_sd']:.5f}",
            f"{row['NDCG@10 inference_sd']:.5f}"]) + LATEX_ROW_END)
    uncertainty_tex.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Separate training- and inference-randomness dispersion. Inference SD is the pooled within-training-seed sample SD over declared inference seeds.}",
        r"\label{tab:ecir-uncertainty}", r"\end{table*}",
    ])
    with (args.out / "uncertainty.tex").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(uncertainty_tex) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, dataset in zip(axes.flat, DATASETS):
        rows = [row for row in all_records if row["dataset"] == dataset]
        errors = [math.hypot(row["NDCG@10 train_sd"],
                             row["NDCG@10 inference_sd"]) for row in rows]
        axis.barh([LABELS[row["method"]] for row in rows],
                  [row["NDCG@10 mean"] for row in rows], xerr=errors, capsize=2)
        axis.set_title(dataset)
        axis.set_xlabel("NDCG@10 (descriptive SD bars)")
        axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(args.out / "ndcg10.pdf", bbox_inches="tight")
    figure.savefig(args.out / "ndcg10.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    files = sorted(path for path in args.out.iterdir() if path.is_file())
    dump(args.out / "manifest.json", {
        "schema": 1, "experiment": "ecir-combined-report",
        "result_manifests": provenance,
        "report_source_sha256": digest(Path(__file__)),
        "bootstrap_replicates": args.bootstrap,
        "files": {path.name: digest(path) for path in files},
    })
    print(f"ECIR tables, figure, intervals, and provenance saved: {args.out}")


if __name__ == "__main__":
    main()
