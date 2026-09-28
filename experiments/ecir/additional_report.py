"""Build verified ECIR add-on tables for V-SKNN, efficiency, and diagnostics."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.revision.protocol import (METRICS, aggregate, digest, dump,
                                           read_jsonl, stable_seed,
                                           verify_manifest)

DATASETS = ("talkplay", "ml1m", "lastfm", "amazon_music")
METHODS = ("learned_centroid", "learned_fixed", "full")
LABELS = {"learned_centroid": "Learned centroid",
          "learned_fixed": "Learned centroid + fixed spread",
          "full": "Full (adaptive spread)"}
LATEX_ROW_END = " " + "\\" * 2


def csv_file(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def index_datasets(folders, experiment):
    indexed = {}
    for folder in folders:
        manifest = verify_manifest(folder)
        dataset = manifest.get("dataset")
        if manifest.get("experiment") != experiment:
            raise ValueError(f"Wrong experiment type in {folder}")
        if dataset not in DATASETS or dataset in indexed:
            raise ValueError(f"Unexpected or duplicate dataset: {dataset}")
        indexed[dataset] = (folder, manifest)
    if set(indexed) != set(DATASETS):
        raise ValueError(f"{experiment} requires exactly one folder per dataset")
    return indexed


def index_efficiency(folders):
    indexed = {}
    for folder in folders:
        manifest = verify_manifest(folder)
        key = (manifest.get("dataset"), manifest.get("method"))
        if manifest.get("experiment") != "ecir-efficiency":
            raise ValueError(f"Wrong experiment type in {folder}")
        if key[0] not in DATASETS or key[1] not in METHODS or key in indexed:
            raise ValueError(f"Unexpected or duplicate efficiency result: {key}")
        indexed[key] = (folder, manifest)
    expected = {(dataset, method) for dataset in DATASETS for method in METHODS}
    if set(indexed) != expected:
        raise ValueError("Efficiency input requires all 4 datasets x 3 methods")
    return indexed


def verify_sources(manifest):
    root = Path(__file__).resolve().parents[2]
    for relative, expected in manifest["source_sha256"].items():
        path = root / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Current source differs from evaluated source: {relative}")


def checked_entry(folder, manifest, entry, reference=None):
    if entry["file"] not in manifest["files"]:
        raise ValueError(f"Unverified per-query file: {entry['file']}")
    rows = sorted(read_jsonl(folder / entry["file"]), key=lambda row: row["event"])
    support = [(row["event"], row["group"]) for row in rows]
    if len(support) != len(set(support)):
        raise ValueError(f"Duplicate query in {entry['file']}")
    if reference is not None and support != reference:
        raise ValueError("V-SKNN and Full must use identical ordered test support")
    recomputed = aggregate(rows)
    if entry["n"] != len(rows) or any(
            abs(recomputed[metric] - entry[metric]) > 1e-10 for metric in METRICS):
        raise ValueError(f"Summary mismatch for {entry['file']}")
    values = np.asarray([[row[metric] for metric in METRICS] for row in rows],
                        dtype=np.float64)
    return rows, support, values


def paired_interval(differences, groups, repeats, seed):
    names = sorted(set(groups))
    grouped = {name: [] for name in names}
    for difference, group in zip(differences, groups):
        grouped[group].append(difference)
    totals = np.asarray([sum(grouped[name]) for name in names])
    counts = np.asarray([len(grouped[name]) for name in names])
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats)
    for index in range(repeats):
        selected = rng.integers(0, len(names), len(names))
        samples[index] = totals[selected].sum() / counts[selected].sum()
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(np.mean(differences)), float(low), float(high)


def tex_escape(text):
    return str(text).replace("_", r"\_")


def optional_number(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def optional_mib(value):
    return "--" if value is None else f"{value / (1024 ** 2):.1f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-knn-results", nargs="+", type=Path, required=True)
    parser.add_argument("--stochastic-results", nargs="+", type=Path, required=True)
    parser.add_argument("--efficiency-results", nargs="+", type=Path, required=True)
    parser.add_argument("--diagnostic-results", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.out.exists() or args.bootstrap < 100:
        parser.error("Use a new output path and at least 100 bootstrap replicates")

    sessions = index_datasets(args.session_knn_results, "ecir-session-knn")
    stochastic = index_datasets(args.stochastic_results,
                                "ecir-stochastic-evaluation")
    diagnostics = index_datasets(args.diagnostic_results,
                                 "ecir-kappa-diagnostics")
    efficiency = index_efficiency(args.efficiency_results)
    session_records, comparisons, efficiency_records, diagnostic_records = [], [], [], []
    provenance = {}

    for dataset in DATASETS:
        session_folder, session_manifest = sessions[dataset]
        stochastic_folder, stochastic_manifest = stochastic[dataset]
        diagnostic_folder, diagnostic_manifest = diagnostics[dataset]
        for folder, manifest in ((session_folder, session_manifest),
                                 (stochastic_folder, stochastic_manifest),
                                 (diagnostic_folder, diagnostic_manifest)):
            verify_sources(manifest)
            provenance[str(folder / "manifest.json")] = digest(folder / "manifest.json")
        if len({session_manifest["data_manifest_sha256"],
                stochastic_manifest["data_manifest_sha256"],
                diagnostic_manifest["data_manifest_sha256"]}) != 1:
            raise ValueError(f"Prepared data differs across {dataset} inputs")
        if diagnostic_manifest["stochastic_run_manifest_sha256"] != stochastic_manifest["robust_run_manifest_sha256"]:
            raise ValueError(f"Diagnostics use the wrong stochastic run for {dataset}")

        session_summary = json.loads((session_folder / "summary.json").read_text())
        if len(session_summary) != 1 or session_summary[0]["method"] != "vsknn":
            raise ValueError(f"Expected one V-SKNN summary row for {dataset}")
        session_rows, support, session_values = checked_entry(
            session_folder, session_manifest, session_summary[0])
        groups = [row["group"] for row in session_rows]

        stochastic_summary = json.loads(
            (stochastic_folder / "summary.json").read_text())
        full_entries = [entry for entry in stochastic_summary
                        if entry["method"] == "full"]
        seeds = stochastic_manifest["revision_config"]["seeds"]
        inference = stochastic_manifest["extension_config"]["inference_seeds"]
        expected = [(training_seed, inference_seed) for training_seed in seeds
                    for inference_seed in inference]
        actual = [(entry["train_seed"], entry["inference_seed"])
                  for entry in full_entries]
        if actual != expected:
            raise ValueError(f"Incomplete Full repetitions for {dataset}")
        repetitions = []
        for entry in full_entries:
            _, _, values = checked_entry(stochastic_folder, stochastic_manifest,
                                         entry, support)
            repetitions.append(values)
        full_values = np.stack(repetitions).mean(axis=0)
        record = {"dataset": dataset,
                  "neighbor_count": session_manifest["selection"]["neighbor_count"],
                  "recency_decay": session_manifest["selection"]["recency_decay"],
                  "queries": len(session_rows)}
        for metric_index, metric in enumerate(METRICS):
            record[f"V-SKNN {metric}"] = float(session_values[:, metric_index].mean())
            record[f"Full {metric}"] = float(full_values[:, metric_index].mean())
            differences = full_values[:, metric_index] - session_values[:, metric_index]
            mean, low, high = paired_interval(
                differences, groups, args.bootstrap,
                stable_seed("ecir-additional-bootstrap", dataset, metric))
            comparisons.append({
                "dataset": dataset, "comparison": "full - vsknn",
                "metric": metric, "mean_difference": mean,
                "bootstrap_95_low": low, "bootstrap_95_high": high,
                "clusters": len(set(groups)), "replicates": args.bootstrap,
            })
        session_records.append(record)

        diagnostic = json.loads((diagnostic_folder / "summary.json").read_text())
        pooled = diagnostic["pooled_seed_query"]
        event = diagnostic["event_mean_across_training_seeds"]
        correlation = diagnostic["spearman_event_level"]
        diagnostic_records.append({
            "dataset": dataset, "events": diagnostic["events"],
            "kappa_mean": pooled["kappa"]["mean"],
            "kappa_q10": pooled["kappa"]["quantiles"]["q10"],
            "kappa_median": pooled["kappa"]["quantiles"]["q50"],
            "kappa_q90": pooled["kappa"]["quantiles"]["q90"],
            "radius_degrees_mean": pooled["radius_degrees"]["mean"],
            "radius_degrees_q10": pooled["radius_degrees"]["quantiles"]["q10"],
            "radius_degrees_median": pooled["radius_degrees"]["quantiles"]["q50"],
            "radius_degrees_q90": pooled["radius_degrees"]["quantiles"]["q90"],
            "centroid_shift_degrees_median": pooled["centroid_shift_degrees"]["quantiles"]["q50"],
            "context_coherence_mean": event["context_coherence"]["mean"],
            "fraction_at_kappa_minimum": diagnostic["boundary_fractions"]["at_minimum"],
            "fraction_at_kappa_maximum": diagnostic["boundary_fractions"]["at_maximum"],
            "spearman_radius_context_length": correlation["radius_vs_context_length"],
            "spearman_radius_context_coherence": correlation["radius_vs_context_coherence"],
        })

        for method in METHODS:
            folder, manifest = efficiency[(dataset, method)]
            verify_sources(manifest)
            provenance[str(folder / "manifest.json")] = digest(folder / "manifest.json")
            if manifest["data_manifest_sha256"] != stochastic_manifest["data_manifest_sha256"]:
                raise ValueError(f"Efficiency data mismatch for {dataset}/{method}")
            if manifest["stochastic_run_manifest_sha256"] != stochastic_manifest["robust_run_manifest_sha256"]:
                raise ValueError(f"Efficiency stochastic-run mismatch for {dataset}/{method}")
            result = json.loads((folder / "efficiency.json").read_text())
            if result["dataset"] != dataset or result["method"] != method:
                raise ValueError(f"Efficiency payload mismatch in {folder}")
            efficiency_records.append({
                "dataset": dataset, "method": method,
                "device": result["device"],
                "torch_version": manifest["environment"]["versions"]["torch"],
                "platform": manifest["environment"]["platform"],
                "probes": result["probes"], "parameters": result["parameters"],
                "catalog_size": result["catalog_size"],
                "batch_size": result["batch_size"],
                "query_median_ms_per_query": result["query_construction"]["median_ms_per_query"],
                "scoring_median_ms_per_query": result["exhaustive_scoring"]["median_ms_per_query"],
                "total_median_ms_per_query": result["total"]["median_ms_per_query"],
                "total_p95_ms_per_query": result["total"]["p95_ms_per_query"],
                "accelerator_memory_observed_delta_bytes": result["accelerator_memory_observed_delta_bytes"],
                "process_peak_rss_bytes": result["process_peak_rss_bytes"],
            })

    args.out.mkdir(parents=True)
    csv_file(args.out / "session_knn.csv", session_records)
    csv_file(args.out / "paired_session_knn.csv", comparisons)
    csv_file(args.out / "efficiency.csv", efficiency_records)
    csv_file(args.out / "kappa_diagnostics.csv", diagnostic_records)

    session_tex = [
        r"\begin{table*}[t]", r"\centering\scriptsize",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Dataset & $K$ / $\lambda$ & V-SKNN NDCG@10 & Full NDCG@10 & $\Delta$ & 95\% CI" + LATEX_ROW_END,
        r"\midrule",
    ]
    for record in session_records:
        comparison = next(row for row in comparisons
                          if row["dataset"] == record["dataset"] and
                          row["metric"] == "NDCG@10")
        session_tex.append(" & ".join([
            tex_escape(record["dataset"]),
            f"{record['neighbor_count']} / {record['recency_decay']:.2f}",
            f"{record['V-SKNN NDCG@10']:.5f}",
            f"{record['Full NDCG@10']:.5f}",
            f"{comparison['mean_difference']:+.5f}",
            f"[{comparison['bootstrap_95_low']:+.5f}, {comparison['bootstrap_95_high']:+.5f}]",
        ]) + LATEX_ROW_END)
    session_tex.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Validation-selected recency-weighted V-SKNN versus Full. Differences are Full minus V-SKNN; exploratory 95\% intervals cluster-resample sessions/users after averaging Full repetitions.}",
        r"\label{tab:ecir-vsknn}", r"\end{table*}",
    ])
    (args.out / "session_knn.tex").write_text("\n".join(session_tex) + "\n")

    efficiency_tex = [
        r"\begin{table*}[t]", r"\centering\scriptsize",
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"Dataset & Method & Probes & Query ms & Score ms & Total ms (p95) & Acc. $\Delta$ MiB & RSS MiB" + LATEX_ROW_END,
        r"\midrule",
    ]
    for row in efficiency_records:
        efficiency_tex.append(" & ".join([
            tex_escape(row["dataset"]), LABELS[row["method"]], str(row["probes"]),
            f"{row['query_median_ms_per_query']:.3f}",
            f"{row['scoring_median_ms_per_query']:.3f}",
            f"{row['total_median_ms_per_query']:.3f} ({row['total_p95_ms_per_query']:.3f})",
            optional_mib(row["accelerator_memory_observed_delta_bytes"]),
            optional_mib(row["process_peak_rss_bytes"]),
        ]) + LATEX_ROW_END)
    efficiency_tex.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Per-query inference cost on the recorded device. Times are medians over synchronized batches; parentheses give batch-level p95. Accelerator memory is the largest observed live-allocation increase and RSS is the process high-water mark.}",
        r"\label{tab:ecir-efficiency}", r"\end{table*}",
    ])
    (args.out / "efficiency.tex").write_text("\n".join(efficiency_tex) + "\n")

    diagnostic_tex = [
        r"\begin{table*}[t]", r"\centering\scriptsize",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Dataset & Median $\kappa$ & Median radius & $\kappa$ at min & $\kappa$ at max & $\rho$(radius,len.) & $\rho$(radius,coh.)" + LATEX_ROW_END,
        r"\midrule",
    ]
    for row in diagnostic_records:
        diagnostic_tex.append(" & ".join([
            tex_escape(row["dataset"]), f"{row['kappa_median']:.2f}",
            f"{row['radius_degrees_median']:.2f}" + r"$^\circ$",
            f"{100 * row['fraction_at_kappa_minimum']:.1f}" + r"\%",
            f"{100 * row['fraction_at_kappa_maximum']:.1f}" + r"\%",
            optional_number(row["spearman_radius_context_length"]),
            optional_number(row["spearman_radius_context_coherence"]),
        ]) + LATEX_ROW_END)
    diagnostic_tex.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Validation-set adaptive-spread diagnostics pooled across training seeds. Radius is $\arctan(1/\sqrt{\kappa})$.}",
        r"\label{tab:ecir-kappa-diagnostics}", r"\end{table*}",
    ])
    (args.out / "kappa_diagnostics.tex").write_text(
        "\n".join(diagnostic_tex) + "\n")

    markdown = [
        "# ECIR additional experiment report", "",
        "All inputs and source hashes were verified before aggregation. V-SKNN",
        "hyperparameters were chosen on validation NDCG@10. Full effectiveness",
        "is averaged over all declared training and inference seeds before paired",
        "cluster bootstrap comparisons. Efficiency is inference-only, and the",
        "κ/radius analysis uses validation queries only.", "",
        "Generated artifacts:", "",
        "- `session_knn.csv` and `session_knn.tex`",
        "- `paired_session_knn.csv`",
        "- `efficiency.csv` and `efficiency.tex`",
        "- `kappa_diagnostics.csv` and `kappa_diagnostics.tex`", "",
        "The paired intervals are exploratory and not multiplicity-corrected.",
    ]
    (args.out / "REPORT.md").write_text("\n".join(markdown) + "\n")
    files = sorted(path for path in args.out.iterdir() if path.is_file())
    dump(args.out / "manifest.json", {
        "schema": 1, "experiment": "ecir-additional-report",
        "result_manifests": provenance,
        "report_source_sha256": digest(Path(__file__)),
        "bootstrap_replicates": args.bootstrap,
        "files": {path.name: digest(path) for path in files},
    })
    print(f"Additional ECIR tables and provenance saved: {args.out}")


if __name__ == "__main__":
    main()
