"""Produce every revised table/plot from the SAME verified per-query results."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .protocol import (LABELS, METHODS, METRICS, aggregate, digest, dump,
                       read_jsonl, verify_manifest)

DATASETS = ("talkplay", "ml1m", "lastfm", "amazon_music")


def paired_interval(differences, groups, rng, repeats):
    """Cluster bootstrap after averaging seeds; seeds are NOT independent users."""
    names = sorted(set(groups))
    by_group = {g: [] for g in names}
    for value, group in zip(differences, groups):
        by_group[group].append(value)
    totals = np.array([sum(by_group[g]) for g in names])
    counts = np.array([len(by_group[g]) for g in names])
    samples = np.empty(repeats)
    for i in range(repeats):
        selected = rng.integers(0, len(names), len(names))
        samples[i] = totals[selected].sum() / counts[selected].sum()
    return float(np.mean(differences)), *map(float, np.quantile(samples, [.025, .975]))


def csv_file(path, rows):
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    a = p.parse_args()
    if a.out.exists() or a.bootstrap < 100:
        p.error("Use a new output directory and at least 100 bootstrap replicates")
    all_rows, comparisons, provenance, populations = [], [], {}, []
    configs, code_hashes, datasets = [], [], set()
    for folder in a.results:
        manifest = verify_manifest(folder)
        dataset, config = manifest["dataset"], manifest["config"]
        if manifest["source_sha256"]["experiments/revision/protocol.py"] != digest(Path(__file__).with_name("protocol.py")):
            raise ValueError("Reporting protocol differs from evaluated protocol")
        if dataset not in DATASETS or dataset in datasets:
            raise ValueError(f"Unexpected/duplicate dataset {dataset}")
        datasets.add(dataset)
        configs.append(config)
        code_hashes.append(manifest["source_sha256"])
        provenance[str(folder / "manifest.json")] = digest(folder / "manifest.json")
        prep = manifest["preparation"]
        for split in ("train", "validation", "test"):
            count = prep["counts"][split]
            populations.append({"dataset": dataset, "split": split,
                                "catalog_size": prep["catalog_size"],
                                "assigned": count.get("assigned", 0),
                                "retained": count.get("retained", 0),
                                "repeated_target": count.get("repeated_target", 0),
                                "missing_target_embedding": count.get("missing_target_embedding", 0),
                                "missing_context_embedding": count.get("missing_context_embedding", 0),
                                "no_context": count.get("no_context", 0)})
        summary = json.loads((folder / "summary.json").read_text())
        expected = {(m, s) for m in METHODS for s in config["seeds"]}
        if len(summary) != len(expected) or {(r["method"], r["seed"]) for r in summary} != expected:
            raise ValueError("Missing/extra method or seed; cannot produce complete paper table")
        arrays, reference_events, groups = {}, None, None
        for method in METHODS:
            seed_rows = []
            for seed in config["seeds"]:
                entry = next(r for r in summary if r["method"] == method and r["seed"] == seed)
                if entry["file"] not in manifest["files"]:
                    raise ValueError("Unverified result file")
                rows = sorted(read_jsonl(folder / entry["file"]), key=lambda r: r["event"])
                events = [(r["event"], r["group"]) for r in rows]
                if len(set(events)) != len(events):
                    raise ValueError("Duplicate evaluation event")
                if reference_events is None:
                    reference_events, groups = events, [r["group"] for r in rows]
                if events != reference_events:
                    raise ValueError("Methods/seeds do not share identical evaluation support")
                recomputed = aggregate(rows)
                if entry["n"] != len(rows) or any(abs(recomputed[k] - entry[k]) > 1e-10 for k in METRICS):
                    raise ValueError("Summary does not match per-query files")
                seed_rows.append([[r[k] for k in METRICS] for r in rows])
            array = np.asarray(seed_rows)
            arrays[method] = array
            means = array.mean(axis=1)
            record = {"dataset": dataset, "method": method, "queries": len(reference_events),
                      "seeds": len(config["seeds"])}
            for j, metric in enumerate(METRICS):
                record[metric + " mean"] = float(means[:, j].mean())
                record[metric + " seed_sd"] = float(means[:, j].std(ddof=1)) if len(means) > 1 else 0.
            all_rows.append(record)
        for baseline in METHODS:
            if baseline == "full":
                continue
            differences = (arrays["full"][:, :, 0] - arrays[baseline][:, :, 0]).mean(axis=0)
            mean, low, high = paired_interval(differences, groups, np.random.default_rng(2026), a.bootstrap)
            comparisons.append({"dataset": dataset, "comparison": "full - " + baseline,
                                "metric": "NDCG@10", "mean_difference": mean,
                                "bootstrap_95_low": low, "bootstrap_95_high": high,
                                "clusters": len(set(groups)), "replicates": a.bootstrap})
    if datasets != set(DATASETS):
        raise ValueError("All four retained datasets are required; no silent cherry-picking")
    if any(c != configs[0] for c in configs) or any(c != code_hashes[0] for c in code_hashes):
        raise ValueError("Datasets were evaluated with different code/configurations")
    a.out.mkdir(parents=True)
    all_rows.sort(key=lambda r: (DATASETS.index(r["dataset"]), METHODS.index(r["method"])))
    csv_file(a.out / "metrics.csv", all_rows)
    csv_file(a.out / "paired_comparisons.csv", comparisons)
    csv_file(a.out / "population_counts.csv", populations)
    count_tex = [r"\begin{table}[t]\centering\small",
                 r"\begin{tabular}{lrrrr}\toprule",
                 r"Dataset & Catalog & Train & Validation & Test \\", r"\midrule"]
    for dataset in DATASETS:
        pop = {r["split"]: r for r in populations if r["dataset"] == dataset}
        count_tex.append(" & ".join([dataset.replace("_", r"\_"),
                                      str(pop["train"]["catalog_size"]),
                                      *[str(pop[s]["retained"]) for s in ("train", "validation", "test")]]) + r" \\")
    count_tex.extend([r"\bottomrule\end{tabular}",
                      r"\caption{Actual retained prediction-event counts after eligibility checks. Detailed exclusions are in population\_counts.csv; training counts are events, not users.}",
                      r"\label{tab:revision-populations}\end{table}"])
    with (a.out / "populations.tex").open("x") as f:
        f.write("\n".join(count_tex) + "\n")
    tex = [r"\begin{table*}[t]", r"\centering\scriptsize",
           r"\begin{tabular}{llrrrr}", r"\toprule",
           r"Dataset & Method & NDCG@10 & Recall@10 & Recall@50 & MRR@50 \\", r"\midrule"]
    markdown = ["# Revised experiment results", "",
                "Generated from verified per-query files; no historical metrics were imported.", "",
                "Values are seed means ± sample standard deviation (not confidence intervals).",
                "Paired bootstrap intervals in paired_comparisons.csv resample session/user clusters",
                "after averaging per-query differences across seeds. These intervals are exploratory,",
                "not multiple-comparison-corrected, and condition on these trained seeds and catalogs.", "",
                "| Dataset | Method | NDCG@10 | Recall@10 | Recall@50 | MRR@50 |",
                "|---|---|---:|---:|---:|---:|"]
    for row in all_rows:
        vals = [f"{row[k + ' mean']:.5f} ± {row[k + ' seed_sd']:.5f}" for k in METRICS]
        markdown.append("| " + " | ".join([row["dataset"], LABELS[row["method"]], *vals]) + " |")
        texvals = [v.replace("±", r"$\pm$") for v in vals]
        tex.append(" & ".join([row["dataset"].replace("_", r"\_"), LABELS[row["method"]], *texvals]) + r" \\")
    tex.extend([r"\bottomrule", r"\end{tabular}",
                r"\caption{Full-catalog results: mean $\pm$ sample standard deviation across training/sampling seeds. All methods share query support. Last.fm is unordered artist retrieval.}",
                r"\label{tab:revision-results}", r"\end{table*}"])
    with (a.out / "results.tex").open("x") as f:
        f.write("\n".join(tex) + "\n")
    with (a.out / "REPORT.md").open("x") as f:
        f.write("\n".join(markdown) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, dataset in zip(axes.flat, DATASETS):
        rows = [r for r in all_rows if r["dataset"] == dataset]
        ax.barh([LABELS[r["method"]] for r in rows], [r["NDCG@10 mean"] for r in rows],
                xerr=[r["NDCG@10 seed_sd"] for r in rows], capsize=3)
        ax.set_title(dataset)
        ax.set_xlabel("NDCG@10 (mean ± seed SD)")
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(a.out / "ndcg10.pdf", bbox_inches="tight")
    fig.savefig(a.out / "ndcg10.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    files = sorted(f for f in a.out.iterdir() if f.is_file())
    dump(a.out / "manifest.json", {"result_manifests": provenance,
                                   "report_source_sha256": digest(Path(__file__)),
                                   "files": {f.name: digest(f) for f in files}})
    print(f"Tables, figure, CSVs, paired intervals and provenance: {a.out}")


if __name__ == "__main__":
    main()
