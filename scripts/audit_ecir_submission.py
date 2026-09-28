"""Read-only preflight audit for the ECIR manuscript and frozen artifacts.

The audit does not train or evaluate a model. It verifies artifact hashes and
provenance links, checks the paper's tables and quantitative prose against the
generated CSV files, validates structural LaTeX invariants, and rechecks the
anonymous review archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "ECIR.tex"
MAIN = ROOT / "paper/generated/ecir-mps-v2"
ADDITIONAL = ROOT / "paper/generated/ecir-additional-v1"
SENSITIVITY = ROOT / "paper/generated/ecir-direction-sensitivity-v1"
FROZEN = ROOT / "paper/generated/ecir-frozen-radius-v1"
RADIUS_V9 = ROOT / "paper/generated/ecir-radius-v9"
REDUNDANCY = ROOT / "paper/generated/ecir-review-v8/probe-redundancy/talkplay"
SUPPORT = ROOT / "paper/generated/ecir-review-v10/target-support"
VALIDATION_REDUNDANCY = ROOT / "paper/generated/ecir-review-v10/validation-probe-redundancy/talkplay"
VALIDATION_RESULT = ROOT / "artifacts/ecir/review-v10/validation-probe-redundancy/talkplay"
DATASETS = ("talkplay", "ml1m", "lastfm", "amazon_music")


class AuditFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_by(rows, **keys):
    matches = [row for row in rows if all(str(row[key]) == str(value)
                                          for key, value in keys.items())]
    require(len(matches) == 1, f"Expected one row in keys {keys}; found {len(matches)}")
    return matches[0]


def close(actual: float, expected: float, label: str, tolerance: float = 5e-12) -> None:
    require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance),
            f"{label}: {actual} != {expected}")


def paper_number(value: float, digits: int = 5, sign: bool = False) -> str:
    text = f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("+0."):
        return "+." + text[3:]
    if text.startswith("-0."):
        return "-." + text[3:]
    return text


def manifest_paths() -> list[Path]:
    patterns = (
        "artifacts/revision/v1/data/*/manifest.json",
        "artifacts/revision/mps-v1/runs/*/manifest.json",
        "artifacts/revision/mps-v1/results/*/manifest.json",
        "artifacts/ecir/mps-v2/**/manifest.json",
        "artifacts/ecir/additional-v1/**/manifest.json",
        "artifacts/ecir/direction-sensitivity-v1/**/manifest.json",
        "artifacts/ecir/frozen-radius-v1/**/manifest.json",
        "artifacts/ecir/review-v8/**/manifest.json",
        "artifacts/ecir/radius-v9/**/manifest.json",
        "artifacts/ecir/review-v10/**/manifest.json",
        "paper/generated/ecir-mps-v2/manifest.json",
        "paper/generated/ecir-additional-v1/manifest.json",
        "paper/generated/ecir-direction-sensitivity-v1/manifest.json",
        "paper/generated/ecir-frozen-radius-v1/*/manifest.json",
        "paper/generated/ecir-review-v8/**/manifest.json",
        "paper/generated/ecir-radius-v9/**/manifest.json",
        "paper/generated/ecir-review-v10/**/manifest.json",
    )
    paths = {path for pattern in patterns for path in ROOT.glob(pattern)}
    return sorted(paths)


def verify_manifest_files(path: Path) -> tuple[int, int]:
    manifest = read_json(path)
    count = total = 0
    for relative, expected in manifest.get("files", {}).items():
        target = path.parent / relative
        require(target.is_file(), f"Missing manifest payload: {target}")
        require(sha256(target) == expected, f"Hash mismatch: {target}")
        count += 1
        total += target.stat().st_size
    sources = manifest.get("source_sha256", {})
    if isinstance(sources, str):
        require(sources == sha256(ROOT / "experiments/ecir/frozen_radius.py"),
                f"Frozen-radius report source mismatch: {path}")
    else:
        for relative, expected in sources.items():
            target = ROOT / relative
            require(target.is_file(), f"Missing frozen source: {relative}")
            require(sha256(target) == expected,
                    f"Current source differs from evaluated source: {relative}")
    if "recovery_source_sha256" in manifest:
        require(manifest["recovery_source_sha256"] == sha256(
            ROOT / "experiments/ecir/resume_frozen_radius.py"),
            f"Frozen-radius recovery source mismatch: {path}")
    for relative, expected in manifest.get("frozen_source_sha256", {}).items():
        require(sha256(ROOT / relative) == expected,
                f"Review analysis frozen dependency changed: {relative}")
    return count, total


def verify_provenance() -> tuple[int, int, int]:
    paths = manifest_paths()
    require(paths, "No experiment manifests found")
    files = total = 0
    for path in paths:
        count, size = verify_manifest_files(path)
        files += count
        total += size

    for dataset in DATASETS:
        data = ROOT / f"artifacts/revision/v1/data/{dataset}/manifest.json"
        original_run = ROOT / f"artifacts/revision/mps-v1/runs/{dataset}/manifest.json"
        original_result = ROOT / f"artifacts/revision/mps-v1/results/{dataset}/manifest.json"
        baseline_run = ROOT / f"artifacts/ecir/mps-v2/baseline-runs/{dataset}/manifest.json"
        baseline_result = ROOT / f"artifacts/ecir/mps-v2/baseline-results/{dataset}/manifest.json"
        stochastic_run = ROOT / f"artifacts/ecir/mps-v2/stochastic-runs/{dataset}/manifest.json"
        stochastic_result = ROOT / f"artifacts/ecir/mps-v2/stochastic-results/{dataset}/manifest.json"
        session_run = ROOT / f"artifacts/ecir/additional-v1/session-knn-runs/{dataset}/manifest.json"
        session_result = ROOT / f"artifacts/ecir/additional-v1/session-knn-results/{dataset}/manifest.json"
        diagnostic = ROOT / f"artifacts/ecir/additional-v1/diagnostics/{dataset}/manifest.json"
        frozen_fit = ROOT / f"artifacts/ecir/frozen-radius-v1/runs/{dataset}/manifest.json"
        frozen_result = ROOT / f"artifacts/ecir/frozen-radius-v1/results/{dataset}/manifest.json"
        frozen_report = FROZEN / dataset / "manifest.json"
        radius_fit = ROOT / f"artifacts/ecir/radius-v9/runs/{dataset}/manifest.json"
        radius_result = ROOT / f"artifacts/ecir/radius-v9/results/{dataset}/manifest.json"
        radius_report = RADIUS_V9 / dataset / "manifest.json"
        radius_config = read_json(ROOT / "experiments/ecir/radius_v9_config.json")
        require(read_json(radius_fit)["config"] == radius_config,
                f"Radius-v9 configuration drift: {dataset}")
        require(radius_config["fixed_kappas"] ==
                [10, 20, 35, 50, 75, 100, 150, 250, 350, 400, 450, 500],
                "Radius grid differs from manuscript")
        require(radius_config["selection_inference_seeds"] == list(range(3101, 3111))
                and radius_config["test_inference_seeds"] == list(range(3101, 3111)),
                "Radius-v9 selection/test inference seeds differ from manuscript")

        links = (
            (original_run, "data_manifest_sha256", data),
            (original_result, "data_manifest_sha256", data),
            (original_result, "run_manifest_sha256", original_run),
            (baseline_run, "data_manifest_sha256", data),
            (baseline_result, "data_manifest_sha256", data),
            (baseline_result, "run_manifest_sha256", baseline_run),
            (stochastic_run, "data_manifest_sha256", data),
            (stochastic_run, "original_run_manifest_sha256", original_run),
            (stochastic_result, "data_manifest_sha256", data),
            (stochastic_result, "robust_run_manifest_sha256", stochastic_run),
            (stochastic_result, "original_run_manifest_sha256", original_run),
            (session_run, "data_manifest_sha256", data),
            (session_result, "data_manifest_sha256", data),
            (session_result, "run_manifest_sha256", session_run),
            (diagnostic, "data_manifest_sha256", data),
            (diagnostic, "original_run_manifest_sha256", original_run),
            (diagnostic, "stochastic_run_manifest_sha256", stochastic_run),
            (frozen_fit, "data_manifest_sha256", data),
            (frozen_fit, "original_run_manifest_sha256", original_run),
            (frozen_result, "data_manifest_sha256", data),
            (frozen_result, "original_run_manifest_sha256", original_run),
            (frozen_result, "fit_manifest_sha256", frozen_fit),
            (frozen_report, "fit_manifest_sha256", frozen_fit),
            (frozen_report, "result_manifest_sha256", frozen_result),
            (radius_fit, "data_manifest_sha256", data),
            (radius_fit, "original_run_manifest_sha256", original_run),
            (radius_fit, "previous_fit_manifest_sha256", frozen_fit),
            (radius_result, "data_manifest_sha256", data),
            (radius_result, "fit_manifest_sha256", radius_fit),
            (radius_report, "fit_manifest_sha256", radius_fit),
            (radius_report, "result_manifest_sha256", radius_result),
        )
        for source, field, target in links:
            require(source.is_file() and target.is_file(),
                    f"Missing provenance endpoint: {source} -> {target}")
            require(read_json(source)[field] == sha256(target),
                    f"Broken provenance link {source}:{field}")

        for method in ("learned_centroid", "learned_fixed", "full"):
            efficiency = ROOT / (
                f"artifacts/ecir/additional-v1/efficiency/{dataset}/{method}/manifest.json")
            payload = read_json(efficiency)
            require(payload["data_manifest_sha256"] == sha256(data),
                    f"Efficiency data mismatch: {dataset}/{method}")
            require(payload["original_run_manifest_sha256"] == sha256(original_run),
                    f"Efficiency original-run mismatch: {dataset}/{method}")
            require(payload["stochastic_run_manifest_sha256"] == sha256(stochastic_run),
                    f"Efficiency stochastic-run mismatch: {dataset}/{method}")

    direction_result = ROOT / "artifacts/ecir/direction-sensitivity-v1/results/manifest.json"
    direction = read_json(direction_result)
    require(direction["data_manifest_sha256"] == sha256(
        ROOT / "artifacts/revision/v1/data/talkplay/manifest.json"),
        "Direction sweep data mismatch")
    require(direction["original_run_manifest_sha256"] == sha256(
        ROOT / "artifacts/revision/mps-v1/runs/talkplay/manifest.json"),
        "Direction sweep original-run mismatch")
    require(direction["stochastic_run_manifest_sha256"] == sha256(
        ROOT / "artifacts/ecir/mps-v2/stochastic-runs/talkplay/manifest.json"),
        "Direction sweep stochastic-run mismatch")

    report_sources = {
        MAIN / "manifest.json": ROOT / "experiments/ecir/report.py",
        ADDITIONAL / "manifest.json": ROOT / "experiments/ecir/additional_report.py",
        SENSITIVITY / "manifest.json": ROOT / "experiments/ecir/query_direction_sensitivity.py",
    }
    for report_manifest, source in report_sources.items():
        payload = read_json(report_manifest)
        require(payload["report_source_sha256"] == sha256(source),
                f"Report source mismatch: {report_manifest}")
        for relative, expected in payload.get("result_manifests", {}).items():
            require(sha256(ROOT / relative) == expected,
                    f"Report input changed: {relative}")
    sensitivity_manifest = read_json(SENSITIVITY / "manifest.json")
    require(sensitivity_manifest["result_manifest_sha256"] == sha256(direction_result),
            "Sensitivity report input changed")
    verify_v10_provenance()
    return len(paths), files, total


def verify_v10_provenance() -> None:
    support = read_json(SUPPORT / "manifest.json")
    require(support["experiment"] == "ecir-target-support" and
            support["datasets"] == list(DATASETS), "Unexpected target-support scope")
    for field, target in (
            ("main_report_manifest_sha256", MAIN / "manifest.json"),
            ("session_report_manifest_sha256", ADDITIONAL / "manifest.json")):
        require(support[field] == sha256(target), f"Target-support link changed: {field}")
    for dataset in DATASETS:
        links = support["inputs"][dataset]
        data = ROOT / f"artifacts/revision/v1/data/{dataset}/manifest.json"
        require(links["data_manifest_sha256"] == sha256(data),
                f"Target-support data changed: {dataset}")
        folders = {
            "primary": "artifacts/revision/mps-v1/results",
            "baseline": "artifacts/ecir/mps-v2/baseline-results",
            "stochastic": "artifacts/ecir/mps-v2/stochastic-results",
            "session": "artifacts/ecir/additional-v1/session-knn-results",
        }
        for name, folder in folders.items():
            require(links["result_manifest_sha256"][name] == sha256(
                ROOT / folder / dataset / "manifest.json"),
                f"Target-support result changed: {dataset}/{name}")
    validation = read_json(VALIDATION_RESULT / "manifest.json")
    require(validation["split"] == "validation" and
            validation["sample_counts"] == [1, 5, 20] and validation["cutoff"] == 10 and
            validation["training_seeds"] == [42, 123, 456, 789, 2026] and
            validation["inference_seeds"] == list(range(3101, 3111)),
            "Validation-redundancy protocol differs from manuscript")
    expected_cells = {f"cells/{method}-train-{seed}-noise-{noise}.json"
                      for method in ("learned_fixed", "full")
                      for seed in validation["training_seeds"]
                      for noise in validation["inference_seeds"]}
    require({name for name in validation["files"] if name.startswith("cells/")}
            == expected_cells, "Validation-redundancy cells are incomplete")
    links = (
        (validation, "data_manifest_sha256",
         ROOT / "artifacts/revision/v1/data/talkplay/manifest.json"),
        (validation, "stochastic_run_manifest_sha256",
         ROOT / "artifacts/ecir/mps-v2/stochastic-runs/talkplay/manifest.json"),
        (validation, "reference_sweep_manifest_sha256",
         ROOT / "artifacts/ecir/direction-sensitivity-v1/results/manifest.json"),
        (read_json(VALIDATION_REDUNDANCY / "manifest.json"), "result_manifest_sha256",
         VALIDATION_RESULT / "manifest.json"),
        (read_json(VALIDATION_REDUNDANCY / "manifest.json"), "test_report_manifest_sha256",
         REDUNDANCY / "manifest.json"),
    )
    for payload, field, target in links:
        require(payload[field] == sha256(target), f"Validation diagnostic link changed: {field}")


def verify_support_table(tex: str, coverage: list[dict[str, str]],
                         conditional: list[dict[str, str]]) -> None:
    require(r"\label{tab:target-support}" in tex, "Missing training-item support table")
    require(len(coverage) == 8 and len(conditional) == 132,
            "Incomplete target-support output")
    names = dict(zip(DATASETS, ("TalkPlay", "MovieLens", "Last.fm", "Amazon Digital Music")))
    compact = re.sub(r"\s+", "", tex)
    for dataset in DATASETS:
        valid = row_by(coverage, dataset=dataset, split="validation")
        test = row_by(coverage, dataset=dataset, split="test")
        for row in (valid, test):
            total, warm, cold = (int(row[key]) for key in
                                 ("events", "warm_events", "cold_events"))
            require(total > 0 and warm + cold == total,
                    f"Invalid support counts: {dataset}/{row['split']}")
            close(float(row["warm_fraction"]), warm / total, "Warm fraction")
            close(float(row["cold_fraction"]), cold / total, "Cold fraction")
        def metric(method: str, partition: str) -> float:
            return float(row_by(conditional, dataset=dataset, method=method,
                                partition=partition)["NDCG@10"])
        cells = [names[dataset], paper_number(100 * float(valid["cold_fraction"]), 2),
                 paper_number(100 * float(test["cold_fraction"]), 2),
                 f"{int(test['warm_events']):,}", paper_number(metric("full", "warm")),
                 paper_number(metric("vsknn", "warm")), paper_number(metric("full", "cold"))]
        require(re.sub(r"\s+", "", " & ".join(cells) + r"\\") in compact,
                f"Training-item support table mismatch: {dataset}")
        close(metric("vsknn", "cold"), 0.0, f"Cold V-SKNN claim: {dataset}")
        methods = {row["method"] for row in conditional if row["dataset"] == dataset}
        require(len(methods) == 11, f"Missing conditional methods: {dataset}")
        for method in methods:
            rows = {part: row_by(conditional, dataset=dataset, method=method, partition=part)
                    for part in ("warm", "cold", "all")}
            for part, count_key in (("warm", "warm_events"), ("cold", "cold_events"),
                                    ("all", "events")):
                require(int(rows[part]["events"]) == int(test[count_key]),
                        f"Conditional event count changed: {dataset}/{method}/{part}")
            for key in ("NDCG@10", "Recall@10", "Recall@50", "MRR@50"):
                combined = sum(int(rows[part]["events"]) * float(rows[part][key])
                               for part in ("warm", "cold")) / int(test["events"])
                close(combined, float(rows["all"][key]), f"Conditional aggregation: {key}")
    flat = " ".join(tex.split())
    for method in ("last_item", "weighted_prf"):
        source = row_by(conditional, dataset="amazon_music", method=method, partition="warm")
        require(paper_number(float(source["NDCG@10"])) in flat,
                f"Missing Amazon warm comparator: {method}")
    require("We do not measure overlap or coverage" not in tex,
            "Obsolete coverage disclaimer remains")


def verify_validation_redundancy(tex: str) -> None:
    combined = read_csv(VALIDATION_REDUNDANCY / "validation_vs_test.csv")
    raw = read_json(VALIDATION_RESULT / "aggregate.json")
    test = read_csv(REDUNDANCY / "probe_redundancy.csv")
    require(len(combined) == 6 and len(raw) == 6, "Incomplete validation comparison")
    for method in ("learned_fixed", "full"):
        for samples in (1, 5, 20):
            row = row_by(combined, method=method, samples=samples, cutoff=10)
            valid = row_by(raw, method=method, samples=samples)
            original = row_by(test, method=method, samples=samples, cutoff=10)
            require(int(row["validation_events"]) == valid["queries_per_cell"] == 1520 and
                    int(row["test_events"]) == int(original["queries_per_cell"]) == 1000 and
                    valid["cells"] == 50, "Probe diagnostic support changed")
            for suffix, column in (("jaccard", "pairwise_topk_jaccard"),
                                   ("unique", "unique_candidates"),
                                   ("union_hit", "target_union_coverage"),
                                   ("merged_ndcg10", "merged_ndcg10")):
                for split, source in (("validation", valid), ("test", original)):
                    observed = row[f"{split}_{suffix}"]
                    if source[column] in (None, ""):
                        require(observed == "", "Undefined one-probe overlap must be blank")
                    else:
                        close(float(observed), float(source[column]), f"{split}/{method}/{suffix}")
        final = row_by(combined, method=method, samples=20)
        for column, digits in (("validation_jaccard", 3), ("validation_unique", 2)):
            require(paper_number(float(final[column]), digits) in tex,
                    f"Validation probe statistic absent from manuscript: {method}/{column}")
    require("1,520 validation events" in " ".join(tex.split()),
            "Validation event count absent from manuscript")


def verify_secondary_bounds(tex: str, metrics_by_dataset: dict[str, list[dict[str, str]]]) -> None:
    contrasts = (
        ("no_expansion", ("fixed_per_center",), (("Recall@10", .00057), ("MRR@50", .00012))),
        ("fixed_per_center", ("adaptive_hard_clip", "adaptive_smooth"),
         (("Recall@10", .00014), ("MRR@50", .00004))),
    )
    for control, methods, bounds in contrasts:
        for metric, bound in bounds:
            differences = []
            for dataset in DATASETS:
                rows = metrics_by_dataset[dataset]
                baseline = float(row_by(rows, method=control)[f"{metric} mean"])
                differences.extend(float(row_by(rows, method=method)[f"{metric} mean"]) - baseline
                                   for method in methods)
            require(all(math.isfinite(value) and abs(value) < bound for value in differences),
                    f"Unsupported secondary-metric effect bound: {metric}/{control}")
            require(min(differences) < 0 < max(differences),
                    f"Secondary effects no longer mixed: {metric}/{control}")
            require(paper_number(bound) in tex and metric in tex,
                    f"Secondary-metric bound absent from manuscript: {metric}/{control}")


def references_start_page(auxiliary: str) -> int:
    matches = re.findall(r"\\newlabel\{page:references-start\}\{\{[^{}]*\}\{(\d+)\}",
                         auxiliary)
    require(len(matches) == 1, "Missing or duplicate reference-page marker; compile twice")
    page = int(matches[0])
    require(2 <= page <= 13,
            f"Content exceeds 12 pages: references start on page {page}")
    return page


def verify_frozen_center_table(tex: str) -> None:
    require(r"\label{tab:frozen-center-radius}" in tex,
            "Frozen-center comparison is absent from manuscript")
    labels = {"talkplay": "TalkPlay", "ml1m": "MovieLens",
              "lastfm": "Last.fm", "amazon_music": "Amazon Digital Music"}
    compact = re.sub(r"\s+", "", tex.replace("$", ""))
    for dataset in DATASETS:
        folder = RADIUS_V9 / dataset
        metrics = read_csv(folder / "metrics.csv")
        comparisons = read_csv(folder / "paired_comparisons.csv")
        selection = read_json(folder / "selection.json")
        kappa = selection["fixed"]["global_kappa"]
        require(kappa == 500.0, f"Unexpected selected fixed radius: {dataset}")
        values = [float(row_by(metrics, method=method)["NDCG@10 mean"])
                  for method in ("no_expansion", "fixed_global", "fixed_per_center",
                                 "adaptive_hard_clip", "adaptive_smooth")]
        row = "&".join((
            labels[dataset],
            *(paper_number(value, digits=6) for value in values),
        )) + r"\\"
        require(re.sub(r"\s+", "", row) in compact,
                f"Frozen-center table row mismatch: {dataset}")
        for comparison in ("fixed_global - no_expansion",
                           "fixed_per_center - no_expansion",
                           "adaptive_hard_clip - fixed_global",
                           "adaptive_hard_clip - fixed_per_center",
                           "adaptive_smooth - fixed_global",
                           "adaptive_smooth - fixed_per_center",
                           "adaptive_smooth - adaptive_hard_clip",
                           "fixed_per_center - fixed_global"):
            interval = row_by(comparisons, comparison=comparison)
            require(float(interval["crossed_95_low"]) <= 0 <=
                    float(interval["crossed_95_high"]),
                    f"Unsupported frozen-center prose: {dataset}/{comparison}")
            require(int(interval["training_seeds"]) == 5 and
                    int(interval["inference_seeds"]) == 10 and
                    int(interval["replicates"]) == 10000,
                    f"Unexpected crossed-bootstrap axes: {dataset}/{comparison}")
        selected = selection["fixed"]["per_center_kappas"]
        require(set(selected) == {"42", "123", "456", "789", "2026"},
                f"Incomplete per-center fixed selection: {dataset}")
        expected_ranges = {"talkplay": (450, 500), "ml1m": (20, 500),
                           "lastfm": (350, 500), "amazon_music": (20, 400)}
        require((min(selected.values()), max(selected.values())) == expected_ranges[dataset],
                f"Per-center radius range differs from manuscript: {dataset}")


def verify_figure_coordinates(tex: str) -> None:
    """Check embedded plot values against the retained aggregate reports."""
    def block(name: str) -> str:
        matches = re.findall(r"% BEGIN " + re.escape(name) + r"\s*\n(.*?)"
                             r"% END " + re.escape(name), tex, flags=re.DOTALL)
        require(len(matches) == 1, f"Missing or duplicate figure block: {name}")
        return matches[0]

    def close(actual: float, expected: float, label: str) -> None:
        require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15),
                f"Figure coordinate mismatch: {label}")

    sensitivity = read_csv(SENSITIVITY / "direction_sensitivity.csv")
    for name, method in (("FIXED", "learned_fixed"), ("ADAPTIVE", "full")):
        pairs = re.findall(r"\((\d+),\s*([\d.eE+\-]+)\)",
                           block("PROBES " + name))
        require([int(s) for s, _ in pairs] == [1, 2, 5, 10, 20],
                f"Incorrect probe-count axis: {name}")
        for samples, value in pairs:
            source = row_by(sensitivity, method=method, samples=int(samples))
            close(float(value), float(source["NDCG@10 mean"]),
                  f"{method}/S={samples}")

    for name, method, offset in (("HARD", "adaptive_hard_clip", .1),
                                 ("SMOOTH", "adaptive_smooth", -.1)):
        lines = block("FROZEN " + name).strip().splitlines()
        require(lines[0].split() == ["y", "mean", "low", "high"] and len(lines) == 5,
                f"Incorrect frozen-radius figure shape: {name}")
        for index, dataset in enumerate(DATASETS):
            values = [float(value) for value in lines[index + 1].split()]
            require(len(values) == 4, f"Malformed figure row: {name}/{dataset}")
            source = row_by(read_csv(RADIUS_V9 / dataset / "paired_comparisons.csv"),
                            comparison=f"{method} - fixed_per_center")
            expected = [4 - index + offset, float(source["mean_difference"]),
                        float(source["crossed_95_low"]),
                        float(source["crossed_95_high"])]
            for actual, reference in zip(values, expected):
                close(actual, reference, f"{name}/{dataset}")

    redundancy = read_csv(REDUNDANCY / "probe_redundancy.csv")
    prose = " ".join(tex.split())
    fixed = row_by(redundancy, method="learned_fixed", samples=20, cutoff=10)
    adaptive = row_by(redundancy, method="full", samples=20, cutoff=10)
    first = row_by(redundancy, method="full", samples=1, cutoff=10)
    for source, column, digits in (
            (fixed, "pairwise_topk_jaccard", 3),
            (adaptive, "pairwise_topk_jaccard", 3),
            (fixed, "unique_candidates", 2),
            (adaptive, "unique_candidates", 2),
            (adaptive, "target_union_coverage", 5),
            (adaptive, "merged_recall", 5),
            (first, "merged_recall", 5)):
        number = paper_number(float(source[column]), digits=digits)
        require(number in prose, f"Missing probe-redundancy value: {column}/{number}")


def verify_data_table(tex: str) -> None:
    names = {"talkplay": "TalkPlay", "ml1m": "MovieLens-1M",
             "lastfm": "Last.fm", "amazon_music": "Amazon Digital Music"}
    representations = {"talkplay": "CLAP audio, 512d",
                       "ml1m": "MiniLM text, 384d",
                       "lastfm": "MiniLM name, 384d",
                       "amazon_music": "MiniLM text, 384d"}
    for dataset in DATASETS:
        manifest = read_json(ROOT / f"artifacts/revision/v1/data/{dataset}/manifest.json")
        counts = manifest["counts"]
        expected = " & ".join((
            names[dataset], f"{manifest['catalog_size']:,}",
            f"{counts['train']['retained']:,}",
            f"{counts['validation']['retained']:,}",
            f"{counts['test']['retained']:,}", representations[dataset])) + r"\\"
        require(expected in tex.replace(" \\\\n", "\\\\\n"),
                f"Dataset table row mismatch: {dataset}")


def verify_main_table(tex: str, metrics: list[dict[str, str]],
                      sessions: list[dict[str, str]]) -> None:
    labels = {
        "popularity": "Popularity", "last_item": "Last item",
        "transition_knn": "Transition", "vsknn": "V-SKNN",
        "content_sasrec": "Content-SASRec", "uniform": "Uniform center",
        "weighted_prf": "Weighted center", "learned_centroid": "Learned center",
        "weighted_fixed": "Weighted+fixed-radius",
        "learned_fixed": "Learned+fixed-radius",
        "full": r"\method{} adaptive",
    }
    best = {"talkplay": "vsknn", "ml1m": "transition_knn",
            "lastfm": "vsknn", "amazon_music": "last_item"}
    values: dict[tuple[str, str], float] = {}
    for row in metrics:
        values[(row["dataset"], row["method"])] = float(row["NDCG@10 mean"])
    for row in sessions:
        values[(row["dataset"], "vsknn")] = float(row["V-SKNN NDCG@10"])
    methods = tuple(labels)
    for method in methods:
        cells = []
        for dataset in DATASETS:
            value = paper_number(values[(dataset, method)])
            if method in {"content_sasrec", "learned_centroid",
                          "learned_fixed", "full"}:
                source = row_by(metrics, dataset=dataset, method=method)
                deviation = paper_number(float(source["NDCG@10 train_sd"]))
                value = "$" + value + r"\!\pm\!" + deviation + "$"
            elif method == "weighted_fixed":
                source = row_by(metrics, dataset=dataset, method=method)
                deviation = paper_number(float(source["NDCG@10 inference_sd"]))
                value = "$" + value + r"\!\pm\!" + deviation + "$"
            cells.append(value)
        expected = " & ".join((labels[method], *cells)) + r"\\"
        require(expected in tex.replace(" \\\\n", "\\\\\n"),
                f"Main result row mismatch: {method}")
    for dataset in DATASETS:
        ranking = sorted(((values[(dataset, method)], method) for method in methods),
                         reverse=True)
        require(ranking[0][1] == best[dataset],
                f"Unexpected best method for {dataset}: {ranking[0]}")


def verify_narrative(tex: str, metrics: list[dict[str, str]],
                     comparisons: list[dict[str, str]],
                     session_comparisons: list[dict[str, str]],
                     sensitivity: list[dict[str, str]],
                     seed_rows: list[dict[str, str]], hierarchy: dict[str, str],
                     diagnostics: list[dict[str, str]],
                     efficiency: list[dict[str, str]]) -> None:
    flat = " ".join(tex.split())
    compact = re.sub(r"\s+", "", tex)

    def contains(fragment: str, label: str) -> None:
        require(fragment in flat, f"Missing or inconsistent manuscript claim: {label}")

    for dataset, display in (("talkplay", "TalkPlay"), ("ml1m", "MovieLens"),
                             ("lastfm", "Last.fm")):
        row = row_by(session_comparisons, dataset=dataset, metric="NDCG@10")
        gain = -float(row["mean_difference"])
        low, high = float(row["bootstrap_95_low"]), float(row["bootstrap_95_high"])
        phrase = (f"{paper_number(gain)} NDCG@10 on {display}" if dataset == "talkplay"
                  else f"{paper_number(gain)} on {display}")
        contains(phrase, f"V-SKNN gain/{dataset}")
        contains(f"[{low:.5f},{high:.5f}]", f"V-SKNN interval/{dataset}")

    amazon_v = row_by(session_comparisons, dataset="amazon_music", metric="NDCG@10")
    contains(paper_number(float(amazon_v["mean_difference"])), "Amazon Full-VSKNN")
    contains(f"[{float(amazon_v['bootstrap_95_low']):.5f},"
             f"{float(amazon_v['bootstrap_95_high']):.5f}]", "Amazon VSKNN interval")

    def comparison(dataset: str, name: str) -> dict[str, str]:
        return row_by(comparisons, dataset=dataset, comparison=name)

    amazon_last = comparison("amazon_music", "full - last_item")
    contains(f"[{float(amazon_last['bootstrap_95_low']):.5f},"
             f"{float(amazon_last['bootstrap_95_high']):.5f}]", "Amazon last-item interval")

    for dataset, display in (("ml1m", "MovieLens"), ("lastfm", "Last.fm")):
        row = comparison(dataset, "learned_centroid - weighted_prf")
        contains("$" + f"{float(row['mean_difference']):+.5f}" + "$" +
                 (" on " + display), f"learned-center gain/{dataset}")
        contains(f"[{float(row['bootstrap_95_low']):.5f},"
                 f"{float(row['bootstrap_95_high']):.5f}]", f"center interval/{dataset}")

    for name, sign in (("learned_fixed - learned_centroid", True),
                       ("weighted_fixed - weighted_prf", True),
                       ("full - learned_fixed", False)):
        row = comparison("talkplay", name)
        prose_value = (f"{float(row['mean_difference']):+.5f}" if sign else
                       f"{float(row['mean_difference']):.6f}")
        contains("$" + prose_value + "$" if sign else prose_value, name)
        contains(f"[{float(row['bootstrap_95_low']):.5f},"
                 f"{float(row['bootstrap_95_high']):.5f}]", name + " interval")

    differences = [float(row["ACE - learned-fixed"]) for row in seed_rows]
    expected_seeds = "[" + ",".join(f"{value:.5f}" for value in differences) + "]"
    contains(expected_seeds, "per-training-seed differences")
    contains(f"{float(hierarchy['mean_difference']):.6f}",
             "hierarchical mean")
    contains(f"[{float(hierarchy['hierarchical_95_low']):.5f},"
             f"{float(hierarchy['hierarchical_95_high']):.5f}]", "hierarchical interval")

    verify_figure_coordinates(tex)
    fixed_values = [float(row["NDCG@10 mean"]) for row in sensitivity
                    if row["method"] == "learned_fixed"]
    full_values = [float(row["NDCG@10 mean"]) for row in sensitivity
                   if row["method"] == "full"]
    contains(f"only {max(fixed_values)-min(fixed_values):.5f}", "fixed S range")
    contains(r"adaptive \method{} by " + f"{max(full_values)-min(full_values):.5f}",
             "ACE S range")

    # Full radius/dispersion tables remain in the artifact; audit the retained
    # boundary claims in the shortened manuscript against their source CSV.
    for dataset, column in (("talkplay", "fraction_at_kappa_maximum"),
                             ("lastfm", "fraction_at_kappa_minimum")):
        row = row_by(diagnostics, dataset=dataset)
        contains(f"{100 * float(row[column]):.1f}\\%", f"Radius boundary/{dataset}")

    # The full timing/memory table remains in the generated artifact; audit
    # representative values retained in the shorter manuscript.
    amazon_center = row_by(efficiency, dataset="amazon_music", method="learned_centroid")
    amazon_full = row_by(efficiency, dataset="amazon_music", method="full")
    for row in (amazon_center, amazon_full):
        contains(paper_number(float(row["scoring_median_ms_per_query"]), digits=3),
                 "Amazon scoring time")
        contains(paper_number(float(row["total_median_ms_per_query"]), digits=3),
                 "Amazon total time")
    score_ratio = (float(amazon_full["scoring_median_ms_per_query"]) /
                   float(amazon_center["scoring_median_ms_per_query"]))
    total_increase = 100 * (float(amazon_full["total_median_ms_per_query"]) /
                            float(amazon_center["total_median_ms_per_query"]) - 1)
    contains(f"{score_ratio:.1f} times", "Amazon scoring ratio")
    contains(f"{total_increase:.0f}\\% higher", "Amazon total latency increase")

    talk_full = float(row_by(metrics, dataset="talkplay", method="full")["NDCG@10 mean"])
    talk_uniform = float(row_by(metrics, dataset="talkplay", method="uniform")["NDCG@10 mean"])
    talk_vsknn = float(row_by(
        read_csv(ADDITIONAL / "session_knn.csv"), dataset="talkplay")["V-SKNN NDCG@10"])
    contains(f"V-SKNN exceeds adaptive \\method{{}} by {talk_vsknn-talk_full:.5f}",
             "discussion V-SKNN difference")
    contains(f"adaptive \\method{{}} exceeds the uniform center by "
             f"{talk_full-talk_uniform:.5f}", "discussion ACE difference")


def verify_selection_and_config(tex: str) -> None:
    revision = read_json(ROOT / "experiments/revision/config.json")
    extension = read_json(ROOT / "experiments/ecir/config.json")
    additional = read_json(ROOT / "experiments/ecir/additional_config.json")
    require(revision["seeds"] == [42, 123, 456, 789, 2026], "Training seed drift")
    require(extension["seeds"] == revision["seeds"], "ECIR training seed drift")
    require(extension["inference_seeds"] == list(range(3101, 3111)),
            "Inference seed drift")
    require(revision["samples"] == 5 and revision["max_context"] == 10,
            "Core query configuration drift")
    require(additional["direction_sensitivity"]["sample_counts"] == [1, 2, 5, 10, 20],
            "Direction sweep drift")
    require(r"\{42,123,456,789,2026\}" in tex, "Training seeds absent from manuscript")
    require(r"\{3101,\ldots,3110\}" in tex, "Inference seeds absent from manuscript")
    require("fixed $S=5$" in tex, "Provenance of the five-probe budget is absent")

    expected = {
        "talkplay": (100, 0.85), "ml1m": (250, 0.85),
        "lastfm": (50, 1.0), "amazon_music": (50, 1.0),
    }
    for dataset, (neighbors, decay) in expected.items():
        selection = read_json(ROOT / (
            f"artifacts/ecir/additional-v1/session-knn-runs/{dataset}/selection.json"))
        chosen = selection["selected"]
        require((chosen["neighbor_count"], chosen["recency_decay"]) == (neighbors, decay),
                f"V-SKNN selection mismatch: {dataset}")


def verify_latex(tex: str) -> None:
    slash = chr(92)
    require(tex.count("{") == tex.count("}"), "Unbalanced LaTeX braces")
    require(tex.count(slash + "begin{") == tex.count(slash + "end{"),
            "Unbalanced LaTeX environments")
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    refs = re.findall(r"\\ref\{([^}]+)\}", tex)
    require(len(labels) == len(set(labels)), "Duplicate LaTeX labels")
    require(not (set(refs) - set(labels)), f"Unresolved labels: {set(refs)-set(labels)}")
    citations = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", tex)
                 for key in group.split(",")}
    bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", tex))
    require(citations == bibitems,
            f"Citation mismatch missing={citations-bibitems}, unused={bibitems-citations}")
    require(r"\author{Anonymous Authors}" in tex and
            r"\institute{Anonymous Institution}" in tex,
            "Manuscript identity fields are not anonymous")
    forbidden = ("TODO", "FIXME", "TBD", "/" + "Users" + "/",
                 "github" + ".com/", "git" + "@")
    require(not [token for token in forbidden if token.lower() in tex.lower()],
            "Manuscript contains placeholder or identifying token")
    require("dialogue understanding" not in tex.lower(),
            "Manuscript revives an unsupported dialogue-understanding claim")
    def section_between(start: str, end: str) -> str:
        require(start in tex and end in tex and tex.index(start) < tex.index(end),
                f"Missing or misplaced manuscript section: {start}")
        return tex.split(start, 1)[1].split(end, 1)[0]

    scoped_sections = {
        "abstract": section_between(r"\begin{abstract}", r"\end{abstract}"),
        "introduction": section_between(r"\section{Introduction}",
                                        r"\section{Related Work}"),
        "discussion": section_between(r"\section{Discussion}",
                                      r"\section{Limitations, Ethics, and Reproducibility}"),
        "conclusion": section_between(r"\section{Conclusion}",
                                      r"\begin{thebibliography}"),
    }
    for name, section in scoped_sections.items():
        lowered = section.lower()
        require(all(token in lowered for token in ("isotropic", "max", "negative")),
                f"Unqualified scope in manuscript {name}")
    require(re.search(r"\\clearpage\s*\\begin\{thebibliography\}\{\d+\}\s*"
                      r"\\label\{page:references-start\}", tex) is not None,
            "References must follow a float-flushing page break and page marker")

    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\keywords\{", tex, flags=re.DOTALL)
    require(abstract_match is not None, "Manuscript abstract could not be parsed")
    registered_abstract = (ROOT / "paper/abstract.txt").read_text(encoding="utf-8")

    def normalized_abstract(value: str) -> str:
        value = value.replace(r"\method", "ACE").replace(r"\%", "%")
        value = value.replace("$", "")
        return " ".join(value.split())

    require(normalized_abstract(abstract_match.group(1)) ==
            normalized_abstract(registered_abstract),
            "paper/abstract.txt does not match the manuscript abstract")


def verify_anonymous_archive(bundle: Path) -> tuple[int, int, str]:
    archive = bundle.with_suffix(".zip")
    checksum_path = Path(str(archive) + ".sha256")
    require(bundle.is_dir() and archive.is_file() and checksum_path.is_file(),
            "Anonymous artifact has not been built")
    synchronized = {
        bundle / "paper/ECIR.tex": TEX,
        bundle / "README.md": ROOT / "ECIR_ARTIFACT_README.md",
        bundle / "scripts/build_ecir_anonymous_artifact.py":
            ROOT / "scripts/build_ecir_anonymous_artifact.py",
    }
    for packaged, source in synchronized.items():
        require(packaged.is_file() and sha256(packaged) == sha256(source),
                f"Anonymous bundle is stale: {packaged.relative_to(bundle)}")
    spec = importlib.util.spec_from_file_location(
        "ecir_artifact_builder", ROOT / "scripts/build_ecir_anonymous_artifact.py")
    require(spec is not None and spec.loader is not None, "Cannot import artifact builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_audit_patterns()
    for source in module.selected_files():
        packaged = bundle / module.destination(source.relative_to(ROOT))
        require(packaged.is_file() and sha256(packaged) == sha256(source),
                f"Anonymous bundle is missing or stale: "
                f"{packaged.relative_to(bundle)}")
    findings = module.audit(bundle)
    require(not findings, "Anonymous bundle audit failed: " + "; ".join(findings))
    module.verify_zip(bundle, archive)
    expected = checksum_path.read_text(encoding="utf-8").split()[0]
    actual = sha256(archive)
    require(actual == expected, "Outer artifact checksum mismatch")
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        require(not source.comment, "ZIP contains a comment")
        require(not any(info.extra for info in infos), "ZIP contains extra metadata fields")
        require(len(infos) == len({info.filename for info in infos}),
                "ZIP contains duplicate members")
        require(not [info.filename for info in infos
                     if info.filename.startswith("/") or
                     ".." in Path(info.filename).parts or
                     "__MACOSX" in Path(info.filename).parts],
                "ZIP contains unsafe or macOS metadata paths")
    return len(infos), archive.stat().st_size, actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path,
                        default=Path("submission/ecir2027_anonymous_artifact_v10_final"),
                        help="Existing anonymous bundle directory (not its ZIP)")
    parser.add_argument("--aux", type=Path,
                        help="Fresh ECIR.aux after compiling this source; checks content-page limit")
    args = parser.parse_args()
    bundle = args.bundle if args.bundle.is_absolute() else ROOT / args.bundle
    try:
        tex = TEX.read_text(encoding="utf-8")
        manifests, files, bytes_checked = verify_provenance()
        metrics = read_csv(MAIN / "metrics.csv")
        comparisons = read_csv(MAIN / "paired_comparisons.csv")
        sessions = read_csv(ADDITIONAL / "session_knn.csv")
        session_comparisons = read_csv(ADDITIONAL / "paired_session_knn.csv")
        diagnostics = read_csv(ADDITIONAL / "kappa_diagnostics.csv")
        efficiency = read_csv(ADDITIONAL / "efficiency.csv")
        sensitivity = read_csv(SENSITIVITY / "direction_sensitivity.csv")
        seed_rows = read_csv(SENSITIVITY / "seed_differences_s5.csv")
        hierarchy_rows = read_csv(SENSITIVITY / "seed_aware_comparison_s5.csv")
        require(len(hierarchy_rows) == 1, "Expected one hierarchical comparison")

        verify_data_table(tex)
        verify_main_table(tex, metrics, sessions)
        verify_narrative(tex, metrics, comparisons, session_comparisons,
                         sensitivity, seed_rows, hierarchy_rows[0], diagnostics,
                         efficiency)
        verify_frozen_center_table(tex)
        verify_support_table(tex, read_csv(SUPPORT / "target_coverage.csv"),
                             read_csv(SUPPORT / "warm_cold_metrics.csv"))
        verify_validation_redundancy(tex)
        verify_secondary_bounds(tex, {dataset: read_csv(RADIUS_V9 / dataset / "metrics.csv")
                                      for dataset in DATASETS})
        verify_selection_and_config(tex)
        verify_latex(tex)
        reference_page = None
        if args.aux is not None:
            auxiliary = args.aux if args.aux.is_absolute() else ROOT / args.aux
            require(auxiliary.stat().st_mtime_ns >= TEX.stat().st_mtime_ns,
                    "Auxiliary file predates manuscript; rebuild the PDF")
            reference_page = references_start_page(auxiliary.read_text(encoding="utf-8"))
        members, archive_bytes, checksum = verify_anonymous_archive(bundle)
    except (AuditFailure, KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ECIR SUBMISSION AUDIT: FAIL\n{error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("ECIR EVIDENCE AUDIT: PASS")
    print(f"Verified manifests: {manifests}")
    print(f"Verified payload files: {files}")
    print(f"Hashed payload bytes: {bytes_checked}")
    print("Paper checks: tables, plot coordinates, quantitative prose, v10 provenance, "
          "warm/cold support, validation overlap, secondary bounds, selections, "
          "configuration, abstract, citations, labels, and archive anonymity")
    if reference_page is None:
        print("PAGE LIMIT: NOT CHECKED (pass --aux ECIR.aux after compilation)")
    else:
        print(f"Content pages: {reference_page - 1}; references start on page {reference_page}")
    print("PDF visual layout and PDF metadata still require inspection.")
    print(f"Anonymous ZIP: {members} members, {archive_bytes} bytes")
    print(f"Anonymous ZIP SHA-256: {checksum}")


if __name__ == "__main__":
    main()
