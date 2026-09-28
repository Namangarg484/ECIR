"""Match validation inference seed without retraining or changing old results.

select: validation-only fixed-radius reselection on the adaptive selection seed.
report: after selection is sealed, reuse the already saved paired test cells.
This matches inference randomness, NOT candidate/epoch counts or total compute.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from experiments.ecir import frozen_radius as frozen
from experiments.ecir.query_direction_sensitivity import hierarchical_interval
from experiments.ecir.review_common import (
    analysis_sources, cell, check_sources, finish, prepare_output, write_or_check,
)
from experiments.revision.protocol import aggregate, digest, stable_seed, verify_manifest
from experiments.revision.run import Data, configure_runtime, environment


def choose_fixed(candidates):
    if not candidates or not all(np.isfinite(row["mean_validation_ndcg10"])
                                 for row in candidates):
        raise ValueError("Empty or nonfinite validation candidates")
    # max returns the first item on ties: frozen, prespecified grid order.
    return max(candidates, key=lambda row: row["mean_validation_ndcg10"])


def select(args):
    original = frozen.verify_parent(args)
    fit = verify_manifest(args.run)
    if fit.get("experiment") != "ecir-frozen-center-radius-fit":
        raise ValueError("Expected frozen-radius fit")
    check_sources(fit["source_sha256"])
    for key, path in (("original_run_manifest_sha256", args.original_run),
                      ("data_manifest_sha256", args.data)):
        if fit[key] != digest(path / "manifest.json"):
            raise ValueError(f"Parent mismatch: {key}")
    if fit["device"] != args.device:
        raise ValueError("Use the original frozen-radius device")
    for key in ("determinism", "mps_cpu_fallback"):
        if fit["environment"][key] != environment()[key]:
            raise ValueError(f"Runtime mismatch: {key}")
    config, revision = fit["config"], fit["revision_config"]
    frozen.validate_config(config, revision)
    inference_seed = config["selection_inference_seed"]
    old_selection = json.loads((args.run / "selection.json").read_text())
    spec = {"experiment": "ecir-matched-radius-selection", "schema": 1,
            "dataset": original["dataset"],
            "fit_manifest_sha256": digest(args.run / "manifest.json"),
            "data_manifest_sha256": digest(args.data / "manifest.json"),
            "original_run_manifest_sha256": digest(args.original_run / "manifest.json"),
            "source_sha256": analysis_sources(__file__),
            "frozen_source_sha256": frozen.source_hashes(),
            "selection_inference_seed": inference_seed,
            "revision_config": revision, "device": args.device,
            "environment": environment(),
            "claim": "matched validation inference seed, not equal search compute"}
    prepare_output(args.out, spec)
    if (args.out / "manifest.json").exists():
        verify_manifest(args.out)
        print("Selection already complete and verified.")
        return
    # Data verifies manifest bytes, but only validation examples are materialized.
    data = Data(args.data, args.device, ["validation"])
    by_seed = []
    for seed in revision["seeds"]:
        def compute(seed=seed):
            center = args.original_run / old_selection["centers"][str(seed)]["checkpoint"]
            if digest(center) != old_selection["centers"][str(seed)]["sha256"]:
                raise ValueError("Selected center checksum mismatch")
            model = frozen.load_model(data, center, args.device, "hard_clip")
            suite = frozen.evaluate_expanded_suite(
                data, "validation", model, model, config["fixed_kappas"],
                revision, inference_seed, include_adaptive=False)
            return {frozen.fixed_name(k): aggregate(suite[frozen.fixed_name(k)])
                    for k in config["fixed_kappas"]}
        by_seed.append(cell(args.out, f"validation-center-{seed}", compute))
    candidates = [{"kappa": k, "mean_validation_ndcg10": float(np.mean([
        values[frozen.fixed_name(k)]["NDCG@10"] for values in by_seed]))}
        for k in config["fixed_kappas"]]
    chosen = choose_fixed(candidates)
    selection = {"selected_kappa": chosen["kappa"], "candidates": candidates,
                 "selection_inference_seed": inference_seed,
                 "training_seeds": revision["seeds"],
                 "test_inference_seeds": config["inference_seeds"],
                 "previous_ten_seed_kappa": old_selection["fixed_radius"]["selected_kappa"],
                 "adaptive_checkpoints": old_selection["adaptive_radius"],
                 "selection_unit": "global fixed radius, mean over five centers; "
                                   f"adaptive epochs selected per center; both seed {inference_seed}"}
    write_or_check(args.out / "selection.json", selection)
    finish(args.out, spec)
    print(f"Validation selection sealed: kappa={chosen['kappa']:g}. No test metrics read.")


def report(args):
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    selected = verify_manifest(args.selection)
    fit = verify_manifest(args.run)
    results = verify_manifest(args.results)
    if selected.get("experiment") != "ecir-matched-radius-selection":
        raise ValueError("Expected sealed matched selection")
    if results.get("experiment") != "ecir-frozen-center-radius-test":
        raise ValueError("Expected frozen-radius test results")
    for manifest in (selected, results):
        if manifest["fit_manifest_sha256"] != digest(args.run / "manifest.json"):
            raise ValueError("Fit/selection/test provenance mismatch")
        for key in ("data_manifest_sha256", "original_run_manifest_sha256"):
            if manifest[key] != fit[key]:
                raise ValueError(f"Input mismatch: {key}")
        if manifest["revision_config"] != fit["revision_config"]:
            raise ValueError("Revision configuration mismatch")
    check_sources(selected["source_sha256"])
    check_sources(selected["frozen_source_sha256"])
    check_sources(fit["source_sha256"])
    check_sources(results["source_sha256"])
    report_spec = {"experiment": "ecir-matched-radius-report",
                   "selection_manifest_sha256": digest(args.selection / "manifest.json"),
                   "result_manifest_sha256": digest(args.results / "manifest.json"),
                   "source_sha256": analysis_sources(__file__), "bootstrap": args.bootstrap}
    if (args.out / "manifest.json").exists():
        previous = verify_manifest(args.out)
        if any(previous.get(key) != value for key, value in report_spec.items()):
            raise ValueError("Completed report inputs changed; choose a new output")
        print(f"Matched-selection report already complete and verified: {args.out}")
        return
    if args.out.exists():
        raise ValueError("Incomplete report output: move it aside before rerunning")
    selection = json.loads((args.selection / "selection.json").read_text())
    if (selection["training_seeds"] != fit["revision_config"]["seeds"] or
            selection["test_inference_seeds"] != results["config"]["inference_seeds"] or
            selection["selection_inference_seed"] != fit["config"]["selection_inference_seed"]):
        raise ValueError("Selection seed axes differ from frozen experiment")
    summary = json.loads((args.results / "summary.json").read_text())
    entries = {(r["method"], r["train_seed"], r["inference_seed"]): r for r in summary}
    if len(entries) != len(summary):
        raise ValueError("Duplicate test cells")
    fixed = frozen.fixed_name(selection["selected_kappa"])
    methods = ["no_expansion", fixed, "adaptive_hard_clip", "adaptive_smooth"]
    values, reference = {}, None
    for method in methods:
        training = []
        for seed in selection["training_seeds"]:
            inference = []
            seeds = [None] if method == "no_expansion" else selection["test_inference_seeds"]
            for noise_seed in seeds:
                support, array = frozen.verified_values(
                    args.results, results, entries[method, seed, noise_seed], reference)
                if reference is None:
                    reference = support
                inference.append(array)
            stacked = np.stack(inference)
            if method == "no_expansion":
                stacked = np.repeat(stacked, len(selection["test_inference_seeds"]), axis=0)
            training.append(stacked)
        values[method] = np.stack(training)
    metrics = [{"method": method, "NDCG@10 mean": float(array.mean()),
                "training_seed_sd": float(array.mean(axis=(1, 2)).std(ddof=1)),
                "queries": array.shape[2]} for method, array in values.items()]
    comparisons = []
    for left, right in ((fixed, "no_expansion"), ("adaptive_hard_clip", fixed),
                        ("adaptive_smooth", fixed), ("adaptive_smooth", "adaptive_hard_clip")):
        mean, low, high = hierarchical_interval(
            values[left] - values[right], [group for _, group in reference],
            args.bootstrap, stable_seed("frozen-center-radius", left, right))
        comparisons.append({"comparison": f"{left} - {right}",
                            "mean_difference": mean, "crossed_95_low": low,
                            "crossed_95_high": high})
    args.out.mkdir(parents=True, exist_ok=False)
    write_or_check(args.out / "selection.json", selection)
    frozen.csv_file(args.out / "metrics.csv", metrics)
    frozen.csv_file(args.out / "paired_comparisons.csv", comparisons)
    lines = ["# Matched validation-inference-seed comparison", "",
             f"Dataset: {selected['dataset']}; selected kappa: {selection['selected_kappa']:g}.",
             f"Fixed radius and adaptive epochs use validation seed {selection['selection_inference_seed']}. Test averages",
             "the original ten paired inference seeds; no model was retrained.",
             "Search spaces and global-fixed/per-center-adaptive selection still differ.",
             "This post-hoc robustness check reuses an already inspected test set.", ""]
    lines += [f"- {r['comparison']}: {r['mean_difference']:+.6f} "
              f"[{r['crossed_95_low']:+.6f}, {r['crossed_95_high']:+.6f}]" for r in comparisons]
    lines += ["", "Descriptive crossed-bootstrap 95% intervals; no multiplicity correction."]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n")
    finish(args.out, report_spec)
    print("\n".join(lines))


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select_parser = sub.add_parser("select")
    for flag in ("data", "original-run", "run", "out"):
        select_parser.add_argument("--" + flag, type=Path, required=True)
    select_parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    select_parser.add_argument("--cpu-threads", type=int, default=4)
    select_parser.add_argument("--interop-threads", type=int, default=1)
    select_parser.add_argument("--determinism", choices=("strict", "warn"), default="warn")
    report_parser = sub.add_parser("report")
    for flag in ("run", "selection", "results", "out"):
        report_parser.add_argument("--" + flag, type=Path, required=True)
    report_parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.command == "select":
        configure_runtime(args)
        select(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
