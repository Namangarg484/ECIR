"""Safely resume an interrupted frozen-radius fit or test.

This recovery entry point preserves the immutable main experiment program.
It validates every completed artifact before reuse and writes the same
selection, result, and manifest schemas as ``experiments.ecir.frozen_radius``.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.ecir import frozen_radius as experiment
from experiments.revision.protocol import (
    aggregate,
    digest,
    dump,
    read_jsonl,
    verify_manifest,
    write_jsonl,
)
from experiments.revision.run import Data, configure_runtime, environment


def completed_head(model, center_checkpoint, parameterization, checkpoint):
    """Return validation score after validating one checkpoint/history pair."""
    history_path = checkpoint.with_suffix(".history.json")
    if not checkpoint.is_file() or not history_path.is_file():
        raise ValueError(
            f"Incomplete checkpoint pair cannot be reused safely: {checkpoint}")
    saved = torch.load(checkpoint, map_location=model.base.residual_gate.device,
                       weights_only=True)
    if saved.get("parameterization") != parameterization:
        raise ValueError(f"Parameterization mismatch: {checkpoint}")
    if saved.get("center_checkpoint_sha256") != digest(center_checkpoint):
        raise ValueError(f"Parent center mismatch: {checkpoint}")
    model.base.kappa_predictor.load_state_dict(saved["head_state"])
    model.assert_center_frozen()
    if saved.get("center_state_sha256") != model.center_digest():
        raise ValueError(f"Frozen center mismatch: {checkpoint}")

    history = json.loads(history_path.read_text(encoding="utf-8"))
    best_epoch = int(history.get("best_epoch", 0))
    matching = [row for row in history.get("epochs", [])
                if int(row.get("epoch", -1)) == best_epoch]
    if len(matching) != 1:
        raise ValueError(f"Invalid best epoch: {history_path}")
    score = float(matching[0]["validation"]["NDCG@10"])
    print(f"resume: verified {checkpoint.name} "
          f"(best epoch {best_epoch})", flush=True)
    return score


def resume_fit(args):
    if not args.out.is_dir() or (args.out / "manifest.json").exists():
        raise ValueError("Fit recovery requires an incomplete output directory")
    original = experiment.verify_parent(args)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    revision = original["config"]
    experiment.validate_config(config, revision)
    selected = experiment.center_selection(args.original_run)
    data = Data(args.data, args.device, ["train", "validation"])

    center_records = {}
    fixed_totals = {float(value): [] for value in config["fixed_kappas"]}
    adaptive = {name: {"checkpoints": {}, "validation": {},
                       "kappa_statistics": {}}
                for name in experiment.PARAMETERIZATIONS}
    reused = []
    for training_seed in revision["seeds"]:
        center_checkpoint = (
            args.original_run / selected["checkpoints"][str(training_seed)])
        center_records[str(training_seed)] = {
            "checkpoint": center_checkpoint.name,
            "sha256": digest(center_checkpoint),
        }

        initial_hard = experiment.load_model(
            data, center_checkpoint, args.device, "hard_clip")
        initial_smooth = experiment.load_model(
            data, center_checkpoint, args.device, "smooth")
        for inference_seed in config["inference_seeds"]:
            suite = experiment.evaluate_expanded_suite(
                data, "validation", initial_hard, initial_smooth,
                config["fixed_kappas"], revision, inference_seed,
                include_adaptive=False)
            for value in config["fixed_kappas"]:
                fixed_totals[float(value)].append(
                    aggregate(suite[experiment.fixed_name(value)])["NDCG@10"])
        del initial_hard, initial_smooth

        for parameterization in experiment.PARAMETERIZATIONS:
            model = experiment.load_model(
                data, center_checkpoint, args.device, parameterization)
            destination = args.out / (
                f"adaptive_{parameterization}.seed-{training_seed}.pt")
            history_path = destination.with_suffix(".history.json")
            if destination.exists() or history_path.exists():
                score = completed_head(
                    model, center_checkpoint, parameterization, destination)
                reused.append(destination.name)
            else:
                score = experiment.train_head(
                    data, model, center_checkpoint, parameterization,
                    revision, config, training_seed, destination)

            saved = torch.load(destination, map_location=args.device,
                               weights_only=True)
            model.base.kappa_predictor.load_state_dict(saved["head_state"])
            adaptive[parameterization]["checkpoints"][str(training_seed)] = (
                destination.name)
            adaptive[parameterization]["validation"][str(training_seed)] = score
            adaptive[parameterization]["kappa_statistics"][str(training_seed)] = (
                experiment.kappa_statistics(
                    data, "validation", model, revision,
                    float(config["boundary_tolerance"])))
            del model

    candidates = [
        {"kappa": value,
         "mean_validation_ndcg10": float(np.mean(fixed_totals[value]))}
        for value in map(float, config["fixed_kappas"])
    ]
    chosen = max(candidates, key=lambda row: (
        row["mean_validation_ndcg10"],
        -config["fixed_kappas"].index(row["kappa"])))
    selection = {
        "center_method": "learned_centroid",
        "center_params": selected["params"],
        "centers": center_records,
        "fixed_radius": {
            "selected_kappa": chosen["kappa"],
            "candidates": candidates,
            "selection_unit": "mean over training and inference seeds",
        },
        "adaptive_radius": adaptive,
    }
    dump(args.out / "config.json", config)
    dump(args.out / "selection.json", selection)
    files = sorted(path for path in args.out.iterdir() if path.is_file())
    dump(args.out / "manifest.json", {
        "schema": 1,
        "experiment": "ecir-frozen-center-radius-fit",
        "dataset": original["dataset"],
        "config": config,
        "revision_config": revision,
        "data_manifest_sha256": digest(args.data / "manifest.json"),
        "original_run_manifest_sha256": digest(
            args.original_run / "manifest.json"),
        "source_sha256": experiment.source_hashes(),
        "recovery_source_sha256": digest(Path(__file__)),
        "environment": environment(),
        "device": args.device,
        "selection_metric": "validation NDCG@10; test never loaded",
        "recovery": {"reused_head_checkpoints": reused},
        "files": {path.name: digest(path) for path in files},
        "command": sys.argv,
    })
    print(f"Recovered frozen-center fit: {args.out}")
    print(f"Reused {len(reused)} completed head checkpoint(s).")
    print("Test data were not loaded.")


def verified_result(path, data):
    rows = read_jsonl(path)
    expected = [(row["event"], row["group"]) for row in data.splits["test"]]
    actual = [(row.get("event"), row.get("group")) for row in rows]
    if actual != expected:
        raise ValueError(f"Incomplete or mismatched result file: {path}")
    aggregate(rows)
    return rows


def summary_record(dataset, method, training_seed, inference_seed, name, rows):
    return {
        "dataset": dataset,
        "method": method,
        "train_seed": training_seed,
        "inference_seed": inference_seed,
        "n": len(rows),
        "file": name,
        **aggregate(rows),
    }


def resume_test(args):
    if not args.out.is_dir() or (args.out / "manifest.json").exists():
        raise ValueError("Test recovery requires an incomplete output directory")
    original = experiment.verify_parent(args)
    manifest = verify_manifest(args.run)
    if manifest.get("experiment") != "ecir-frozen-center-radius-fit":
        raise ValueError("--run is not a frozen-center radius fit")
    if experiment.source_hashes() != manifest["source_sha256"]:
        raise ValueError("Frozen-radius source changed since fitting")
    if digest(args.original_run / "manifest.json") != manifest[
            "original_run_manifest_sha256"]:
        raise ValueError("Parent run differs from the frozen-radius fit")
    selection = json.loads((args.run / "selection.json").read_text(
        encoding="utf-8"))
    config, revision = manifest["config"], manifest["revision_config"]
    experiment.validate_config(config, revision)
    data = Data(args.data, args.device, ["test"])

    summary, files, reused = [], [], []
    total = len(revision["seeds"]) * (1 + len(config["inference_seeds"]))
    completed = 0
    for training_seed in revision["seeds"]:
        hard = experiment.load_fitted_model(
            data, args, selection, training_seed, "hard_clip")
        smooth = experiment.load_fitted_model(
            data, args, selection, training_seed, "smooth")
        name = f"no_expansion.train-{training_seed}.jsonl"
        path = args.out / name
        if path.exists():
            rows = verified_result(path, data)
            reused.append(name)
        else:
            rows = experiment.evaluate_no_expansion(data, "test", hard, revision)
            write_jsonl(path, rows)
        files.append(name)
        summary.append(summary_record(
            original["dataset"], "no_expansion", training_seed, None,
            name, rows))
        completed += 1
        print(f"frozen-radius test {completed}/{total}: "
              f"center seed={training_seed} no expansion", flush=True)

        for inference_seed in config["inference_seeds"]:
            methods = [experiment.fixed_name(value)
                       for value in config["fixed_kappas"]]
            methods += ["adaptive_hard_clip", "adaptive_smooth"]
            names = {
                method: (f"{method}.train-{training_seed}.inference-"
                         f"{inference_seed}.jsonl")
                for method in methods
            }
            missing = [method for method in methods
                       if not (args.out / names[method]).exists()]
            suite = (experiment.evaluate_expanded_suite(
                data, "test", hard, smooth, config["fixed_kappas"], revision,
                inference_seed) if missing else {})
            for method in methods:
                name = names[method]
                path = args.out / name
                if path.exists():
                    method_rows = verified_result(path, data)
                    reused.append(name)
                else:
                    method_rows = suite[method]
                    write_jsonl(path, method_rows)
                files.append(name)
                summary.append(summary_record(
                    original["dataset"], method, training_seed,
                    inference_seed, name, method_rows))
            completed += 1
            print(f"frozen-radius test {completed}/{total}: center "
                  f"seed={training_seed} inference={inference_seed}", flush=True)
        del hard, smooth

    summary_path = args.out / "summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing != summary:
            raise ValueError("Existing summary conflicts with recovered results")
    else:
        dump(summary_path, summary)
    files.append("summary.json")
    dump(args.out / "manifest.json", {
        "schema": 1,
        "experiment": "ecir-frozen-center-radius-test",
        "dataset": original["dataset"],
        "config": config,
        "revision_config": revision,
        "data_manifest_sha256": digest(args.data / "manifest.json"),
        "original_run_manifest_sha256": digest(
            args.original_run / "manifest.json"),
        "fit_manifest_sha256": digest(args.run / "manifest.json"),
        "source_sha256": experiment.source_hashes(),
        "recovery_source_sha256": digest(Path(__file__)),
        "environment": environment(),
        "device": args.device,
        "recovery": {"reused_result_files": reused},
        "files": {name: digest(args.out / name) for name in files},
        "command": sys.argv,
    })
    print(f"Recovered frozen-center test results: {args.out}")
    print(f"Reused {len(reused)} completed result file(s).")
    print("No test result was used for selection or training.")


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("fit", "test"):
        part = subparsers.add_parser(command)
        part.add_argument("--data", type=Path, required=True)
        part.add_argument("--original-run", type=Path, required=True)
        part.add_argument("--out", type=Path, required=True)
        part.add_argument("--device", choices=("cpu", "cuda", "mps"),
                          default="cpu")
        part.add_argument("--cpu-threads", type=int,
                          default=min(4, os.cpu_count() or 1))
        part.add_argument("--interop-threads", type=int, default=1)
        part.add_argument("--determinism", choices=("strict", "warn"),
                          default="strict")
        if command == "fit":
            part.add_argument(
                "--config", type=Path,
                default=Path("experiments/ecir/frozen_radius_config.json"))
        else:
            part.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        configure_runtime(args)
        (resume_fit if args.command == "fit" else resume_test)(args)
    except (KeyError, ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
