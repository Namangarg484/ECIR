"""Decouple training seeds from inference seeds and robustly select fixed spread."""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

from experiments.revision.protocol import aggregate, digest, dump, verify_manifest, write_jsonl
from experiments.revision.run import (Data, configure_runtime, environment, evaluate,
                                      seed_all, sources as revision_sources, train)
from src.models.vce_model import VCEModel

METHODS = ("weighted_fixed", "learned_fixed", "full")
LEARNED = {"learned_fixed", "full"}


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root / "experiments/revision/run.py",
             root / "experiments/revision/protocol.py", root / "src/models/vce_model.py"]
    return {str(path.relative_to(root)): digest(path) for path in paths}


def validate_extension_config(extension, revision):
    inference = extension.get("inference_seeds", [])
    if len(inference) < 2 or len(set(inference)) != len(inference):
        raise ValueError("Use at least two unique inference seeds")
    if extension.get("seeds") != revision.get("seeds"):
        raise ValueError("ECIR training seeds must exactly match the frozen revision run")
    if extension.get("selection_seed") != revision.get("selection_seed"):
        raise ValueError("ECIR selection seed must match the frozen revision run")


def load_model(data, checkpoint, device):
    model = VCEModel(embed_dim=data.emb.shape[1]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


def expected_validation(data, method, model, params, revision_config, inference_seeds):
    values = []
    for inference_seed in inference_seeds:
        rows = evaluate(data, "validation", method, model, params,
                        revision_config, inference_seed)
        values.append(aggregate(rows))
    return {metric: sum(value[metric] for value in values) / len(values)
            for metric in values[0]}


def copy_checkpoint(source, destination):
    shutil.copy2(source, destination)
    history = source.with_suffix(".history.json")
    if history.exists():
        shutil.copy2(history, destination.with_suffix(".history.json"))


def fit(args):
    original_manifest = verify_manifest(args.original_run)
    if args.device != original_manifest["device"]:
        raise ValueError("Robust selection must use the original fit device")
    runtime = environment()
    for key in ("determinism", "mps_cpu_fallback"):
        if runtime[key] != original_manifest["environment"].get(key, runtime[key]):
            raise ValueError(f"Robust-selection runtime {key} differs from original fit")
    if revision_sources() != original_manifest["source_sha256"]:
        raise ValueError("Frozen revision source no longer matches the original fit")
    if digest(args.data / "manifest.json") != original_manifest["data_manifest_sha256"]:
        raise ValueError("Prepared data differs from the original fit")
    extension = json.loads(args.config.read_text())
    revision_config = original_manifest["config"]
    validate_extension_config(extension, revision_config)
    original_selection = json.loads(
        (args.original_run / "selection.json").read_text())
    data = Data(args.data, args.device, ["train", "validation"])
    args.out.mkdir(parents=True, exist_ok=False)
    inference_seeds = extension["inference_seeds"]
    selection = {"inference_seeds": inference_seeds, "methods": {}, "candidates": []}

    # Weighted fixed-spread selection averages validation performance over inference noise.
    weighted_candidates = [row for row in original_selection["candidates"]
                           if row["method"] == "weighted_fixed"]
    best_score, weighted_choice = -1., None
    for row in weighted_candidates:
        metrics = expected_validation(data, "weighted_fixed", None, row["params"],
                                      revision_config, inference_seeds)
        record = {"method": "weighted_fixed", "params": row["params"],
                  "expected_validation": metrics}
        selection["candidates"].append(record)
        if metrics["NDCG@10"] > best_score:
            best_score, weighted_choice = metrics["NDCG@10"], record
    selection["methods"]["weighted_fixed"] = {
        "params": weighted_choice["params"],
        "expected_validation": weighted_choice["expected_validation"]}

    # Candidate checkpoints exist for every fixed kappa at the original selection seed.
    learned_candidates = [row for row in original_selection["candidates"]
                          if row["method"] == "learned_fixed"]
    best_score, learned_choice = -1., None
    selection_seed = revision_config["selection_seed"]
    for index, row in enumerate(learned_candidates):
        checkpoint = args.original_run / (
            f"learned_fixed.candidate-{index}.seed-{selection_seed}.pt")
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing candidate checkpoint: {checkpoint}")
        model = load_model(data, checkpoint, args.device)
        metrics = expected_validation(data, "learned_fixed", model, row["params"],
                                      revision_config, inference_seeds)
        record = {"method": "learned_fixed", "params": row["params"],
                  "source_checkpoint": checkpoint.name,
                  "expected_validation": metrics}
        selection["candidates"].append(record)
        if metrics["NDCG@10"] > best_score:
            best_score, learned_choice = metrics["NDCG@10"], record
        del model
    chosen_kappa = learned_choice["params"]["kappa"]
    original_learned = original_selection["methods"]["learned_fixed"]
    learned_checkpoints = {}
    chosen_index = next(i for i, row in enumerate(learned_candidates)
                        if row["params"] == learned_choice["params"])
    for seed in revision_config["seeds"]:
        destination = args.out / f"learned_fixed.seed-{seed}.pt"
        if seed == selection_seed:
            source = args.original_run / (
                f"learned_fixed.candidate-{chosen_index}.seed-{selection_seed}.pt")
            copy_checkpoint(source, destination)
        elif chosen_kappa == original_learned["params"]["kappa"]:
            source = args.original_run / original_learned["checkpoints"][str(seed)]
            copy_checkpoint(source, destination)
        else:
            # Only this branch retrains: the robust validation choice changed.
            train(data, "learned_fixed", learned_choice["params"], revision_config,
                  seed, destination)
        learned_checkpoints[str(seed)] = destination.name
    selection["methods"]["learned_fixed"] = {
        "params": learned_choice["params"],
        "expected_validation": learned_choice["expected_validation"],
        "original_kappa": original_learned["params"]["kappa"],
        "selection_changed": chosen_kappa != original_learned["params"]["kappa"],
        "checkpoints": learned_checkpoints}

    # Full checkpoints need no new hyperparameter selection; copy them immutably.
    full_original = original_selection["methods"]["full"]
    full_checkpoints = {}
    for seed in revision_config["seeds"]:
        destination = args.out / f"full.seed-{seed}.pt"
        copy_checkpoint(args.original_run / full_original["checkpoints"][str(seed)],
                        destination)
        full_checkpoints[str(seed)] = destination.name
    selection["methods"]["full"] = {
        "params": full_original["params"], "checkpoints": full_checkpoints}

    dump(args.out / "config.json", extension)
    dump(args.out / "selection.json", selection)
    files = sorted(path for path in args.out.iterdir() if path.is_file())
    dump(args.out / "manifest.json", {
        "schema": 1, "experiment": "ecir-stochastic-evaluation",
        "dataset": data.manifest["dataset"], "extension_config": extension,
        "revision_config": revision_config,
        "data_manifest_sha256": digest(args.data / "manifest.json"),
        "original_run_manifest_sha256": digest(args.original_run / "manifest.json"),
        "source_sha256": source_hashes(), "environment": environment(),
        "device": args.device, "files": {path.name: digest(path) for path in files},
        "command": sys.argv})
    print(f"Frozen robust stochastic selection: {args.out / 'selection.json'}")
    if selection["methods"]["learned_fixed"]["selection_changed"]:
        print("Validation-averaged learned-fixed kappa changed; non-selection seeds were retrained.")


def test(args):
    fit_manifest = verify_manifest(args.run)
    verify_manifest(args.original_run)
    if args.device != fit_manifest["device"]:
        raise ValueError("Test device differs from robust-selection device")
    if source_hashes() != fit_manifest["source_sha256"]:
        raise ValueError("Stochastic-evaluation source changed since fit")
    if digest(args.original_run / "manifest.json") != fit_manifest["original_run_manifest_sha256"]:
        raise ValueError("Original frozen run changed since robust selection")
    if digest(args.data / "manifest.json") != fit_manifest["data_manifest_sha256"]:
        raise ValueError("Prepared data changed since robust selection")
    runtime = environment()
    for key in ("determinism", "mps_cpu_fallback"):
        if runtime[key] != fit_manifest["environment"].get(key, runtime[key]):
            raise ValueError(f"Test runtime {key} differs from robust-selection runtime")
    data = Data(args.data, args.device, ["test"])
    selection = json.loads((args.run / "selection.json").read_text())
    revision_config = fit_manifest["revision_config"]
    inference_seeds = selection["inference_seeds"]
    args.out.mkdir(parents=True, exist_ok=False)
    summary, files = [], []
    for method in METHODS:
        method_selection = selection["methods"][method]
        train_seeds = [None] if method == "weighted_fixed" else revision_config["seeds"]
        for train_seed in train_seeds:
            model = None
            if method in LEARNED:
                model = load_model(
                    data, args.run / method_selection["checkpoints"][str(train_seed)],
                    args.device)
            for inference_seed in inference_seeds:
                seed_all(inference_seed)
                rows = evaluate(data, "test", method, model,
                                method_selection["params"], revision_config,
                                inference_seed)
                train_label = "none" if train_seed is None else str(train_seed)
                name = (f"{method}.train-{train_label}.inference-"
                        f"{inference_seed}.jsonl")
                write_jsonl(args.out / name, rows)
                files.append(name)
                summary.append({
                    "dataset": data.manifest["dataset"], "method": method,
                    "train_seed": train_seed, "inference_seed": inference_seed,
                    "n": len(rows), "file": name, **aggregate(rows)})
            if model is not None:
                del model
    dump(args.out / "summary.json", summary)
    files.append("summary.json")
    dump(args.out / "manifest.json", {
        "schema": 1, "experiment": "ecir-stochastic-evaluation",
        "dataset": data.manifest["dataset"],
        "extension_config": fit_manifest["extension_config"],
        "revision_config": revision_config,
        "data_manifest_sha256": digest(args.data / "manifest.json"),
        "robust_run_manifest_sha256": digest(args.run / "manifest.json"),
        "original_run_manifest_sha256": digest(args.original_run / "manifest.json"),
        "source_sha256": source_hashes(), "environment": environment(),
        "device": args.device, "files": {name: digest(args.out / name) for name in files},
        "command": sys.argv})
    print(f"Repeated stochastic test results saved: {args.out}")


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("fit", "test"):
        part = sub.add_parser(command)
        part.add_argument("--data", type=Path, required=True)
        part.add_argument("--original-run", type=Path, required=True)
        part.add_argument("--out", type=Path, required=True)
        part.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
        part.add_argument("--cpu-threads", type=int, default=min(4, os.cpu_count() or 1))
        part.add_argument("--interop-threads", type=int, default=1)
        part.add_argument("--determinism", choices=["strict", "warn"], default="strict")
        if command == "fit":
            part.add_argument("--config", type=Path,
                              default=Path("experiments/ecir/config.json"))
        else:
            part.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; choose a new versioned path")
    try:
        configure_runtime(args)
        (fit if args.command == "fit" else test)(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
