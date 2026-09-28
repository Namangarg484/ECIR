"""Frozen centers, finer fixed radii, and ten-seed adaptive epoch selection.

grid/fit load training or validation only. test requires sealed fit selections.
All old sources and results are read-only. Resume uses atomic epoch checkpoints.
"""
import argparse
import csv
import io
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.ecir import frozen_radius as frozen
from experiments.ecir import radius_v9_storage as storage
from experiments.ecir.query_direction_sensitivity import hierarchical_interval
from experiments.revision.protocol import METRICS, aggregate, digest, stable_seed, verify_manifest
from experiments.revision.run import Data, configure_runtime, environment, negatives, seed_all

METHODS = ("no_expansion", "fixed_global", "fixed_per_center",
           "adaptive_hard_clip", "adaptive_smooth")


def sources():
    root = Path(__file__).resolve().parents[2]
    return {**frozen.source_hashes(), **{
        str(path.relative_to(root)): digest(path)
        for path in (Path(__file__), Path(storage.__file__))}}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(config, previous, revision):
    check(config.get("schema") == 1, "Unknown configuration schema")
    for key in ("parameterizations", "radius_head_learning_rate", "epochs", "patience"):
        check(config.get(key) == previous[key], f"Preserve the original head-training {key}")
    for key in ("selection_inference_seeds", "test_inference_seeds"):
        check(config.get(key) == previous["inference_seeds"] == list(range(3101, 3111)),
              "Both validation and test must use all original ten inference seeds")
    grid = config.get("fixed_kappas", [])
    check(len(grid) > len(previous["fixed_kappas"]) and
          all(isinstance(k, (int, float)) and math.isfinite(k) and 10 <= k <= 500 for k in grid),
          "Use a finite finer radius grid within [10, 500]")
    check(grid == sorted(set(grid)) and set(previous["fixed_kappas"]).issubset(grid),
          "Grid must be sorted, unique, and include every original fixed radius")
    check(revision["seeds"] == [42, 123, 456, 789, 2026] and revision["samples"] == 5,
          "Preserve original center seeds and five probes")


def inputs(args):
    parent = frozen.verify_parent(args)
    previous = verify_manifest(args.previous_run)
    check(previous.get("experiment") == "ecir-frozen-center-radius-fit", "Expected original radius fit")
    check(previous["source_sha256"] == frozen.source_hashes(), "Frozen-radius source changed")
    for key, folder in (("data_manifest_sha256", args.data),
                        ("original_run_manifest_sha256", args.original_run)):
        check(previous[key] == digest(folder / "manifest.json"), f"Parent mismatch: {key}")
    check(previous["revision_config"] == parent["config"], "Original training configurations differ")
    check(previous["device"] == args.device, "Preserve original radius-training device")
    for key in ("determinism", "mps_cpu_fallback"):
        check(previous["environment"][key] == environment()[key], f"Runtime mismatch: {key}")
    centers = json.loads((args.previous_run / "selection.json").read_text())["centers"]
    for seed in parent["config"]["seeds"]:
        entry = centers[str(seed)]
        check(digest(args.original_run / entry["checkpoint"]) == entry["sha256"], "Center checksum differs")
    return parent, previous, centers


def best_index(values):
    check(len(values) > 0 and all(math.isfinite(float(v)) for v in values), "Invalid validation scores")
    return max(range(len(values)), key=lambda index: values[index])  # First declared candidate on ties.


def choose_fixed(grid, matrix):
    """matrix: [center seed, inference seed, kappa], validation NDCG only."""
    array = np.asarray(matrix, dtype=float)
    check(array.ndim == 3 and array.shape[2] == len(grid) and np.isfinite(array).all(),
          "Fixed validation matrix has invalid shape or values")
    per_center = array.mean(axis=1)
    global_means = per_center.mean(axis=0)
    return {"global_kappa": grid[best_index(global_means)],
            "per_center_kappas": [grid[best_index(row)] for row in per_center],
            "global_validation_means": global_means.tolist(),
            "per_center_validation_means": per_center.tolist()}


def update_best(score, epoch, best_score, best_epoch):
    check(math.isfinite(score), "Nonfinite validation score")
    return (score, epoch) if score > best_score else (best_score, best_epoch)


def fixed_grid(args, data, centers, config, revision):
    matrix = []
    for seed in revision["seeds"]:
        checkpoint = args.original_run / centers[str(seed)]["checkpoint"]
        model = frozen.load_model(data, checkpoint, args.device, "hard_clip")
        seed_scores = []
        for noise_seed in config["selection_inference_seeds"]:
            path = args.out / "fixed-validation" / f"center-{seed}.noise-{noise_seed}.json"
            def compute():
                suite = frozen.evaluate_expanded_suite(data, "validation", model, model,
                    config["fixed_kappas"], revision, noise_seed, include_adaptive=False)
                return {frozen.fixed_name(k): aggregate(suite[frozen.fixed_name(k)])
                        for k in config["fixed_kappas"]}
            values = storage.cell(path, compute)
            seed_scores.append([values[frozen.fixed_name(k)]["NDCG@10"] for k in config["fixed_kappas"]])
            print(f"grid center={seed} validation-noise={noise_seed}: verified", flush=True)
        matrix.append(seed_scores)
        del model
    selected = choose_fixed(config["fixed_kappas"], matrix)
    selected["per_center_kappas"] = dict(zip(map(str, revision["seeds"]), selected["per_center_kappas"]))
    selected["grid"] = config["fixed_kappas"]
    storage.json_write(args.out / "fixed_selection.json", selected)
    print(f"Validation-only grid done: global={selected['global_kappa']}; "
          f"per-center={selected['per_center_kappas']}", flush=True)
    return selected


def train_epoch(data, model, optimizer, revision, seed, epoch):
    """Original objective/negatives/noise; log gradient diagnostics as well."""
    model.set_radius_training(True)
    rows = data.splits["train"]
    check(bool(rows), "Empty training split")
    pool = np.asarray(sorted({i for row in rows for i in row["context"] + [row["target"]]}))
    rng = np.random.default_rng(stable_seed(seed, "radius-head", epoch))
    order = rng.permutation(len(rows))
    loss_total, grad_total, zero_batches, batches = 0., 0., 0, 0
    for start in range(0, len(rows), revision["batch_size"]):
        batch = [rows[index] for index in order[start:start + revision["batch_size"]]]
        candidates = torch.tensor([[r["target"]] + negatives(pool, r, revision["negative_count"], rng)
                                   for r in batch], device=data.emb.device)
        optimizer.zero_grad(set_to_none=True)
        _, anchors, mask, query = data.batch(batch, revision["max_context"])
        center, kappa = model(query, anchors, mask)
        epsilon = frozen.event_noise(batch, revision["samples"], center.shape[-1], seed,
                                     f"radius-train-{epoch}", center.device)
        queries = frozen.expand(center, kappa, epsilon)
        logits = torch.einsum("bsd,bnd->bsn", queries, data.emb[candidates]).max(1).values
        loss = F.cross_entropy(logits / revision["temperature"],
                               torch.zeros(len(batch), dtype=torch.long, device=data.emb.device))
        check(bool(torch.isfinite(loss)), "Nonfinite radius training loss")
        loss.backward()
        parameters = list(model.radius_parameters())
        check(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),
              "Missing or nonfinite radius gradient")
        grad = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        model.assert_center_frozen()
        check(all(bool(torch.isfinite(p).all()) for p in parameters), "Nonfinite updated head")
        loss_total += float(loss.detach()) * len(batch)
        grad_total += float(grad)
        zero_batches += int(float(grad) == 0.)
        batches += 1
    return {"training_loss": loss_total / len(rows), "mean_unclipped_gradient_norm": grad_total / batches,
            "zero_gradient_batch_fraction": zero_batches / batches, "batches": batches}


def restore(model, optimizer, saved, center_checkpoint, parameterization, seed, epoch):
    check(saved["center_checkpoint_sha256"] == digest(center_checkpoint) and
          saved["center_state_sha256"] == model.center_digest(), "Checkpoint center mismatch")
    check((saved["parameterization"], saved["training_seed"], saved["epoch"]) ==
          (parameterization, seed, epoch), "Checkpoint identity mismatch")
    model.base.kappa_predictor.load_state_dict(saved["head_state"])
    if optimizer is not None:
        optimizer.load_state_dict(saved["optimizer_state"])
    model.assert_center_frozen()


def fit_head(args, data, checkpoint, config, revision, seed, parameterization):
    seed_all(seed)
    model = frozen.load_model(data, checkpoint, args.device, parameterization)
    # Initialization is inherited from the original learned-center checkpoint,
    # exactly as in v1; do not initialize from its trained adaptive winner.
    seed_all(seed)
    optimizer = torch.optim.AdamW(list(model.radius_parameters()),
        lr=config["radius_head_learning_rate"], weight_decay=revision["weight_decay"])
    folder = args.out / "heads" / parameterization / f"seed-{seed}"
    best_score, best_epoch, history = -1., 0, []
    for epoch in range(1, config["epochs"] + 1):
        epoch_folder = folder / f"epoch-{epoch:03d}"
        state_folder = epoch_folder / "training-state"
        started = time.monotonic()
        if state_folder.exists():
            saved = storage.state_load(state_folder)
            restore(model, optimizer, saved, checkpoint, parameterization, seed, epoch)
        else:
            statistics = train_epoch(data, model, optimizer, revision, seed, epoch)
            saved = {**frozen.head_checkpoint(model, checkpoint, parameterization),
                     "training_seed": seed, "epoch": epoch, "training": statistics,
                     "optimizer_state": optimizer.state_dict()}
            storage.state_save(state_folder, saved)
        # Current head has no dropout; negatives/order/noise have explicit per-epoch
        # generators. Resuming a completed epoch needs no mutable global RNG stream.
        validation = []
        for noise_seed in config["selection_inference_seeds"]:
            path = epoch_folder / f"validation-noise-{noise_seed}.json"
            def compute():
                suite = frozen.evaluate_expanded_suite(data, "validation", model, model, [],
                    revision, noise_seed, adaptive_conditions=(parameterization,))
                return aggregate(suite[f"adaptive_{parameterization}"])
            validation.append(storage.cell(path, compute))
            print(f"head={parameterization} center={seed} epoch={epoch} "
                  f"validation-noise={noise_seed}: verified", flush=True)
        averaged = {metric: float(np.mean([v[metric] for v in validation])) for metric in METRICS}
        score = averaged["NDCG@10"]
        best_score, best_epoch = update_best(score, epoch, best_score, best_epoch)
        record = {"epoch": epoch, "validation": averaged, "training": saved["training"],
                  "best_epoch_so_far": best_epoch}
        storage.json_write(epoch_folder / "summary.json", record)
        history.append(record)
        print(f"v9 {parameterization} center={seed} epoch={epoch} "
              f"ten_seed_val_ndcg10={score:.6f} best_epoch={best_epoch} "
              f"elapsed={time.monotonic()-started:.1f}s", flush=True)
        if epoch - best_epoch >= config["patience"]:
            break
    selected_path = folder / f"epoch-{best_epoch:03d}" / "training-state"
    selected = storage.state_load(selected_path)
    restore(model, None, selected, checkpoint, parameterization, seed, best_epoch)
    stats = storage.cell(folder / "selected_kappa.json", lambda: frozen.kappa_statistics(
        data, "validation", model, revision, .001))
    result = {"checkpoint_directory": selected_path.relative_to(args.out).as_posix(),
              "best_epoch": best_epoch, "validation_ndcg10": best_score,
              "epochs_completed": len(history), "kappa_statistics": stats}
    storage.json_write(folder / "selection.json", result)
    return result


def fit(args):
    parent, previous, centers = inputs(args)
    config = json.loads(args.config.read_text())
    revision = parent["config"]
    validate_config(config, previous["config"], revision)
    spec = {"schema": 1, "experiment": "ecir-radius-v9-fit", "dataset": parent["dataset"],
            "config": config, "revision_config": revision, "source_sha256": sources(),
            "data_manifest_sha256": digest(args.data / "manifest.json"),
            "original_run_manifest_sha256": digest(args.original_run / "manifest.json"),
            "previous_fit_manifest_sha256": digest(args.previous_run / "manifest.json"),
            "environment": environment(), "device": args.device}
    storage.prepare(args.out, spec)
    if storage.complete(args.out, spec):
        print("Fit already complete and verified.")
        return
    splits = ["validation"] if args.command == "grid" else ["train", "validation"]
    data = Data(args.data, args.device, splits)
    fixed = fixed_grid(args, data, centers, config, revision)
    if args.command == "grid":
        print("Grid-only stage complete. No head training or test evaluation performed.")
        return
    heads = {kind: {} for kind in config["parameterizations"]}
    for seed in revision["seeds"]:
        checkpoint = args.original_run / centers[str(seed)]["checkpoint"]
        for kind in config["parameterizations"]:
            heads[kind][str(seed)] = fit_head(args, data, checkpoint, config, revision, seed, kind)
    storage.json_write(args.out / "selection.json", {"centers": centers, "fixed": fixed, "heads": heads})
    storage.finish(args.out, spec)
    print("All validation selections sealed. Test examples and metrics were not loaded.")


def test(args):
    parent, previous, _ = inputs(args)
    fitted = verify_manifest(args.run)
    check(fitted.get("experiment") == "ecir-radius-v9-fit", "Expected sealed v9 fit")
    check(fitted["source_sha256"] == sources(), "New experiment source changed after fitting")
    for key, folder in (("data_manifest_sha256", args.data),
                        ("original_run_manifest_sha256", args.original_run),
                        ("previous_fit_manifest_sha256", args.previous_run)):
        check(fitted[key] == digest(folder / "manifest.json"), f"Fit parent mismatch: {key}")
    check(fitted["device"] == args.device, "Fit/test devices differ")
    config, revision = fitted["config"], fitted["revision_config"]
    validate_config(config, previous["config"], revision)
    spec = {"schema": 1, "experiment": "ecir-radius-v9-test", "dataset": parent["dataset"],
            "fit_manifest_sha256": digest(args.run / "manifest.json"),
            "data_manifest_sha256": digest(args.data / "manifest.json"),
            "source_sha256": sources(), "config": config, "revision_config": revision,
            "environment": environment(), "device": args.device}
    storage.prepare(args.out, spec)
    if storage.complete(args.out, spec):
        print("Test already complete and verified.")
        return
    selected = json.loads((args.run / "selection.json").read_text())
    data = Data(args.data, args.device, ["test"])
    summary = []
    for seed in revision["seeds"]:
        checkpoint = args.original_run / selected["centers"][str(seed)]["checkpoint"]
        models = {}
        for kind in config["parameterizations"]:
            head = selected["heads"][kind][str(seed)]
            model = frozen.load_model(data, checkpoint, args.device, kind)
            saved = storage.state_load(args.run / head["checkpoint_directory"])
            restore(model, None, saved, checkpoint, kind, seed, head["best_epoch"])
            models[kind] = model
        hard, smooth = models["hard_clip"], models["smooth"]
        name = f"cells/center-{seed}.json"
        rows = storage.cell(args.out / name, lambda: {
            "no_expansion": frozen.evaluate_no_expansion(data, "test", hard, revision)})
        summary.append({"method": "no_expansion", "train_seed": seed, "inference_seed": None,
                        "file": name, "n": len(rows["no_expansion"]), **aggregate(rows["no_expansion"])})
        global_k = selected["fixed"]["global_kappa"]
        local_k = selected["fixed"]["per_center_kappas"][str(seed)]
        for noise_seed in config["test_inference_seeds"]:
            name = f"cells/center-{seed}.noise-{noise_seed}.json"
            def compute():
                suite = frozen.evaluate_expanded_suite(data, "test", hard, smooth,
                    sorted({global_k, local_k}), revision, noise_seed)
                return {"fixed_global": suite[frozen.fixed_name(global_k)],
                        "fixed_per_center": suite[frozen.fixed_name(local_k)],
                        "adaptive_hard_clip": suite["adaptive_hard_clip"],
                        "adaptive_smooth": suite["adaptive_smooth"]}
            evaluated = storage.cell(args.out / name, compute)
            for method, rows in evaluated.items():
                summary.append({"method": method, "train_seed": seed, "inference_seed": noise_seed,
                                "file": name, "n": len(rows), **aggregate(rows)})
            print(f"test center={seed} inference={noise_seed}: verified", flush=True)
        del hard, smooth, models
    storage.json_write(args.out / "summary.json", summary)
    storage.finish(args.out, spec)
    print("Test complete; only validation-selected fixed radii were tested.")


def arrays_from_results(folder, manifest, summary, seeds, noise_seeds):
    entries = {(r["method"], r["train_seed"], r["inference_seed"]): r for r in summary}
    expected = {(m, s, n) for m in METHODS for s in seeds
                for n in ([None] if m == "no_expansion" else noise_seeds)}
    check(len(entries) == len(summary) and set(entries) == expected, "Missing, duplicate, or unexpected test cells")
    reference, values, cache = None, {}, {}
    for method in METHODS:
        training = []
        for seed in seeds:
            inference = []
            for noise_seed in ([None] if method == "no_expansion" else noise_seeds):
                entry = entries[method, seed, noise_seed]
                check(entry["file"] in manifest["files"], "Unverified test payload")
                # One cached suite at a time avoids retaining the whole JSON corpus.
                if entry["file"] not in cache:
                    cache = {entry["file"]: storage.cell_read(folder / entry["file"])}
                rows = sorted(cache[entry["file"]][method], key=lambda r: r["event"])
                support = [(r["event"], r["group"]) for r in rows]
                check(len({r["event"] for r in rows}) == len(rows), "Duplicate prediction event")
                if reference is None:
                    reference = support
                check(support == reference, "Methods do not share exact paired support")
                recomputed = aggregate(rows)
                check(entry["n"] == len(rows) and all(abs(recomputed[k] - entry[k]) < 1e-10 for k in METRICS),
                      "Test summary does not match per-event metrics")
                inference.append(np.asarray([[r[k] for k in METRICS] for r in rows], dtype=np.float64))
            stacked = np.stack(inference)
            if method == "no_expansion":
                stacked = np.repeat(stacked, len(noise_seeds), axis=0)
            training.append(stacked)
        values[method] = np.stack(training)
    return values, reference


def text_write(path, text):
    storage.text_write(path, text)


def csv_write(path, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    text_write(path, buffer.getvalue())


def report(args):
    check(args.bootstrap >= 100, "Use at least 100 bootstrap replicates")
    fitted, tested = verify_manifest(args.run), verify_manifest(args.results)
    check(fitted.get("experiment") == "ecir-radius-v9-fit" and
          tested.get("experiment") == "ecir-radius-v9-test", "Unexpected fit/test artifacts")
    check(tested["fit_manifest_sha256"] == digest(args.run / "manifest.json"), "Fit/test provenance mismatch")
    check(fitted["source_sha256"] == tested["source_sha256"] == sources(), "Evaluated source changed")
    check(fitted["config"] == tested["config"] and fitted["revision_config"] == tested["revision_config"] and
          fitted["data_manifest_sha256"] == tested["data_manifest_sha256"], "Fit/test protocol mismatch")
    spec = {"schema": 1, "experiment": "ecir-radius-v9-report", "dataset": fitted["dataset"],
            "source_sha256": sources(), "fit_manifest_sha256": digest(args.run / "manifest.json"),
            "result_manifest_sha256": digest(args.results / "manifest.json"), "bootstrap": args.bootstrap}
    storage.prepare(args.out, spec)
    if storage.complete(args.out, spec):
        print("Report already complete and verified.")
        return
    selection = json.loads((args.run / "selection.json").read_text())
    summary = json.loads((args.results / "summary.json").read_text())
    config, revision = fitted["config"], fitted["revision_config"]
    values, support = arrays_from_results(args.results, tested, summary, revision["seeds"],
                                         config["test_inference_seeds"])
    records = []
    for method, array in values.items():
        record = {"method": method, "queries": array.shape[2], "training_seeds": array.shape[0],
                  "inference_seeds": 0 if method == "no_expansion" else array.shape[1]}
        for index, metric in enumerate(METRICS):
            scores = array[..., index]
            record[f"{metric} mean"] = float(scores.mean())
            record[f"{metric} train_sd"] = float(scores.mean(axis=(1, 2)).std(ddof=1))
        records.append(record)
    pairs = [(fixed, "no_expansion") for fixed in ("fixed_global", "fixed_per_center")]
    pairs += [(adaptive, fixed) for adaptive in ("adaptive_hard_clip", "adaptive_smooth")
              for fixed in ("fixed_global", "fixed_per_center")]
    pairs += [("adaptive_smooth", "adaptive_hard_clip"), ("fixed_per_center", "fixed_global")]
    comparisons = []
    for left, right in pairs:
        difference = values[left][..., 0] - values[right][..., 0]
        mean, low, high = hierarchical_interval(difference, [g for _, g in support], args.bootstrap,
                                                stable_seed("radius-v9", left, right))
        comparisons.append({"comparison": f"{left} - {right}", "mean_difference": mean,
                            "crossed_95_low": low, "crossed_95_high": high,
                            "training_seeds": len(revision["seeds"]),
                            "inference_seeds": len(config["test_inference_seeds"]),
                            "clusters": len(set(g for _, g in support)), "replicates": args.bootstrap})
    csv_write(args.out / "metrics.csv", records)
    csv_write(args.out / "paired_comparisons.csv", comparisons)
    storage.json_write(args.out / "selection.json", selection)
    csv_write(args.out / "fixed_validation.csv", [
        {"training_seed": seed, "kappa": kappa, "mean_validation_ndcg10":
         selection["fixed"]["per_center_validation_means"][index][j]}
        for index, seed in enumerate(revision["seeds"]) for j, kappa in enumerate(config["fixed_kappas"])])
    lines = [f"# Ten-seed radius selection: {fitted['dataset']}", "",
             f"Global fixed kappa: {selection['fixed']['global_kappa']}",
             f"Per-center kappas: {selection['fixed']['per_center_kappas']}", "",
             "Heads and fixed radii use the same ten validation inference seeds.",
             "Head early stopping also uses the ten-seed mean. Centers remain frozen.",
             "Per-center fixed selection matches the adaptive selection unit, not its search space or compute.",
             "Test uses ten paired inference seeds. Only validation-selected fixed radii are tested.", "",
             "## NDCG@10", ""]
    lines += [f"- {r['method']}: {r['NDCG@10 mean']:.6f}" for r in records]
    lines += ["", "## Paired crossed-bootstrap contrasts", ""]
    lines += [f"- {r['comparison']}: {r['mean_difference']:+.6f} "
              f"[{r['crossed_95_low']:+.6f}, {r['crossed_95_high']:+.6f}]" for r in comparisons]
    lines += ["", "Post-hoc robustness analysis on an already examined test set. Intervals are descriptive,",
              "conditional on the selected configurations, and uncorrected for multiple comparisons.",
              "An interval containing zero does not demonstrate equivalence."]
    text_write(args.out / "REPORT.md", "\n".join(lines) + "\n")
    storage.finish(args.out, spec)
    print("\n".join(lines))


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("grid", "fit", "test"):
        part = sub.add_parser(command)
        for flag in ("data", "original-run", "previous-run", "out"):
            part.add_argument("--" + flag, type=Path, required=True)
        part.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
        part.add_argument("--cpu-threads", type=int, default=4)
        part.add_argument("--interop-threads", type=int, default=1)
        part.add_argument("--determinism", choices=("strict", "warn"), default="warn")
        if command == "test":
            part.add_argument("--run", type=Path, required=True)
        else:
            part.add_argument("--config", type=Path, default=Path("experiments/ecir/radius_v9_config.json"))
    part = sub.add_parser("report")
    for flag in ("run", "results", "out"):
        part.add_argument("--" + flag, type=Path, required=True)
    part.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.command != "report":
        configure_runtime(args)
    with storage.output_lock(args.out):
        if args.command in ("grid", "fit"):
            fit(args)
        elif args.command == "test":
            test(args)
        else:
            report(args)


if __name__ == "__main__":
    main()
