"""Frozen-center identification experiment for fixed and adaptive radii.

For every existing learned-center training seed, this experiment freezes the
entire selected center and trains only a radius head.  The exact same center is
then evaluated with no expansion, every declared fixed kappa, the legacy
hard-clipped parameterization, and a smooth bounded parameterization.  Test is
a separate command and never participates in fitting or selection.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.ecir.query_direction_sensitivity import hierarchical_interval
from experiments.revision.protocol import (
    METRICS,
    aggregate,
    digest,
    dump,
    metrics_from_scores,
    read_jsonl,
    stable_seed,
    verify_manifest,
    write_jsonl,
)
from experiments.revision.run import (
    Data,
    configure_runtime,
    environment,
    negatives,
    seed_all,
    sources as revision_sources,
)
from src.models.vce_model import VCEModel

PARAMETERIZATIONS = ("hard_clip", "smooth")
LATEX_ROW_END = r"\\"


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root / "experiments/revision/run.py",
             root / "experiments/revision/protocol.py",
             root / "experiments/ecir/query_direction_sensitivity.py",
             root / "src/models/vce_model.py"]
    return {str(path.relative_to(root)): digest(path) for path in paths}


def tensor_state_digest(state):
    """Stable digest of named CPU tensors, independent of torch serialization."""
    checksum = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        checksum.update(name.encode("utf-8"))
        checksum.update(str(tensor.dtype).encode("ascii"))
        checksum.update(json.dumps(list(tensor.shape)).encode("ascii"))
        checksum.update(tensor.numpy().tobytes())
    return checksum.hexdigest()


class FrozenCenterRadius(nn.Module):
    """Exact evaluated center with only its legacy-shaped radius head trainable."""

    def __init__(self, base, parameterization):
        super().__init__()
        if parameterization not in PARAMETERIZATIONS:
            raise ValueError(f"Unknown parameterization: {parameterization}")
        self.base = base
        self.parameterization = parameterization
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        for parameter in self.base.kappa_predictor.parameters():
            parameter.requires_grad_(True)
        self.base.eval()
        self._center_digest = self.center_digest()

    def center_state(self):
        return {name: value for name, value in self.base.state_dict().items()
                if not name.startswith("kappa_predictor.")}

    def center_digest(self):
        return tensor_state_digest(self.center_state())

    def assert_center_frozen(self):
        if self.center_digest() != self._center_digest:
            raise ValueError("A frozen center parameter changed")

    def radius_parameters(self):
        return self.base.kappa_predictor.parameters()

    def set_radius_training(self, enabled):
        # Center dropout remains disabled, so every condition sees exactly the
        # same deterministic center during both head fitting and evaluation.
        self.base.eval()
        self.base.kappa_predictor.train(enabled)

    def center_and_features(self, query_emb, anchor_embs, anchor_mask=None):
        query = query_emb.unsqueeze(1)
        attended, _ = self.base.attn(
            query=query, key=anchor_embs, value=anchor_embs,
            key_padding_mask=anchor_mask)
        hidden = self.base.layer_norm1(query + attended)
        shift = self.base.layer_norm2(attended + self.base.ffn(hidden)).squeeze(1)
        shift = F.normalize(shift, p=2, dim=-1)
        gate = 0.2 * torch.sigmoid(self.base.residual_gate)
        features = query.squeeze(1) + gate * shift
        center = F.normalize(features, p=2, dim=-1)
        return center, features

    def kappa_from_features(self, features):
        raw = self.base.kappa_predictor(features)
        if self.parameterization == "hard_clip":
            return torch.clamp(500.0 * torch.sigmoid(raw) + 1.0,
                               min=10.0, max=500.0)
        return 10.0 + 490.0 * torch.sigmoid(raw)

    def forward(self, query_emb, anchor_embs, anchor_mask=None):
        center, features = self.center_and_features(
            query_emb, anchor_embs, anchor_mask)
        return center, self.kappa_from_features(features)


def validate_config(config, revision):
    if config.get("parameterizations") != list(PARAMETERIZATIONS):
        raise ValueError(f"parameterizations must equal {list(PARAMETERIZATIONS)}")
    kappas = config.get("fixed_kappas", [])
    if kappas != revision.get("fixed_kappas"):
        raise ValueError("fixed_kappas must match the original experiment")
    seeds = config.get("inference_seeds", [])
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("Use at least two unique inference seeds")
    if config.get("selection_inference_seed") not in seeds:
        raise ValueError("selection_inference_seed must be an inference seed")
    for key in ("epochs", "patience"):
        if not isinstance(config.get(key), int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if float(config.get("radius_head_learning_rate", 0)) <= 0:
        raise ValueError("radius_head_learning_rate must be positive")
    if float(config.get("boundary_tolerance", -1)) < 0:
        raise ValueError("boundary_tolerance must be nonnegative")


def verify_parent(args):
    original = verify_manifest(args.original_run)
    if revision_sources() != original["source_sha256"]:
        raise ValueError("Frozen revision source no longer matches the parent run")
    if digest(args.data / "manifest.json") != original["data_manifest_sha256"]:
        raise ValueError("Prepared data differs from the parent run")
    if args.device != original["device"]:
        raise ValueError("Device must match the original center-training device")
    runtime = environment()
    for key in ("determinism", "mps_cpu_fallback"):
        expected = original["environment"].get(key, runtime[key])
        if runtime[key] != expected:
            raise ValueError(f"Runtime {key} differs from the parent run")
    return original


def center_selection(original_run):
    selection = json.loads((original_run / "selection.json").read_text())
    try:
        return selection["methods"]["learned_centroid"]
    except KeyError as error:
        raise ValueError("Parent run lacks selected learned-center checkpoints") from error


def load_model(data, checkpoint, device, parameterization, head_state=None):
    base = VCEModel(embed_dim=data.emb.shape[1]).to(device)
    base.load_state_dict(torch.load(checkpoint, map_location=device,
                                    weights_only=True))
    model = FrozenCenterRadius(base, parameterization).to(device)
    if head_state is not None:
        model.base.kappa_predictor.load_state_dict(head_state)
    model.set_radius_training(False)
    return model


def event_noise(rows, samples, dimension, seed, phase, device):
    noise = []
    for row in rows:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stable_seed(seed, phase, row["event"]))
        noise.append(torch.randn(samples, dimension, generator=generator))
    return torch.stack(noise).to(device)


def expand(center, kappa, epsilon):
    center = center.unsqueeze(1)
    tangent = F.normalize(
        epsilon - (epsilon * center).sum(-1, keepdim=True) * center, dim=-1)
    return F.normalize(
        center + torch.rsqrt(kappa.unsqueeze(1) + 1e-6) * tangent, dim=-1)


def score_queries(data, rows, queries):
    scores = queries[:, 0] @ data.emb.T
    for probe in range(1, queries.shape[1]):
        scores = torch.maximum(scores, queries[:, probe] @ data.emb.T)
    return [
        {"event": row["event"], "group": row["group"],
         **metrics_from_scores(score, row["target"], row["excluded"])}
        for row, score in zip(rows, scores.cpu().numpy())
    ]


@torch.no_grad()
def evaluate_no_expansion(data, split, model, revision):
    model.set_radius_training(False)
    output = []
    rows = data.splits[split]
    for start in range(0, len(rows), revision["batch_size"]):
        batch = rows[start:start + revision["batch_size"]]
        _, anchors, mask, query = data.batch(batch, revision["max_context"])
        center, _ = model.center_and_features(query, anchors, mask)
        output.extend(score_queries(data, batch, center.unsqueeze(1)))
    return output


@torch.no_grad()
def evaluate_expanded_suite(data, split, hard_model, smooth_model, kappas,
                            revision, inference_seed, include_adaptive=True,
                            adaptive_conditions=PARAMETERIZATIONS):
    """Evaluate all expanded conditions with common centers and directions."""
    hard_model.set_radius_training(False)
    smooth_model.set_radius_training(False)
    names = [fixed_name(value) for value in kappas]
    active = tuple(adaptive_conditions) if include_adaptive else ()
    if any(value not in PARAMETERIZATIONS for value in active):
        raise ValueError("Unknown adaptive condition")
    names += [f"adaptive_{value}" for value in active]
    output = {name: [] for name in names}
    rows = data.splits[split]
    for start in range(0, len(rows), revision["batch_size"]):
        batch = rows[start:start + revision["batch_size"]]
        _, anchors, mask, query = data.batch(batch, revision["max_context"])
        center, features = hard_model.center_and_features(query, anchors, mask)
        if "hard_clip" in active:
            hard_kappa = hard_model.kappa_from_features(features)
        if "smooth" in active:
            smooth_kappa = smooth_model.kappa_from_features(features)
        epsilon = event_noise(batch, revision["samples"], center.shape[-1],
                              inference_seed, split, center.device)
        for value in kappas:
            kappa = torch.full((len(batch), 1), float(value),
                               device=center.device)
            output[fixed_name(value)].extend(
                score_queries(data, batch, expand(center, kappa, epsilon)))
        if "hard_clip" in active:
            output["adaptive_hard_clip"].extend(
                score_queries(data, batch, expand(center, hard_kappa, epsilon)))
        if "smooth" in active:
            output["adaptive_smooth"].extend(
                score_queries(data, batch, expand(center, smooth_kappa, epsilon)))
    return output


def fixed_name(value):
    return f"fixed_kappa_{float(value):g}".replace(".", "p")


@torch.no_grad()
def kappa_statistics(data, split, model, revision, tolerance):
    model.set_radius_training(False)
    values = []
    rows = data.splits[split]
    for start in range(0, len(rows), revision["batch_size"]):
        batch = rows[start:start + revision["batch_size"]]
        _, anchors, mask, query = data.batch(batch, revision["max_context"])
        _, kappa = model(query, anchors, mask)
        values.extend(kappa[:, 0].detach().cpu().tolist())
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)), "mean": float(array.mean()),
        "q10": float(np.quantile(array, .1)),
        "median": float(np.quantile(array, .5)),
        "q90": float(np.quantile(array, .9)),
        "fraction_near_minimum": float(np.mean(array <= 10.0 + tolerance)),
        "fraction_near_maximum": float(np.mean(array >= 500.0 - tolerance)),
    }


def head_checkpoint(model, center_checkpoint, parameterization):
    model.assert_center_frozen()
    return {
        "parameterization": parameterization,
        "center_checkpoint_sha256": digest(center_checkpoint),
        "center_state_sha256": model.center_digest(),
        "head_state": copy.deepcopy(model.base.kappa_predictor.state_dict()),
    }


def train_head(data, model, center_checkpoint, parameterization, revision,
               config, training_seed, destination):
    seed_all(training_seed)
    optimizer = torch.optim.AdamW(
        list(model.radius_parameters()),
        lr=float(config["radius_head_learning_rate"]),
        weight_decay=revision["weight_decay"])
    rows = data.splits["train"]
    pool = np.asarray(sorted({item for row in rows
                              for item in row["context"] + [row["target"]]}))
    best_score, best_epoch, best, history = -1.0, 0, None, []
    for epoch in range(1, config["epochs"] + 1):
        model.set_radius_training(True)
        rng = np.random.default_rng(stable_seed(training_seed, "radius-head", epoch))
        order = rng.permutation(len(rows))
        total_loss = 0.0
        for start in range(0, len(rows), revision["batch_size"]):
            batch = [rows[index]
                     for index in order[start:start + revision["batch_size"]]]
            candidates = torch.tensor(
                [[row["target"]] + negatives(
                    pool, row, revision["negative_count"], rng)
                 for row in batch], device=data.emb.device)
            optimizer.zero_grad(set_to_none=True)
            _, anchors, mask, query = data.batch(batch, revision["max_context"])
            center, kappa = model(query, anchors, mask)
            epsilon = event_noise(
                batch, revision["samples"], center.shape[-1], training_seed,
                f"radius-train-{epoch}", center.device)
            queries = expand(center, kappa, epsilon)
            logits = torch.einsum(
                "bsd,bnd->bsn", queries, data.emb[candidates]).max(1).values
            loss = F.cross_entropy(
                logits / revision["temperature"],
                torch.zeros(len(batch), dtype=torch.long,
                            device=data.emb.device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite frozen-radius training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.radius_parameters()), 1.0)
            optimizer.step()
            model.assert_center_frozen()
            total_loss += float(loss.detach()) * len(batch)

        # One declared inference seed chooses the radius-head epoch on validation.
        model.set_radius_training(False)
        rows_by_method = evaluate_expanded_suite(
            data, "validation", model, model, [], revision,
            config["selection_inference_seed"],
            adaptive_conditions=(parameterization,))
        key = ("adaptive_hard_clip" if parameterization == "hard_clip"
               else "adaptive_smooth")
        validation = aggregate(rows_by_method[key])
        score = validation["NDCG@10"]
        history.append({"epoch": epoch,
                        "training_loss": total_loss / len(rows),
                        "validation": validation})
        if score > best_score:
            best_score, best_epoch = score, epoch
            best = head_checkpoint(model, center_checkpoint, parameterization)
        print(f"frozen_radius {parameterization} seed={training_seed} "
              f"epoch={epoch} val_ndcg10={score:.6f}", flush=True)
        if epoch - best_epoch >= config["patience"]:
            break
    if best is None:
        raise ValueError("Radius-head training produced no checkpoint")
    torch.save(best, destination)
    dump(destination.with_suffix(".history.json"),
         {"best_epoch": best_epoch, "epochs": history})
    model.base.kappa_predictor.load_state_dict(best["head_state"])
    model.assert_center_frozen()
    return best_score


def fit(args):
    original = verify_parent(args)
    config = json.loads(args.config.read_text())
    revision = original["config"]
    validate_config(config, revision)
    selected = center_selection(args.original_run)
    data = Data(args.data, args.device, ["train", "validation"])
    args.out.mkdir(parents=True, exist_ok=False)

    center_records = {}
    fixed_totals = {float(value): [] for value in config["fixed_kappas"]}
    adaptive = {name: {"checkpoints": {}, "validation": {},
                       "kappa_statistics": {}}
                for name in PARAMETERIZATIONS}
    for training_seed in revision["seeds"]:
        checkpoint = args.original_run / selected["checkpoints"][str(training_seed)]
        center_records[str(training_seed)] = {
            "checkpoint": checkpoint.name, "sha256": digest(checkpoint)}

        # Select one global fixed radius by averaging the exact same frozen
        # centers over every training and inference seed on validation.
        initial_hard = load_model(data, checkpoint, args.device, "hard_clip")
        initial_smooth = load_model(data, checkpoint, args.device, "smooth")
        for inference_seed in config["inference_seeds"]:
            suite = evaluate_expanded_suite(
                data, "validation", initial_hard, initial_smooth,
                config["fixed_kappas"], revision, inference_seed,
                include_adaptive=False)
            for value in config["fixed_kappas"]:
                fixed_totals[float(value)].append(
                    aggregate(suite[fixed_name(value)])["NDCG@10"])
        del initial_hard, initial_smooth

        for parameterization in PARAMETERIZATIONS:
            model = load_model(data, checkpoint, args.device, parameterization)
            destination = args.out / (
                f"adaptive_{parameterization}.seed-{training_seed}.pt")
            score = train_head(
                data, model, checkpoint, parameterization, revision, config,
                training_seed, destination)
            saved = torch.load(destination, map_location=args.device,
                               weights_only=True)
            model.base.kappa_predictor.load_state_dict(saved["head_state"])
            adaptive[parameterization]["checkpoints"][str(training_seed)] = (
                destination.name)
            adaptive[parameterization]["validation"][str(training_seed)] = score
            adaptive[parameterization]["kappa_statistics"][str(training_seed)] = (
                kappa_statistics(data, "validation", model, revision,
                                 float(config["boundary_tolerance"])))
            del model

    candidates = [
        {"kappa": value,
         "mean_validation_ndcg10": float(np.mean(fixed_totals[value]))}
        for value in map(float, config["fixed_kappas"])
    ]
    chosen = max(candidates, key=lambda row: (
        row["mean_validation_ndcg10"], -config["fixed_kappas"].index(row["kappa"])))
    selection = {
        "center_method": "learned_centroid",
        "center_params": selected["params"],
        "centers": center_records,
        "fixed_radius": {"selected_kappa": chosen["kappa"],
                         "candidates": candidates,
                         "selection_unit": "mean over training and inference seeds"},
        "adaptive_radius": adaptive,
    }
    dump(args.out / "config.json", config)
    dump(args.out / "selection.json", selection)
    files = sorted(path for path in args.out.iterdir() if path.is_file())
    dump(args.out / "manifest.json", {
        "schema": 1, "experiment": "ecir-frozen-center-radius-fit",
        "dataset": original["dataset"], "config": config,
        "revision_config": revision,
        "data_manifest_sha256": digest(args.data / "manifest.json"),
        "original_run_manifest_sha256": digest(args.original_run / "manifest.json"),
        "source_sha256": source_hashes(), "environment": environment(),
        "device": args.device,
        "selection_metric": "validation NDCG@10; test never loaded",
        "files": {path.name: digest(path) for path in files},
        "command": sys.argv,
    })
    print(f"Frozen-center radius selection saved: {args.out / 'selection.json'}")
    print("Test data were not loaded.")


def load_fitted_model(data, args, selection, training_seed, parameterization):
    center = args.original_run / selection["centers"][str(training_seed)]["checkpoint"]
    checkpoint = args.run / selection["adaptive_radius"][parameterization][
        "checkpoints"][str(training_seed)]
    saved = torch.load(checkpoint, map_location=args.device, weights_only=True)
    if saved["parameterization"] != parameterization:
        raise ValueError("Radius-head checkpoint parameterization mismatch")
    if saved["center_checkpoint_sha256"] != digest(center):
        raise ValueError("Radius head refers to a different center checkpoint")
    model = load_model(data, center, args.device, parameterization,
                       saved["head_state"])
    if model.center_digest() != saved["center_state_sha256"]:
        raise ValueError("Frozen center state differs from head-training state")
    return model


def test(args):
    original = verify_parent(args)
    manifest = verify_manifest(args.run)
    if manifest.get("experiment") != "ecir-frozen-center-radius-fit":
        raise ValueError("--run is not a frozen-center radius fit")
    if source_hashes() != manifest["source_sha256"]:
        raise ValueError("Frozen-radius source changed since fitting")
    if digest(args.original_run / "manifest.json") != manifest[
            "original_run_manifest_sha256"]:
        raise ValueError("Parent run differs from the frozen-radius fit")
    selection = json.loads((args.run / "selection.json").read_text())
    config, revision = manifest["config"], manifest["revision_config"]
    validate_config(config, revision)
    data = Data(args.data, args.device, ["test"])
    args.out.mkdir(parents=True, exist_ok=False)
    summary, files = [], []
    total = len(revision["seeds"]) * (1 + len(config["inference_seeds"]))
    completed = 0
    for training_seed in revision["seeds"]:
        hard = load_fitted_model(
            data, args, selection, training_seed, "hard_clip")
        smooth = load_fitted_model(
            data, args, selection, training_seed, "smooth")
        rows = evaluate_no_expansion(data, "test", hard, revision)
        name = f"no_expansion.train-{training_seed}.jsonl"
        write_jsonl(args.out / name, rows)
        files.append(name)
        summary.append({
            "dataset": original["dataset"], "method": "no_expansion",
            "train_seed": training_seed, "inference_seed": None,
            "n": len(rows), "file": name, **aggregate(rows)})
        completed += 1
        print(f"frozen-radius test {completed}/{total}: "
              f"center seed={training_seed} no expansion", flush=True)
        for inference_seed in config["inference_seeds"]:
            suite = evaluate_expanded_suite(
                data, "test", hard, smooth, config["fixed_kappas"], revision,
                inference_seed)
            for method, method_rows in suite.items():
                name = (f"{method}.train-{training_seed}.inference-"
                        f"{inference_seed}.jsonl")
                write_jsonl(args.out / name, method_rows)
                files.append(name)
                summary.append({
                    "dataset": original["dataset"], "method": method,
                    "train_seed": training_seed,
                    "inference_seed": inference_seed,
                    "n": len(method_rows), "file": name,
                    **aggregate(method_rows)})
            completed += 1
            print(f"frozen-radius test {completed}/{total}: center "
                  f"seed={training_seed} inference={inference_seed}", flush=True)
        del hard, smooth
    dump(args.out / "summary.json", summary)
    files.append("summary.json")
    dump(args.out / "manifest.json", {
        "schema": 1, "experiment": "ecir-frozen-center-radius-test",
        "dataset": original["dataset"], "config": config,
        "revision_config": revision,
        "data_manifest_sha256": digest(args.data / "manifest.json"),
        "original_run_manifest_sha256": digest(args.original_run / "manifest.json"),
        "fit_manifest_sha256": digest(args.run / "manifest.json"),
        "source_sha256": source_hashes(), "environment": environment(),
        "device": args.device,
        "files": {name: digest(args.out / name) for name in files},
        "command": sys.argv,
    })
    print(f"Frozen-center test results saved: {args.out}")
    print("No test result was used for selection or training.")


def csv_file(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verified_values(folder, manifest, entry, reference=None):
    if entry["file"] not in manifest["files"]:
        raise ValueError(f"Unverified result file: {entry['file']}")
    rows = sorted(read_jsonl(folder / entry["file"]), key=lambda row: row["event"])
    support = [(row["event"], row["group"]) for row in rows]
    if len(support) != len(set(support)):
        raise ValueError(f"Duplicate support in {entry['file']}")
    if reference is not None and support != reference:
        raise ValueError("Frozen-radius result cells do not share ordered support")
    recomputed = aggregate(rows)
    if entry["n"] != len(rows) or any(
            abs(recomputed[metric] - entry[metric]) > 1e-10
            for metric in METRICS):
        raise ValueError(f"Summary mismatch: {entry['file']}")
    return support, np.asarray([row["NDCG@10"] for row in rows], dtype=np.float64)


def report(args):
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    result_manifest = verify_manifest(args.results)
    fit_manifest = verify_manifest(args.run)
    if result_manifest.get("experiment") != "ecir-frozen-center-radius-test":
        raise ValueError("--results is not a frozen-center radius test")
    if fit_manifest.get("experiment") != "ecir-frozen-center-radius-fit":
        raise ValueError("--run is not a frozen-center radius fit")
    if digest(args.run / "manifest.json") != result_manifest["fit_manifest_sha256"]:
        raise ValueError("Result and fit manifests are not linked")
    if source_hashes() != result_manifest["source_sha256"]:
        raise ValueError("Frozen-radius source changed since result generation")
    config, revision = result_manifest["config"], result_manifest["revision_config"]
    selection = json.loads((args.run / "selection.json").read_text())
    summary = json.loads((args.results / "summary.json").read_text())
    entries = {(row["method"], row["train_seed"], row["inference_seed"]): row
               for row in summary}
    methods = ["no_expansion"] + [fixed_name(value)
                                   for value in config["fixed_kappas"]]
    methods += ["adaptive_hard_clip", "adaptive_smooth"]
    values, reference = {}, None
    for method in methods:
        by_training = []
        for training_seed in revision["seeds"]:
            if method == "no_expansion":
                keys = [(method, training_seed, None)]
            else:
                keys = [(method, training_seed, inference_seed)
                        for inference_seed in config["inference_seeds"]]
            by_inference = []
            for key in keys:
                if key not in entries:
                    raise ValueError(f"Missing frozen-radius result cell: {key}")
                support, array = verified_values(
                    args.results, result_manifest, entries[key], reference)
                if reference is None:
                    reference = support
                by_inference.append(array)
            stacked = np.stack(by_inference)
            if method == "no_expansion":
                stacked = np.repeat(stacked, len(config["inference_seeds"]), axis=0)
            by_training.append(stacked)
        values[method] = np.stack(by_training)

    records = []
    for method in methods:
        array = values[method]
        train_means = array.mean(axis=(1, 2))
        inference_means = array.mean(axis=2)
        records.append({
            "method": method, "queries": array.shape[2],
            "training_seeds": array.shape[0],
            "inference_seeds": (0 if method == "no_expansion" else array.shape[1]),
            "NDCG@10 mean": float(array.mean()),
            "NDCG@10 train_sd": float(train_means.std(ddof=1)),
            "NDCG@10 inference_sd": float(np.sqrt(
                inference_means.var(axis=1, ddof=1).mean())),
        })

    chosen_fixed = fixed_name(selection["fixed_radius"]["selected_kappa"])
    comparisons = []
    for left, right in [
            (chosen_fixed, "no_expansion"),
            ("adaptive_hard_clip", chosen_fixed),
            ("adaptive_smooth", chosen_fixed),
            ("adaptive_smooth", "adaptive_hard_clip")]:
        difference = values[left] - values[right]
        mean, low, high = hierarchical_interval(
            difference, [group for _, group in reference], args.bootstrap,
            stable_seed("frozen-center-radius", left, right))
        comparisons.append({
            "comparison": f"{left} - {right}", "mean_difference": mean,
            "hierarchical_95_low": low, "hierarchical_95_high": high,
            "training_seeds": difference.shape[0],
            "inference_seeds": difference.shape[1],
            "clusters": len(set(group for _, group in reference)),
            "replicates": args.bootstrap,
        })

    kappa_rows = []
    for parameterization in PARAMETERIZATIONS:
        statistics = selection["adaptive_radius"][parameterization][
            "kappa_statistics"]
        for training_seed in revision["seeds"]:
            kappa_rows.append({
                "parameterization": parameterization,
                "training_seed": training_seed,
                **statistics[str(training_seed)],
            })

    args.out.mkdir(parents=True, exist_ok=False)
    csv_file(args.out / "metrics.csv", records)
    csv_file(args.out / "paired_comparisons.csv", comparisons)
    csv_file(args.out / "validation_kappa.csv", kappa_rows)
    tex = [
        r"\begin{table}[t]", r"\centering\small",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"Frozen-center condition & NDCG@10 & Train SD" + LATEX_ROW_END,
        r"\midrule",
    ]
    labels = {
        "no_expansion": "No expansion",
        "adaptive_hard_clip": "Adaptive, hard clip",
        "adaptive_smooth": "Adaptive, smooth bound",
    }
    for row in records:
        label = labels.get(row["method"], row["method"].replace("_", r"\_"))
        tex.append(f"{label} & {row['NDCG@10 mean']:.5f} & "
                   f"{row['NDCG@10 train_sd']:.5f}" + LATEX_ROW_END)
    tex.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Frozen-center comparison. Every row uses the identical "
        r"learned center within training seed; expanded rows also share paired "
        r"directions. Fixed radius is selected on validation only.}",
        r"\label{tab:frozen-center-radius}", r"\end{table}",
    ])
    (args.out / "frozen_center_radius.tex").write_text(
        "\n".join(tex) + "\n", encoding="utf-8")
    report_lines = [
        "# Frozen-center radius experiment", "",
        f"Dataset: `{result_manifest['dataset']}`", "",
        f"Validation-selected fixed kappa: "
        f"`{selection['fixed_radius']['selected_kappa']:g}`", "",
        "All conditions use the identical learned center within each training",
        "seed. Only the adaptive radius head is trained; fixed and adaptive",
        "conditions share event-, split-, and inference-seed-specific directions.",
        "", "## Paired hierarchical comparisons", "",
    ]
    for row in comparisons:
        report_lines.append(
            f"- {row['comparison']}: {row['mean_difference']:+.6f}, "
            f"95% [{row['hierarchical_95_low']:+.6f}, "
            f"{row['hierarchical_95_high']:+.6f}]")
    report_lines += ["", "Intervals are descriptive with five center-training",
                     "seeds; they are not multiplicity corrected."]
    (args.out / "REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8")
    files = sorted(path for path in args.out.iterdir() if path.is_file())
    dump(args.out / "manifest.json", {
        "schema": 1, "experiment": "ecir-frozen-center-radius-report",
        "fit_manifest_sha256": digest(args.run / "manifest.json"),
        "result_manifest_sha256": digest(args.results / "manifest.json"),
        "source_sha256": digest(Path(__file__)),
        "bootstrap_replicates": args.bootstrap,
        "files": {path.name: digest(path) for path in files},
    })
    print(f"Frozen-center radius report saved: {args.out}")


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
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--run", type=Path, required=True)
    report_parser.add_argument("--results", type=Path, required=True)
    report_parser.add_argument("--out", type=Path, required=True)
    report_parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; choose a new versioned path")
    try:
        if args.command in {"fit", "test"}:
            configure_runtime(args)
            (fit if args.command == "fit" else test)(args)
        else:
            report(args)
    except (KeyError, ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
