"""Representative non-centroid and sequential baselines on canonical splits."""
import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.revision.protocol import (aggregate, digest, dump,
                                           metrics_from_scores, stable_seed,
                                           verify_manifest, write_jsonl)
from experiments.revision.run import (Data, configure_runtime, environment,
                                      negatives, seed_all)
from src.models.sasrec import SASRecDualEncoder

METHODS = ("popularity", "last_item", "transition_knn", "content_sasrec")
TRAINED = "content_sasrec"


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root / "experiments/revision/protocol.py",
             root / "experiments/revision/run.py", root / "src/models/sasrec.py"]
    return {str(path.relative_to(root)): digest(path) for path in paths}


def validate_config(config):
    if (not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]) or
            config["selection_seed"] not in config["seeds"]):
        raise ValueError("Selection seed must occur in a nonempty seed list")
    section = config["sasrec"]
    for key in ("hidden_dim", "num_layers", "num_heads", "epochs", "patience",
                "batch_size", "negative_count", "max_context"):
        if not isinstance(section[key], int) or section[key] <= 0:
            raise ValueError(f"sasrec.{key} must be a positive integer")
    if section["hidden_dim"] % section["num_heads"]:
        raise ValueError("SASRec hidden dimension must be divisible by its heads")
    if not section["learning_rates"] or any(x <= 0 for x in section["learning_rates"]):
        raise ValueError("SASRec learning rates must be positive")
    if not section["dropouts"] or any(not 0 <= x < 1 for x in section["dropouts"]):
        raise ValueError("SASRec dropout values must be in [0,1)")
    if section["temperature"] <= 0 or section["weight_decay"] < 0:
        raise ValueError("Invalid SASRec loss/optimizer configuration")


def training_statistics(data):
    """Reconstruct each group's longest observed train prefix without held-out rows."""
    popularity = np.zeros(len(data.ids), dtype=np.float32)
    transitions = defaultdict(Counter)
    sequences = {}
    for row in data.splits["train"]:
        sequence = row["context"] + [row["target"]]
        if len(sequence) > len(sequences.get(row["group"], ())):
            sequences[row["group"]] = sequence
    for sequence in sequences.values():
        for item in sequence:
            popularity[item] += 1
        for source, target in zip(sequence, sequence[1:]):
            transitions[source][target] += 1
    if not popularity.any():
        raise ValueError("No training interactions for baseline statistics")
    return popularity, transitions


@torch.no_grad()
def evaluate_rule(data, split, method, popularity, transitions, batch_size):
    rows, results = data.splits[split], []
    pop = popularity / max(float(popularity.max()), 1.)
    if method == "last_item":
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            last = torch.tensor([row["context"][-1] for row in batch],
                                dtype=torch.long, device=data.emb.device)
            scores = (data.emb[last] @ data.emb.T).cpu().numpy()
            for row, score in zip(batch, scores):
                results.append({"event": row["event"], "group": row["group"],
                                **metrics_from_scores(score, row["target"], row["excluded"])})
        return results
    for row in rows:
        if method == "popularity":
            score = pop.copy()
        elif method == "transition_knn":
            # Positive transition evidence dominates; popularity is a deterministic backoff.
            score = pop * 1e-6
            for target, count in transitions.get(row["context"][-1], {}).items():
                score[target] += math.log1p(count)
        else:
            raise ValueError(f"Unknown rule baseline: {method}")
        results.append({"event": row["event"], "group": row["group"],
                        **metrics_from_scores(score, row["target"], row["excluded"])})
    return results


def model_for(data, params, config):
    section = config["sasrec"]
    return SASRecDualEncoder(
        input_dim=data.emb.shape[1], hidden_dim=section["hidden_dim"],
        num_layers=section["num_layers"], num_heads=section["num_heads"],
        max_seq_len=section["max_context"], dropout=params["dropout"]
    ).to(data.emb.device)


def sas_query(data, rows, model, config):
    _, anchors, mask, _ = data.batch(rows, config["sasrec"]["max_context"])
    lengths = (~mask).sum(1)
    return model(anchors, lengths)


@torch.no_grad()
def evaluate_sasrec(data, split, model, config):
    model.eval()
    section, rows, results = config["sasrec"], data.splits[split], []
    for start in range(0, len(rows), section["batch_size"]):
        batch = rows[start:start + section["batch_size"]]
        scores = (sas_query(data, batch, model, config) @ data.emb.T).cpu().numpy()
        for row, score in zip(batch, scores):
            results.append({"event": row["event"], "group": row["group"],
                            **metrics_from_scores(score, row["target"], row["excluded"])})
    return results


def train_sasrec(data, params, config, seed, destination):
    seed_all(seed)
    section = config["sasrec"]
    model = model_for(data, params, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                                  weight_decay=section["weight_decay"])
    rows = data.splits["train"]
    pool = np.asarray(sorted({i for row in rows for i in row["context"] + [row["target"]]}))
    best, best_epoch, best_state, history = -1., 0, None, []
    for epoch in range(1, section["epochs"] + 1):
        model.train()
        rng = np.random.default_rng(stable_seed("sasrec", seed, epoch))
        order, total = rng.permutation(len(rows)), 0.
        for start in range(0, len(rows), section["batch_size"]):
            batch = [rows[i] for i in order[start:start + section["batch_size"]]]
            candidates = torch.tensor(
                [[row["target"]] + negatives(pool, row, section["negative_count"], rng)
                 for row in batch], device=data.emb.device)
            optimizer.zero_grad(set_to_none=True)
            query = sas_query(data, batch, model, config)
            logits = torch.einsum("bd,bnd->bn", query, data.emb[candidates])
            loss = F.cross_entropy(
                logits / section["temperature"],
                torch.zeros(len(batch), dtype=torch.long, device=data.emb.device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite ContentSASRec training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        validation = aggregate(evaluate_sasrec(data, "validation", model, config))
        score = validation["NDCG@10"]
        history.append({"epoch": epoch, "training_loss": total / len(rows),
                        "validation": validation})
        if score > best:
            best, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        print(f"content_sasrec seed={seed} epoch={epoch} val_ndcg10={score:.6f}",
              flush=True)
        if epoch - best_epoch >= section["patience"]:
            break
    torch.save(best_state, destination)
    dump(destination.with_suffix(".history.json"),
         {"best_epoch": best_epoch, "params": params, "epochs": history})
    return best, sum(parameter.numel() for parameter in model.parameters())


def fit(args):
    config = json.loads(args.config.read_text())
    validate_config(config)
    data = Data(args.data, args.device, ["train", "validation"])
    args.out.mkdir(parents=True, exist_ok=False)
    popularity, transitions = training_statistics(data)
    selection = {"methods": {}, "candidates": []}
    for method in METHODS[:-1]:
        metrics = aggregate(evaluate_rule(data, "validation", method, popularity,
                                          transitions, config["sasrec"]["batch_size"]))
        selection["methods"][method] = {"validation": metrics}
    seed, best, chosen = config["selection_seed"], -1., None
    candidate_index = 0
    for lr in config["sasrec"]["learning_rates"]:
        for dropout in config["sasrec"]["dropouts"]:
            params = {"lr": lr, "dropout": dropout}
            checkpoint = args.out / f"content_sasrec.candidate-{candidate_index}.seed-{seed}.pt"
            score, count = train_sasrec(data, params, config, seed, checkpoint)
            record = {"params": params, "seed": seed, "validation_ndcg10": score,
                      "checkpoint": checkpoint.name, "parameter_count": count}
            selection["candidates"].append(record)
            if score > best:
                best, chosen = score, record
            candidate_index += 1
    checkpoints = {str(seed): chosen["checkpoint"]}
    for other_seed in config["seeds"]:
        if other_seed == seed:
            continue
        checkpoint = args.out / f"content_sasrec.selected.seed-{other_seed}.pt"
        train_sasrec(data, chosen["params"], config, other_seed, checkpoint)
        checkpoints[str(other_seed)] = checkpoint.name
    selection["methods"][TRAINED] = {
        "params": chosen["params"], "validation_ndcg10": best,
        "parameter_count": chosen["parameter_count"], "checkpoints": checkpoints}
    dump(args.out / "config.json", config)
    dump(args.out / "selection.json", selection)
    files = sorted(path for path in args.out.iterdir() if path.is_file())
    dump(args.out / "manifest.json", {
        "schema": 1, "experiment": "ecir-representative-baselines",
        "dataset": data.manifest["dataset"], "config": config,
        "data_manifest_sha256": digest(args.data / "manifest.json"),
        "source_sha256": source_hashes(), "environment": environment(),
        "device": args.device, "files": {path.name: digest(path) for path in files},
        "command": sys.argv})
    print(f"Frozen baseline selection: {args.out / 'selection.json'}")


def test(args):
    manifest = verify_manifest(args.run)
    if args.device != manifest["device"]:
        raise ValueError("Test device differs from baseline-fit device")
    runtime = environment()
    for key in ("determinism", "mps_cpu_fallback"):
        if runtime[key] != manifest["environment"].get(key, runtime[key]):
            raise ValueError(f"Test runtime {key} differs from baseline-fit runtime")
    if source_hashes() != manifest["source_sha256"]:
        raise ValueError("Baseline source changed since fit; start a new versioned run")
    if digest(args.data / "manifest.json") != manifest["data_manifest_sha256"]:
        raise ValueError("Prepared data differs from baseline fit")
    data = Data(args.data, args.device, ["train", "test"])
    config, selection = manifest["config"], json.loads(
        (args.run / "selection.json").read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    popularity, transitions = training_statistics(data)
    summary, files = [], []
    for method in METHODS[:-1]:
        rows = evaluate_rule(data, "test", method, popularity, transitions,
                             config["sasrec"]["batch_size"])
        name = f"{method}.jsonl"
        write_jsonl(args.out / name, rows)
        files.append(name)
        summary.append({"dataset": data.manifest["dataset"], "method": method,
                        "seed": None, "n": len(rows), "file": name, **aggregate(rows)})
    selected = selection["methods"][TRAINED]
    for seed in config["seeds"]:
        seed_all(seed)
        model = model_for(data, selected["params"], config)
        model.load_state_dict(torch.load(
            args.run / selected["checkpoints"][str(seed)], map_location=args.device,
            weights_only=True))
        rows = evaluate_sasrec(data, "test", model, config)
        name = f"content_sasrec.seed-{seed}.jsonl"
        write_jsonl(args.out / name, rows)
        files.append(name)
        summary.append({"dataset": data.manifest["dataset"], "method": TRAINED,
                        "seed": seed, "n": len(rows), "file": name, **aggregate(rows)})
    dump(args.out / "summary.json", summary)
    files.append("summary.json")
    dump(args.out / "manifest.json", {
        "schema": 1, "experiment": "ecir-representative-baselines",
        "dataset": data.manifest["dataset"], "config": config,
        "data_manifest_sha256": digest(args.data / "manifest.json"),
        "run_manifest_sha256": digest(args.run / "manifest.json"),
        "source_sha256": source_hashes(), "environment": environment(),
        "device": args.device, "files": {name: digest(args.out / name) for name in files},
        "command": sys.argv})
    print(f"Baseline test results saved: {args.out}")


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("fit", "test"):
        part = sub.add_parser(command)
        part.add_argument("--data", type=Path, required=True)
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
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
