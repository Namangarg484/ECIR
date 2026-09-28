"""Fit on train/validation, then evaluate frozen selections in a separate command."""
import argparse
import importlib.metadata
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models.vce_model import VCEModel
from .protocol import (METHODS, ItemBM25, aggregate, digest, dump,
                       metrics_from_scores, read_jsonl, stable_seed,
                       verify_manifest, write_jsonl)

LEARNED = {"learned_centroid", "learned_fixed", "full"}


def sources():
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), Path(__file__).with_name("protocol.py"),
             Path(__file__).with_name("prepare.py"), root / "src/models/vce_model.py"]
    return {str(p.relative_to(root)): digest(p) for p in paths}


def environment():
    versions = {}
    for package in ("torch", "numpy", "sentence-transformers", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": sys.version, "platform": platform.platform(), "versions": versions,
            "cuda": torch.version.cuda,
            "mps_available": torch.backends.mps.is_available(),
            "cpu_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "determinism": "warn" if torch.is_deterministic_algorithms_warn_only_enabled() else "strict",
            "mps_cpu_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def configure_runtime(a):
    if a.cpu_threads < 1 or a.interop_threads < 1:
        raise ValueError("CPU and inter-op thread counts must be positive")
    if a.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable in this PyTorch/macOS installation")
    if a.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; Apple GPUs use --device mps")
    torch.set_num_threads(a.cpu_threads)
    torch.set_num_interop_threads(a.interop_threads)
    torch.use_deterministic_algorithms(True, warn_only=a.determinism == "warn")
    print(f"Device={a.device}; CPU tensor threads={torch.get_num_threads()}; "
          f"inter-op threads={torch.get_num_interop_threads()}; determinism={a.determinism}", flush=True)
    if a.determinism == "warn":
        print("WARNING: nondeterministic operations may run with warnings; exact repeatability is not guaranteed.", flush=True)
    if a.device == "mps":
        print("Apple GPU: model/catalog tensor computation. CPU: input processing, random directions and metrics. "
              "Python loops are not parallelized by the tensor thread setting.", flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(
        True, warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False


class Data:
    def __init__(self, folder, device, split_names):
        self.folder = folder
        self.manifest = verify_manifest(folder)
        catalog = json.loads((folder / "catalog.json").read_text())
        self.ids, self.texts = catalog["ids"], catalog["texts"]
        self.index = {iid: i for i, iid in enumerate(self.ids)}
        self.emb = torch.tensor(np.load(folder / "embeddings.npy", allow_pickle=False), device=device)
        self.bm25 = ItemBM25(self.texts)
        self.splits = {}
        for split in split_names:
            self.splits[split] = [dict(r, context=[self.index[i] for i in r["context"]],
                                       target=self.index[r["target"]],
                                       excluded=[self.index[i] for i in r["excluded"]])
                                  for r in read_jsonl(folder / f"{split}.jsonl")]

    def batch(self, rows, max_context):
        contexts = [r["context"][-max_context:] for r in rows]
        size = max(map(len, contexts))
        indices = torch.zeros((len(rows), size), dtype=torch.long, device=self.emb.device)
        mask = torch.ones_like(indices, dtype=torch.bool)
        for n, ctx in enumerate(contexts):
            indices[n, :len(ctx)] = torch.tensor(ctx, device=self.emb.device)
            mask[n, :len(ctx)] = False
        anchors = self.emb[indices]
        query = F.normalize((anchors * (~mask).unsqueeze(-1)).sum(1) /
                            (~mask).sum(1, keepdim=True), dim=-1)
        return contexts, anchors, mask, query


def directions(data, rows, method, model, params, config, seed, phase):
    contexts, anchors, mask, query = data.batch(rows, config["max_context"])
    if method in LEARNED:
        mu, kappa, _ = model(query, anchors, mask)
    elif method.startswith("weighted"):
        mu = torch.stack([F.normalize((data.emb[ctx] * torch.tensor(
            data.bm25.weights(ctx), device=data.emb.device).unsqueeze(-1)).sum(0), dim=0)
            for ctx in contexts])
        kappa = None
    else:
        mu, kappa = query, None
    if method in {"uniform", "weighted_prf", "learned_centroid"}:
        return mu.unsqueeze(1)
    if method != "full":
        kappa = torch.full((len(rows), 1), params["kappa"], device=data.emb.device)
    # Common per-event random directions across methods; invariant to batch size/order.
    noise = []
    for row in rows:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stable_seed(seed, phase, row["event"]))
        noise.append(torch.randn(config["samples"], mu.shape[-1], generator=generator))
    epsilon = torch.stack(noise).to(mu.device)
    center = mu.unsqueeze(1)
    tangent = F.normalize(epsilon - (epsilon * center).sum(-1, keepdim=True) * center, dim=-1)
    # Fixed-radius tangent perturbation; NOT exact von Mises-Fisher sampling.
    return F.normalize(center + torch.rsqrt(kappa.unsqueeze(1) + 1e-6) * tangent, dim=-1)


@torch.no_grad()
def evaluate(data, split, method, model, params, config, seed):
    if model is not None:
        model.eval()
    result = []
    rows = data.splits[split]
    for start in range(0, len(rows), config["batch_size"]):
        batch = rows[start:start + config["batch_size"]]
        queries = directions(data, batch, method, model, params, config, seed, split)
        # Avoid a B x S x catalog allocation; exact exhaustive candidate scores.
        scores = queries[:, 0] @ data.emb.T
        for s in range(1, queries.shape[1]):
            scores = torch.maximum(scores, queries[:, s] @ data.emb.T)
        for row, score in zip(batch, scores.cpu().numpy()):
            result.append({"event": row["event"], "group": row["group"],
                           **metrics_from_scores(score, row["target"], row["excluded"])})
    return result


def negatives(pool, row, count, rng):
    forbidden = set(row["excluded"]) | {row["target"]}
    found = []
    for _ in range(20):
        candidates = rng.choice(pool, size=max(2 * count, 128), replace=True)
        found.extend(int(i) for i in candidates if i not in forbidden)
        if len(found) >= count:
            return found[:count]
    allowed = np.asarray([i for i in pool if i not in forbidden], dtype=np.int64)
    if not len(allowed):
        raise ValueError(f"No legal training negatives for {row['event']}")
    return rng.choice(allowed, size=count, replace=True).tolist()


def train(data, method, params, config, seed, destination):
    seed_all(seed)
    model = VCEModel(embed_dim=data.emb.shape[1]).to(data.emb.device)
    if method != "full":
        for parameter in model.kappa_predictor.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=params["lr"], weight_decay=config["weight_decay"])
    rows = data.splits["train"]
    # No held-out labels used even for negative sampling.
    pool = np.asarray(sorted({i for row in rows for i in row["context"] + [row["target"]]}))
    best, best_epoch, best_state, history = -1., 0, None, []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        rng = np.random.default_rng(stable_seed(seed, "training", epoch))
        order = rng.permutation(len(rows))
        total_loss = 0.
        for start in range(0, len(rows), config["batch_size"]):
            batch = [rows[i] for i in order[start:start + config["batch_size"]]]
            candidates = torch.tensor([[r["target"]] + negatives(pool, r, config["negative_count"], rng)
                                       for r in batch], device=data.emb.device)
            optimizer.zero_grad(set_to_none=True)
            queries = directions(data, batch, method, model, params, config, seed, f"train-{epoch}")
            logits = torch.einsum("bsd,bnd->bsn", queries, data.emb[candidates]).max(1).values
            loss = F.cross_entropy(logits / config["temperature"],
                                   torch.zeros(len(batch), dtype=torch.long, device=data.emb.device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        validation = aggregate(evaluate(data, "validation", method, model, params, config, seed))
        score = validation["NDCG@10"]
        history.append({"epoch": epoch, "training_loss": total_loss / len(rows), "validation": validation})
        if score > best:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"{method} seed={seed} epoch={epoch} val_ndcg10={score:.6f}", flush=True)
        if epoch - best_epoch >= config["patience"]:
            break
    torch.save(best_state, destination)
    dump(destination.with_suffix(".history.json"), {"best_epoch": best_epoch, "epochs": history})
    return best


def validate_config(config):
    for key in ("epochs", "batch_size", "patience", "max_context", "negative_count", "samples"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not config["learning_rates"] or any(x <= 0 for x in config["learning_rates"]):
        raise ValueError("Learning rates must be positive")
    if not config["fixed_kappas"] or any(x <= 0 for x in config["fixed_kappas"]):
        raise ValueError("Fixed kappas must be positive")
    if config["temperature"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Invalid optimizer/loss configuration")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Seeds must be nonempty and unique")
    if config["selection_seed"] not in config["seeds"]:
        raise ValueError("Selection seed must be included in reported seeds")


def fit(a):
    config = json.loads(a.config.read_text())
    validate_config(config)
    data = Data(a.data, a.device, ["train", "validation"])
    a.out.mkdir(parents=True, exist_ok=False)
    dump(a.out / "config.json", config)
    selected, candidates_log = {}, []
    seed = config["selection_seed"]
    for method in METHODS:
        params_grid = [{}]
        if method in LEARNED:
            params_grid = [{"lr": lr} for lr in config["learning_rates"]]
        if method in {"weighted_fixed", "learned_fixed"}:
            params_grid = [dict(params, kappa=k) for params in params_grid for k in config["fixed_kappas"]]
        best_score, chosen = -1., None
        for index, params in enumerate(params_grid):
            checkpoint = a.out / f"{method}.candidate-{index}.seed-{seed}.pt"
            if method in LEARNED:
                score = train(data, method, params, config, seed, checkpoint)
            else:
                score = aggregate(evaluate(data, "validation", method, None, params, config, seed))["NDCG@10"]
            candidates_log.append({"method": method, "params": params, "seed": seed, "validation_ndcg10": score})
            # Grid order is the predeclared deterministic tie-breaker.
            if score > best_score:
                best_score = score
                chosen = {"params": params, "validation_ndcg10": score, "checkpoints": {}}
                if method in LEARNED:
                    chosen["checkpoints"][str(seed)] = checkpoint.name
        selected[method] = chosen
        if method in LEARNED:
            for other_seed in config["seeds"]:
                if other_seed == seed:
                    continue
                checkpoint = a.out / f"{method}.selected.seed-{other_seed}.pt"
                train(data, method, chosen["params"], config, other_seed, checkpoint)
                chosen["checkpoints"][str(other_seed)] = checkpoint.name
    dump(a.out / "selection.json", {"methods": selected, "candidates": candidates_log})
    files = sorted(f for f in a.out.iterdir() if f.is_file())
    dump(a.out / "manifest.json", {
        "schema": 1, "dataset": data.manifest["dataset"], "config": config,
        "data_manifest_sha256": digest(a.data / "manifest.json"),
        "source_sha256": sources(), "environment": environment(), "device": a.device,
        "selection_metric": "validation NDCG@10; never test",
        "files": {f.name: digest(f) for f in files}, "command": sys.argv,
    })
    print(f"Frozen selection: {a.out / 'selection.json'}. Test has not been evaluated.")


def test(a):
    manifest = verify_manifest(a.run)
    runtime = environment()
    for key in ("determinism", "mps_cpu_fallback"):
        if runtime[key] != manifest["environment"].get(key, runtime[key]):
            raise ValueError(f"Test runtime {key} differs from fit; use the same setting")
    if sources() != manifest["source_sha256"]:
        raise ValueError("Code changed since fit; restore source or start a new run")
    if digest(a.data / "manifest.json") != manifest["data_manifest_sha256"]:
        raise ValueError("Data preparation differs from fit")
    data = Data(a.data, a.device, ["test"])
    config = manifest["config"]
    selected = json.loads((a.run / "selection.json").read_text())["methods"]
    a.out.mkdir(parents=True, exist_ok=False)
    summary, files = [], []
    for method in METHODS:
        for seed in config["seeds"]:
            seed_all(seed)
            model = None
            if method in LEARNED:
                model = VCEModel(embed_dim=data.emb.shape[1]).to(a.device)
                model.load_state_dict(torch.load(a.run / selected[method]["checkpoints"][str(seed)],
                                                 map_location=a.device, weights_only=True))
            rows = evaluate(data, "test", method, model, selected[method]["params"], config, seed)
            name = f"{method}.seed-{seed}.jsonl"
            write_jsonl(a.out / name, rows)
            files.append(name)
            summary.append({"dataset": data.manifest["dataset"], "method": method, "seed": seed,
                            "n": len(rows), "file": name, **aggregate(rows)})
    dump(a.out / "summary.json", summary)
    files.append("summary.json")
    dump(a.out / "manifest.json", {
        "schema": 1, "dataset": data.manifest["dataset"], "config": config,
        "data_manifest_sha256": digest(a.data / "manifest.json"),
        "run_manifest_sha256": digest(a.run / "manifest.json"),
        "preparation": {k: data.manifest.get(k) for k in
                        ("counts", "catalog_size", "embedding_dim", "notes")},
        "source_sha256": sources(), "environment": environment(), "device": a.device,
        "files": {f: digest(a.out / f) for f in files}, "command": sys.argv,
    })
    print(f"Test results saved: {a.out}. Do not tune against these results.")


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("fit", "test"):
        q = sub.add_parser(name)
        q.add_argument("--data", type=Path, required=True)
        q.add_argument("--out", type=Path, required=True)
        q.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
        q.add_argument("--cpu-threads", type=int, default=min(4, os.cpu_count() or 1),
                       help="CPU tensor worker threads; does not parallelize Python preprocessing")
        q.add_argument("--interop-threads", type=int, default=1)
        q.add_argument("--determinism", choices=["strict", "warn"], default="strict",
                       help="warn permits nondeterministic operations with warnings; recorded in manifest")
        if name == "fit":
            q.add_argument("--config", type=Path, default=Path("experiments/revision/config.json"))
        else:
            q.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error("Output exists; choose a new path (no overwrite/resume)")
    try:
        configure_runtime(a)
    except ValueError as error:
        p.error(str(error))
    (fit if a.command == "fit" else test)(a)


if __name__ == "__main__":
    main()
