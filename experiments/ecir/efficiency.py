"""Inference-only latency and memory benchmark for the three learned variants.

Run one method per process so process and accelerator memory measurements are
not contaminated by models benchmarked earlier in the same interpreter.
"""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.revision.protocol import digest, dump, verify_manifest
from experiments.revision.run import (Data, configure_runtime, directions,
                                      environment, seed_all,
                                      sources as revision_sources)
from src.models.vce_model import VCEModel

METHODS = ("learned_centroid", "learned_fixed", "full")


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root / "experiments/revision/run.py",
             root / "experiments/revision/protocol.py",
             root / "src/models/vce_model.py"]
    return {str(path.relative_to(root)): digest(path) for path in paths}


def verify_current_sources(manifest):
    root = Path(__file__).resolve().parents[2]
    for relative, expected in manifest["source_sha256"].items():
        path = root / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Current source differs from frozen run: {relative}")


def validate_inputs(args):
    original = verify_manifest(args.original_run)
    stochastic = verify_manifest(args.stochastic_run)
    if stochastic.get("experiment") != "ecir-stochastic-evaluation":
        raise ValueError("--stochastic-run is not an ECIR stochastic run")
    if revision_sources() != original["source_sha256"]:
        raise ValueError("Current revision sources differ from the original fit")
    verify_current_sources(stochastic)
    data_hash = digest(args.data / "manifest.json")
    if data_hash != original["data_manifest_sha256"] or data_hash != stochastic["data_manifest_sha256"]:
        raise ValueError("Prepared data differs across benchmark inputs")
    if stochastic["original_run_manifest_sha256"] != digest(args.original_run / "manifest.json"):
        raise ValueError("Stochastic run was not derived from this original run")
    if original["dataset"] != stochastic["dataset"]:
        raise ValueError("Original and stochastic runs refer to different datasets")
    if args.device != original["device"] or args.device != stochastic["device"]:
        raise ValueError("Benchmark device must match both frozen training runs")
    runtime = environment()
    for manifest in (original, stochastic):
        for key in ("determinism", "mps_cpu_fallback"):
            expected = manifest["environment"].get(key, runtime[key])
            if runtime[key] != expected:
                raise ValueError(f"Benchmark runtime {key} differs from frozen run")
    return original, stochastic


def selected_model(args, data, original_manifest, stochastic_manifest, section):
    training_seed = str(section["training_seed"])
    if args.method == "learned_centroid":
        selection = json.loads((args.original_run / "selection.json").read_text())
        selected = selection["methods"][args.method]
        checkpoint = args.original_run / selected["checkpoints"][training_seed]
    else:
        selection = json.loads((args.stochastic_run / "selection.json").read_text())
        selected = selection["methods"][args.method]
        checkpoint = args.stochastic_run / selected["checkpoints"][training_seed]
    model = VCEModel(embed_dim=data.emb.shape[1]).to(args.device)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device,
                                     weights_only=True))
    model.eval()
    return model, selected["params"], checkpoint


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def accelerator_memory(device):
    if device == "cuda":
        return int(torch.cuda.memory_allocated())
    if device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return int(torch.mps.current_allocated_memory())
    return None


def max_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes; Linux and most other Unix platforms report KiB.
    return int(value if sys.platform == "darwin" else value * 1024)


def timing_summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {"mean_ms_per_query": float(array.mean()),
            "median_ms_per_query": float(np.median(array)),
            "p95_ms_per_query": float(np.quantile(array, 0.95)),
            "batch_observations": int(len(array))}


@torch.no_grad()
def run_batch(data, rows, method, model, params, revision_config, inference_seed):
    device_type = data.emb.device.type
    synchronize(device_type)
    before = accelerator_memory(device_type)
    started = time.perf_counter()
    queries = directions(data, rows, method, model, params, revision_config,
                         inference_seed, "efficiency")
    synchronize(device_type)
    query_elapsed = time.perf_counter() - started
    after_query = accelerator_memory(device_type)

    started = time.perf_counter()
    scores = queries[:, 0] @ data.emb.T
    for probe in range(1, queries.shape[1]):
        scores = torch.maximum(scores, queries[:, probe] @ data.emb.T)
    synchronize(device_type)
    score_elapsed = time.perf_counter() - started
    after_score = accelerator_memory(device_type)
    # Force the score tensor to remain live until after synchronization/memory sampling.
    _ = scores.shape
    observed = [value for value in (before, after_query, after_score)
                if value is not None]
    peak = max(observed) if observed else None
    del scores, queries
    return query_elapsed, score_elapsed, before, peak


def benchmark(args):
    config = json.loads(args.config.read_text())
    section = config.get("efficiency", {})
    for key in ("training_seed", "inference_seed", "batch_size",
                "warmup_batches", "measured_batches"):
        if not isinstance(section.get(key), int) or section[key] <= 0:
            raise ValueError(f"efficiency.{key} must be a positive integer")
    original, stochastic = validate_inputs(args)
    data = Data(args.data, args.device, ["test"])
    model, params, checkpoint = selected_model(
        args, data, original, stochastic, section)
    seed_all(section["inference_seed"])
    rows = data.splits["test"]
    batch_size = section["batch_size"]
    batches = [rows[start:start + batch_size]
               for start in range(0, len(rows), batch_size)]
    if not batches:
        raise ValueError("Cannot benchmark an empty test split")

    for index in range(section["warmup_batches"]):
        run_batch(data, batches[index % len(batches)], args.method, model, params,
                  original["config"], section["inference_seed"])

    query_times, score_times, total_times = [], [], []
    memory_baselines, memory_peaks = [], []
    query_count = 0
    for index in range(section["measured_batches"]):
        batch = batches[(section["warmup_batches"] + index) % len(batches)]
        query_elapsed, score_elapsed, baseline, peak = run_batch(
            data, batch, args.method, model, params, original["config"],
            section["inference_seed"])
        size = len(batch)
        query_count += size
        query_times.append(1000.0 * query_elapsed / size)
        score_times.append(1000.0 * score_elapsed / size)
        total_times.append(1000.0 * (query_elapsed + score_elapsed) / size)
        if baseline is not None:
            memory_baselines.append(baseline)
        if peak is not None:
            memory_peaks.append(peak)

    probes = 1 if args.method == "learned_centroid" else original["config"]["samples"]
    accelerator_baseline = min(memory_baselines) if memory_baselines else None
    accelerator_peak = max(memory_peaks) if memory_peaks else None
    result = {
        "dataset": original["dataset"], "method": args.method,
        "device": args.device, "training_seed": section["training_seed"],
        "inference_seed": section["inference_seed"], "checkpoint": checkpoint.name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "probes": probes, "catalog_size": len(data.ids),
        "embedding_dimension": int(data.emb.shape[1]),
        "batch_size": batch_size, "warmup_batches": section["warmup_batches"],
        "measured_batches": section["measured_batches"],
        "measured_queries_including_repeats": query_count,
        "query_construction": timing_summary(query_times),
        "exhaustive_scoring": timing_summary(score_times),
        "total": timing_summary(total_times),
        "accelerator_memory_baseline_bytes": accelerator_baseline,
        "accelerator_memory_peak_observed_bytes": accelerator_peak,
        "accelerator_memory_observed_delta_bytes": (
            accelerator_peak - accelerator_baseline
            if accelerator_peak is not None and accelerator_baseline is not None
            else None),
        "process_peak_rss_bytes": max_rss_bytes(),
        "notes": [
            "Latency includes query construction and exact full-catalog scoring but excludes metric computation and result serialization.",
            "Measured batches may cycle over test queries; no relevance labels or metrics are inspected.",
            "Observed accelerator memory is sampled after synchronized stages and is not an allocator high-water mark on MPS.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out / "efficiency.json", result)
    dump(args.out / "manifest.json", {
        "schema": 1, "experiment": "ecir-efficiency",
        "dataset": original["dataset"], "method": args.method,
        "config": config,
        "data_manifest_sha256": digest(args.data / "manifest.json"),
        "original_run_manifest_sha256": digest(args.original_run / "manifest.json"),
        "stochastic_run_manifest_sha256": digest(args.stochastic_run / "manifest.json"),
        "source_sha256": source_hashes(), "environment": environment(),
        "device": args.device,
        "files": {"efficiency.json": digest(args.out / "efficiency.json")},
        "command": sys.argv,
    })
    print(f"Efficiency result saved: {args.out / 'efficiency.json'}")


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--stochastic-run", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=Path("experiments/ecir/additional_config.json"))
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--cpu-threads", type=int,
                        default=min(4, os.cpu_count() or 1))
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--determinism", choices=("strict", "warn"),
                        default="strict")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; choose a new versioned path")
    try:
        configure_runtime(args)
        benchmark(args)
    except (KeyError, ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
