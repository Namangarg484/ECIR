"""Inference-only diagnostics for learned concentration and angular geometry."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.revision.protocol import digest, dump, verify_manifest, write_jsonl
from experiments.revision.run import (Data, configure_runtime, environment,
                                      seed_all, sources as revision_sources)
from src.models.vce_model import VCEModel


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


def validate_inputs(args, section):
    original = verify_manifest(args.original_run)
    stochastic = verify_manifest(args.stochastic_run)
    if stochastic.get("experiment") != "ecir-stochastic-evaluation":
        raise ValueError("--stochastic-run is not an ECIR stochastic run")
    if revision_sources() != original["source_sha256"]:
        raise ValueError("Current revision sources differ from the original fit")
    verify_current_sources(stochastic)
    data_hash = digest(args.data / "manifest.json")
    if data_hash != original["data_manifest_sha256"] or data_hash != stochastic["data_manifest_sha256"]:
        raise ValueError("Prepared data differs across diagnostic inputs")
    if stochastic["original_run_manifest_sha256"] != digest(args.original_run / "manifest.json"):
        raise ValueError("Stochastic run was not derived from this original run")
    if original["dataset"] != stochastic["dataset"]:
        raise ValueError("Original and stochastic runs refer to different datasets")
    if args.device != original["device"] or args.device != stochastic["device"]:
        raise ValueError("Diagnostic device must match the frozen training runs")
    if section.get("split") != "validation":
        raise ValueError("Mechanism diagnostics are predeclared on validation only")
    seeds = section.get("training_seeds", [])
    if seeds != original["config"]["seeds"]:
        raise ValueError("Diagnostic training seeds must equal the frozen run seeds")
    if not isinstance(section.get("batch_size"), int) or section["batch_size"] <= 0:
        raise ValueError("diagnostics.batch_size must be positive")
    if not isinstance(section.get("max_context"), int) or section["max_context"] <= 0:
        raise ValueError("diagnostics.max_context must be positive")
    if not 0 <= float(section.get("boundary_tolerance", -1)) < 1:
        raise ValueError("diagnostics.boundary_tolerance must be in [0, 1)")
    runtime = environment()
    for manifest in (original, stochastic):
        for key in ("determinism", "mps_cpu_fallback"):
            expected = manifest["environment"].get(key, runtime[key])
            if runtime[key] != expected:
                raise ValueError(f"Diagnostic runtime {key} differs from frozen run")
    return original, stochastic


def rankdata(values):
    """Average ranks for exact ties; equivalent to the usual Spearman ranks."""
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def spearman(left, right):
    if len(left) != len(right) or len(left) < 2:
        return None
    left_rank, right_rank = rankdata(left), rankdata(right)
    if left_rank.std() == 0 or right_rank.std() == 0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def distribution(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Diagnostic distribution is empty or non-finite")
    quantiles = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    return {"count": int(len(array)), "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "quantiles": {f"q{int(q * 100):02d}": float(np.quantile(array, q))
                          for q in quantiles}}


def coherence(anchors, mask):
    """Mean pairwise cosine among the actually used, normalized context items."""
    normalized = F.normalize(anchors, dim=-1)
    valid = (~mask).unsqueeze(-1)
    totals = (normalized * valid).sum(1)
    lengths = (~mask).sum(1)
    numerator = (totals * totals).sum(-1) - lengths
    denominator = lengths * (lengths - 1)
    return torch.where(lengths > 1, numerator / denominator,
                       torch.full_like(numerator, float("nan")))


def grouped_summary(values, keys):
    grouped = {}
    for value, key in zip(values, keys):
        grouped.setdefault(str(key), []).append(value)
    return {key: distribution(grouped[key]) for key in sorted(grouped)}


@torch.no_grad()
def diagnose(args):
    config = json.loads(args.config.read_text())
    section = config.get("diagnostics", {})
    original, stochastic = validate_inputs(args, section)
    data = Data(args.data, args.device, [section["split"]])
    selection = json.loads((args.stochastic_run / "selection.json").read_text())
    selected = selection["methods"]["full"]
    rows = data.splits[section["split"]]
    records = [{"event": row["event"], "group": row["group"],
                "context_length_full": len(row["context"]),
                "context_length_used": min(len(row["context"]), section["max_context"]),
                "kappa_by_training_seed": [],
                "radius_degrees_by_training_seed": [],
                "centroid_shift_degrees_by_training_seed": []}
               for row in rows]
    coherence_values = [None] * len(rows)
    gate_values = {}
    kappa_min, kappa_max = 10.0, 500.0

    for seed in section["training_seeds"]:
        seed_all(seed)
        model = VCEModel(embed_dim=data.emb.shape[1]).to(args.device)
        checkpoint = args.stochastic_run / selected["checkpoints"][str(seed)]
        model.load_state_dict(torch.load(checkpoint, map_location=args.device,
                                         weights_only=True))
        model.eval()
        gate_values[str(seed)] = float((0.2 * torch.sigmoid(
            model.residual_gate.detach())).cpu())
        kappa_max = float(model.max_kappa)
        for start in range(0, len(rows), section["batch_size"]):
            batch = rows[start:start + section["batch_size"]]
            _, anchors, mask, query = data.batch(batch, section["max_context"])
            mu, kappa, _ = model(query, anchors, mask)
            radius = torch.rad2deg(torch.atan(torch.rsqrt(kappa)))
            cosine = (query * mu).sum(-1).clamp(-1.0, 1.0)
            shift = torch.rad2deg(torch.acos(cosine))
            coherent = coherence(anchors, mask)
            for offset in range(len(batch)):
                record = records[start + offset]
                record["kappa_by_training_seed"].append(float(kappa[offset, 0].cpu()))
                record["radius_degrees_by_training_seed"].append(
                    float(radius[offset, 0].cpu()))
                record["centroid_shift_degrees_by_training_seed"].append(
                    float(shift[offset].cpu()))
                if coherence_values[start + offset] is None:
                    value = float(coherent[offset].cpu())
                    coherence_values[start + offset] = (
                        value if math.isfinite(value) else "undefined")
        print(f"Diagnostics training seed {seed}: {len(rows)}/{len(rows)} events",
              flush=True)
        del model

    for record, coherent in zip(records, coherence_values):
        record["context_coherence"] = (
            None if coherent == "undefined" else coherent)
        for source, destination in (
                ("kappa_by_training_seed", "kappa"),
                ("radius_degrees_by_training_seed", "radius_degrees"),
                ("centroid_shift_degrees_by_training_seed",
                 "centroid_shift_degrees")):
            values = np.asarray(record[source], dtype=np.float64)
            record[destination + "_mean"] = float(values.mean())
            record[destination + "_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0)

    pooled_kappa = [value for record in records
                    for value in record["kappa_by_training_seed"]]
    pooled_radius = [value for record in records
                     for value in record["radius_degrees_by_training_seed"]]
    pooled_shift = [value for record in records
                    for value in record["centroid_shift_degrees_by_training_seed"]]
    event_kappa = [record["kappa_mean"] for record in records]
    event_radius = [record["radius_degrees_mean"] for record in records]
    event_shift = [record["centroid_shift_degrees_mean"] for record in records]
    context_lengths = [record["context_length_used"] for record in records]
    coherent_pairs = [(record["radius_degrees_mean"],
                       record["context_coherence"])
                      for record in records
                      if record["context_coherence"] is not None]
    if not coherent_pairs:
        raise ValueError("No context has at least two items for coherence analysis")
    coherent_radii = [pair[0] for pair in coherent_pairs]
    coherences = [pair[1] for pair in coherent_pairs]
    tolerance = float(section["boundary_tolerance"])
    quartile_edges = np.quantile(np.asarray(coherences), (0.25, 0.5, 0.75))
    coherence_bins = [f"Q{np.searchsorted(quartile_edges, value, side='right') + 1}"
                      for value in coherences]
    summary = {
        "dataset": original["dataset"], "split": section["split"],
        "events": len(records), "training_seeds": section["training_seeds"],
        "kappa_bounds": {"minimum": kappa_min, "maximum": kappa_max,
                         "tolerance": tolerance},
        "pooled_seed_query": {
            "kappa": distribution(pooled_kappa),
            "radius_degrees": distribution(pooled_radius),
            "centroid_shift_degrees": distribution(pooled_shift),
        },
        "event_mean_across_training_seeds": {
            "kappa": distribution(event_kappa),
            "radius_degrees": distribution(event_radius),
            "centroid_shift_degrees": distribution(event_shift),
            "context_coherence": distribution(coherences),
        },
        "boundary_fractions": {
            "at_minimum": float(np.mean(np.asarray(pooled_kappa) <=
                                         kappa_min + tolerance)),
            "at_maximum": float(np.mean(np.asarray(pooled_kappa) >=
                                         kappa_max - tolerance)),
        },
        "spearman_event_level": {
            "radius_vs_context_length": spearman(event_radius, context_lengths),
            "radius_vs_context_coherence": spearman(coherent_radii, coherences),
            "kappa_vs_context_length": spearman(event_kappa, context_lengths),
            "kappa_vs_context_coherence": spearman(
                [record["kappa_mean"] for record in records
                 if record["context_coherence"] is not None], coherences),
        },
        "radius_by_context_length": grouped_summary(event_radius, context_lengths),
        "radius_by_coherence_quartile": grouped_summary(coherent_radii,
                                                          coherence_bins),
        "coherence_quartile_edges": [float(value) for value in quartile_edges],
        "residual_gate_by_training_seed": gate_values,
        "notes": [
            "Radius is atan(1/sqrt(kappa)) in degrees, matching the implemented fixed-radius tangent perturbation.",
            "Distributions are descriptive validation-set mechanism diagnostics and are not test-set model selection.",
            "Event-level correlations use each query's mean prediction across training seeds.",
            "Context coherence is undefined and omitted for singleton contexts.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "diagnostics.jsonl", records)
    dump(args.out / "summary.json", summary)
    files = ["diagnostics.jsonl", "summary.json"]
    dump(args.out / "manifest.json", {
        "schema": 1, "experiment": "ecir-kappa-diagnostics",
        "dataset": original["dataset"], "config": config,
        "data_manifest_sha256": digest(args.data / "manifest.json"),
        "original_run_manifest_sha256": digest(args.original_run / "manifest.json"),
        "stochastic_run_manifest_sha256": digest(args.stochastic_run / "manifest.json"),
        "source_sha256": source_hashes(), "environment": environment(),
        "device": args.device,
        "files": {name: digest(args.out / name) for name in files},
        "command": sys.argv,
    })
    print(f"Kappa/radius diagnostics saved: {args.out}")


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--stochastic-run", type=Path, required=True)
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
        diagnose(args)
    except (KeyError, ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
