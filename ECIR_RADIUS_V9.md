# Final radius-selection robustness package (v9)

Status: user-completed fit, test, and aggregate reports are present for all four
datasets. The final manuscript incorporates these saved results as its primary
frozen-center comparison. Every adaptive-versus-per-center-fixed interval
includes zero; this is not an equivalence finding. Existing experiment sources,
manifests, checkpoints, and results remain unchanged. The assistant has not run
training, tests, audits, or LaTeX compilation for this integration. The updated
preflight, rebuilt archive, and rendered page count still require author checks.

## Question and fixed protocol

Does the existing conditional result persist with a finer fixed-radius grid and
ten-inference-seed validation for adaptive checkpoint selection and early stopping?

- Datasets: TalkPlay, MovieLens-1M, Last.fm, Amazon Digital Music.
- Centers: the same five selected learned-center checkpoints, never updated.
- Center seeds: 42, 123, 456, 789, 2026.
- Validation and test inference seeds: 3101 through 3110. Random directions also
  depend on split and event, so sharing seed labels does not reuse validation
  examples or their exact event-specific probes for test.
- Probe budget: five, unchanged.
- Fixed kappa grid: 10, 20, 35, 50, 75, 100, 150, 250, 350, 400, 450, 500.
  Larger kappa means a smaller angular radius. The grid stays within the
  adaptive heads' existing range; it does not expand that range.
- Fixed selection: report both one global radius (averaged over the five
  centers and ten inference seeds) and a separate radius for each center
  (averaged over ten inference seeds). Exact ties choose the first grid value.
- Adaptive heads: retrain hard-clipped and smooth heads from the original
  learned-center checkpoint's head initialization, **not** the old adaptive
  winner. Keep the original objective, negatives, LR, regularization, training
  noise construction, maximum 50 epochs, and patience seven.
- Epoch selection and early stopping both use mean validation NDCG@10 over all
  ten inference seeds. Exact ties keep the earlier epoch.
- Test compares no expansion, global fixed, per-center fixed, hard adaptation,
  and smooth adaptation. Only the selected fixed radii are tested. All expanded
  conditions share event-specific directions and use the same exact evaluator.

The protocol is specified in `experiments/ecir/radius_v9_config.json`. Do not
change it after inspecting new test results. Source/configuration drift requires
a new versioned output, not bypassing the checks.

## What this does and does not resolve

This matches validation inference seeds and adds a matched per-center selection
unit. It does **not** equalize grid-size versus epoch-search compute, prove good
optimization, or remove conditional uncertainty from selecting configurations.
It remains a post-hoc robustness study on an already examined test set.
Report all declared comparisons, not just those supporting the paper's story.

The grid is inference-only. The head stage is genuinely new training: four
datasets × five centers × two parameterizations = 40 radius-head fits.
Ten-seed full-catalog validation can dominate runtime. Do not assume an
hours-not-days bound; epoch logs provide actual elapsed time. In the worst case,
40 × 50 × 10 = 20,000 head-validation passes, before early stopping reduces them.
This package deliberately does not add model families, datasets, or probe sweeps.

## Run from the repository root

```bash
source .venv-revision/bin/activate

# Synthetic tests and shell syntax checks first; these do not launch real training.
python -m unittest experiments.ecir.test_radius_v9 -v &&
bash -n scripts/run_ecir_radius_v9.sh &&
DEVICE=mps CPU_THREADS=4 bash scripts/run_ecir_radius_v9.sh all
```

The default runs each stage across all four datasets before advancing:
all validation grids → all head fits → all tests → all reports. On Apple MPS,
tensor scoring/model operations use the GPU; input processing, noise generation,
rank metrics, hashing, and bootstrap reporting use the CPU. The tensor thread
setting does not turn Python loops into multithreaded code.

To run stages separately:

```bash
# Inference-only fixed-radius selection (no heads trained, no test scores loaded).
DEVICE=mps CPU_THREADS=4 bash scripts/run_ecir_radius_v9.sh grid

# This stage DOES retrain only radius heads and reuses completed grid cells.
DEVICE=mps CPU_THREADS=4 bash scripts/run_ecir_radius_v9.sh fit

# Only after fits finish and selections are sealed.
DEVICE=mps CPU_THREADS=4 bash scripts/run_ecir_radius_v9.sh test
bash scripts/run_ecir_radius_v9.sh report
```

Preserve MPS, warn-determinism, and the original fallback setting to match the
parent runs. A mismatch stops execution. This is not a reason to disable checks.
Optional macOS sleep prevention: prefix the long command with `caffeinate -i`,
for example `DEVICE=mps CPU_THREADS=4 caffeinate -i bash scripts/run_ecir_radius_v9.sh all`.

## Resume and storage

Run the same command again after an interruption. Never launch two copies for
the same output; a filesystem lock rejects concurrent writers.

- Fixed validation cells resume by center/inference seed.
- Every epoch saves its head, optimizer state, center identity, and gradient/loss
  diagnostics before validation. Checkpoint plus checksum is committed together.
- Validation resumes individual completed inference seeds within that epoch.
- If training was interrupted before the epoch checkpoint committed, only that
  epoch is repeated from its predecessor, not the entire head fit.
- Checkpoint selection is reconstructed from saved ten-seed validation means;
  no epoch is selected from partially completed validation.
- Every completed epoch is retained, not just the winner. Disk use grows with
  epochs; optimizer states are saved, but the full frozen center is not duplicated.
- Test cells, reports, and completed manifests are verified before reuse.
- Hidden `.partial-*` crash remnants are ignored; no existing data are deleted.

Current heads contain no dropout; epoch order/negatives and event probe noise
use explicit generators. This supports epoch-boundary resume, but MPS warn-mode
nondeterminism still prevents a promise of bit-identical reruns on hardware.

## Outputs and interpretation

For each dataset:

- `artifacts/ecir/radius-v9/runs/DATASET/selection.json`
- `artifacts/ecir/radius-v9/runs/DATASET/heads/TYPE/seed-N/epoch-NNN/`
- `artifacts/ecir/radius-v9/results/DATASET/manifest.json`
- `paper/generated/ecir-radius-v9/DATASET/REPORT.md`
- `paper/generated/ecir-radius-v9/DATASET/metrics.csv`
- `paper/generated/ecir-radius-v9/DATASET/paired_comparisons.csv`
- `paper/generated/ecir-radius-v9/DATASET/fixed_validation.csv`

Send the four REPORT.md files and paired-comparison CSVs for manuscript integration.
Reports include NDCG, recall, and MRR means; NDCG contrasts use the same crossed
bootstrap implementation with shared session/user draws. Intervals are descriptive,
not multiplicity corrected, and conditional on fitted/selected models. Crossing
zero is not evidence of equivalence. If outcomes change, revise the conclusion
rather than choosing whichever protocol produces the preferred result.

Default outputs are new `radius-v9` directories. If a source/config change requires
a new tag, use `RADIUS_TAG=v9b` consistently; preserve the prior run. The artifact
allowlist and audit below discover the default v9 outputs only. Update those
explicitly if a different tag becomes the retained analysis.

## After result integration, not before

```bash
python scripts/build_ecir_anonymous_artifact.py --out submission/ecir2027_anonymous_artifact_v9
python scripts/audit_ecir_submission.py --bundle submission/ecir2027_anonymous_artifact_v9
latexmk -pdf -interaction=nonstopmode -halt-on-error ECIR.tex
```

Archive creation is create-only; use a fresh suffix if the destination exists.
The new source/tests/launcher and default generated reports are allowlisted, not
the large training-state directories. Compile and inspect the actual page count,
figures, references, and PDF anonymity. Experiments do not replace these checks
and cannot guarantee acceptance.
