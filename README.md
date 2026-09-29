# ECIR 2027 Code and Reproducibility Materials

Repository: [Namangarg484/ECIR](https://github.com/Namangarg484/ECIR)

This artifact accompanies *Learning the Center or Exploring Its Neighborhood?
A Controlled Study of Multi-Probe Session Retrieval*. It contains the
implementation, frozen experiment configurations, exact launch scripts, unit
tests, and aggregate files used by the manuscript.

This public repository identifies its owner. For double-blind review, submit
the separately packaged anonymous ZIP below; do not use this repository URL
as an anonymous artifact link.

## Manuscript and review package

- [Latest manuscript](paper/ECIR.tex): includes all final wording edits and the
  acknowledgement, "The paper is partially supported by the University of
  Piraeus Research Center." This version is non-anonymous and self-contained.
- [Anonymous manuscript](paper/ECIR_anonymous.tex): preserved source matching
  the audited v13 review package, with acknowledgements disabled.
- [Submission abstract](paper/abstract.txt): synchronized with both versions.
- [Anonymous review ZIP](review-artifact/ecir2027_anonymous_artifact_v13_final.zip)
  and its [SHA-256 checksum](review-artifact/ecir2027_anonymous_artifact_v13_final.zip.sha256).

The ZIP contains the review source, code, configurations, aggregate results,
and provenance. It excludes the institutional acknowledgement and Git history.
It contains no compiled PDF. The final PDF's rendering, metadata, page count,
and uploaded version must be checked separately.

To compile either manuscript with Springer LNCS and PGFPlots installed:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error ECIR.tex
# For the anonymous version instead:
latexmk -pdf -interaction=nonstopmode -halt-on-error ECIR_anonymous.tex
```

## What the artifact supports

The code separates three operations under one full-catalog evaluator:

1. constructing a weighted or learned query center;
2. using one query or fixed-radius local probes; and
3. predicting a radius-control parameter per query.

It also contains the V-SKNN and content-SASRec controls, the probe-count sweep,
radius diagnostics, efficiency benchmark, seed-aware comparison, and report
generators. The aggregate CSV files and generated LaTeX tables are included so
the reported results can be inspected without retraining.

The primary frozen-center comparison is `ecir-radius-v9`: a twelve-value fixed
grid, global and per-center fixed selection, and newly fitted hard/smooth heads
whose epoch selection and early stopping use the same ten validation inference
seeds. It matches selection unit and inference seeds, not search spaces or
tuning compute. The v8 probe-redundancy analysis uses the original separately
trained models. Both are post-hoc robustness analyses, not independent test sets.

The completed v10 follow-ups add training-item support, warm/cold test metrics,
and validation/test probe-overlap comparisons. They preserve the full catalog
and selected checkpoints. `paper/IMPLEMENTATION_DETAILS.md` retains the exact
metadata weighting and V-SKNN definitions omitted from the compact manuscript.

The hash-frozen implementation retains the legacy Python class name
`VCEModel` and a few historical comments referring to “variational” sampling
or uncertainty. The executed code normalizes fixed-length tangent perturbations;
it does not sample an exact von Mises--Fisher distribution or produce calibrated
uncertainty. The manuscript's ACE terminology and geometric definition describe
the evaluated computation. The frozen source is left unchanged so its hashes
continue to match every recorded run.

Raw datasets, pretrained encoders, embedding caches, and learned checkpoints
are not redistributed because of size and third-party access or license terms.
Reproducing training therefore requires obtaining those inputs separately.

## Environment

Python 3.11--3.13 is recommended. Clone the repository first:

```bash
git clone https://github.com/Namangarg484/ECIR.git
cd ECIR
```

Then, from the repository or extracted artifact root:

```bash
python -m venv .venv-revision
source .venv-revision/bin/activate
python -m pip install -r requirements-revision.txt
```

`requirements-revision.lock.txt` records the environment used for the final
runs. The shorter requirements file is preferable on a different platform.

Run the non-training checks with:

```bash
python -m unittest \
  experiments.revision.test_protocol \
  experiments.revision.test_runner \
  experiments.revision.test_runtime \
  experiments.ecir.test_ecir \
  experiments.ecir.test_additional \
  experiments.ecir.test_frozen_radius \
  experiments.ecir.test_submission_audit -v
```

## Required input layout

Place the separately obtained inputs at these paths relative to the artifact
root:

```text
train_dataset.jsonl
test_dataset.jsonl
cache/embeddings_track_embeddings_track_id_audio-laion_clap_2295310bfe98.pkl
data/TalkPlayData-Challenge-Track-Metadata/track_metadata.jsonl
data/ml-1m/ml-1m/movies.dat
data/ml-1m/ml-1m/ratings.dat
hetrec2011-lastfm-2k.zip
meta_Digital_Music.jsonl.gz
Digital_Music.jsonl.gz
models/minilm/                    # complete local SentenceTransformer snapshot
```

The preparation command performs no download. TalkPlay uses the supplied frozen
CLAP vectors. MovieLens, Last.fm, and Amazon Digital Music metadata are encoded
by a complete local snapshot of
`sentence-transformers/all-MiniLM-L6-v2` (mean pooling, 384 dimensions, output
normalization). Review the terms of each upstream dataset/model before use.

## Exact experiment sequence

The following commands reconstruct the prepared splits, primary four-method
ablation, behavioral/sequential controls, stochastic evaluation, V-SKNN,
diagnostics, timing measurements, and probe-count analysis. No test result is
used for fitting or hyperparameter selection.

```bash
# Prepare immutable train/validation/test events.
REVISION_DATA=artifacts/revision/v1/data \
  bash scripts/run_revision.sh prepare

# Fit on training events, select on validation, then test once.
REVISION_BASE=artifacts/revision/mps-v1 \
REVISION_DATA=artifacts/revision/v1/data \
REVISION_DEVICE=mps \
REVISION_DETERMINISM=warn \
  bash scripts/run_revision.sh fit

REVISION_BASE=artifacts/revision/mps-v1 \
REVISION_DATA=artifacts/revision/v1/data \
REVISION_DEVICE=mps \
REVISION_DETERMINISM=warn \
  bash scripts/run_revision.sh test

REVISION_BASE=artifacts/revision/mps-v1 \
REVISION_DATA=artifacts/revision/v1/data \
  bash scripts/run_revision.sh report

# Strong controls and repeated stochastic inference.
ECIR_TAG=mps-v2 bash scripts/run_ecir_experiments.sh all

# Session-kNN, radius diagnostics, and efficiency.
ECIR_TAG=mps-v2 ECIR_ADDITIONAL_TAG=v1 \
  bash scripts/run_ecir_additional.sh all

# Frozen-checkpoint S = 1, 2, 5, 10, 20 analysis and hierarchical interval.
ECIR_TAG=mps-v2 ECIR_SENSITIVITY_TAG=v1 \
  bash scripts/run_ecir_additional.sh direction-sensitivity

# Decisive identification check: freeze each selected learned center, train
# only hard-clipped or smoothly bounded radius heads, and compare against no
# expansion and every fixed kappa with paired directions.
FROZEN_RADIUS_TAG=v1 DEVICE=mps DETERMINISM=warn \
  bash scripts/run_ecir_frozen_radius.sh
```

On non-Apple systems, replace the `mps` runtime in the two ECIR shell scripts
with `cuda` or `cpu` as supported by the local PyTorch installation. This is a
runtime choice only; do not alter seeds or configurations when reproducing the
reported experiment.

## Result map

- `paper/generated/ecir-mps-v2/`: combined effectiveness results and paired
  model-conditional intervals.
- `paper/generated/ecir-additional-v1/`: V-SKNN, radius diagnostics, and
  efficiency results.
- `paper/generated/ecir-direction-sensitivity-v1/`: probe-count sensitivity,
  per-training-seed differences, and the seed-aware hierarchical interval.
- `paper/generated/ecir-radius-v9/`: primary frozen-center table, paired effect
  figure, validation selections, and crossed intervals for all four datasets.
  Exact reproduction commands are in `ECIR_RADIUS_V9.md`.
- `paper/generated/ecir-review-v8/`: single-validation-seed selection control
  and TalkPlay top-k probe overlap/union diagnostics; see `ECIR_REVIEW_V8.md`.
- `paper/generated/ecir-review-v10/`: validation/test training-item support,
  warm/cold test effectiveness, and TalkPlay validation/test overlap comparison;
  reproduction commands are in `ECIR_REVIEW_V10.md`.
- `paper/generated/ecir-frozen-radius-v1/`: historical frozen-center metrics,
  validation radius diagnostics, and paired hierarchical comparisons for each
  dataset. The corresponding validation selections are in
  `provenance/frozen-radius-selection/`.
- `provenance/data/`: retained split counts, catalog dimensions, input hashes,
  and prepared-file hashes for each dataset.
- `provenance/*-selection/`: validation scores and the selected primary,
  content-SASRec, stochastic, and V-SKNN configurations.
- `paper/ECIR.tex`: latest manuscript with the institutional acknowledgement.
- `paper/ECIR_anonymous.tex`: anonymous review manuscript.

The manuscript tables and the human-readable reports derive from the same CSV
files. Report manifests record hashes of their source result manifests and all
generated outputs. The raw per-query result directories are intentionally not
inside this lightweight repository; they can be released with checkpoints
after double-blind review, subject to venue policy and upstream licenses.

## Protocol safeguards

- Validation and test prediction events are absent from supervised training.
- Hyperparameters and checkpoints are selected using validation data only.
- All compared methods rank the same eligible full catalog and mask the full
  observed history.
- Inference seeds are distinct from training seeds.
- Deterministic baselines are not given artificial seed variance.
- Model-conditional and retraining-aware uncertainty are named separately.
- Last.fm is explicitly treated as hash-ordered artist retrieval because the
  source counts do not provide an event chronology.

## Review archive and repository integrity

The separate ZIP in `review-artifact/` was built from the anonymous author
workspace with `scripts/build_ecir_anonymous_artifact.py`. The
builder uses an explicit allowlist, omits `.git`, raw data, checkpoints, caches,
Python bytecode, PDFs, PNGs, and filesystem extended attributes, assigns fixed
ZIP timestamps and permissions, and scans text and member names for home paths,
hostnames, email addresses, and repository-owner URLs. The archive's internal
`SHA256SUMS` and `ANONYMITY_AUDIT.txt` apply only to its anonymous payload.
The outer archive checksum is written beside the ZIP.

The repository-root `SHA256SUMS` covers the public snapshot's files except
itself and Git metadata. It is an integrity manifest, not an anonymity claim.
The root `ANONYMITY_AUDIT.txt` explains this distinction.

To verify the downloaded review ZIP on macOS:

```bash
cd review-artifact
shasum -a 256 -c ecir2027_anonymous_artifact_v13_final.zip.sha256
```

The submission PDF is not part of this archive and must be checked separately
after compilation for author fields, acknowledgments, PDF metadata, and page
count.

The artifact builder and `scripts/audit_ecir_submission.py` are author-workspace
utilities: they need the original layout, manifests, and payloads omitted from
this lightweight repository. They are not clone-only checks. The builder
rejects enabled acknowledgements in review material.
After a clean compilation it accepts `--aux ECIR.aux` to check that references
start by page 13. The manuscript flushes content floats before references.
This checks the content-page boundary, not PDF metadata or visual quality.
