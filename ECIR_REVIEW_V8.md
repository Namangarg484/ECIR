# Review-response implementation (v8)

Status: code written and inspected, **not executed**. No new results are claimed
in the manuscript. Run from the repository root using the existing environment.
Do not edit frozen training/report sources to accommodate these analyses.

## 1. Matched validation-inference-seed control — priority

This implements the reviewer's single-seed option, not a new training campaign:

- Retain the existing hard/smooth heads, whose epochs were selected on seed 3101.
- Evaluate all four fixed radii on validation using the same seed 3101 and each
  of the five original learned-center checkpoints.
- Select one fixed radius per dataset by mean validation NDCG@10 across centers,
  breaking ties by the original declared grid order.
- Seal selections for all datasets before producing any new test comparisons.
- Reuse the existing per-event test results for the selected fixed radius and
  both adaptive heads over the same ten inference seeds. Original testing saved
  every declared fixed radius, so no new test forward passes are needed here.
- Report no expansion, selected fixed, hard adaptation, and smooth adaptation,
  with the original crossed bootstrap and paired support checks.

This matches **validation inference randomness**, not total model-selection
compute. Fixed selection is global across center seeds; adaptive epochs are
selected per center, with a different candidate search space. It is a post-hoc
robustness analysis on an already examined test set, not a new confirmatory test.
Do not describe it as an equal-compute or perfectly matched search-budget study.

## 2. Probe redundancy — inference only, TalkPlay

Use the existing separately optimized learned-fixed and adaptive checkpoints,
all five training seeds, all ten inference seeds, S in {1,2,5,10,20}, k in {10,50}.
No new choices are made from the test results. The implementation reports:

- mean pairwise Jaccard overlap of each probe's exact top-k results;
- unique candidates in their union and candidates added by the last probe;
- target coverage in the union, gains over the first probe, and gains from S-1;
- actual Recall@k and NDCG@10 after the original max-score aggregation;
- mean pairwise cosine and angle between query probes.

Union coverage uses up to S*k candidates and is **not** fixed-budget Recall@k.
Compare it with actual merged Recall@k before interpreting redundancy. Geometric
diversity alone cannot establish useful semantic diversity. S=1 pairwise values
are undefined, not zero. Output is descriptive; no new significance claim is made.
Each seed/count NDCG must reproduce the saved probe-count sweep within 1e-10;
the run stops on a mismatch rather than quietly reporting inconsistent evidence.

The GPU scores one probe at a time, avoiding a batch-by-probes-by-catalog score
allocation. CPU code handles exact tie-breaking, exclusions, and diagnostics.
The CPU thread flag does not parallelize Python loops.

## Commands

```bash
source .venv-revision/bin/activate
python -m unittest experiments.ecir.test_review_v8 -v
bash -n scripts/run_ecir_review_v8.sh

# Recommended: finish the priority control first.
DEVICE=mps CPU_THREADS=4 bash scripts/run_ecir_review_v8.sh select
bash scripts/run_ecir_review_v8.sh report

# Then the optional mechanism diagnostic (no training).
DEVICE=mps CPU_THREADS=4 bash scripts/run_ecir_review_v8.sh redundancy
```

Or, after tests, run all stages in that order:

```bash
DEVICE=mps CPU_THREADS=4 bash scripts/run_ecir_review_v8.sh all
```

Defaults match the retained MPS/warn-determinism runs. Runtime mismatches fail
explicitly; do not change the device of frozen experiments merely to bypass a
check. Inputs are the existing v1 data/frozen-radius runs, mps-v1 parent runs,
mps-v2 stochastic checkpoints, and direction-sensitivity-v1 results.

Selections and diagnostic inference resume verified cells on the same command.
Completed reports are verified and skipped. An interrupted report (as opposed
to inference) must be moved aside before rerunning; nothing is deleted for you.
Do not run two copies against the same output directory simultaneously.
If source/configuration changes after execution starts, use a new REVIEW_TAG.
The artifact allowlist below includes the default v8 outputs only; update it
explicitly if a different tag becomes the retained experiment.

## Send back these outputs

- `paper/generated/ecir-review-v8/matched-selection/*/REPORT.md`
- `paper/generated/ecir-review-v8/matched-selection/*/selection.json`
- `paper/generated/ecir-review-v8/matched-selection/*/paired_comparisons.csv`
- `paper/generated/ecir-review-v8/probe-redundancy/talkplay/REPORT.md`
- `paper/generated/ecir-review-v8/probe-redundancy/talkplay/probe_redundancy.csv`

The manuscript intentionally still reports the completed v7 comparisons and
their selection asymmetry. After these outputs are available, integrate the
actual results and decide which diagnostic belongs in the main text versus
the supplement. Do not replace historical numbers preemptively.

## Manuscript changes already made

- Controlled-study title, shorter abstract, and three scientific contributions.
- RQ2a (expansion versus center) separated from RQ2b (adaptive versus fixed).
- Stronger tangent-score interpretation and rationale for isotropic controls.
- Baselines as context, Last.fm imposed-order stress-test wording, explicit SD
  caption, softer inferential language, and less prescriptive discussion.
- Shorter reproducibility, ethics, and repeated discussion passages; numerical
  evidence and both figures retained.

Remaining editorial work: verify recent primary literature before adding new
references (none were invented), integrate the new analysis outputs, and compile
to establish the actual page count. This edit does not certify the page limit.

## Packaging and rendering, after result integration

```bash
python scripts/build_ecir_anonymous_artifact.py --out submission/ecir2027_anonymous_artifact_v8
python scripts/audit_ecir_submission.py --bundle submission/ecir2027_anonymous_artifact_v8
latexmk -pdf -interaction=nonstopmode -halt-on-error ECIR.tex
```

Existing archives are stale. Choose a fresh archive suffix if v8 already exists.
Inspect the rendered PDF, official submission rules, and actual PDF/ZIP metadata.
Source consistency checks do not prove visual correctness or guarantee acceptance.
