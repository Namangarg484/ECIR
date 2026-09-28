# Final ECIR reviewer diagnostics (v10)

These are descriptive follow-ups to the saved ECIR experiments. They do not
fit models, change checkpoints, select hyperparameters, or replace the existing
full-catalog results. Both diagnostics have now been completed by the author:
the coverage reports and all 100 validation cells are saved. Their results are
integrated into `ECIR.tex`. The assistant has not executed either diagnostic.

The commands below are retained for reproduction, not a request to rerun the
completed package. Output directories are create-only. For the current final
checks, use `ECIR_SUBMISSION_HANDOFF.md` instead.

Check the small protocol tests first:

```bash
python -m unittest experiments.ecir.test_review_v10 -v
```

Then run the complete diagnostic package with:

```bash
DEVICE=mps CPU_THREADS=4 bash scripts/run_ecir_review_v10.sh all
```

The separate commands below are useful if you want to inspect coverage before
running the validation diagnostic, or resume an interrupted validation run.

## 1. Training-item support and warm/cold test results

```bash
python -m experiments.ecir.target_support \
  --out paper/generated/ecir-review-v10/target-support
```

Outputs:

- `target_coverage.csv`: validation/test event counts and percentages whose
  target occurred anywhere in the prepared training interactions.
- `warm_cold_metrics.csv`: original saved full-catalog rankings for eleven
  configurations, partitioned by test-target support, plus the overall mean.
- `REPORT.md` and `manifest.json`: definition, limitations, and hashes.

Interpretation: warm means an item occurred as a training context item or
training target. The catalog and ranking exclusions remain unchanged. This
analysis measures item support, not a causal content-versus-behavior effect.
Inspect Amazon Digital Music first, then all four datasets. If cold targets
are common or warm-only method order changes, revise Table 2's interpretation
before submission. Do not use the test partition to retune models.

## 2. Probe overlap on TalkPlay validation events

This reuses the separately trained learned+fixed and adaptive checkpoints,
five training seeds, ten inference seeds, and the 1/5/20-probe settings.
It generates validation-specific random directions and compares aggregate
overlap with the already saved test diagnostic. No retraining occurs.

```bash
python -m experiments.ecir.validation_probe_redundancy run \
  --data artifacts/revision/v1/data/talkplay \
  --original-run artifacts/revision/mps-v1/runs/talkplay \
  --stochastic-run artifacts/ecir/mps-v2/stochastic-runs/talkplay \
  --reference-sweep artifacts/ecir/direction-sensitivity-v1/results \
  --out artifacts/ecir/review-v10/validation-probe-redundancy/talkplay \
  --device mps --cpu-threads 4 --interop-threads 1 --determinism warn

python -m experiments.ecir.validation_probe_redundancy report \
  --results artifacts/ecir/review-v10/validation-probe-redundancy/talkplay \
  --test-results artifacts/ecir/review-v8/probe-redundancy/talkplay \
  --test-report paper/generated/ecir-review-v8/probe-redundancy/talkplay \
  --out paper/generated/ecir-review-v10/validation-probe-redundancy/talkplay
```

`run` can resume verified completed cells after interruption; `report` creates
a new output directory. Review `validation_vs_test.csv`. Similar overlap on
validation and test supports the descriptive redundancy interpretation, but
does not turn the post-hoc check into a preregistered mechanism test.

## Integrated findings

- Test cold-target fractions are 46.00% on TalkPlay, 0.08% on MovieLens,
  11.40% on Last.fm, and 87.06% on Amazon. The paper includes both held-out
  splits' support, warm ACE/V-SKNN comparisons, cold ACE results, and the
  Amazon warm last-item/weighted-center scores. Full-catalog rankings are unchanged.
- At 20 probes, validation Jaccard is .826/.913 for learned+fixed/adaptive,
  with union sizes 13.75/11.70; this closely agrees with the test diagnostic.
  These remain post-hoc, descriptive checks on selected checkpoints.
- The paper summarizes frozen-center Recall@10/MRR@50 point estimates without
  claiming uncomputed secondary-metric significance tests.
- The manuscript now calls the same-group V-SKNN exclusion a conservative
  comparator choice, including its loss of legitimate per-user history.
- Rebuilding the artifact, running the updated audit, compiling, and inspecting
  the final PDF remain author-side checks. Existing v9 bundles are stale.

Additional head retraining with hard negatives, standard-residual center
retraining, more seeds, or a document-retrieval benchmark would require a new
post-hoc study and validation budget. They are not required for the manuscript's
current conditional claims; they remain explicitly untested limitations.
