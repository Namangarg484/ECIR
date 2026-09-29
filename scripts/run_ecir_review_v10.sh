#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. No model fitting or checkpoint selection.
V10_STAGE="${1:-all}"
V10_DEVICE="${DEVICE:-mps}"
V10_THREADS="${CPU_THREADS:-4}"

case "${V10_STAGE}" in
  coverage|validation|report|all) ;;
  *) echo "Usage: bash scripts/run_ecir_review_v10.sh [coverage|validation|report|all]" >&2; exit 2 ;;
esac

if [[ "${V10_STAGE}" == coverage || "${V10_STAGE}" == all ]]; then
  python -m experiments.ecir.target_support \
    --out paper/generated/ecir-review-v10/target-support
fi

if [[ "${V10_STAGE}" == validation || "${V10_STAGE}" == all ]]; then
  python -m experiments.ecir.validation_probe_redundancy run \
    --data artifacts/revision/v1/data/talkplay \
    --original-run artifacts/revision/mps-v1/runs/talkplay \
    --stochastic-run artifacts/ecir/mps-v2/stochastic-runs/talkplay \
    --reference-sweep artifacts/ecir/direction-sensitivity-v1/results \
    --out artifacts/ecir/review-v10/validation-probe-redundancy/talkplay \
    --device "${V10_DEVICE}" --cpu-threads "${V10_THREADS}" \
    --interop-threads 1 --determinism warn
fi

if [[ "${V10_STAGE}" == report || "${V10_STAGE}" == all ]]; then
  python -m experiments.ecir.validation_probe_redundancy report \
    --results artifacts/ecir/review-v10/validation-probe-redundancy/talkplay \
    --test-results artifacts/ecir/review-v8/probe-redundancy/talkplay \
    --test-report paper/generated/ecir-review-v8/probe-redundancy/talkplay \
    --out paper/generated/ecir-review-v10/validation-probe-redundancy/talkplay
fi
