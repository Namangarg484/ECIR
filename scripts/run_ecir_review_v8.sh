#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. No training is performed by this script.
# Same command resumes checksum-verified inference cells after an interruption.
REVIEW_TAG="${REVIEW_TAG:-v8}"
REVIEW_STAGE="${1:-all}"  # select | report | redundancy | all
REVIEW_DEVICE="${DEVICE:-mps}"
REVIEW_THREADS="${CPU_THREADS:-4}"
REVIEW_DETERMINISM="${DETERMINISM:-warn}"
REVIEW_DATASETS="${DATASETS:-talkplay ml1m lastfm amazon_music}"
REVIEW_BASE="artifacts/ecir/review-${REVIEW_TAG}"
REVIEW_REPORT="paper/generated/ecir-review-${REVIEW_TAG}"

case "${REVIEW_STAGE}" in
  select|report|redundancy|all) ;;
  *) echo "Usage: bash scripts/run_ecir_review_v8.sh [select|report|redundancy|all]" >&2; exit 2 ;;
esac

if [[ "${REVIEW_STAGE}" == select || "${REVIEW_STAGE}" == all ]]; then
  # Seal validation selections for all datasets before examining new contrasts.
  for dataset in ${REVIEW_DATASETS}; do
    python -m experiments.ecir.matched_radius_selection select \
      --data "artifacts/revision/v1/data/${dataset}" \
      --original-run "artifacts/revision/mps-v1/runs/${dataset}" \
      --run "artifacts/ecir/frozen-radius-v1/runs/${dataset}" \
      --out "${REVIEW_BASE}/matched-selection/${dataset}" \
      --device "${REVIEW_DEVICE}" --cpu-threads "${REVIEW_THREADS}" \
      --interop-threads 1 --determinism "${REVIEW_DETERMINISM}"
  done
fi

if [[ "${REVIEW_STAGE}" == report || "${REVIEW_STAGE}" == all ]]; then
  for dataset in ${REVIEW_DATASETS}; do
    python -m experiments.ecir.matched_radius_selection report \
      --run "artifacts/ecir/frozen-radius-v1/runs/${dataset}" \
      --selection "${REVIEW_BASE}/matched-selection/${dataset}" \
      --results "artifacts/ecir/frozen-radius-v1/results/${dataset}" \
      --out "${REVIEW_REPORT}/matched-selection/${dataset}" --bootstrap 10000
  done
fi

if [[ "${REVIEW_STAGE}" == redundancy || "${REVIEW_STAGE}" == all ]]; then
  python -m experiments.ecir.probe_redundancy run \
    --data artifacts/revision/v1/data/talkplay \
    --original-run artifacts/revision/mps-v1/runs/talkplay \
    --stochastic-run artifacts/ecir/mps-v2/stochastic-runs/talkplay \
    --reference-sweep artifacts/ecir/direction-sensitivity-v1/results \
    --out "${REVIEW_BASE}/probe-redundancy/talkplay" \
    --device "${REVIEW_DEVICE}" --cpu-threads "${REVIEW_THREADS}" \
    --interop-threads 1 --determinism "${REVIEW_DETERMINISM}"
  python -m experiments.ecir.probe_redundancy report \
    --results "${REVIEW_BASE}/probe-redundancy/talkplay" \
    --out "${REVIEW_REPORT}/probe-redundancy/talkplay"
fi
