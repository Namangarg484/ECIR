#!/usr/bin/env bash
set -euo pipefail

# Frozen-center identification experiment requested in the final ECIR review.
# Nothing is overwritten: change TAG to create a new immutable run.
TAG="${FROZEN_RADIUS_TAG:-v1}"
DEVICE="${DEVICE:-mps}"
CPU_THREADS="${CPU_THREADS:-4}"
DETERMINISM="${DETERMINISM:-warn}"
DATA_ROOT="${REVISION_DATA:-artifacts/revision/v1/data}"
PARENT_ROOT="${REVISION_BASE:-artifacts/revision/mps-v1}/runs"
BASE="artifacts/ecir/frozen-radius-${TAG}"
DATASET_LIST="${DATASETS:-talkplay ml1m lastfm amazon_music}"

for dataset in ${DATASET_LIST}; do
  run_dir="${BASE}/runs/${dataset}"
  result_dir="${BASE}/results/${dataset}"
  report_dir="paper/generated/ecir-frozen-radius-${TAG}/${dataset}"

  if [[ -f "${run_dir}/manifest.json" ]]; then
    echo "${dataset}: fit already complete; skipping"
  elif [[ -d "${run_dir}" ]]; then
    echo "${dataset}: resuming interrupted fit"
    python -m experiments.ecir.resume_frozen_radius fit \
      --data "${DATA_ROOT}/${dataset}" \
      --original-run "${PARENT_ROOT}/${dataset}" \
      --out "${run_dir}" \
      --config experiments/ecir/frozen_radius_config.json \
      --device "${DEVICE}" --cpu-threads "${CPU_THREADS}" \
      --interop-threads 1 --determinism "${DETERMINISM}"
  else
    python -m experiments.ecir.frozen_radius fit \
      --data "${DATA_ROOT}/${dataset}" \
      --original-run "${PARENT_ROOT}/${dataset}" \
      --out "${run_dir}" \
      --config experiments/ecir/frozen_radius_config.json \
      --device "${DEVICE}" --cpu-threads "${CPU_THREADS}" \
      --interop-threads 1 --determinism "${DETERMINISM}"
  fi

  if [[ -f "${result_dir}/manifest.json" ]]; then
    echo "${dataset}: test already complete; skipping"
  elif [[ -d "${result_dir}" ]]; then
    echo "${dataset}: resuming interrupted test"
    python -m experiments.ecir.resume_frozen_radius test \
      --data "${DATA_ROOT}/${dataset}" \
      --original-run "${PARENT_ROOT}/${dataset}" \
      --run "${run_dir}" \
      --out "${result_dir}" \
      --device "${DEVICE}" --cpu-threads "${CPU_THREADS}" \
      --interop-threads 1 --determinism "${DETERMINISM}"
  else
    python -m experiments.ecir.frozen_radius test \
      --data "${DATA_ROOT}/${dataset}" \
      --original-run "${PARENT_ROOT}/${dataset}" \
      --run "${run_dir}" \
      --out "${result_dir}" \
      --device "${DEVICE}" --cpu-threads "${CPU_THREADS}" \
      --interop-threads 1 --determinism "${DETERMINISM}"
  fi

  if [[ -f "${report_dir}/manifest.json" ]]; then
    echo "${dataset}: report already complete; skipping"
  elif [[ -e "${report_dir}" ]]; then
    echo "ERROR: incomplete report directory exists: ${report_dir}" >&2
    echo "Move it aside, then rerun this command." >&2
    exit 1
  else
    python -m experiments.ecir.frozen_radius report \
      --run "${run_dir}" \
      --results "${result_dir}" \
      --out "${report_dir}" \
      --bootstrap 10000
  fi
done
