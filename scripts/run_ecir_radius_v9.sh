#!/usr/bin/env bash
set -euo pipefail

# New output namespace only. This DOES train radius heads; centers remain frozen.
# Run one copy at a time. Every command verifies and resumes completed work.
RADIUS_STAGE="${1:-all}"
RADIUS_TAG="${RADIUS_TAG:-v9}"
RADIUS_DEVICE="${DEVICE:-mps}"
RADIUS_THREADS="${CPU_THREADS:-4}"
RADIUS_DETERMINISM="${DETERMINISM:-warn}"
RADIUS_DATASETS="${DATASETS:-talkplay ml1m lastfm amazon_music}"
RADIUS_BASE="artifacts/ecir/radius-${RADIUS_TAG}"
RADIUS_REPORT="paper/generated/ecir-radius-${RADIUS_TAG}"

case "${RADIUS_STAGE}" in
  grid|fit|test|report|all) ;;
  *) echo "Usage: bash scripts/run_ecir_radius_v9.sh [grid|fit|test|report|all]" >&2; exit 2 ;;
esac

for phase in grid fit test report; do
  if [[ "${RADIUS_STAGE}" != all && "${RADIUS_STAGE}" != "${phase}" ]]; then
    continue
  fi
  # In all mode, complete this phase across ALL datasets before proceeding.
  # No new test evaluation starts until all head and radius selections are sealed.
  for dataset in ${RADIUS_DATASETS}; do
    echo "radius-${RADIUS_TAG}: ${phase} / ${dataset}"
    run_dir="${RADIUS_BASE}/runs/${dataset}"
    result_dir="${RADIUS_BASE}/results/${dataset}"
    if [[ "${phase}" == report ]]; then
      python -m experiments.ecir.radius_v9 report \
        --run "${run_dir}" --results "${result_dir}" \
        --out "${RADIUS_REPORT}/${dataset}" --bootstrap 10000
      continue
    fi
    args=(
      --data "artifacts/revision/v1/data/${dataset}"
      --original-run "artifacts/revision/mps-v1/runs/${dataset}"
      --previous-run "artifacts/ecir/frozen-radius-v1/runs/${dataset}"
      --device "${RADIUS_DEVICE}" --cpu-threads "${RADIUS_THREADS}"
      --interop-threads 1 --determinism "${RADIUS_DETERMINISM}"
    )
    if [[ "${phase}" == test ]]; then
      args+=(--run "${run_dir}" --out "${result_dir}")
    else
      args+=(--config experiments/ecir/radius_v9_config.json --out "${run_dir}")
    fi
    python -m experiments.ecir.radius_v9 "${phase}" "${args[@]}"
  done
done
