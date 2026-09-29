#!/usr/bin/env bash
# Invoke with bash from the repository root. Never called automatically.
set -euo pipefail
stage="${1:-}"
revision_base="${REVISION_BASE:-artifacts/revision/v1}"
revision_device="${REVISION_DEVICE:-cpu}"
revision_config="${REVISION_CONFIG:-experiments/revision/config.json}"
revision_paper="${REVISION_PAPER:-paper/generated/revision}"
revision_data="${REVISION_DATA:-$revision_base/data}"
revision_threads="${REVISION_CPU_THREADS:-4}"
revision_interop="${REVISION_INTEROP_THREADS:-1}"
revision_determinism="${REVISION_DETERMINISM:-strict}"
runtime_args=(--device "$revision_device" --cpu-threads "$revision_threads" \
  --interop-threads "$revision_interop" --determinism "$revision_determinism")
if [[ ! -f experiments/revision/config.json ]]; then
  echo "Run this script from the NeuraIPS repository root." >&2
  exit 2
fi
case "$stage" in
  prepare)
    for dataset in talkplay ml1m lastfm amazon_music; do
      python -m experiments.revision.prepare --dataset "$dataset" \
        --out "$revision_data/$dataset" --embedding-device "$revision_device"
    done
    ;;
  fit)
    for dataset in talkplay ml1m lastfm amazon_music; do
      python -m experiments.revision.run fit --data "$revision_data/$dataset" \
        --out "$revision_base/runs/$dataset" --config "$revision_config" "${runtime_args[@]}"
    done
    ;;
  test)
    for dataset in talkplay ml1m lastfm amazon_music; do
      python -m experiments.revision.run test --data "$revision_data/$dataset" \
        --run "$revision_base/runs/$dataset" --out "$revision_base/results/$dataset" "${runtime_args[@]}"
    done
    ;;
  report)
    python -m experiments.revision.report --results \
      "$revision_base/results/talkplay" "$revision_base/results/ml1m" \
      "$revision_base/results/lastfm" "$revision_base/results/amazon_music" \
      --out "$revision_paper"
    ;;
  *)
    echo "Usage: bash scripts/run_revision.sh {prepare|fit|test|report}" >&2
    exit 2
    ;;
esac
