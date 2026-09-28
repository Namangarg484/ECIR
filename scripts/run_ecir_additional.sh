#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

stage="${1:-}"
if [[ -z "$stage" ]]; then
  echo "Usage: bash scripts/run_ecir_additional.sh {session-knn-fit|session-knn-test|diagnostics|efficiency|report|direction-sensitivity-run|direction-sensitivity-report|direction-sensitivity|all}" >&2
  exit 2
fi

datasets=(talkplay ml1m lastfm amazon_music)
methods=(learned_centroid learned_fixed full)
data_root="artifacts/revision/v1/data"
original_run_root="artifacts/revision/mps-v1/runs"
ecir_tag="${ECIR_TAG:-mps-v2}"
existing_root="artifacts/ecir/$ecir_tag"
additional_tag="${ECIR_ADDITIONAL_TAG:-v1}"
additional_root="artifacts/ecir/additional-$additional_tag"
report_out="paper/generated/ecir-additional-$additional_tag"
sensitivity_tag="${ECIR_SENSITIVITY_TAG:-v1}"
sensitivity_root="artifacts/ecir/direction-sensitivity-$sensitivity_tag"
sensitivity_report="paper/generated/ecir-direction-sensitivity-$sensitivity_tag"
config="experiments/ecir/additional_config.json"
cpu_runtime=(--cpu-threads 4 --interop-threads 1 --determinism strict)
mps_runtime=(--device mps --cpu-threads 4 --interop-threads 1 --determinism warn)

export PYTHONHASHSEED=0
export PYTORCH_ENABLE_MPS_FALLBACK=0
export PYTHONUNBUFFERED=1

session_knn_fit() {
  for dataset in "${datasets[@]}"; do
    python -m experiments.ecir.session_knn fit \
      --data "$data_root/$dataset" \
      --out "$additional_root/session-knn-runs/$dataset" \
      --config "$config" \
      "${cpu_runtime[@]}"
  done
}

session_knn_test() {
  for dataset in "${datasets[@]}"; do
    python -m experiments.ecir.session_knn test \
      --data "$data_root/$dataset" \
      --run "$additional_root/session-knn-runs/$dataset" \
      --out "$additional_root/session-knn-results/$dataset" \
      "${cpu_runtime[@]}"
  done
}

diagnostics() {
  for dataset in "${datasets[@]}"; do
    python -m experiments.ecir.kappa_diagnostics \
      --data "$data_root/$dataset" \
      --original-run "$original_run_root/$dataset" \
      --stochastic-run "$existing_root/stochastic-runs/$dataset" \
      --out "$additional_root/diagnostics/$dataset" \
      --config "$config" \
      "${mps_runtime[@]}"
  done
}

efficiency() {
  for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
      python -m experiments.ecir.efficiency \
        --data "$data_root/$dataset" \
        --original-run "$original_run_root/$dataset" \
        --stochastic-run "$existing_root/stochastic-runs/$dataset" \
        --method "$method" \
        --out "$additional_root/efficiency/$dataset/$method" \
        --config "$config" \
        "${mps_runtime[@]}"
    done
  done
}

make_report() {
  python -m experiments.ecir.additional_report \
    --session-knn-results \
      "$additional_root/session-knn-results/talkplay" \
      "$additional_root/session-knn-results/ml1m" \
      "$additional_root/session-knn-results/lastfm" \
      "$additional_root/session-knn-results/amazon_music" \
    --stochastic-results \
      "$existing_root/stochastic-results/talkplay" \
      "$existing_root/stochastic-results/ml1m" \
      "$existing_root/stochastic-results/lastfm" \
      "$existing_root/stochastic-results/amazon_music" \
    --efficiency-results \
      "$additional_root/efficiency/talkplay/learned_centroid" \
      "$additional_root/efficiency/talkplay/learned_fixed" \
      "$additional_root/efficiency/talkplay/full" \
      "$additional_root/efficiency/ml1m/learned_centroid" \
      "$additional_root/efficiency/ml1m/learned_fixed" \
      "$additional_root/efficiency/ml1m/full" \
      "$additional_root/efficiency/lastfm/learned_centroid" \
      "$additional_root/efficiency/lastfm/learned_fixed" \
      "$additional_root/efficiency/lastfm/full" \
      "$additional_root/efficiency/amazon_music/learned_centroid" \
      "$additional_root/efficiency/amazon_music/learned_fixed" \
      "$additional_root/efficiency/amazon_music/full" \
    --diagnostic-results \
      "$additional_root/diagnostics/talkplay" \
      "$additional_root/diagnostics/ml1m" \
      "$additional_root/diagnostics/lastfm" \
      "$additional_root/diagnostics/amazon_music" \
    --out "$report_out" \
    --bootstrap 10000
}

direction_sensitivity_run() {
  python -m experiments.ecir.query_direction_sensitivity run \
    --data "$data_root/talkplay" \
    --original-run "$original_run_root/talkplay" \
    --stochastic-run "$existing_root/stochastic-runs/talkplay" \
    --out "$sensitivity_root/results" \
    --config "$config" \
    "${mps_runtime[@]}"
}

direction_sensitivity_report() {
  python -m experiments.ecir.query_direction_sensitivity report \
    --results "$sensitivity_root/results" \
    --out "$sensitivity_report" \
    --bootstrap 10000
}

case "$stage" in
  session-knn-fit) session_knn_fit ;;
  session-knn-test) session_knn_test ;;
  diagnostics) diagnostics ;;
  efficiency) efficiency ;;
  report) make_report ;;
  direction-sensitivity-run) direction_sensitivity_run ;;
  direction-sensitivity-report) direction_sensitivity_report ;;
  direction-sensitivity)
    direction_sensitivity_run
    direction_sensitivity_report
    ;;
  all)
    session_knn_fit
    session_knn_test
    diagnostics
    efficiency
    make_report
    ;;
  *)
    echo "Unknown stage: $stage" >&2
    exit 2
    ;;
esac
