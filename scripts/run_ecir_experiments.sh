#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

stage="${1:-}"
if [[ -z "$stage" ]]; then
  echo "Usage: bash scripts/run_ecir_experiments.sh {baselines-fit|baselines-test|stochastic-fit|stochastic-test|report|all}" >&2
  exit 2
fi

datasets=(talkplay ml1m lastfm amazon_music)
data_root="artifacts/revision/v1/data"
original_run_root="artifacts/revision/mps-v1/runs"
original_result_root="artifacts/revision/mps-v1/results"
run_tag="${ECIR_TAG:-mps-v2}"
ecir_root="artifacts/ecir/$run_tag"
report_out="paper/generated/ecir-$run_tag"
runtime=(--device mps --cpu-threads 4 --interop-threads 1 --determinism warn)

export PYTHONHASHSEED=0
export PYTORCH_ENABLE_MPS_FALLBACK=0
export PYTHONUNBUFFERED=1

baselines_fit() {
  for dataset in "${datasets[@]}"; do
    python -m experiments.ecir.baselines fit \
      --data "$data_root/$dataset" \
      --out "$ecir_root/baseline-runs/$dataset" \
      --config experiments/ecir/config.json \
      "${runtime[@]}"
  done
}

baselines_test() {
  for dataset in "${datasets[@]}"; do
    python -m experiments.ecir.baselines test \
      --data "$data_root/$dataset" \
      --run "$ecir_root/baseline-runs/$dataset" \
      --out "$ecir_root/baseline-results/$dataset" \
      "${runtime[@]}"
  done
}

stochastic_fit() {
  for dataset in "${datasets[@]}"; do
    python -m experiments.ecir.stochastic_eval fit \
      --data "$data_root/$dataset" \
      --original-run "$original_run_root/$dataset" \
      --out "$ecir_root/stochastic-runs/$dataset" \
      --config experiments/ecir/config.json \
      "${runtime[@]}"
  done
}

stochastic_test() {
  for dataset in "${datasets[@]}"; do
    python -m experiments.ecir.stochastic_eval test \
      --data "$data_root/$dataset" \
      --original-run "$original_run_root/$dataset" \
      --run "$ecir_root/stochastic-runs/$dataset" \
      --out "$ecir_root/stochastic-results/$dataset" \
      "${runtime[@]}"
  done
}

make_report() {
  python -m experiments.ecir.report \
    --original-results \
      "$original_result_root/talkplay" \
      "$original_result_root/ml1m" \
      "$original_result_root/lastfm" \
      "$original_result_root/amazon_music" \
    --baseline-results \
      "$ecir_root/baseline-results/talkplay" \
      "$ecir_root/baseline-results/ml1m" \
      "$ecir_root/baseline-results/lastfm" \
      "$ecir_root/baseline-results/amazon_music" \
    --stochastic-results \
      "$ecir_root/stochastic-results/talkplay" \
      "$ecir_root/stochastic-results/ml1m" \
      "$ecir_root/stochastic-results/lastfm" \
      "$ecir_root/stochastic-results/amazon_music" \
    --out "$report_out"
}

case "$stage" in
  baselines-fit) baselines_fit ;;
  baselines-test) baselines_test ;;
  stochastic-fit) stochastic_fit ;;
  stochastic-test) stochastic_test ;;
  report) make_report ;;
  all)
    baselines_fit
    baselines_test
    stochastic_fit
    stochastic_test
    make_report
    ;;
  *)
    echo "Unknown stage: $stage" >&2
    exit 2
    ;;
esac
