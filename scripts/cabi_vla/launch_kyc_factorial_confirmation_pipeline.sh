#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${KYC_SCALING_DATA_ROOT:-/share/longjunyu/cabi-vla/kyc-scaling-v3}
FACTOR_EVAL_ROOT=${KYC_FACTORIAL_EVAL_ROOT:-$DATA_ROOT/eval/factorial}
SCALING_EVAL_ROOT=${KYC_SCALING_EVAL_ROOT:-$DATA_ROOT/eval/stage-b1}
FOV_ROOT=${KYC_BOUNDARY_FOV_ROOT:-/share/longjunyu/cabi-vla/camera-viewpoint-study-v2/fov_guard_test40-49_v5}
PYTHON=${KYC_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}

stage_b1_summary="$SCALING_EVAL_ROOT/analysis/stage_b1_scaling_summary.json"
while [[ ! -s "$stage_b1_summary" ]]; do
  if ! tmux has-session -t kyc-factorial-pipeline 2>/dev/null; then
    echo "factorial pipeline stopped before Stage B1 budget selection" >&2
    exit 1
  fi
  sleep 60
done
budget=$(jq -r '.factorial_budget_selection.selected_budget // empty' "$stage_b1_summary")
if [[ -z "$budget" ]]; then
  echo "factorial budget is unavailable because Stage B1 baseline is invalid" >&2
  exit 1
fi
seed41_summary="$FACTOR_EVAL_ROOT/n${budget}/analysis/seed41/summary.json"
while [[ ! -s "$seed41_summary" ]]; do
  if ! tmux has-session -t kyc-factorial-pipeline 2>/dev/null; then
    echo "seed-41 factorial pipeline stopped before its summary" >&2
    exit 1
  fi
  sleep 60
done

train_manager="kyc-factorial-confirm-train-n${budget}"
if ! tmux has-session -t "$train_manager" 2>/dev/null; then
  tmux new-session -d -s "$train_manager" -c "$REPO_ROOT" \
    "exec bash scripts/cabi_vla/launch_kyc_factorial_confirmation_train.sh \
      '$seed41_summary'"
fi
while tmux has-session -t "$train_manager" 2>/dev/null; do
  sleep 60
done

eval_manager="kyc-factorial-confirm-eval-n${budget}"
if ! tmux has-session -t "$eval_manager" 2>/dev/null; then
  tmux new-session -d -s "$eval_manager" -c "$REPO_ROOT" \
    "exec bash scripts/cabi_vla/launch_kyc_factorial_confirmation_eval.sh \
      '$seed41_summary'"
fi
while tmux has-session -t "$eval_manager" 2>/dev/null; do
  sleep 60
done

fov_json=("$FOV_ROOT"/*.json)
output="$FACTOR_EVAL_ROOT/n${budget}/analysis/confirmed/summary.json"
if [[ ! -s "$output" ]]; then
  "$PYTHON" scripts/cabi_vla/summarize_kyc_factorial_confirmed.py \
    --seed41-summary "$seed41_summary" \
    --factor-eval-root "$FACTOR_EVAL_ROOT" \
    --scaling-eval-root "$SCALING_EVAL_ROOT" \
    --fov-json "${fov_json[@]}" \
    --output "$output"
fi

echo "KYC multi-seed scene-cue by wrist confirmation complete: n=$budget"
