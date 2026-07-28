#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${KYC_SCALING_DATA_ROOT:-/share/longjunyu/cabi-vla/kyc-scaling-v3}
RUN_ROOT=${KYC_OUTPUT_ROOT:-$DATA_ROOT/runs}
EVAL_ROOT=${KYC_SCALING_EVAL_ROOT:-$DATA_ROOT/eval/stage-b1}
PYTHON=${KYC_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
STEPS=${KYC_SCALING_STEPS:-33000}

stage_b1_summary="$EVAL_ROOT/analysis/stage_b1_scaling_summary.json"
while [[ ! -s "$stage_b1_summary" ]]; do
  if ! tmux has-session -t kyc-camera-generalization-pipeline 2>/dev/null; then
    for catalog_size in 10 45 215 1000; do
      if [[ ! -s "$EVAL_ROOT/analysis/n${catalog_size}/summary.json" ]]; then
        echo "Stage B1 pipeline stopped before scaling analyses completed" >&2
        exit 1
      fi
    done
    "$PYTHON" scripts/cabi_vla/summarize_kyc_scaling_stage_b1.py \
      --analysis-root "$EVAL_ROOT/analysis" \
      --output "$stage_b1_summary"
    break
  fi
  sleep 60
done

selection="$EVAL_ROOT/analysis/stage_b2_selection.json"
if [[ ! -s "$selection" ]]; then
  "$PYTHON" scripts/cabi_vla/select_kyc_scaling_stage_b2.py \
    --stage-b1-summary "$stage_b1_summary" \
    --output "$selection"
fi

train_manager=kyc-scaling-b2-train-manager
if ! tmux has-session -t "$train_manager" 2>/dev/null; then
  tmux new-session -d -s "$train_manager" -c "$REPO_ROOT" \
    "exec bash scripts/cabi_vla/launch_kyc_scaling_stage_b2_train.sh '$selection'"
fi
while tmux has-session -t "$train_manager" 2>/dev/null; do
  sleep 60
done

mapfile -t budgets < <(jq -r '.training_budgets[]' "$selection")
mapfile -t seeds < <(jq -r '.confirmation_seeds[]' "$selection")
for catalog_size in "${budgets[@]}"; do
  for seed in "${seeds[@]}"; do
    for arm in poseaug_control kyc; do
      checkpoint="$RUN_ROOT/kyc_${arm}_scale-n${catalog_size}-fixed-wrist-on_h20_seed${seed}_steps${STEPS}/final_model/model.safetensors"
      if [[ ! -s "$checkpoint" ]]; then
        echo "Stage B2 training ended without checkpoint: $checkpoint" >&2
        exit 1
      fi
    done
  done
done

eval_manager=kyc-scaling-b2-eval-manager
if ! tmux has-session -t "$eval_manager" 2>/dev/null; then
  tmux new-session -d -s "$eval_manager" -c "$REPO_ROOT" \
    "exec bash scripts/cabi_vla/launch_kyc_scaling_stage_b2_eval.sh '$selection'"
fi
while tmux has-session -t "$eval_manager" 2>/dev/null; do
  sleep 60
done

for catalog_size in "${budgets[@]}"; do
  for seed in "${seeds[@]}"; do
    analysis="$EVAL_ROOT/analysis/n${catalog_size}/seed${seed}/summary.json"
    if [[ ! -s "$analysis" ]]; then
      echo "Stage B2 evaluation ended without analysis: $analysis" >&2
      exit 1
    fi
  done
done

summary="$EVAL_ROOT/analysis/stage_b2_summary.json"
if [[ ! -s "$summary" ]]; then
  "$PYTHON" scripts/cabi_vla/summarize_kyc_scaling_stage_b2.py \
    --analysis-root "$EVAL_ROOT/analysis" \
    --selection "$selection" \
    --output "$summary"
fi

echo "KYC scaling Stage B2 complete"
