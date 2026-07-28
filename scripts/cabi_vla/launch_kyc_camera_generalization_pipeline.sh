#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${KYC_SCALING_DATA_ROOT:-/share/longjunyu/cabi-vla/kyc-scaling-v3}
RUN_ROOT=${KYC_OUTPUT_ROOT:-$DATA_ROOT/runs}
EVAL_ROOT=${KYC_SCALING_EVAL_ROOT:-$DATA_ROOT/eval/stage-b1}
OFFICIAL_ROOT=${KYC_OFFICIAL_OUTPUT_ROOT:-/share/longjunyu/kyc-official-data/runs}
PYTHON=${KYC_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
SEED=${KYC_SCALING_SEED:-41}
STEPS=${KYC_SCALING_STEPS:-33000}

view_path() {
  local catalog_size=$1
  printf '%s/views/libero-bind-kyc-n%s-fixed-h20/manifest.json' \
    "$DATA_ROOT" "$catalog_size"
}

checkpoint_path() {
  local catalog_size=$1
  local arm=$2
  local tag="scale-n${catalog_size}-fixed-wrist-on"
  printf '%s/kyc_%s_%s_h20_seed%s_steps%s/final_model/model.safetensors' \
    "$RUN_ROOT" "$arm" "$tag" "$SEED" "$STEPS"
}

evaluation_path() {
  local catalog_size=$1
  local arm=$2
  printf '%s/n%s/n%s-%s-s%s-fixed-wrist-on/camera_sweep_test.json' \
    "$EVAL_ROOT" "$catalog_size" "$catalog_size" "$arm" "$SEED"
}

wait_for_data_view() {
  local catalog_size=$1
  local parent_session="kyc-data-build-n${catalog_size}"
  local manifest
  manifest=$(view_path "$catalog_size")
  while [[ ! -s "$manifest" ]]; do
    if ! tmux has-session -t "$parent_session" 2>/dev/null; then
      echo "data generation stopped without a manifest: $parent_session" >&2
      exit 1
    fi
    sleep 30
  done
}

wait_for_training_wave() {
  local specifications=("$@")
  while true; do
    local complete=1
    for specification in "${specifications[@]}"; do
      read -r catalog_size arm <<<"$specification"
      if [[ ! -s "$(checkpoint_path "$catalog_size" "$arm")" ]]; then
        complete=0
        break
      fi
    done
    if [[ "$complete" == 1 ]]; then
      return
    fi
    if ! tmux has-session -t kyc-scaling-b1-manager 2>/dev/null; then
      echo "Stage B1 manager stopped before all checkpoints completed" >&2
      exit 1
    fi
    sleep 60
  done
}

wait_for_official_matrix() {
  while true; do
    local complete=1
    for arm in image kyc; do
      for seed in 0 1 2; do
        checkpoint="$OFFICIAL_ROOT/official_act_lift_randomized_${arm}_seed${seed}/epoch_20000.pth"
        if [[ ! -s "$checkpoint" ]]; then
          complete=0
          session="kyc-official-${arm}-s${seed}"
          if ! tmux has-session -t "$session" 2>/dev/null; then
            echo "official KYC run stopped without checkpoint: $session" >&2
            exit 1
          fi
        fi
      done
    done
    [[ "$complete" == 1 ]] && return
    sleep 60
  done
}

wait_for_evaluation() {
  local catalog_size=$1
  local arm=$2
  local output
  local session="kyc-b1-eval-n${catalog_size}-${arm}-s${SEED}"
  output=$(evaluation_path "$catalog_size" "$arm")
  while true; do
    if [[ -s "$output" ]] && jq -e \
      '.status == "complete" and (.rows | length) == 520' \
      "$output" >/dev/null; then
      return
    fi
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "evaluation stopped without a complete result: $session" >&2
      exit 1
    fi
    sleep 30
  done
}

for catalog_size in 10 45 215 1000; do
  wait_for_data_view "$catalog_size"
done

validation="$DATA_ROOT/diagnostics/fixed_scaling_views_validation.json"
validation_args=()
for catalog_size in 10 45 215 1000; do
  validation_args+=(
    --view "$catalog_size=fixed=$DATA_ROOT/views/libero-bind-kyc-n${catalog_size}-fixed-h20"
  )
done
if [[ ! -s "$validation" ]]; then
  "$PYTHON" scripts/cabi_vla/validate_kyc_scaling_views.py \
    "${validation_args[@]}" \
    --output "$validation"
else
  "$PYTHON" scripts/cabi_vla/validate_kyc_scaling_views.py \
    "${validation_args[@]}" >/dev/null
fi

if ! tmux has-session -t kyc-scaling-b1-manager 2>/dev/null; then
  tmux new-session -d -s kyc-scaling-b1-manager -c "$REPO_ROOT" \
    "exec bash scripts/cabi_vla/launch_kyc_scaling_stage_b1.sh"
fi

wave_one=(
  "10 poseaug_control"
  "10 kyc"
  "10 poseaug_rgb"
  "45 poseaug_control"
  "45 kyc"
  "45 poseaug_rgb"
  "215 poseaug_control"
  "215 kyc"
)
all_training=(
  "${wave_one[@]}"
  "1000 poseaug_control"
  "1000 kyc"
)
wait_for_training_wave "${wave_one[@]}"

KYC_OFFICIAL_GPU_IDS=2,3,4,5,6,7 \
  bash scripts/cabi_vla/launch_kyc_official_act_matrix.sh

wait_for_training_wave "${all_training[@]}"

for specification in "1000 poseaug_control 0" "1000 kyc 1"; do
  read -r catalog_size arm gpu_id <<<"$specification"
  output=$(evaluation_path "$catalog_size" "$arm")
  session="kyc-b1-eval-n${catalog_size}-${arm}-s${SEED}"
  if [[ ! -s "$output" ]] && ! tmux has-session -t "$session" 2>/dev/null; then
    tmux new-session -d -s "$session" -c "$REPO_ROOT" \
      "exec bash scripts/cabi_vla/run_kyc_scaling_eval.sh \
        '$catalog_size' '$arm' '$SEED' '$gpu_id'"
  fi
done
wait_for_evaluation 1000 poseaug_control
wait_for_evaluation 1000 kyc
wait_for_official_matrix

if ! tmux has-session -t kyc-scaling-b1-eval-manager 2>/dev/null; then
  tmux new-session -d -s kyc-scaling-b1-eval-manager -c "$REPO_ROOT" \
    "exec bash scripts/cabi_vla/launch_kyc_scaling_stage_b1_eval.sh"
fi

for catalog_size in 10 45 215 1000; do
  while [[ ! -s "$EVAL_ROOT/analysis/n${catalog_size}/summary.json" ]]; do
    if ! tmux has-session -t kyc-scaling-b1-eval-manager 2>/dev/null; then
      echo "Stage B1 evaluation manager stopped before analysis completed" >&2
      exit 1
    fi
    sleep 60
  done
done

scaling_summary="$EVAL_ROOT/analysis/stage_b1_scaling_summary.json"
if [[ ! -s "$scaling_summary" ]]; then
  "$PYTHON" scripts/cabi_vla/summarize_kyc_scaling_stage_b1.py \
    --analysis-root "$EVAL_ROOT/analysis" \
    --output "$scaling_summary"
fi

echo "KYC camera generalization scaling and official positive control complete"
