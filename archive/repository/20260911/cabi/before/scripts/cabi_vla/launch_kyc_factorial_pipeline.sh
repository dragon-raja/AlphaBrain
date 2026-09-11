#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${KYC_SCALING_DATA_ROOT:-/share/longjunyu/cabi-vla/kyc-scaling-v3}
RUN_ROOT=${KYC_OUTPUT_ROOT:-$DATA_ROOT/runs}
SCALING_EVAL_ROOT=${KYC_SCALING_EVAL_ROOT:-$DATA_ROOT/eval/stage-b1}
FACTORIAL_EVAL_ROOT=${KYC_FACTORIAL_EVAL_ROOT:-$DATA_ROOT/eval/factorial}
FOV_ROOT=${KYC_BOUNDARY_FOV_ROOT:-/share/longjunyu/cabi-vla/camera-viewpoint-study-v2/fov_guard_test40-49_v5}
PYTHON=${KYC_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
SEED=${KYC_FACTORIAL_SEED:-41}
STEPS=${KYC_SCALING_STEPS:-33000}

for catalog_size in 10 45 215 1000; do
  analysis="$SCALING_EVAL_ROOT/analysis/n${catalog_size}/summary.json"
  while [[ ! -s "$analysis" ]]; do
    if ! tmux has-session -t kyc-camera-generalization-pipeline 2>/dev/null; then
      echo "scaling pipeline stopped before Stage B1 analysis completed" >&2
      exit 1
    fi
    sleep 60
  done
done

scaling_summary="$SCALING_EVAL_ROOT/analysis/stage_b1_scaling_summary.json"
if [[ ! -s "$scaling_summary" ]]; then
  "$PYTHON" scripts/cabi_vla/summarize_kyc_scaling_stage_b1.py \
    --analysis-root "$SCALING_EVAL_ROOT/analysis" \
    --output "$scaling_summary"
fi
budget=$(jq -r '.factorial_budget_selection.selected_budget // empty' "$scaling_summary")
if [[ -z "$budget" ]]; then
  echo "Stage B1 baseline is invalid; factorial training is not permitted" >&2
  exit 1
fi

stage_b2_summary="$SCALING_EVAL_ROOT/analysis/stage_b2_summary.json"
while [[ ! -s "$stage_b2_summary" ]]; do
  if ! tmux has-session -t kyc-scaling-stage-b2-pipeline 2>/dev/null; then
    echo "Stage B2 confirmation pipeline stopped before its summary" >&2
    exit 1
  fi
  sleep 60
done

cue_cell="n${budget}-cue"
cue_view="$DATA_ROOT/views/libero-bind-kyc-${cue_cell}-h20"
cue_session="kyc-data-build-${cue_cell}"
if [[ ! -s "$cue_view/manifest.json" ]]; then
  if [[ -e "$cue_view" ]]; then
    echo "incomplete cue-randomized data view requires inspection: $cue_view" >&2
    exit 1
  fi
  if ! tmux has-session -t "$cue_session" 2>/dev/null; then
    tmux new-session -d -s "$cue_session" -c "$REPO_ROOT" \
      "exec bash scripts/cabi_vla/build_kyc_scaling_data_view.sh \
        '$cue_cell' '$budget' cue_randomized 0"
  fi
fi
while [[ ! -s "$cue_view/manifest.json" ]]; do
  if ! tmux has-session -t "$cue_session" 2>/dev/null; then
    echo "cue-randomized data generation stopped without a manifest" >&2
    exit 1
  fi
  sleep 30
done

fixed_view="$DATA_ROOT/views/libero-bind-kyc-n${budget}-fixed-h20"
validation="$DATA_ROOT/diagnostics/factorial-n${budget}-views-validation.json"
validation_args=(
  --view "$budget=fixed=$fixed_view"
  --view "$budget=cue_randomized=$cue_view"
)
if [[ ! -s "$validation" ]]; then
  "$PYTHON" scripts/cabi_vla/validate_kyc_scaling_views.py \
    "${validation_args[@]}" \
    --output "$validation"
else
  "$PYTHON" scripts/cabi_vla/validate_kyc_scaling_views.py \
    "${validation_args[@]}" >/dev/null
fi

train_manager="kyc-factorial-train-n${budget}-s${SEED}"
if ! tmux has-session -t "$train_manager" 2>/dev/null; then
  tmux new-session -d -s "$train_manager" -c "$REPO_ROOT" \
    "exec bash scripts/cabi_vla/launch_kyc_factorial_train.sh '$budget' '$SEED'"
fi

factorial_checkpoint() {
  local scene=$1
  local wrist=$2
  local arm=$3
  local tag="factorial-n${budget}-${scene}-wrist-${wrist}"
  printf '%s/kyc_%s_%s_h20_seed%s_steps%s/final_model/model.safetensors' \
    "$RUN_ROOT" "$arm" "$tag" "$SEED" "$STEPS"
}

for specification in \
  "fixed off poseaug_control" \
  "fixed off kyc" \
  "cue_randomized on poseaug_control" \
  "cue_randomized on kyc" \
  "cue_randomized off poseaug_control" \
  "cue_randomized off kyc"; do
  read -r scene wrist arm <<<"$specification"
  checkpoint=$(factorial_checkpoint "$scene" "$wrist" "$arm")
  while [[ ! -s "$checkpoint" ]]; do
    if ! tmux has-session -t "$train_manager" 2>/dev/null; then
      echo "factorial training manager stopped before completion" >&2
      exit 1
    fi
    sleep 60
  done
done

eval_manager="kyc-factorial-eval-n${budget}-s${SEED}"
if ! tmux has-session -t "$eval_manager" 2>/dev/null; then
  tmux new-session -d -s "$eval_manager" -c "$REPO_ROOT" \
    "exec bash scripts/cabi_vla/launch_kyc_factorial_eval.sh '$budget' '$SEED'"
fi
while tmux has-session -t "$eval_manager" 2>/dev/null; do
  sleep 60
done

fov_json=("$FOV_ROOT"/*.json)
factorial_summary="$FACTORIAL_EVAL_ROOT/n${budget}/analysis/seed${SEED}/summary.json"
if [[ ! -s "$factorial_summary" ]]; then
  "$PYTHON" scripts/cabi_vla/summarize_kyc_factorial.py \
    --evaluation-root "$FACTORIAL_EVAL_ROOT" \
    --fov-json "${fov_json[@]}" \
    --budget "$budget" \
    --seed "$SEED" \
    --output "$factorial_summary"
fi

short_name() {
  local scene=$1
  local arm=$2
  local scene_tag
  local arm_tag
  case "$scene" in
    fixed) scene_tag=fx ;;
    cue_randomized) scene_tag=cue ;;
  esac
  case "$arm" in
    poseaug_control) arm_tag=ctrl ;;
    kyc) arm_tag=kyc ;;
  esac
  printf 'n%s-tr%s-woff-m%s-s%s-ev%s' \
    "$budget" "$scene_tag" "$arm_tag" "$SEED" "$scene_tag"
}

for scene in fixed cue_randomized; do
  for arm in poseaug_control kyc; do
    name=$(short_name "$scene" "$arm")
    run_dir="$FACTORIAL_EVAL_ROOT/n${budget}/$name"
    output_dir="$run_dir/videos_av1"
    if [[ ! -e "$output_dir" ]]; then
      PYTHONPATH="$REPO_ROOT/scripts/cabi_vla:$REPO_ROOT/scripts/fresh_vla" \
        "$PYTHON" scripts/cabi_vla/render_libero_bind_eval_frames.py \
        --evaluation "$run_dir/camera_sweep_test.json" \
        --frame-dir "$run_dir/frames" \
        --output-dir "$output_dir" \
        --codecs av1 \
        --fps 20
    fi
  done
done

echo "KYC seed-${SEED} scene-cue by wrist factorial complete: n=$budget"
