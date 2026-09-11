#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PROTOCOL=${DSOL_ACCEL_PROTOCOL:-/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m0-v1/operational-three-task-scan-v2/constructed_m1_protocol_v1.json}
CHECKPOINT=${DSOL_ACCEL_CHECKPOINT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_broad64-quick-gate-v1_seed41_g8_gb32_steps2000/final_model}
OUTPUT_DIR=${DSOL_ACCEL_OUTPUT_DIR:-/share/longjunyu/alphabrain/experiments/dsol-accel-constructed-v2/broad64-practical-seed41-full}
RENDER_SOURCE_DIR=${DSOL_ACCEL_RENDER_SOURCE_DIR:-$OUTPUT_DIR}
RUNTIME=${DSOL_ACCEL_RUNTIME:-/share/longjunyu/alphabrain/datasets/libero-plus/runtime/LIBERO-plus}
CONFIG_ROOT=${DSOL_ACCEL_CONFIG_ROOT:-/share/longjunyu/alphabrain/envs/libero-plus-runtime-config-v1}
SIM_PYTHON=${DSOL_ACCEL_SIM_PYTHON:-/workspace/envs/fresh-libero/bin/python}
MODEL_PYTHON=${DSOL_ACCEL_MODEL_PYTHON:-/alphabrain/.venv/bin/python}
DEVICE=${DSOL_ACCEL_DEVICE:-cuda:7}
RENDER_GPU=${DSOL_ACCEL_RENDER_GPU:-7}
BATCH_SIZE=${DSOL_ACCEL_BATCH_SIZE:-8}
MAX_STATES=${DSOL_ACCEL_MAX_STATES:-}
SEED=${DSOL_ACCEL_SEED:-20260820}

for required in \
  "$PROTOCOL" \
  "$CHECKPOINT/model.safetensors" \
  "$RUNTIME/libero/libero/bddl_files" \
  "$CONFIG_ROOT" \
  "$SIM_PYTHON" \
  "$MODEL_PYTHON"; do
  [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 2; }
done
[[ "$RENDER_GPU" =~ ^[0-7]$ ]] || { echo "render GPU must be in [0,7]" >&2; exit 2; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "batch size must be positive" >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be a non-negative integer" >&2; exit 2; }
[[ -z "$MAX_STATES" || "$MAX_STATES" =~ ^[1-9][0-9]*$ ]] || {
  echo "max states must be empty or positive" >&2
  exit 2
}
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
  echo "refusing scientific Accel run from a dirty worktree" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/controller.log") 2>&1
printf 'constructed_accel_start=%s git_commit=%s\n' \
  "$(date -u +%FT%TZ)" "$(git -C "$REPO_ROOT" rev-parse HEAD)"

max_state_args=()
if [[ -n "$MAX_STATES" ]]; then
  max_state_args=(--max-states "$MAX_STATES")
fi

if [[ "$RENDER_SOURCE_DIR" == "$OUTPUT_DIR" ]]; then
  PYTHONPATH="$REPO_ROOT:/projects/openpi/packages/openpi-client/src:$REPO_ROOT/scripts/cabi_vla:$REPO_ROOT/scripts/dsol_paper1" \
    "$SIM_PYTHON" "$REPO_ROOT/scripts/dsol_paper1/run_constructed_accel_bank.py" \
      --stage render \
      --protocol "$PROTOCOL" \
      --runtime "$RUNTIME" \
      --config-root "$CONFIG_ROOT" \
      --output-dir "$OUTPUT_DIR" \
      --render-gpu "$RENDER_GPU" \
      --resize-size 224 \
      --seed "$SEED" \
      "${max_state_args[@]}"
else
  [[ -s "$RENDER_SOURCE_DIR/render_summary.json" ]] || {
    echo "missing shared render summary: $RENDER_SOURCE_DIR/render_summary.json" >&2
    exit 2
  }
  printf 'reusing_render_source=%s\n' "$RENDER_SOURCE_DIR"
fi

PRETRAINED_MODELS_DIR=/share/longjunyu/alphabrain/pretrained_models \
ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1 \
PYTHONPATH="$REPO_ROOT:/projects/openpi/src:/projects/openpi/packages/openpi-client/src" \
  "$MODEL_PYTHON" "$REPO_ROOT/scripts/dsol_paper1/run_constructed_accel_bank.py" \
    --stage rank \
    --protocol "$PROTOCOL" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --render-source-dir "$RENDER_SOURCE_DIR" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    "${max_state_args[@]}"

printf 'constructed_accel_complete=%s summary=%s\n' \
  "$(date -u +%FT%TZ)" "$OUTPUT_DIR/summary.json"
