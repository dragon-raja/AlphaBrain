#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
POLICY_PYTHON=${CABI_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
SIM_PYTHON=${CABI_LIBERO_PYTHON:-/share/longjunyu/capt-vla/envs/libero/bin/python}
LIBERO_SOURCE=${CABI_LIBERO_SOURCE:-/share/longjunyu/capt-vla/vendor/LIBERO}
LIBERO_CONFIG=${CABI_LIBERO_CONFIG:-/share/longjunyu/capt-vla/config/libero}
SUITE_ROOT=${CABI_SUITE_ROOT:-/share/longjunyu/cabi-vla/libero-bind-v0}
CAMERA_CONFIG=${CABI_CAMERA_CONFIG:-$REPO_ROOT/docs/cabi_vla/configs/camera_pose_grid_v1.json}
OUTPUT_ROOT=${CABI_CAMERA_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/camera-viewpoint-study-v1}
SPLIT=${CABI_EVAL_SPLIT:-train}
STATE_INDICES=${CABI_EVAL_STATE_INDICES:-}
EDGES=${CABI_EVAL_EDGES:-all}
POSES=${CABI_CAMERA_POSES:-all}
RAY_MODE=${CABI_CAMERA_RAY_MODE:-correct}
WRIST_RAY_MODE=${CABI_WRIST_RAY_MODE:-correct}
HORIZONS=${CABI_EVAL_HORIZONS:-3}
MAX_STEPS=${CABI_EVAL_MAX_STEPS:-320}
RESOLUTION=${CABI_CAMERA_RESOLUTION:-224}
MINIMUM_VISIBLE_PIXELS=${CABI_CAMERA_MINIMUM_VISIBLE_PIXELS:-64}
EVAL_SEED=${CABI_EVAL_SEED:-20260722}
SCENE_CUE_MODE=${CABI_SCENE_CUE_MODE:-fixed}
SCENE_CUE_SEED=${CABI_SCENE_CUE_SEED:-20260728}
FRAME_POSES=${CABI_CAMERA_FRAME_POSES:-baseline}
FRAME_EDGES=${CABI_CAMERA_FRAME_EDGES:-all}
FRAME_EPISODES=${CABI_EVAL_FRAME_EPISODES:-0}
SERVER_TIMEOUT=${CABI_POLICY_SERVER_TIMEOUT:-600}

CHECKPOINT=${1:?usage: run_libero_bind_camera_sweep.sh CHECKPOINT RUN_NAME GPU_ID}
RUN_NAME=${2:?usage: run_libero_bind_camera_sweep.sh CHECKPOINT RUN_NAME GPU_ID}
GPU_ID=${3:?usage: run_libero_bind_camera_sweep.sh CHECKPOINT RUN_NAME GPU_ID}

if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9_.-]+$ || ${#RUN_NAME} -gt 64 ]]; then
  echo "RUN_NAME must be at most 64 safe filename characters" >&2
  exit 2
fi
if [[ "$SPLIT" != train && "$SPLIT" != val && "$SPLIT" != test ]]; then
  echo "CABI_EVAL_SPLIT must be train, val, or test" >&2
  exit 2
fi
if [[ ! "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CABI_EVAL_MAX_STEPS must be a positive integer" >&2
  exit 2
fi
if [[ ! "$RESOLUTION" =~ ^[1-9][0-9]*$ || ! "$MINIMUM_VISIBLE_PIXELS" =~ ^[1-9][0-9]*$ ]]; then
  echo "camera resolution and minimum visible pixels must be positive integers" >&2
  exit 2
fi
if [[ ! "$FRAME_EPISODES" =~ ^[0-9]+$ ]]; then
  echo "CABI_EVAL_FRAME_EPISODES must be a non-negative integer" >&2
  exit 2
fi
if [[ "$SCENE_CUE_MODE" != fixed && "$SCENE_CUE_MODE" != cue_randomized ]]; then
  echo "CABI_SCENE_CUE_MODE must be fixed or cue_randomized" >&2
  exit 2
fi
if [[ ! -s "$CHECKPOINT/model.safetensors" ]]; then
  echo "missing checkpoint: $CHECKPOINT/model.safetensors" >&2
  exit 1
fi
if [[ ! -f "$CAMERA_CONFIG" ]]; then
  echo "missing camera config: $CAMERA_CONFIG" >&2
  exit 1
fi
if [[ ! -x "$POLICY_PYTHON" || ! -x "$SIM_PYTHON" ]]; then
  echo "missing policy or LIBERO Python" >&2
  exit 1
fi

read -r -a horizon_args <<<"$HORIZONS"
for value in "${horizon_args[@]}"; do
  if [[ "$value" != 1 && "$value" != 2 && "$value" != 3 ]]; then
    echo "CABI_EVAL_HORIZONS only supports 1, 2, and 3" >&2
    exit 2
  fi
done

RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
OUTPUT="$RUN_DIR/camera_sweep_${SPLIT}.json"
PARTIAL="$RUN_DIR/camera_sweep_${SPLIT}.partial.json"
SOCKET="/tmp/cabi-cam-${RUN_NAME:0:40}-$$.sock"
SERVER_LOG="$RUN_DIR/policy_server.log"
SESSION="gpu-keepalive-${GPU_ID}"
SERVER_PID=""
WAS_RUNNING=0

if [[ -e "$OUTPUT" || -e "$PARTIAL" ]]; then
  echo "refusing to overwrite existing evaluation: $OUTPUT or $PARTIAL" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$SOCKET"
  if [[ "$WAS_RUNNING" == 1 ]]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      "${AI2R_KEEPALIVE_EXTRA_GIB:-1}" "${AI2R_KEEPALIVE_N:-8192}" \
      "$SESSION" "$GPU_ID" >/dev/null || true
  fi
}
trap cleanup EXIT

if tmux has-session -t "$SESSION" 2>/dev/null; then
  WAS_RUNNING=1
  tmux kill-session -t "$SESSION"
fi

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PRETRAINED_MODELS_DIR=/share/longjunyu/alphabrain/pretrained_models \
PYTHONDONTWRITEBYTECODE=1 \
"$POLICY_PYTHON" scripts/fresh_vla/pi05_policy_server.py \
  --checkpoint "$CHECKPOINT" \
  --socket "$SOCKET" \
  --device cuda:0 >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 "$SERVER_TIMEOUT"); do
  [[ -S "$SOCKET" ]] && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -n 80 "$SERVER_LOG" >&2
    exit 1
  fi
  sleep 1
done
if [[ ! -S "$SOCKET" ]]; then
  echo "timed out waiting for Pi0.5 policy server" >&2
  exit 1
fi

state_args=()
if [[ -n "$STATE_INDICES" ]]; then
  state_args=(--state-indices "$STATE_INDICES")
fi
frame_args=()
if [[ "$FRAME_EPISODES" -gt 0 ]]; then
  frame_args=(
    --frame-dir "$RUN_DIR/frames"
    --frame-poses "$FRAME_POSES"
    --frame-edges "$FRAME_EDGES"
    --frame-episodes-per-edge "$FRAME_EPISODES"
  )
fi

PYTHONPATH="$REPO_ROOT/scripts/cabi_vla:$LIBERO_SOURCE${PYTHONPATH:+:$PYTHONPATH}" \
LIBERO_CONFIG_PATH="$LIBERO_CONFIG" \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTHONDONTWRITEBYTECODE=1 \
"$SIM_PYTHON" scripts/cabi_vla/evaluate_libero_bind_camera_viewpoints.py \
  --suite-root "$SUITE_ROOT" \
  --policy-socket "$SOCKET" \
  --camera-config "$CAMERA_CONFIG" \
  --output "$OUTPUT" \
  --split "$SPLIT" \
  --edges "$EDGES" \
  --poses "$POSES" \
  --ray-mode "$RAY_MODE" \
  --wrist-ray-mode "$WRIST_RAY_MODE" \
  --execution-horizons "${horizon_args[@]}" \
  --max-steps "$MAX_STEPS" \
  --resolution "$RESOLUTION" \
  --minimum-visible-pixels "$MINIMUM_VISIBLE_PIXELS" \
  --seed "$EVAL_SEED" \
  --scene-cue-mode "$SCENE_CUE_MODE" \
  --scene-cue-seed "$SCENE_CUE_SEED" \
  "${state_args[@]}" \
  "${frame_args[@]}"
