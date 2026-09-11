#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
SIM_PYTHON=${FRESH_LIBERO_PYTHON:-/workspace/envs/fresh-libero/bin/python}
LIBERO_SOURCE=${FRESH_LIBERO_SOURCE:-/projects/openpi/third_party/libero}
OUTPUT_ROOT=${FRESH_CLOSED_LOOP_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2}
EPISODE_ROOT=${FRESH_EPISODE_ROOT:-/share/longjunyu/fresh-vla/libero-full-episode-v2-128}
PRETRAINED_MODELS_DIR=${PRETRAINED_MODELS_DIR:-/share/longjunyu/alphabrain/pretrained_models}
EVAL_ONLY=${FRESH_EVAL_ONLY:-all}
EVAL_SPLIT=${FRESH_EVAL_SPLIT:-test}
EVAL_MAX_STEPS=${FRESH_EVAL_MAX_STEPS:-320}
OUTPUT_TAG=${FRESH_EVAL_OUTPUT_TAG:-}
MAX_GROUPS=${FRESH_EVAL_MAX_GROUPS:-}
REACH_MAX_STEPS=${FRESH_REACH_MAX_STEPS:-20}
REACH_TARGET_STEP=${FRESH_REACH_TARGET_STEP:-20}
SERVER_START_TIMEOUT=${FRESH_POLICY_SERVER_TIMEOUT:-600}

METHOD=${1:?usage: run_libero_closed_loop_eval.sh METHOD SEED GPU_ID}
SEED=${2:?usage: run_libero_closed_loop_eval.sh METHOD SEED GPU_ID}
GPU_ID=${3:?usage: run_libero_closed_loop_eval.sh METHOD SEED GPU_ID}
RUN_ID=${FRESH_RUN_ID:-fresh_closed_loop_${METHOD}_seed${SEED}}
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
CHECKPOINT="$RUN_DIR/final_model"
SESSION="gpu-keepalive-${GPU_ID}"
WAS_RUNNING=0
SERVER_PID=""
SOCKET_PATH="/tmp/fresh-pi05-${METHOD}-${SEED}-$$.sock"

case "$EVAL_ONLY" in
  all|closed_loop|isolated|end_to_end|reach) ;;
  *)
    echo "FRESH_EVAL_ONLY must be one of: all, closed_loop, isolated, end_to_end, reach" >&2
    exit 2
    ;;
esac
case "$EVAL_SPLIT" in
  train|val|test) ;;
  *)
    echo "FRESH_EVAL_SPLIT must be one of: train, val, test" >&2
    exit 2
    ;;
esac
if [[ ! "$EVAL_MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "FRESH_EVAL_MAX_STEPS must be a positive integer" >&2
  exit 2
fi
if [ -n "$OUTPUT_TAG" ] && [[ ! "$OUTPUT_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "FRESH_EVAL_OUTPUT_TAG may only contain letters, numbers, dots, underscores, and dashes" >&2
  exit 2
fi
if [ -n "$MAX_GROUPS" ] && [[ ! "$MAX_GROUPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "FRESH_EVAL_MAX_GROUPS must be a positive integer" >&2
  exit 2
fi
if [[ ! "$REACH_MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "FRESH_REACH_MAX_STEPS must be a positive integer" >&2
  exit 2
fi
if [[ ! "$REACH_TARGET_STEP" =~ ^[1-9][0-9]*$ ]]; then
  echo "FRESH_REACH_TARGET_STEP must be a positive integer" >&2
  exit 2
fi
if [[ ! "$SERVER_START_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "FRESH_POLICY_SERVER_TIMEOUT must be a positive integer" >&2
  exit 2
fi

OUTPUT_SUFFIX=""
if [ -n "$OUTPUT_TAG" ]; then
  OUTPUT_SUFFIX="_${OUTPUT_TAG}"
fi
max_group_args=()
if [ -n "$MAX_GROUPS" ]; then
  max_group_args=(--max-groups "$MAX_GROUPS")
fi

if [ ! -f "$CHECKPOINT/model.safetensors" ]; then
  echo "missing checkpoint: $CHECKPOINT/model.safetensors" >&2
  exit 1
fi

restore_keepalive() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ "$WAS_RUNNING" = 1 ]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      "${AI2R_KEEPALIVE_EXTRA_GIB:-1}" "${AI2R_KEEPALIVE_N:-8192}" "$SESSION" "$GPU_ID" >/dev/null || true
  fi
}
trap restore_keepalive EXIT
if tmux has-session -t "$SESSION" 2>/dev/null; then
  WAS_RUNNING=1
  tmux kill-session -t "$SESSION"
fi

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PRETRAINED_MODELS_DIR="$PRETRAINED_MODELS_DIR" \
PYTHONDONTWRITEBYTECODE=1 \
"$PYTHON" scripts/fresh_vla/pi05_policy_server.py \
  --checkpoint "$CHECKPOINT" \
  --socket "$SOCKET_PATH" \
  --device cuda:0 >"$RUN_DIR/policy_server${OUTPUT_SUFFIX}.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 "$SERVER_START_TIMEOUT"); do
  if [ -S "$SOCKET_PATH" ]; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Pi0.5 policy server exited before becoming ready" >&2
    exit 1
  fi
  sleep 1
done
if [ ! -S "$SOCKET_PATH" ]; then
  echo "timed out waiting for Pi0.5 policy server" >&2
  exit 1
fi
for evaluation in isolated end_to_end; do
  if [ "$EVAL_ONLY" != all ] && [ "$EVAL_ONLY" != closed_loop ] && [ "$EVAL_ONLY" != "$evaluation" ]; then
    continue
  fi
  output="$RUN_DIR/closed_loop_${evaluation}${OUTPUT_SUFFIX}.json"
  if [ -f "$output" ]; then
    echo "skip completed $evaluation method=$METHOD seed=$SEED"
    continue
  fi
  video_args=()
  if [ "${FRESH_SAVE_EVAL_VIDEOS:-0}" = 1 ] && \
     { [ "$evaluation" = end_to_end ] || [ "${FRESH_SAVE_ISOLATED_VIDEOS:-0}" = 1 ]; }; then
    video_args=(
      --video-dir "$RUN_DIR/closed_loop_videos${OUTPUT_SUFFIX}"
      --video-groups "${FRESH_EVAL_VIDEO_GROUPS:-2}"
    )
  fi
  PYTHONPATH="$REPO_ROOT/scripts/fresh_vla:$LIBERO_SOURCE${PYTHONPATH:+:$PYTHONPATH}" \
  LIBERO_CONFIG_PATH="$REPO_ROOT/scripts/fresh_vla/libero_config" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  PRETRAINED_MODELS_DIR="$PRETRAINED_MODELS_DIR" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$SIM_PYTHON" scripts/fresh_vla/evaluate_libero_closed_loop.py \
    --policy-socket "$SOCKET_PATH" \
    --episode-root "$EPISODE_ROOT" \
    --output "$output" \
    --evaluation "$evaluation" \
    --execution-horizons 1 2 3 \
    --device cuda:0 \
    --max-steps "$EVAL_MAX_STEPS" \
    --split "$EVAL_SPLIT" \
    --seed "$((314159 + SEED))" \
    "${max_group_args[@]}" \
    "${video_args[@]}"
done

if [ "$EVAL_ONLY" = all ] || [ "$EVAL_ONLY" = reach ]; then
  reach_output="$RUN_DIR/deterministic_reach${OUTPUT_SUFFIX}.json"
  if [ -f "$reach_output" ]; then
    if "$PYTHON" - "$reach_output" "$EVAL_SPLIT" "$EPISODE_ROOT" "$MAX_GROUPS" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
rows = payload.get("rows", [])
manifest = json.load(open(f"{sys.argv[3]}/manifest.json"))
group_count = sum(group["split"] == sys.argv[2] for group in manifest["groups"])
if sys.argv[4]:
    group_count = min(group_count, int(sys.argv[4]))
identities = {
    (int(row["execution_horizon"]), str(row["pair_id"]))
    for row in rows
}
valid = (
    payload.get("evaluation") == "deterministic_reach"
    and payload.get("split") == sys.argv[2]
    and len(rows) == 3 * group_count
    and len(identities) == len(rows)
    and {int(row["execution_horizon"]) for row in rows} == {1, 2, 3}
)
raise SystemExit(0 if valid else 1)
PY
    then
      echo "skip completed deterministic reach method=$METHOD seed=$SEED split=$EVAL_SPLIT"
      exit 0
    fi
    echo "existing deterministic reach output does not match split=$EVAL_SPLIT: $reach_output" >&2
    exit 1
  fi
  PYTHONPATH="$REPO_ROOT/scripts/fresh_vla:$LIBERO_SOURCE${PYTHONPATH:+:$PYTHONPATH}" \
  LIBERO_CONFIG_PATH="$REPO_ROOT/scripts/fresh_vla/libero_config" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  PRETRAINED_MODELS_DIR="$PRETRAINED_MODELS_DIR" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$SIM_PYTHON" scripts/fresh_vla/evaluate_libero_deterministic_reach.py \
    --policy-socket "$SOCKET_PATH" \
    --episode-root "$EPISODE_ROOT" \
    --output "$reach_output" \
    --execution-horizons 1 2 3 \
    --max-steps "$REACH_MAX_STEPS" \
    --reference-target-step "$REACH_TARGET_STEP" \
    --device cuda:0 \
    --split "$EVAL_SPLIT" \
    --seed "$((271828 + SEED))" \
    "${max_group_args[@]}"
fi
