#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
VARIANT=${1:?usage: run_recovery_prompt_eval.sh VARIANT SEED GPU_ID}
SEED=${2:?usage: run_recovery_prompt_eval.sh VARIANT SEED GPU_ID}
GPU_ID=${3:?usage: run_recovery_prompt_eval.sh VARIANT SEED GPU_ID}
RUN_ROOT=${FRESH_RESEARCH_RESET_ROOT:-/share/longjunyu/fresh-vla/research-reset}/recovery_prompt
BASELINE_ROOT=${FRESH_CLOSED_LOOP_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2}
EPISODE_ROOT=${FRESH_EPISODE_ROOT:-/share/longjunyu/fresh-vla/libero-full-episode-v2-128}
PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
SIM_PYTHON=${FRESH_LIBERO_PYTHON:-/workspace/envs/fresh-libero/bin/python}
LIBERO_SOURCE=${FRESH_LIBERO_SOURCE:-/projects/openpi/third_party/libero}
KEEPALIVE_SESSION="gpu-keepalive-${GPU_ID}"
SOCKET_PATH="/tmp/fresh-recovery-prompt-${VARIANT}-${SEED}-$$.sock"
SERVER_PID=""
WAS_RUNNING=0

case "$VARIANT" in
  explicit_recovery)
    PROMPT="the grasp failed. open the gripper, move to the cream cheese, grasp it again, then put it in the bowl"
    ;;
  false_success_assumption)
    PROMPT="the cream cheese is grasped. lift it and put it in the bowl"
    ;;
  *) echo "unknown prompt variant: $VARIANT" >&2; exit 2 ;;
esac
case "$SEED" in
  41|42|43) ;;
  *) echo "seed must be one of 41, 42, 43" >&2; exit 2 ;;
esac
if [[ ! "$GPU_ID" =~ ^[0-7]$ ]]; then
  echo "GPU_ID must be in [0, 7]" >&2
  exit 2
fi

restore_runtime() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$SOCKET_PATH"
  if [ "$WAS_RUNNING" = 1 ]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      "${AI2R_KEEPALIVE_EXTRA_GIB:-1}" "${AI2R_KEEPALIVE_N:-8192}" "$KEEPALIVE_SESSION" "$GPU_ID" >/dev/null || true
  fi
}
trap restore_runtime EXIT

if tmux has-session -t "$KEEPALIVE_SESSION" 2>/dev/null; then
  WAS_RUNNING=1
  tmux kill-session -t "$KEEPALIVE_SESSION"
fi

RUN_DIR="$RUN_ROOT/${VARIANT}_seed${SEED}"
OUTPUT="$RUN_DIR/closed_loop_isolated.json"
mkdir -p "$RUN_DIR/videos"
if [ -f "$OUTPUT" ]; then
  echo "refusing to overwrite existing output: $OUTPUT" >&2
  exit 1
fi

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PRETRAINED_MODELS_DIR=/share/longjunyu/alphabrain/pretrained_models \
PYTHONDONTWRITEBYTECODE=1 \
"$PYTHON" scripts/fresh_vla/pi05_policy_server.py \
  --checkpoint "$BASELINE_ROOT/fresh_closed_loop_full_h_seed${SEED}/final_model" \
  --socket "$SOCKET_PATH" \
  --device cuda:0 >"$RUN_DIR/policy_server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 600); do
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

PYTHONPATH="$REPO_ROOT/scripts/fresh_vla:$LIBERO_SOURCE${PYTHONPATH:+:$PYTHONPATH}" \
LIBERO_CONFIG_PATH="$REPO_ROOT/scripts/fresh_vla/libero_config" \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTHONDONTWRITEBYTECODE=1 \
"$SIM_PYTHON" scripts/fresh_vla/evaluate_libero_closed_loop.py \
  --policy-socket "$SOCKET_PATH" \
  --episode-root "$EPISODE_ROOT" \
  --output "$OUTPUT" \
  --evaluation isolated \
  --execution-horizons 3 \
  --max-steps 320 \
  --split test \
  --seed "$((314159 + SEED))" \
  --language "$PROMPT" \
  --video-dir "$RUN_DIR/videos" \
  --video-groups 13
