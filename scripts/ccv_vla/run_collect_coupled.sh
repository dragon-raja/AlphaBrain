#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MODEL_PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
SIM_PYTHON=${FRESH_LIBERO_PYTHON:-/workspace/envs/fresh-libero/bin/python}
LIBERO_SOURCE=${FRESH_LIBERO_SOURCE:-/projects/openpi/third_party/libero}
EPISODE_ROOT=${CCV_EPISODE_ROOT:-/share/longjunyu/fresh-vla/libero-full-episode-v2-128}
OUTPUT_ROOT=${CCV_OUTPUT_ROOT:-}
CHECKPOINT=${CCV_CHECKPOINT:-/share/longjunyu/fresh-vla/runs/baseline-repair-v1/baseline_repair_full_h_ddp8_seed41_steps13804_formal-v2/checkpoints/steps_10353}
PREREGISTRATION=$REPO_ROOT/docs/ccv_vla/gate0_preregistration.md
GPU_ID=${1:?usage: run_collect_coupled.sh GPU_ID RUN_KIND [MAX_GROUPS] [GROUP_OFFSET]}
RUN_KIND=${2:?run kind must be smoke or formal}
MAX_GROUPS=${3:-}
GROUP_OFFSET=${4:-0}
SOCKET=/tmp/ccv-collect-${RUN_KIND}-${GPU_ID}-$$.sock
SESSION=gpu-keepalive-${GPU_ID}
WAS_RUNNING=0
SERVER_PID=""

export OMP_NUM_THREADS=${CCV_CPU_THREADS:-1}
export OPENBLAS_NUM_THREADS=${CCV_CPU_THREADS:-1}
export MKL_NUM_THREADS=${CCV_CPU_THREADS:-1}
export NUMEXPR_NUM_THREADS=${CCV_CPU_THREADS:-1}

if [ -z "$OUTPUT_ROOT" ]; then
  if [ "$RUN_KIND" = smoke ]; then
    OUTPUT_ROOT=/share/longjunyu/fresh-vla/ccv-vla/gate0-coupled-smoke-v3
  else
    OUTPUT_ROOT=/share/longjunyu/fresh-vla/ccv-vla/gate0-coupled-v3
  fi
fi

cleanup() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ "$WAS_RUNNING" = 1 ]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh 1 2048 "$SESSION" "$GPU_ID" >/dev/null || true
  fi
}
trap cleanup EXIT

mkdir -p "$OUTPUT_ROOT/logs"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  WAS_RUNNING=1
  tmux kill-session -t "$SESSION"
fi

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" PRETRAINED_MODELS_DIR=/share/longjunyu/alphabrain/pretrained_models \
  "$MODEL_PYTHON" scripts/fresh_vla/pi05_policy_server.py \
  --checkpoint "$CHECKPOINT" --socket "$SOCKET" --device cuda:0 \
  >"$OUTPUT_ROOT/logs/policy-server-${RUN_KIND}-g${GPU_ID}-o${GROUP_OFFSET}.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 600); do
  [ -S "$SOCKET" ] && break
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -80 "$OUTPUT_ROOT/logs/policy-server-${RUN_KIND}-g${GPU_ID}-o${GROUP_OFFSET}.log"; exit 1; }
  sleep 1
done
[ -S "$SOCKET" ] || { echo "policy server timeout" >&2; exit 1; }

extra_args=()
if [ -n "$MAX_GROUPS" ]; then
  extra_args+=(--max-groups "$MAX_GROUPS")
fi
if [ "$RUN_KIND" = smoke ]; then
  extra_args+=(--lookahead-actions 8 --continuation-repeats 2 --max-actions 40)
fi

PYTHONPATH="$REPO_ROOT/scripts/ccv_vla:$REPO_ROOT/scripts/cora_vla:$REPO_ROOT/scripts/fresh_vla:$LIBERO_SOURCE" \
LIBERO_CONFIG_PATH="$REPO_ROOT/scripts/fresh_vla/libero_config" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES="$GPU_ID" "$SIM_PYTHON" scripts/ccv_vla/collect_coupled_continuations.py \
  --policy-socket "$SOCKET" --episode-root "$EPISODE_ROOT" --output-root "$OUTPUT_ROOT" \
  --preregistration "$PREREGISTRATION" --seed 41 --group-offset "$GROUP_OFFSET" \
  --run-kind "$RUN_KIND" "${extra_args[@]}"
