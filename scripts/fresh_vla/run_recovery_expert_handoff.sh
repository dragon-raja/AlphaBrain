#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MODEL_PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
SIM_PYTHON=${FRESH_LIBERO_PYTHON:-/workspace/envs/fresh-libero/bin/python}
LIBERO_SOURCE=${FRESH_LIBERO_SOURCE:-/projects/openpi/third_party/libero}
EPISODE_ROOT=${FRESH_EPISODE_ROOT:-/share/longjunyu/fresh-vla/libero-full-episode-v2-128}
BASELINE_ROOT=${FRESH_CLOSED_LOOP_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2}
OUTPUT_ROOT=${FRESH_HANDOFF_ROOT:-/share/longjunyu/fresh-vla/research-reset/recovery-expert-handoff-v1}
PRETRAINED_MODELS_DIR=${PRETRAINED_MODELS_DIR:-/share/longjunyu/alphabrain/pretrained_models}
RUN_KIND=${FRESH_HANDOFF_RUN_KIND:-decision}
SPLIT=${FRESH_HANDOFF_SPLIT:-val}
GROUP_OFFSET=${FRESH_HANDOFF_GROUP_OFFSET:-0}
MAX_GROUPS=${FRESH_HANDOFF_MAX_GROUPS:-1}
EXECUTION_HORIZON=${FRESH_HANDOFF_EXECUTION_HORIZON:-3}
TOTAL_ACTION_BUDGET=${FRESH_HANDOFF_TOTAL_ACTION_BUDGET:-120}
MAX_TEACHER_ACTIONS=${FRESH_HANDOFF_MAX_TEACHER_ACTIONS:-90}
CONTINUATIONS=${FRESH_HANDOFF_CONTINUATIONS:-5}
STAGE_DWELL_STEPS=${FRESH_HANDOFF_STAGE_DWELL_STEPS:-2}
VIDEO_REPEATS=${FRESH_HANDOFF_VIDEO_REPEATS:-1}
SERVER_START_TIMEOUT=${FRESH_POLICY_SERVER_TIMEOUT:-600}
ALLOW_DIRTY=${FRESH_ALLOW_DIRTY:-0}

SEED=${1:-41}
GPU_ID=${2:-0}
case "$SEED" in
  41) EXPECTED_CHECKPOINT_SHA256=144a3b3d3dcc8421418564a62059a1038c9a7ef3196ac157f5f9ea1997a31f30 ;;
  42) EXPECTED_CHECKPOINT_SHA256=98dc52d2ed1983776d218fee7666f3131053d1a55296e93e9f521b1c088ce875 ;;
  43) EXPECTED_CHECKPOINT_SHA256=5db16350d9835c1f28d01b660dd6e9234bcab3da79abbce1f092e92b08ac9149 ;;
  *) echo "unsupported Full-H seed: $SEED" >&2; exit 2 ;;
esac
if [[ ! "$GPU_ID" =~ ^[0-7]$ ]]; then
  echo "GPU_ID must be in [0, 7]" >&2
  exit 2
fi
if [ "$RUN_KIND" != decision ] && [ "$RUN_KIND" != smoke ]; then
  echo "FRESH_HANDOFF_RUN_KIND must be decision or smoke" >&2
  exit 2
fi
if [[ ! "$MAX_GROUPS" =~ ^[1-9][0-9]*$ ]] || [[ ! "$GROUP_OFFSET" =~ ^[0-9]+$ ]]; then
  echo "MAX_GROUPS must be positive and GROUP_OFFSET must be non-negative" >&2
  exit 2
fi

CHECKPOINT="$BASELINE_ROOT/fresh_closed_loop_full_h_seed${SEED}/final_model"
RUN_NAME="${RUN_KIND}-seed${SEED}-${SPLIT}-o${GROUP_OFFSET}-g${MAX_GROUPS}"
RUN_NAME="${RUN_NAME}-k${EXECUTION_HORIZON}-b${TOTAL_ACTION_BUDGET}"
RUN_NAME="${RUN_NAME}-t${MAX_TEACHER_ACTIONS}-c${CONTINUATIONS}-d${STAGE_DWELL_STEPS}"
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
OUTPUT="$RUN_DIR/recovery_expert_handoff.json"
VIDEO_DIR="$RUN_DIR/videos"
SOCKET_PATH="/tmp/fresh-recovery-handoff-${SEED}-${GPU_ID}-$$.sock"
SESSION="gpu-keepalive-${GPU_ID}"
SERVER_PID=""
WAS_RUNNING=0

if [ ! -f "$CHECKPOINT/model.safetensors" ]; then
  echo "missing frozen Full-H checkpoint: $CHECKPOINT/model.safetensors" >&2
  exit 1
fi
if [ -e "$OUTPUT" ]; then
  echo "refusing to overwrite existing output: $OUTPUT" >&2
  exit 1
fi
if ! command -v flock >/dev/null || ! command -v sha256sum >/dev/null; then
  echo "required command missing: flock and sha256sum are mandatory" >&2
  exit 1
fi

cd "$REPO_ROOT"
export FRESH_GIT_SHA
FRESH_GIT_SHA=$(git rev-parse HEAD)
export FRESH_GIT_DIRTY=0
if [ -n "$(git status --porcelain)" ]; then
  FRESH_GIT_DIRTY=1
fi
if [ "$FRESH_GIT_DIRTY" = 1 ]; then
  if [ "$RUN_KIND" = decision ]; then
    echo "refusing to run a decision experiment from a dirty worktree" >&2
    exit 1
  fi
  if [ "$ALLOW_DIRTY" != 1 ]; then
    echo "dirty smoke requires FRESH_ALLOW_DIRTY=1" >&2
    exit 1
  fi
fi

export FRESH_CHECKPOINT_SHA256
PREVERIFIED_NAME="FRESH_PREVERIFIED_CHECKPOINT_SHA256_${SEED}"
FRESH_CHECKPOINT_SHA256=${!PREVERIFIED_NAME:-}
if [ -z "$FRESH_CHECKPOINT_SHA256" ]; then
  FRESH_CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT/model.safetensors" | awk '{print $1}')
fi
if [ "$FRESH_CHECKPOINT_SHA256" != "$EXPECTED_CHECKPOINT_SHA256" ]; then
  echo "frozen checkpoint SHA256 mismatch for seed $SEED" >&2
  exit 1
fi

restore_runtime() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$SOCKET_PATH"
  if [ "$WAS_RUNNING" = 1 ]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      "${AI2R_KEEPALIVE_EXTRA_GIB:-1}" "${AI2R_KEEPALIVE_N:-8192}" "$SESSION" "$GPU_ID" >/dev/null || true
  fi
}
trap restore_runtime EXIT

if tmux has-session -t "$SESSION" 2>/dev/null; then
  WAS_RUNNING=1
  tmux kill-session -t "$SESSION"
fi

mkdir -p "$RUN_DIR" "$VIDEO_DIR"
exec 9>"$RUN_DIR/.run.lock"
if ! flock -n 9; then
  echo "recovery handoff evaluation is already running: $RUN_DIR" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" \
PRETRAINED_MODELS_DIR="$PRETRAINED_MODELS_DIR" \
PYTHONDONTWRITEBYTECODE=1 \
"$MODEL_PYTHON" scripts/fresh_vla/pi05_policy_server.py \
  --checkpoint "$CHECKPOINT" \
  --socket "$SOCKET_PATH" \
  --device cuda:0 >"$RUN_DIR/policy_server.log" 2>&1 &
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

PYTHONPATH="$REPO_ROOT/scripts/fresh_vla:$LIBERO_SOURCE${PYTHONPATH:+:$PYTHONPATH}" \
LIBERO_CONFIG_PATH="$REPO_ROOT/scripts/fresh_vla/libero_config" \
MUJOCO_GL=egl \
PYTHONDONTWRITEBYTECODE=1 \
"$SIM_PYTHON" scripts/fresh_vla/evaluate_recovery_expert_handoff.py \
  --policy-socket "$SOCKET_PATH" \
  --episode-root "$EPISODE_ROOT" \
  --output "$OUTPUT" \
  --video-dir "$VIDEO_DIR" \
  --video-repeats "$VIDEO_REPEATS" \
  --run-kind "$RUN_KIND" \
  --split "$SPLIT" \
  --group-offset "$GROUP_OFFSET" \
  --max-groups "$MAX_GROUPS" \
  --execution-horizon "$EXECUTION_HORIZON" \
  --total-action-budget "$TOTAL_ACTION_BUDGET" \
  --max-teacher-actions "$MAX_TEACHER_ACTIONS" \
  --continuations "$CONTINUATIONS" \
  --stage-dwell-steps "$STAGE_DWELL_STEPS" \
  --seed "$SEED"

echo "recovery expert handoff complete: $OUTPUT"
