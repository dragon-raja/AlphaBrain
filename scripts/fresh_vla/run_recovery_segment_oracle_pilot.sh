#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
SIM_PYTHON=${FRESH_LIBERO_PYTHON:-/workspace/envs/fresh-libero/bin/python}
LIBERO_SOURCE=${FRESH_LIBERO_SOURCE:-/projects/openpi/third_party/libero}
EPISODE_ROOT=${FRESH_EPISODE_ROOT:-/share/longjunyu/fresh-vla/libero-full-episode-v2-128}
BASELINE_ROOT=${FRESH_CLOSED_LOOP_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2}
OUTPUT_ROOT=${FRESH_SEGMENT_ORACLE_ROOT:-/share/longjunyu/fresh-vla/research-reset/recovery-segment-oracle-v2}
PRETRAINED_MODELS_DIR=${PRETRAINED_MODELS_DIR:-/share/longjunyu/alphabrain/pretrained_models}
RUN_KIND=${FRESH_SEGMENT_RUN_KIND:-decision}
MAX_GROUPS=${FRESH_SEGMENT_MAX_GROUPS:-1}
GROUP_OFFSET=${FRESH_SEGMENT_GROUP_OFFSET:-0}
SAMPLE_COUNT=${FRESH_SEGMENT_SAMPLE_COUNT:-4}
SEGMENT_REPLANS=${FRESH_SEGMENT_REPLANS:-4}
EXECUTION_HORIZON=${FRESH_SEGMENT_EXECUTION_HORIZON:-3}
TOTAL_ACTION_BUDGET=${FRESH_SEGMENT_TOTAL_ACTION_BUDGET:-120}
LOOKAHEAD_STEPS=${FRESH_SEGMENT_LOOKAHEAD_STEPS:-30}
SELECTION_CONTINUATIONS=${FRESH_SEGMENT_SELECTION_CONTINUATIONS:-3}
DECISION_HELDOUT_CONTINUATIONS=${FRESH_SEGMENT_DECISION_HELDOUT_CONTINUATIONS:-5}
FULL_HELDOUT_CONTINUATIONS=${FRESH_SEGMENT_FULL_HELDOUT_CONTINUATIONS:-5}
RANDOM_SCHEDULES=${FRESH_SEGMENT_RANDOM_SCHEDULES:-3}
STAGE_DWELL_STEPS=${FRESH_SEGMENT_STAGE_DWELL_STEPS:-2}
SPLIT=${FRESH_SEGMENT_SPLIT:-val}
SERVER_START_TIMEOUT=${FRESH_POLICY_SERVER_TIMEOUT:-600}
ALLOW_DIRTY=${FRESH_ALLOW_DIRTY:-0}
PREREGISTRATION="$REPO_ROOT/docs/embodied_research_reset/recovery_segment_preregistration.md"
EXPECTED_PREREGISTRATION_SHA256=81986ee652f6d1466a31b98fa56eeda8b474f6fe42d8a25dd9f497cb48c5937e

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
  echo "FRESH_SEGMENT_RUN_KIND must be decision or smoke" >&2
  exit 2
fi
if [[ ! "$MAX_GROUPS" =~ ^[1-9][0-9]*$ ]] || [[ ! "$GROUP_OFFSET" =~ ^[0-9]+$ ]]; then
  echo "MAX_GROUPS must be positive and GROUP_OFFSET must be non-negative" >&2
  exit 2
fi

CHECKPOINT="$BASELINE_ROOT/fresh_closed_loop_full_h_seed${SEED}/final_model"
RUN_NAME="seed${SEED}-${SPLIT}-o${GROUP_OFFSET}-g${MAX_GROUPS}-n${SAMPLE_COUNT}"
RUN_NAME="${RUN_NAME}-r${SEGMENT_REPLANS}-k${EXECUTION_HORIZON}-b${TOTAL_ACTION_BUDGET}"
RUN_NAME="${RUN_KIND}-${RUN_NAME}-l${LOOKAHEAD_STEPS}-s${SELECTION_CONTINUATIONS}"
RUN_NAME="${RUN_NAME}-d${DECISION_HELDOUT_CONTINUATIONS}-f${FULL_HELDOUT_CONTINUATIONS}"
RUN_NAME="${RUN_NAME}-q${RANDOM_SCHEDULES}"
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
OUTPUT="$RUN_DIR/recovery_segment_oracle.json"
BANK_DIR="$RUN_DIR/training_bank"
AUDIT_BANK_DIR="$RUN_DIR/privileged_audit_bank"
VIDEO_DIR="$RUN_DIR/videos"
SOCKET_PATH="/tmp/fresh-recovery-segment-${SEED}-${GPU_ID}-$$.sock"
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
if [ ! -f "$PREREGISTRATION" ]; then
  echo "missing preregistration: $PREREGISTRATION" >&2
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

export FRESH_PREREGISTRATION_SHA256
FRESH_PREREGISTRATION_SHA256=$(sha256sum "$PREREGISTRATION" | awk '{print $1}')
if [ "$RUN_KIND" = decision ] && [ "$FRESH_PREREGISTRATION_SHA256" != "$EXPECTED_PREREGISTRATION_SHA256" ]; then
  echo "preregistration SHA256 does not match frozen decision config" >&2
  exit 1
fi

export FRESH_CHECKPOINT_SHA256
PREVERIFIED_NAME="FRESH_PREVERIFIED_CHECKPOINT_SHA256_${SEED}"
FRESH_CHECKPOINT_SHA256=${!PREVERIFIED_NAME:-}
export FRESH_CHECKPOINT_SHA256_SOURCE=preverified_environment
if [ -z "$FRESH_CHECKPOINT_SHA256" ]; then
  FRESH_CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT/model.safetensors" | awk '{print $1}')
  FRESH_CHECKPOINT_SHA256_SOURCE=sha256sum_at_launch
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

mkdir -p "$RUN_DIR" "$BANK_DIR" "$AUDIT_BANK_DIR" "$VIDEO_DIR"
exec 9>"$RUN_DIR/.run.lock"
if ! flock -n 9; then
  echo "recovery-segment pilot is already running: $RUN_DIR" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" \
PRETRAINED_MODELS_DIR="$PRETRAINED_MODELS_DIR" \
PYTHONDONTWRITEBYTECODE=1 \
"$PYTHON" scripts/fresh_vla/pi05_policy_server.py \
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
"$SIM_PYTHON" scripts/fresh_vla/evaluate_recovery_segment_oracle.py \
  --policy-socket "$SOCKET_PATH" \
  --episode-root "$EPISODE_ROOT" \
  --output "$OUTPUT" \
  --bank-dir "$BANK_DIR" \
  --audit-bank-dir "$AUDIT_BANK_DIR" \
  --video-dir "$VIDEO_DIR" \
  --run-kind "$RUN_KIND" \
  --split "$SPLIT" \
  --group-offset "$GROUP_OFFSET" \
  --max-groups "$MAX_GROUPS" \
  --sample-count "$SAMPLE_COUNT" \
  --segment-replans "$SEGMENT_REPLANS" \
  --execution-horizon "$EXECUTION_HORIZON" \
  --total-action-budget "$TOTAL_ACTION_BUDGET" \
  --lookahead-steps "$LOOKAHEAD_STEPS" \
  --selection-continuations "$SELECTION_CONTINUATIONS" \
  --decision-heldout-continuations "$DECISION_HELDOUT_CONTINUATIONS" \
  --full-heldout-continuations "$FULL_HELDOUT_CONTINUATIONS" \
  --random-schedules "$RANDOM_SCHEDULES" \
  --stage-dwell-steps "$STAGE_DWELL_STEPS" \
  --seed "$SEED"

echo "recovery-segment pilot complete: $OUTPUT"
