#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
SIM_PYTHON=${FRESH_LIBERO_PYTHON:-/workspace/envs/fresh-libero/bin/python}
LIBERO_SOURCE=${FRESH_LIBERO_SOURCE:-/projects/openpi/third_party/libero}
BASELINE_VIEW_ROOT=${FRESH_BASELINE_VIEW_ROOT:-/share/longjunyu/fresh-vla/runs/baseline-repair-v1/eval_views}
BASELINE_GATE=${FRESH_BASELINE_GATE:-/share/longjunyu/fresh-vla/runs/baseline-repair-v1/baseline_repair_v2_three_seed_gate.json}
BASELINE_STEPS=${FRESH_BASELINE_STEPS:-10353}
EPISODE_ROOT=${FRESH_EPISODE_ROOT:-/share/longjunyu/fresh-vla/libero-full-episode-v2-128}
OUTPUT_ROOT=${FRESH_RECOVERY_SUPPORT_ROOT:-/share/longjunyu/fresh-vla/runs/recovery-support-repaired-v2-step10353}
PRETRAINED_MODELS_DIR=${PRETRAINED_MODELS_DIR:-/share/longjunyu/alphabrain/pretrained_models}

SEED=${1:?usage: run_policy_state_recovery_collection.sh SEED GPU_ID [MAX_GROUPS]}
GPU_ID=${2:?usage: run_policy_state_recovery_collection.sh SEED GPU_ID [MAX_GROUPS]}
MAX_GROUPS=${3:-}

case "$SEED" in
  41) EXPECTED_SHA256=732da869fe5aab23ae83f6b517bb33a83bb0b5e7cea9c2535edc9388f07d61c4 ;;
  42) EXPECTED_SHA256=73d23cc8659ab7510eecdd013b1ffdc48c2ea97304ec14b3cf886906fc4da90a ;;
  43) EXPECTED_SHA256=cfd9547bde803ca83430bae37675aeb61e534f0d490f2b1d233ab3289baec4c4 ;;
  *) echo "seed must be one of 41, 42, or 43" >&2; exit 2 ;;
esac
if [[ ! "$GPU_ID" =~ ^[0-7]$ ]]; then
  echo "GPU_ID must be in [0,7]" >&2
  exit 2
fi
if [ -n "$MAX_GROUPS" ] && [[ ! "$MAX_GROUPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_GROUPS must be positive" >&2
  exit 2
fi

CHECKPOINT="$BASELINE_VIEW_ROOT/fresh_closed_loop_repair_step${BASELINE_STEPS}_seed${SEED}/final_model"
TAG="policy_state_repaired_seed${SEED}"
if [ -n "$MAX_GROUPS" ]; then
  TAG="${TAG}_smoke${MAX_GROUPS}"
fi
OUTPUT_DIR="$OUTPUT_ROOT/corrections/$TAG"
LOG_DIR="$OUTPUT_ROOT/corrections/logs"
KEEPALIVE_SESSION="gpu-keepalive-${GPU_ID}"
SOCKET_PATH="/tmp/fresh-policy-state-${SEED}-$$.sock"
SERVER_PID=""
KEEPALIVE_WAS_RUNNING=0

if [ ! -f "$CHECKPOINT/model.safetensors" ]; then
  echo "missing checkpoint: $CHECKPOINT/model.safetensors" >&2
  exit 1
fi
if [ -e "$OUTPUT_DIR" ]; then
  echo "refusing to overwrite existing output: $OUTPUT_DIR" >&2
  exit 1
fi

if [ "${FRESH_ALLOW_UNGATED_SUPPORT_SMOKE:-0}" != 1 ]; then
  "$PYTHON" - "$BASELINE_GATE" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing repaired baseline gate: {path}")
payload = json.loads(path.read_text())
if payload.get("decision") != "BASELINE_VALID_PROCEED_TO_RECOVERY_CONTROLS":
    raise SystemExit(f"repaired baseline gate did not pass: {payload.get('decision')}")
if payload.get("test_split_opened", False):
    raise SystemExit("repaired baseline gate unexpectedly opened test")
PY
fi

cd "$REPO_ROOT"
GIT_SHA=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "formal policy-state collection requires a clean Git worktree" >&2
  exit 1
fi
ACTUAL_SHA256=$(sha256sum "$CHECKPOINT/model.safetensors" | awk '{print $1}')
if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
  echo "checkpoint SHA256 mismatch for seed $SEED" >&2
  exit 1
fi

restore_runtime() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$SOCKET_PATH"
  if [ "$KEEPALIVE_WAS_RUNNING" = 1 ]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      "${AI2R_KEEPALIVE_EXTRA_GIB:-1}" "${AI2R_KEEPALIVE_N:-8192}" \
      "$KEEPALIVE_SESSION" "$GPU_ID" >/dev/null || true
  fi
}
trap restore_runtime EXIT

if tmux has-session -t "$KEEPALIVE_SESSION" 2>/dev/null; then
  KEEPALIVE_WAS_RUNNING=1
  tmux kill-session -t "$KEEPALIVE_SESSION"
fi
mkdir -p "$LOG_DIR"
export PRETRAINED_MODELS_DIR
export ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1
export NO_ALBUMENTATIONS_UPDATE=1

CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONDONTWRITEBYTECODE=1 \
"$PYTHON" scripts/fresh_vla/pi05_policy_server.py \
  --checkpoint "$CHECKPOINT" \
  --socket "$SOCKET_PATH" \
  --device cuda:0 >"$LOG_DIR/${TAG}_server.log" 2>&1 &
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

max_group_args=()
if [ -n "$MAX_GROUPS" ]; then
  max_group_args=(--max-groups "$MAX_GROUPS" --video-groups "$MAX_GROUPS" --policy-audit-stride 1 --minimum-correction-group-rate 0.0)
fi

FRESH_GIT_SHA="$GIT_SHA" \
FRESH_GIT_DIRTY=0 \
FRESH_CHECKPOINT_SHA256="$ACTUAL_SHA256" \
PYTHONPATH="$REPO_ROOT/scripts/fresh_vla:$LIBERO_SOURCE${PYTHONPATH:+:$PYTHONPATH}" \
LIBERO_CONFIG_PATH="$REPO_ROOT/scripts/fresh_vla/libero_config" \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PRETRAINED_MODELS_DIR="$PRETRAINED_MODELS_DIR" \
PYTHONDONTWRITEBYTECODE=1 \
"$SIM_PYTHON" scripts/fresh_vla/collect_policy_state_recovery.py \
  --policy-socket "$SOCKET_PATH" \
  --episode-root "$EPISODE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  "${max_group_args[@]}" \
  2>&1 | tee "$LOG_DIR/${TAG}_collector.log"
