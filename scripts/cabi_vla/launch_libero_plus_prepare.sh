#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${LIBERO_PLUS_DATA_ROOT:-/share/longjunyu/alphabrain/datasets/libero-plus}
VERIFIED="$DATA_ROOT/verified"
RUNTIME="$DATA_ROOT/runtime/LIBERO-plus"
RLDS_ROOT="$DATA_ROOT/extracted/camparam-rlds-v1"
AUDIT="$DATA_ROOT/audit/camparam-rlds-v1.json"
SCREEN_ROOT=${KYC_DUAL_EVAL_ROOT:-/share/longjunyu/cabi-vla/dual-camera-kyc-screen-v1}
DATA_ENV=${LIBERO_PLUS_DATA_ENV:-/share/longjunyu/alphabrain/envs/libero-plus-data-v1}

ASSETS_SIZE=6395849578
ASSETS_SHA=96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf
RLDS_SIZE=16607835331
RLDS_SHA=a99466a1bb7eab4d0c55094d64d53ef6794ee835ba0db003fcee3e3fa6568e73

wait_for_size() {
  local path=$1
  local expected=$2
  while [[ ! -f "$path" || $(stat -c %s "$path" 2>/dev/null || true) != "$expected" ]]; do
    echo "waiting for verified resource: $path"
    sleep 30
  done
}

screen_is_running() {
  tmux has-session -t dualcam-screen-orchestrator 2>/dev/null && return 0
  tmux list-sessions -F '#{session_name}' 2>/dev/null | rg -q '^dualcam-eval-' && return 0
  return 1
}

wait_for_size "$VERIFIED/assets.zip" "$ASSETS_SIZE"
wait_for_size "$VERIFIED/libero_plus_camparam_rlds.zip" "$RLDS_SIZE"
while screen_is_running; do
  echo "waiting for dual-camera evaluation and video rendering"
  sleep 30
done

"$REPO_ROOT/scripts/cabi_vla/create_libero_plus_data_env.sh" "$DATA_ENV"

if [[ ! -s "$RUNTIME/libero_plus_runtime_manifest.json" ]]; then
  PYTHONPATH="$REPO_ROOT/scripts/cabi_vla" "$REPO_ROOT/.venv/bin/python" \
    "$REPO_ROOT/scripts/cabi_vla/prepare_libero_plus_runtime.py" \
    --source-repo /projects/LIBERO-plus \
    --assets-archive "$VERIFIED/assets.zip" \
    --output "$RUNTIME" \
    --expected-sha256 "$ASSETS_SHA"
else
  echo "LIBERO-Plus runtime already prepared: $RUNTIME"
fi

if [[ ! -s "$RLDS_ROOT/libero_plus_camparam_rlds_manifest.json" ]]; then
  PYTHONPATH="$REPO_ROOT/scripts/cabi_vla" "$REPO_ROOT/.venv/bin/python" \
    "$REPO_ROOT/scripts/cabi_vla/prepare_libero_plus_camparam_rlds.py" \
    --archive "$VERIFIED/libero_plus_camparam_rlds.zip" \
    --output "$RLDS_ROOT" \
    --expected-sha256 "$RLDS_SHA"
else
  echo "LIBERO-Plus camparam RLDS already prepared: $RLDS_ROOT"
fi

if [[ ! -s "$AUDIT" ]]; then
  PYTHONPATH="$REPO_ROOT/scripts/cabi_vla" "$DATA_ENV/bin/python" \
    "$REPO_ROOT/scripts/cabi_vla/inspect_libero_plus_camparam_rlds.py" \
    --dataset-root "$RLDS_ROOT" \
    --hand-eye-config "$REPO_ROOT/docs/cabi_vla/configs/libero_wrist_hand_eye_v1.json" \
    --output "$AUDIT"
else
  echo "LIBERO-Plus camparam audit already complete: $AUDIT"
fi

echo "LIBERO-Plus resources, runtime, and data audit are complete"
