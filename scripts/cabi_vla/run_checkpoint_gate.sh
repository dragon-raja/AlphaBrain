#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${CABI_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
CHECKPOINT=${1:?usage: run_checkpoint_gate.sh CHECKPOINT RUN_PREFIX GPU_ID}
RUN_PREFIX=${2:?usage: run_checkpoint_gate.sh CHECKPOINT RUN_PREFIX GPU_ID}
GPU_ID=${3:?usage: run_checkpoint_gate.sh CHECKPOINT RUN_PREFIX GPU_ID}
TRAIN_SESSION=${CABI_TRAIN_SESSION:-}
POLL_SECONDS=${CABI_GATE_POLL_SECONDS:-15}
ANCHORS=${CABI_RELOAD_ANCHORS:-/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v5-loss-balanced/anchors.npz}
OFFLINE_OUTPUT=/share/longjunyu/cabi-vla/offline-evaluations/$RUN_PREFIX/offline.json
EVAL_RUN=${RUN_PREFIX}_train0_k3
EVAL_OUTPUT=/share/longjunyu/cabi-vla/evaluations/$EVAL_RUN/closed_loop_train.json
DIAGNOSTIC_RUN=${RUN_PREFIX}_policy_diagnosis
DIAGNOSTIC_OUTPUT=/share/longjunyu/cabi-vla/diagnostics/$DIAGNOSTIC_RUN/policy_diagnosis.json
GEOMETRY_OUTPUT=/share/longjunyu/cabi-vla/diagnostics/${RUN_PREFIX}_binding_geometry/binding_geometry.json

if [[ ! "$RUN_PREFIX" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "RUN_PREFIX contains unsupported characters" >&2
  exit 2
fi
if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CABI_GATE_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

if [[ -n "$TRAIN_SESSION" ]]; then
  echo "waiting for training session: $TRAIN_SESSION"
  while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
fi
if [[ ! -s "$CHECKPOINT/model.safetensors" || ! -s "$CHECKPOINT/framework_config.yaml" ]]; then
  echo "training ended without a complete checkpoint: $CHECKPOINT" >&2
  exit 1
fi

cd "$REPO_ROOT"
echo "stage=reload checkpoint=$CHECKPOINT"
CABI_RELOAD_ANCHORS="$ANCHORS" \
  scripts/cabi_vla/verify_policy_reload.sh "$CHECKPOINT" "$GPU_ID"

if [[ -s "$OFFLINE_OUTPUT" ]]; then
  echo "stage=offline status=skip_complete output=$OFFLINE_OUTPUT"
else
  echo "stage=offline status=run"
  scripts/cabi_vla/run_libero_bind_offline_eval.sh \
    "$CHECKPOINT" "$RUN_PREFIX" "$GPU_ID"
fi

if [[ -s "$EVAL_OUTPUT" ]]; then
  echo "stage=train0_k3 status=skip_complete output=$EVAL_OUTPUT"
else
  echo "stage=train0_k3 status=run"
  CABI_EVAL_SPLIT=train \
  CABI_EVAL_STATE_INDICES=0 \
  CABI_EVAL_HORIZONS=3 \
  CABI_EVAL_FRAME_EPISODES=1 \
    scripts/cabi_vla/run_libero_bind_eval.sh \
      "$CHECKPOINT" "$EVAL_RUN" "$GPU_ID"
fi

if [[ -s "$DIAGNOSTIC_OUTPUT" ]]; then
  echo "stage=policy_diagnosis status=skip_complete output=$DIAGNOSTIC_OUTPUT"
else
  echo "stage=policy_diagnosis status=run"
  CABI_DIAGNOSTIC_STATE_INDICES=0 \
  CABI_DIAGNOSTIC_FRAME_STRIDE=20 \
    scripts/cabi_vla/run_libero_bind_policy_diagnosis.sh \
      "$CHECKPOINT" "$DIAGNOSTIC_RUN" "$GPU_ID"
fi

cabi_enabled=$("$PYTHON" - "$CHECKPOINT/framework_config.yaml" <<'PY'
import sys
import yaml

payload = yaml.safe_load(open(sys.argv[1]))
print("true" if payload.get("framework", {}).get("cabi", {}).get("enabled", False) else "false")
PY
)
if [[ "$cabi_enabled" == true ]]; then
  if [[ -s "$GEOMETRY_OUTPUT" ]]; then
    echo "stage=binding_geometry status=skip_complete output=$GEOMETRY_OUTPUT"
  else
    echo "stage=binding_geometry status=run"
    mkdir -p "$(dirname "$GEOMETRY_OUTPUT")"
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    PRETRAINED_MODELS_DIR=/share/longjunyu/alphabrain/pretrained_models \
    PYTHONDONTWRITEBYTECODE=1 \
      "$PYTHON" scripts/cabi_vla/diagnose_binding_geometry.py \
        --checkpoint "$CHECKPOINT" \
        --training-view "$(dirname "$ANCHORS")" \
        --output "$GEOMETRY_OUTPUT" \
        --device cuda:0
  fi
else
  echo "stage=binding_geometry status=not_applicable"
fi

echo "checkpoint_gate_complete run=$RUN_PREFIX"
