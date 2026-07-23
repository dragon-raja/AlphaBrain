#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${CABI_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
CHECKPOINT=${1:?usage: run_checkpoint_gate.sh CHECKPOINT RUN_PREFIX GPU_ID}
RUN_PREFIX=${2:?usage: run_checkpoint_gate.sh CHECKPOINT RUN_PREFIX GPU_ID}
GPU_ID=${3:?usage: run_checkpoint_gate.sh CHECKPOINT RUN_PREFIX GPU_ID}
TRAIN_SESSION=${CABI_TRAIN_SESSION:-}
POLL_SECONDS=${CABI_GATE_POLL_SECONDS:-15}
ANCHORS=${CABI_RELOAD_ANCHORS:-/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v7-coverage-phase-loss-balanced/anchors.npz}
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

RUN_ROOT=$(dirname "$CHECKPOINT")
METRICS=$RUN_ROOT/metrics.jsonl
TRAINING_VIEW=$(dirname "$ANCHORS")
echo "stage=training_invariants"
"$PYTHON" - "$CHECKPOINT/framework_config.yaml" "$TRAINING_VIEW/manifest.json" "$METRICS" <<'PY'
import json
import math
import sys

import yaml

config_path, manifest_path, metrics_path = sys.argv[1:]
config = yaml.safe_load(open(config_path))
manifest = json.load(open(manifest_path))
model_horizon = int(config["framework"]["action_model"]["action_horizon"])
data_horizon = int(manifest["action_horizon"])
if model_horizon != data_horizon:
    raise ValueError(
        f"checkpoint/data horizon mismatch: model={model_horizon} data={data_horizon}"
    )

cafc_enabled = bool(
    config.get("framework", {})
    .get("counterfactual_action_completion", {})
    .get("enabled", False)
)
cafc_values = []
with open(metrics_path) as stream:
    for line in stream:
        row = json.loads(line)
        if "counterfactual_action_completion" in row:
            cafc_values.append(float(row["counterfactual_action_completion"]))
if cafc_enabled and not cafc_values:
    raise ValueError("CAFC checkpoint contains no sampled CAFC training batch")
if any(not math.isfinite(value) for value in cafc_values):
    raise ValueError("CAFC metrics contain a non-finite value")
print(
    json.dumps(
        {
            "model_horizon": model_horizon,
            "data_horizon": data_horizon,
            "cafc_enabled": cafc_enabled,
            "cafc_batches": len(cafc_values),
            "cafc_all_finite": all(math.isfinite(value) for value in cafc_values),
        },
        sort_keys=True,
    )
)
PY

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
