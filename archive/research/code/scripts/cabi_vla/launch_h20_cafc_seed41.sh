#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${H20_CAFC_DATA_ROOT:-/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v15-decision-observed-edge-phase-loss-balanced-h20}
RUN_ROOT=${CABI_TRAIN_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/runs}
GATE_ROOT=${CABI_GATE_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/gates}
COMPARISON_ROOT=${CABI_COMPARISON_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/comparisons}

SEED=41
STEPS=33000
BC_GPU=${H20_BC_GPU:-0}
BRIDGE_GPU=${H20_BRIDGE_GPU:-1}
PLAIN_GPU=${H20_PLAIN_GPU:-4}
GROUNDED_GPU=${H20_GROUNDED_GPU:-5}

BC_MODE=cabi_bind_pi05_bc_h20_smoke
BRIDGE_MODE=cabi_bind_pi05_action_bridge_h20_smoke
PLAIN_MODE=cabi_bind_pi05_action_completion_h20_smoke
GROUNDED_MODE=cabi_bind_pi05_action_bridge_completion_h20_smoke
BC_TAG=h20-edge-balanced-3epoch-v15
BRIDGE_TAG=h20-bridge-edge-balanced-3epoch-v15
PLAIN_TAG=h20-cafc-edge-balanced-3epoch-v15
GROUNDED_TAG=h20-bridge-cafc-edge-balanced-3epoch-v15

BC_RUN_ID=${BC_MODE}_seed${SEED}_steps${STEPS}_${BC_TAG}
BRIDGE_RUN_ID=${BRIDGE_MODE}_seed${SEED}_steps${STEPS}_${BRIDGE_TAG}
PLAIN_RUN_ID=${PLAIN_MODE}_seed${SEED}_steps${STEPS}_${PLAIN_TAG}
GROUNDED_RUN_ID=${GROUNDED_MODE}_seed${SEED}_steps${STEPS}_${GROUNDED_TAG}
BC_PREFIX=bc_h20_33000_s41_v15
BRIDGE_PREFIX=bridge_h20_33000_s41_v15
PLAIN_PREFIX=cafc_h20_33000_s41_v15
GROUNDED_PREFIX=bridge_cafc_h20_33000_s41_v15
COMPARISON_NAME=cafc_h20_migration_seed41_v15

BC_TRAIN_SESSION=cabi-bc-h20-33000-v15-s41
BRIDGE_TRAIN_SESSION=cabi-bridge-h20-33000-v15-s41
PLAIN_TRAIN_SESSION=cabi-cafc-h20-33000-v15-s41
GROUNDED_TRAIN_SESSION=cabi-bridge-cafc-h20-33000-v15-s41
BC_GATE_SESSION=cabi-bc-h20-gate-v15-s41
BRIDGE_GATE_SESSION=cabi-bridge-h20-gate-v15-s41
PLAIN_GATE_SESSION=cabi-cafc-h20-gate-v15-s41
GROUNDED_GATE_SESSION=cabi-bridge-cafc-h20-gate-v15-s41
MASTER_SESSION=cabi-h20-migration-gate-v15-s41

for value in "$BC_GPU" "$BRIDGE_GPU" "$PLAIN_GPU" "$GROUNDED_GPU"; do
  if [[ ! "$value" =~ ^[0-7]$ ]]; then
    echo "H20 GPU ids must be integers in [0, 7]" >&2
    exit 2
  fi
done
if [[ $(printf '%s\n' "$BC_GPU" "$BRIDGE_GPU" "$PLAIN_GPU" "$GROUNDED_GPU" | sort -u | wc -l) -ne 4 ]]; then
  echo "H20 arms require four distinct GPUs" >&2
  exit 2
fi
if [[ ! -s "$DATA_ROOT/manifest.json" || ! -s "$DATA_ROOT/anchors.npz" ]]; then
  echo "H20 v15 training view is incomplete: $DATA_ROOT" >&2
  exit 1
fi
if ! jq -e \
  '.action_horizon == 20 and .leakage_guard.fourth_corner_actions_loaded == false and .leakage_guard.horizon_reslice_uses_teacher_qa == false' \
  "$DATA_ROOT/manifest.json" >/dev/null; then
  echo "H20 v15 manifest violates the frozen leakage/horizon contract" >&2
  exit 1
fi

BC_CHECKPOINT=$RUN_ROOT/$BC_RUN_ID/final_model
BRIDGE_CHECKPOINT=$RUN_ROOT/$BRIDGE_RUN_ID/final_model
PLAIN_CHECKPOINT=$RUN_ROOT/$PLAIN_RUN_ID/final_model
GROUNDED_CHECKPOINT=$RUN_ROOT/$GROUNDED_RUN_ID/final_model
DECISION_OUTPUT=$COMPARISON_ROOT/${COMPARISON_NAME}_orchestration.json
for path in \
  "$RUN_ROOT/$BC_RUN_ID" "$RUN_ROOT/$BRIDGE_RUN_ID" \
  "$RUN_ROOT/$PLAIN_RUN_ID" "$RUN_ROOT/$GROUNDED_RUN_ID" "$DECISION_OUTPUT"; do
  if [[ -e "$path" ]]; then
    echo "refusing to overwrite H20 output: $path" >&2
    exit 1
  fi
done
for session in \
  "$BC_TRAIN_SESSION" "$BRIDGE_TRAIN_SESSION" \
  "$PLAIN_TRAIN_SESSION" "$GROUNDED_TRAIN_SESSION" \
  "$BC_GATE_SESSION" "$BRIDGE_GATE_SESSION" \
  "$PLAIN_GATE_SESSION" "$GROUNDED_GATE_SESSION" "$MASTER_SESSION"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "H20 session already exists: $session" >&2
    exit 1
  fi
done

mkdir -p "$GATE_ROOT"
launch_train() {
  local session=$1
  local mode=$2
  local tag=$3
  local gpu=$4
  tmux new-session -d -s "$session" \
    "cd '$REPO_ROOT' && env CABI_DATA_ROOT='$DATA_ROOT' CABI_RUN_TAG='$tag' scripts/cabi_vla/run_pilot_train.sh '$mode' '$SEED' '$gpu' '$STEPS'"
}
launch_gate() {
  local session=$1
  local train_session=$2
  local checkpoint=$3
  local prefix=$4
  local gpu=$5
  tmux new-session -d -s "$session" \
    "cd '$REPO_ROOT' && env CABI_TRAIN_SESSION='$train_session' CABI_DATA_ROOT='$DATA_ROOT' CABI_RELOAD_ANCHORS='$DATA_ROOT/anchors.npz' scripts/cabi_vla/run_checkpoint_gate.sh '$checkpoint' '$prefix' '$gpu' > '$GATE_ROOT/${prefix}.log' 2>&1"
}

launch_train "$BC_TRAIN_SESSION" "$BC_MODE" "$BC_TAG" "$BC_GPU"
launch_train "$BRIDGE_TRAIN_SESSION" "$BRIDGE_MODE" "$BRIDGE_TAG" "$BRIDGE_GPU"
launch_train "$PLAIN_TRAIN_SESSION" "$PLAIN_MODE" "$PLAIN_TAG" "$PLAIN_GPU"
launch_train "$GROUNDED_TRAIN_SESSION" "$GROUNDED_MODE" "$GROUNDED_TAG" "$GROUNDED_GPU"
launch_gate "$BC_GATE_SESSION" "$BC_TRAIN_SESSION" "$BC_CHECKPOINT" "$BC_PREFIX" "$BC_GPU"
launch_gate "$BRIDGE_GATE_SESSION" "$BRIDGE_TRAIN_SESSION" "$BRIDGE_CHECKPOINT" "$BRIDGE_PREFIX" "$BRIDGE_GPU"
launch_gate "$PLAIN_GATE_SESSION" "$PLAIN_TRAIN_SESSION" "$PLAIN_CHECKPOINT" "$PLAIN_PREFIX" "$PLAIN_GPU"
launch_gate "$GROUNDED_GATE_SESSION" "$GROUNDED_TRAIN_SESSION" "$GROUNDED_CHECKPOINT" "$GROUNDED_PREFIX" "$GROUNDED_GPU"

tmux new-session -d -s "$MASTER_SESSION" \
  "cd '$REPO_ROOT' && scripts/cabi_vla/run_h20_cafc_migration_gate.sh '$BC_CHECKPOINT' '$BC_PREFIX' '$BC_GATE_SESSION' '$BC_GPU' '$BRIDGE_CHECKPOINT' '$BRIDGE_PREFIX' '$BRIDGE_GATE_SESSION' '$BRIDGE_GPU' '$PLAIN_CHECKPOINT' '$PLAIN_PREFIX' '$PLAIN_GATE_SESSION' '$PLAIN_GPU' '$GROUNDED_CHECKPOINT' '$GROUNDED_PREFIX' '$GROUNDED_GATE_SESSION' '$GROUNDED_GPU' '$COMPARISON_NAME' > '$GATE_ROOT/${COMPARISON_NAME}_master.log' 2>&1"

echo "h20_cafc_launch=STARTED"
echo "bc_run=$BC_RUN_ID session=$BC_TRAIN_SESSION gpu=$BC_GPU"
echo "bridge_run=$BRIDGE_RUN_ID session=$BRIDGE_TRAIN_SESSION gpu=$BRIDGE_GPU"
echo "plain_run=$PLAIN_RUN_ID session=$PLAIN_TRAIN_SESSION gpu=$PLAIN_GPU"
echo "grounded_run=$GROUNDED_RUN_ID session=$GROUNDED_TRAIN_SESSION gpu=$GROUNDED_GPU"
echo "master_session=$MASTER_SESSION decision=$DECISION_OUTPUT"

