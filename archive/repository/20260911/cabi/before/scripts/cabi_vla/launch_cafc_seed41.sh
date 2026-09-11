#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${CAFC_DATA_ROOT:-/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v13-decision-observed-edge-phase-loss-balanced}
RUN_ROOT=${CABI_TRAIN_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/runs}
GATE_ROOT=${CABI_GATE_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/gates}
COMPARISON_ROOT=${CABI_COMPARISON_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/comparisons}

SEED=${CAFC_SEED:-41}
STEPS=${CAFC_STEPS:-33000}
PLAIN_GPU=${CAFC_PLAIN_GPU:-4}
GROUNDED_GPU=${CAFC_GROUNDED_GPU:-5}
PLAIN_MODE=cabi_bind_pi05_action_completion_smoke
GROUNDED_MODE=cabi_bind_pi05_action_bridge_completion_smoke
PLAIN_TAG=cafc-edge-balanced-3epoch-v14
GROUNDED_TAG=cafc-bridge-edge-balanced-3epoch-v14
PLAIN_RUN_ID=${PLAIN_MODE}_seed${SEED}_steps${STEPS}_${PLAIN_TAG}
GROUNDED_RUN_ID=${GROUNDED_MODE}_seed${SEED}_steps${STEPS}_${GROUNDED_TAG}
PLAIN_PREFIX=cafc33000_edge_balanced_seed${SEED}_v14
GROUNDED_PREFIX=cabi_bridge_cafc33000_edge_balanced_seed${SEED}_v14
COMPARISON_NAME=cafc_action_field_migration_seed${SEED}_v14

PLAIN_TRAIN_SESSION=cabi-cafc33000-v14-s${SEED}
GROUNDED_TRAIN_SESSION=cabi-bridge-cafc33000-v14-s${SEED}
PLAIN_GATE_SESSION=cabi-cafc33000-gate-v14-s${SEED}
GROUNDED_GATE_SESSION=cabi-bridge-cafc33000-gate-v14-s${SEED}
MASTER_SESSION=cabi-cafc-migration-gate-v14-s${SEED}

for value in "$SEED" "$STEPS" "$PLAIN_GPU" "$GROUNDED_GPU"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "seed, steps, and GPU ids must be non-negative integers" >&2
    exit 2
  fi
done
if [[ ! -s "$DATA_ROOT/manifest.json" || ! -s "$DATA_ROOT/anchors.npz" ]]; then
  echo "CAFC v13 training view is incomplete: $DATA_ROOT" >&2
  exit 1
fi

PLAIN_CHECKPOINT=$RUN_ROOT/$PLAIN_RUN_ID/final_model
GROUNDED_CHECKPOINT=$RUN_ROOT/$GROUNDED_RUN_ID/final_model
DECISION_OUTPUT=$COMPARISON_ROOT/${COMPARISON_NAME}_orchestration.json
for path in "$RUN_ROOT/$PLAIN_RUN_ID" "$RUN_ROOT/$GROUNDED_RUN_ID" "$DECISION_OUTPUT"; do
  if [[ -e "$path" ]]; then
    echo "refusing to overwrite CAFC output: $path" >&2
    exit 1
  fi
done
for session in \
  "$PLAIN_TRAIN_SESSION" "$GROUNDED_TRAIN_SESSION" \
  "$PLAIN_GATE_SESSION" "$GROUNDED_GATE_SESSION" "$MASTER_SESSION"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "CAFC session already exists: $session" >&2
    exit 1
  fi
done

mkdir -p "$GATE_ROOT"
tmux new-session -d -s "$PLAIN_TRAIN_SESSION" \
  "cd '$REPO_ROOT' && env CABI_DATA_ROOT='$DATA_ROOT' CABI_RUN_TAG='$PLAIN_TAG' scripts/cabi_vla/run_pilot_train.sh '$PLAIN_MODE' '$SEED' '$PLAIN_GPU' '$STEPS'"
tmux new-session -d -s "$GROUNDED_TRAIN_SESSION" \
  "cd '$REPO_ROOT' && env CABI_DATA_ROOT='$DATA_ROOT' CABI_RUN_TAG='$GROUNDED_TAG' scripts/cabi_vla/run_pilot_train.sh '$GROUNDED_MODE' '$SEED' '$GROUNDED_GPU' '$STEPS'"

tmux new-session -d -s "$PLAIN_GATE_SESSION" \
  "cd '$REPO_ROOT' && env CABI_TRAIN_SESSION='$PLAIN_TRAIN_SESSION' CABI_DATA_ROOT='$DATA_ROOT' CABI_RELOAD_ANCHORS='$DATA_ROOT/anchors.npz' scripts/cabi_vla/run_checkpoint_gate.sh '$PLAIN_CHECKPOINT' '$PLAIN_PREFIX' '$PLAIN_GPU' > '$GATE_ROOT/${PLAIN_PREFIX}.log' 2>&1"
tmux new-session -d -s "$GROUNDED_GATE_SESSION" \
  "cd '$REPO_ROOT' && env CABI_TRAIN_SESSION='$GROUNDED_TRAIN_SESSION' CABI_DATA_ROOT='$DATA_ROOT' CABI_RELOAD_ANCHORS='$DATA_ROOT/anchors.npz' scripts/cabi_vla/run_checkpoint_gate.sh '$GROUNDED_CHECKPOINT' '$GROUNDED_PREFIX' '$GROUNDED_GPU' > '$GATE_ROOT/${GROUNDED_PREFIX}.log' 2>&1"
tmux new-session -d -s "$MASTER_SESSION" \
  "cd '$REPO_ROOT' && scripts/cabi_vla/run_cafc_migration_gate.sh '$PLAIN_CHECKPOINT' '$PLAIN_PREFIX' '$PLAIN_GATE_SESSION' '$PLAIN_GPU' '$GROUNDED_CHECKPOINT' '$GROUNDED_PREFIX' '$GROUNDED_GATE_SESSION' '$GROUNDED_GPU' '$COMPARISON_NAME' > '$GATE_ROOT/${COMPARISON_NAME}_master.log' 2>&1"

echo "cafc_launch=STARTED"
echo "plain_run=$PLAIN_RUN_ID session=$PLAIN_TRAIN_SESSION gpu=$PLAIN_GPU"
echo "grounded_run=$GROUNDED_RUN_ID session=$GROUNDED_TRAIN_SESSION gpu=$GROUNDED_GPU"
echo "master_session=$MASTER_SESSION decision=$DECISION_OUTPUT"
