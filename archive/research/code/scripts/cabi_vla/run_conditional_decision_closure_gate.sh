#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
POLL_SECONDS=${CABI_CONDITIONAL_POLL_SECONDS:-30}
UPSTREAM_SESSION=${CABI_UPSTREAM_SESSION:-cabi-v10b-3epoch-migration-gate}
UPSTREAM_DECISION=${CABI_UPSTREAM_DECISION:-/share/longjunyu/cabi-vla/comparisons/bc_vs_cabi_grounded_coverage_phase_3epoch_seed41_v10b_orchestration.json}
DATA_ROOT=${CABI_DECISION_DATA_ROOT:-/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v8-decision-coverage-phase-loss-balanced}
RUN_ROOT=${CABI_TRAIN_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/runs}
GATE_ROOT=${CABI_GATE_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/gates}
COMPARISON_ROOT=${CABI_COMPARISON_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/comparisons}

SEED=${CABI_DECISION_SEED:-41}
STEPS=${CABI_DECISION_STEPS:-33000}
BC_GPU=${CABI_DECISION_BC_GPU:-4}
METHOD_GPU=${CABI_DECISION_METHOD_GPU:-5}

BC_MODE=cabi_bind_pi05_bc_smoke
METHOD_MODE=cabi_bind_pi05_decision_closure_smoke
BC_TAG=decision-equal-data-v11
METHOD_TAG=decision-closure-v11
BC_RUN_ID=${BC_MODE}_seed${SEED}_steps${STEPS}_${BC_TAG}
METHOD_RUN_ID=${METHOD_MODE}_seed${SEED}_steps${STEPS}_${METHOD_TAG}
BC_PREFIX=bc33000_decision_equal_data_seed${SEED}_v11
METHOD_PREFIX=cabi_decision33000_seed${SEED}_v11
COMPARISON_NAME=bc_vs_cabi_decision_closure_seed${SEED}_v11

BC_TRAIN_SESSION=cabi-bc33000-decision-v11
METHOD_TRAIN_SESSION=cabi-decision33000-v11
BC_GATE_SESSION=cabi-bc33000-decision-gate-v11
METHOD_GATE_SESSION=cabi-decision33000-gate-v11
PAIR_SESSION=cabi-v11-decision-migration-gate

if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CABI_CONDITIONAL_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi
for value in "$SEED" "$STEPS" "$BC_GPU" "$METHOD_GPU"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "seed, steps, and GPU ids must be non-negative integers" >&2
    exit 2
  fi
done
if [[ ! -s "$DATA_ROOT/manifest.json" || ! -s "$DATA_ROOT/anchors.npz" ]]; then
  echo "decision-point training view is incomplete: $DATA_ROOT" >&2
  exit 1
fi

echo "waiting upstream migration decision: $UPSTREAM_DECISION"
while [[ ! -s "$UPSTREAM_DECISION" ]]; do
  if ! tmux has-session -t "$UPSTREAM_SESSION" 2>/dev/null; then
    echo "upstream gate ended without a decision: $UPSTREAM_SESSION" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

upstream=$(jq -er '.decision' "$UPSTREAM_DECISION")
case "$upstream" in
  BASELINE_INVALID)
    echo "conditional_decision=SKIP_AMENDMENT reason=BASELINE_INVALID"
    exit 0
    ;;
  ADVANCE_TO_FULL_CONTROLS)
    echo "conditional_decision=SKIP_AMENDMENT reason=STATIC_CABI_PASSED"
    exit 0
    ;;
  PILOT_DOES_NOT_CLEAR_MIGRATION_GATE)
    baseline_rate=$(jq -er '.baseline_supervised_rate' "$UPSTREAM_DECISION")
    if ! jq -e -n --argjson rate "$baseline_rate" '$rate >= 0.70' >/dev/null; then
      echo "refusing amendment because the upstream baseline is invalid: $baseline_rate" >&2
      exit 1
    fi
    ;;
  *)
    echo "unsupported upstream decision: $upstream" >&2
    exit 1
    ;;
esac

BC_CHECKPOINT=$RUN_ROOT/$BC_RUN_ID/final_model
METHOD_CHECKPOINT=$RUN_ROOT/$METHOD_RUN_ID/final_model
DECISION_OUTPUT=$COMPARISON_ROOT/${COMPARISON_NAME}_orchestration.json
for path in "$RUN_ROOT/$BC_RUN_ID" "$RUN_ROOT/$METHOD_RUN_ID" "$DECISION_OUTPUT"; do
  if [[ -e "$path" ]]; then
    echo "refusing to overwrite conditional-run output: $path" >&2
    exit 1
  fi
done
for session in \
  "$BC_TRAIN_SESSION" "$METHOD_TRAIN_SESSION" \
  "$BC_GATE_SESSION" "$METHOD_GATE_SESSION" "$PAIR_SESSION"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "conditional-run session already exists: $session" >&2
    exit 1
  fi
done

mkdir -p "$GATE_ROOT"
tmux new-session -d -s "$BC_TRAIN_SESSION" \
  "cd '$REPO_ROOT' && env CABI_DATA_ROOT='$DATA_ROOT' CABI_RUN_TAG='$BC_TAG' scripts/cabi_vla/run_pilot_train.sh '$BC_MODE' '$SEED' '$BC_GPU' '$STEPS'"
tmux new-session -d -s "$METHOD_TRAIN_SESSION" \
  "cd '$REPO_ROOT' && env CABI_DATA_ROOT='$DATA_ROOT' CABI_RUN_TAG='$METHOD_TAG' scripts/cabi_vla/run_pilot_train.sh '$METHOD_MODE' '$SEED' '$METHOD_GPU' '$STEPS'"

tmux new-session -d -s "$BC_GATE_SESSION" \
  "cd '$REPO_ROOT' && env CABI_TRAIN_SESSION='$BC_TRAIN_SESSION' CABI_DATA_ROOT='$DATA_ROOT' CABI_RELOAD_ANCHORS='$DATA_ROOT/anchors.npz' scripts/cabi_vla/run_checkpoint_gate.sh '$BC_CHECKPOINT' '$BC_PREFIX' '$BC_GPU' > '$GATE_ROOT/${BC_PREFIX}.log' 2>&1"
tmux new-session -d -s "$METHOD_GATE_SESSION" \
  "cd '$REPO_ROOT' && env CABI_TRAIN_SESSION='$METHOD_TRAIN_SESSION' CABI_DATA_ROOT='$DATA_ROOT' CABI_RELOAD_ANCHORS='$DATA_ROOT/anchors.npz' scripts/cabi_vla/run_checkpoint_gate.sh '$METHOD_CHECKPOINT' '$METHOD_PREFIX' '$METHOD_GPU' > '$GATE_ROOT/${METHOD_PREFIX}.log' 2>&1"
tmux new-session -d -s "$PAIR_SESSION" \
  "cd '$REPO_ROOT' && env CABI_DATA_ROOT='$DATA_ROOT' scripts/cabi_vla/run_paired_migration_gate.sh '$BC_CHECKPOINT' '$BC_PREFIX' '$BC_GATE_SESSION' '$BC_GPU' '$METHOD_CHECKPOINT' '$METHOD_PREFIX' '$METHOD_GATE_SESSION' '$METHOD_GPU' '$COMPARISON_NAME' > '$GATE_ROOT/${COMPARISON_NAME}_master.log' 2>&1"

echo "conditional_decision=START_DECISION_CLOSURE bc=$BC_RUN_ID method=$METHOD_RUN_ID"
while tmux has-session -t "$PAIR_SESSION" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done
if [[ ! -s "$DECISION_OUTPUT" ]]; then
  echo "decision-closure migration gate ended without a decision: $DECISION_OUTPUT" >&2
  exit 1
fi
echo "decision_closure_result=$(jq -r '.decision' "$DECISION_OUTPUT") output=$DECISION_OUTPUT"
