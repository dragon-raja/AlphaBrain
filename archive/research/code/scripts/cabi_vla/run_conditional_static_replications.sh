#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
POLL_SECONDS=${CABI_REPLICATION_POLL_SECONDS:-30}
UPSTREAM_SESSION=${CABI_UPSTREAM_SESSION:-cabi-v10b-3epoch-migration-gate}
UPSTREAM_DECISION=${CABI_UPSTREAM_DECISION:-/share/longjunyu/cabi-vla/comparisons/bc_vs_cabi_grounded_coverage_phase_3epoch_seed41_v10b_orchestration.json}
DATA_ROOT=${CABI_REPLICATION_DATA_ROOT:-/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v7-coverage-phase-loss-balanced}
RUN_ROOT=${CABI_TRAIN_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/runs}
GATE_ROOT=${CABI_GATE_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/gates}
COMPARISON_ROOT=${CABI_COMPARISON_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/comparisons}

STEPS=33000
TAG=coverage-phase-3epoch-v10b
BC_MODE=cabi_bind_pi05_bc_smoke
METHOD_MODE=cabi_bind_pi05_grounded_contrastive_smoke

if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CABI_REPLICATION_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

echo "waiting seed-41 static migration decision: $UPSTREAM_DECISION"
while [[ ! -s "$UPSTREAM_DECISION" ]]; do
  if ! tmux has-session -t "$UPSTREAM_SESSION" 2>/dev/null; then
    echo "seed-41 gate ended without a decision: $UPSTREAM_SESSION" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

upstream=$(jq -er '.decision' "$UPSTREAM_DECISION")
if [[ "$upstream" != "ADVANCE_TO_FULL_CONTROLS" ]]; then
  echo "replication_decision=SEALED_NOT_EVALUATED reason=$upstream"
  exit 0
fi

launch_seed_gate() {
  local seed=$1
  local bc_gpu=$2
  local method_gpu=$3
  local bc_train_session=cabi-bc33000-coverage-phase-v10b-s${seed}
  local method_train_session=cabi-grounded33000-coverage-phase-v10b-s${seed}
  local bc_gate_session=cabi-bc33000-gate-v10b-s${seed}
  local method_gate_session=cabi-grounded33000-gate-v10b-s${seed}
  local pair_session=cabi-v10b-replication-gate-s${seed}
  local bc_run_id=${BC_MODE}_seed${seed}_steps${STEPS}_${TAG}
  local method_run_id=${METHOD_MODE}_seed${seed}_steps${STEPS}_${TAG}
  local bc_checkpoint=$RUN_ROOT/$bc_run_id/final_model
  local method_checkpoint=$RUN_ROOT/$method_run_id/final_model
  local bc_prefix=bc33000_coverage_phase_3epoch_seed${seed}_v10b
  local method_prefix=cabi_grounded33000_coverage_phase_3epoch_seed${seed}_v10b
  local comparison_name=bc_vs_cabi_grounded_coverage_phase_3epoch_seed${seed}_v10b
  local decision_output=$COMPARISON_ROOT/${comparison_name}_orchestration.json

  if [[ -e "$decision_output" ]]; then
    echo "refusing to overwrite replication decision: $decision_output" >&2
    return 1
  fi
  for session in "$bc_gate_session" "$method_gate_session" "$pair_session"; do
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "replication gate session already exists: $session" >&2
      return 1
    fi
  done

  tmux new-session -d -s "$bc_gate_session" \
    "cd '$REPO_ROOT' && env CABI_TRAIN_SESSION='$bc_train_session' CABI_DATA_ROOT='$DATA_ROOT' CABI_RELOAD_ANCHORS='$DATA_ROOT/anchors.npz' scripts/cabi_vla/run_checkpoint_gate.sh '$bc_checkpoint' '$bc_prefix' '$bc_gpu' > '$GATE_ROOT/${bc_prefix}.log' 2>&1"
  tmux new-session -d -s "$method_gate_session" \
    "cd '$REPO_ROOT' && env CABI_TRAIN_SESSION='$method_train_session' CABI_DATA_ROOT='$DATA_ROOT' CABI_RELOAD_ANCHORS='$DATA_ROOT/anchors.npz' scripts/cabi_vla/run_checkpoint_gate.sh '$method_checkpoint' '$method_prefix' '$method_gpu' > '$GATE_ROOT/${method_prefix}.log' 2>&1"
  tmux new-session -d -s "$pair_session" \
    "cd '$REPO_ROOT' && env CABI_DATA_ROOT='$DATA_ROOT' scripts/cabi_vla/run_paired_migration_gate.sh '$bc_checkpoint' '$bc_prefix' '$bc_gate_session' '$bc_gpu' '$method_checkpoint' '$method_prefix' '$method_gate_session' '$method_gpu' '$comparison_name' > '$GATE_ROOT/${comparison_name}_master.log' 2>&1"
  echo "$pair_session|$decision_output"
}

mkdir -p "$GATE_ROOT"
seed42=$(launch_seed_gate 42 0 1)
seed43=$(launch_seed_gate 43 2 3)
echo "replication_decision=START_EVALUATION seeds=42,43"

for item in "$seed42" "$seed43"; do
  session=${item%%|*}
  output=${item#*|}
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
  if [[ ! -s "$output" ]]; then
    echo "replication gate ended without decision: $output" >&2
    exit 1
  fi
  echo "replication_result seed=${session##*-s} decision=$(jq -r '.decision' "$output") output=$output"
done
