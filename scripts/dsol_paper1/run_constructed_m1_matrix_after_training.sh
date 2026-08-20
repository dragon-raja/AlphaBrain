#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TRAIN_SESSION=${DSOL_M1_TRAIN_SESSION:-dsol-broad64-pairing-m-a-v2}
EARLY_PRACTICAL_SESSION=${DSOL_M1_EARLY_PRACTICAL_SESSION:-dsol-constructed-m1-practical-early-v2}
POLL_SECONDS=${DSOL_WAIT_POLL_SECONDS:-30}
PROTOCOL=${DSOL_CONSTRUCTED_M1_PROTOCOL:-/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m0-v1/operational-three-task-scan-v2/constructed_m1_protocol_v1.json}
TRAIN_ROOT=${DSOL_PAIR_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs}
OUTPUT_ROOT=${DSOL_CONSTRUCTED_M1_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m1-v2}
ANALYZER=$REPO_ROOT/scripts/dsol_paper1/summarize_dsol_libero_m1_visibility.py
EVALUATOR=$REPO_ROOT/scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh
OFFICIAL=/share/longjunyu/alphabrain/pretrained_models/openpi/pi05_libero_pytorch
PRACTICAL=$TRAIN_ROOT/dsol_broad_unpaired_practical_broad64-quick-gate-v1_seed41_g8_gb32_steps2000/final_model
STATE_MATCHED=$TRAIN_ROOT/dsol_broad_unpaired_state_matched_broad64-parallel-formal-v1_seed41_g2_gb32_steps2000/final_model
PAIRED_FM=$TRAIN_ROOT/dsol_broad_paired_fm_broad64-parallel-formal-v1_seed41_g2_gb32_steps2000/final_model
PAIRED_CONSISTENCY=$TRAIN_ROOT/dsol_broad_paired_consistency_broad64-parallel-formal-v1_seed41_g2_gb32_steps2000/final_model

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid DSOL_WAIT_POLL_SECONDS" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT/logs"
exec > >(tee -a "$OUTPUT_ROOT/controller.log") 2>&1
printf 'constructed_m1_controller_start=%s git_commit=%s\n' \
  "$(date -u +%FT%TZ)" "$(git -C "$REPO_ROOT" rev-parse HEAD)"

while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
  printf 'waiting_for_training=%s at=%s\n' "$TRAIN_SESSION" "$(date -u +%FT%TZ)"
  sleep "$POLL_SECONDS"
done
while [[ ! -s "$PROTOCOL" ]]; do
  printf 'waiting_for_protocol=%s at=%s\n' "$PROTOCOL" "$(date -u +%FT%TZ)"
  sleep "$POLL_SECONDS"
done
while [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; do
  printf 'waiting_for_clean_worktree=%s at=%s\n' "$REPO_ROOT" "$(date -u +%FT%TZ)"
  sleep "$POLL_SECONDS"
done

jq -e '
  .schema == "dsol_constructed_m1_frozen_closed_loop_protocol_v1"
  and .status == "PASS"
  and .manual_audit_verified == true
  and .selected_state_count >= 20
  and .condition_count == 10
' "$PROTOCOL" >/dev/null
for checkpoint in "$OFFICIAL" "$PRACTICAL" "$STATE_MATCHED" "$PAIRED_FM" "$PAIRED_CONSISTENCY"; do
  [[ -s "$checkpoint/model.safetensors" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 1; }
done

run_eval() {
  local name=$1 checkpoint=$2 backend=$3 devices=$4 port=$5 output=$6 max_per_shard=${7:-}
  mkdir -p "$output"
  CHECKPOINT="$checkpoint" \
  OUTPUT_DIR="$output" \
  PROTOCOL="$PROTOCOL" \
  POLICY_BACKEND="$backend" \
  GPU_COUNT=2 \
  DSOL_GPU_DEVICES="$devices" \
  BASE_PORT="$port" \
  REPLAN_STEPS=5 \
  WAIT_STEPS=0 \
  EVAL_SEED=20260820 \
  VIDEO_EPISODES=999 \
  MAX_EPISODES_PER_SHARD="$max_per_shard" \
  RUN_ANALYSIS=$([[ -z "$max_per_shard" ]] && echo 1 || echo 0) \
  ANALYZER="$ANALYZER" \
    "$EVALUATOR" > "$OUTPUT_ROOT/logs/${name}.log" 2>&1
}

SMOKE=$OUTPUT_ROOT/protocol-smoke-broad64-practical
smoke_existing_count=$(awk 'NF {n++} END {print n+0}' "$SMOKE"/episodes-shard-*.jsonl 2>/dev/null || true)
while [[ "$smoke_existing_count" != 20 ]] && tmux has-session -t "$EARLY_PRACTICAL_SESSION" 2>/dev/null; do
  printf 'waiting_for_early_protocol_smoke=%s episodes=%s at=%s\n' \
    "$EARLY_PRACTICAL_SESSION" "$smoke_existing_count" "$(date -u +%FT%TZ)"
  sleep "$POLL_SECONDS"
  smoke_existing_count=$(awk 'NF {n++} END {print n+0}' "$SMOKE"/episodes-shard-*.jsonl 2>/dev/null || true)
done
if [[ "$smoke_existing_count" != 20 ]]; then
  printf 'protocol_smoke_start=%s\n' "$(date -u +%FT%TZ)"
  run_eval broad64-practical-smoke "$PRACTICAL" alphabrain 0,1 19100 "$SMOKE" 10
fi
smoke_count=$(awk 'NF {n++} END {print n+0}' "$SMOKE"/episodes-shard-*.jsonl)
[[ "$smoke_count" == 20 ]] || { echo "protocol smoke expected 20 episodes, found $smoke_count" >&2; exit 1; }
SMOKE="$SMOKE" /alphabrain/.venv/bin/python - <<'PY'
import glob
import json
import os
from collections import defaultdict

rows = []
for path in glob.glob(os.path.join(os.environ["SMOKE"], "episodes-shard-*.jsonl")):
    with open(path, encoding="utf-8") as handle:
        rows.extend(json.loads(line) for line in handle if line.strip())
by_pair = defaultdict(list)
for row in rows:
    by_pair[row["pair_key"]].append(row)
if len(by_pair) != 2 or any(len(values) != 10 for values in by_pair.values()):
    raise SystemExit("protocol smoke does not contain two complete ten-condition groups")
for pair, values in by_pair.items():
    hashes = {row["initial_metrics"]["physics_state_sha256"] for row in values}
    stages = {row["initial_metrics"]["physics_state_stage"] for row in values}
    if len(hashes) != 1 or stages != {"after_set_init_state_before_camera_install_and_wait"}:
        raise SystemExit(f"invalid exact-state pairing for {pair}")
PY
printf 'protocol_smoke_complete=%s episodes=%s\n' "$(date -u +%FT%TZ)" "$smoke_count"

declare -a pids=()
launch_eval() {
  local name=$1 checkpoint=$2 backend=$3 devices=$4 port=$5
  local output=$OUTPUT_ROOT/$name
  if [[ -s "$output/analysis/metrics.json" ]]; then
    printf 'reuse_complete_eval=%s\n' "$name"
    return
  fi
  (
    run_eval "$name" "$checkpoint" "$backend" "$devices" "$port" "$output"
  ) &
  pids+=("$!")
  printf 'launched_eval=%s pid=%s devices=%s\n' "$name" "$!" "$devices"
}

launch_eval official "$OFFICIAL" openpi 0,1 19200
launch_eval broad64-state-matched "$STATE_MATCHED" alphabrain 2,3 19240
launch_eval broad64-paired-fm "$PAIRED_FM" alphabrain 4,5 19260
(
  while tmux has-session -t "$EARLY_PRACTICAL_SESSION" 2>/dev/null; do
    printf 'waiting_for_early_practical=%s at=%s\n' "$EARLY_PRACTICAL_SESSION" "$(date -u +%FT%TZ)"
    sleep "$POLL_SECONDS"
  done
  if [[ ! -s "$OUTPUT_ROOT/broad64-practical/analysis/metrics.json" ]]; then
    run_eval broad64-practical "$PRACTICAL" alphabrain 6,7 19220 "$OUTPUT_ROOT/broad64-practical"
  else
    printf 'reuse_complete_eval=broad64-practical\n'
  fi
  [[ -s "$OUTPUT_ROOT/broad64-practical/analysis/metrics.json" ]] || {
    echo "broad64-practical evaluation did not complete" >&2
    exit 1
  }
  if [[ ! -s "$OUTPUT_ROOT/broad64-paired-consistency/analysis/metrics.json" ]]; then
    run_eval broad64-paired-consistency "$PAIRED_CONSISTENCY" alphabrain 6,7 19300 "$OUTPUT_ROOT/broad64-paired-consistency"
  else
    printf 'reuse_complete_eval=broad64-paired-consistency\n'
  fi
) &
pids+=("$!")
printf 'launched_eval_lane=practical_then_consistency pid=%s devices=6,7\n' "$!"
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
[[ "$failed" == 0 ]] || { echo "constructed M1 parallel lanes failed" >&2; exit 1; }

comparison=$OUTPUT_ROOT/cross-model-analysis
/alphabrain/.venv/bin/python "$REPO_ROOT/scripts/dsol_paper1/compare_dsol_libero_m1_models.py" \
  --run official="$OUTPUT_ROOT/official" \
  --run broad64-practical="$OUTPUT_ROOT/broad64-practical" \
  --run broad64-state-matched="$OUTPUT_ROOT/broad64-state-matched" \
  --run broad64-paired-fm="$OUTPUT_ROOT/broad64-paired-fm" \
  --run broad64-paired-consistency="$OUTPUT_ROOT/broad64-paired-consistency" \
  --baseline official \
  --output-dir "$comparison"
printf 'constructed_m1_controller_complete=%s output=%s\n' \
  "$(date -u +%FT%TZ)" "$comparison/metrics.json"
