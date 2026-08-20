#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PROTOCOL=${DSOL_CONSTRUCTED_M1_PROTOCOL:-/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m0-v1/operational-three-task-scan-v2/constructed_m1_protocol_v1.json}
TRAIN_ROOT=${DSOL_PAIR_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs}
OUTPUT_ROOT=${DSOL_CONSTRUCTED_M1_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m1-v2}
PRACTICAL=$TRAIN_ROOT/dsol_broad_unpaired_practical_broad64-quick-gate-v1_seed41_g8_gb32_steps2000/final_model
EVALUATOR=$REPO_ROOT/scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh
ANALYZER=$REPO_ROOT/scripts/dsol_paper1/summarize_dsol_libero_m1_visibility.py
DEVICES=${DSOL_M1_EARLY_DEVICES:-6,7}

mkdir -p "$OUTPUT_ROOT/logs"
exec > >(tee -a "$OUTPUT_ROOT/early-practical-controller.log") 2>&1
printf 'early_practical_start=%s git_commit=%s devices=%s\n' \
  "$(date -u +%FT%TZ)" "$(git -C "$REPO_ROOT" rev-parse HEAD)" "$DEVICES"

[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
  echo "worktree must be clean before early M1 evaluation" >&2
  exit 1
}
[[ -s "$PROTOCOL" ]] || { echo "missing protocol: $PROTOCOL" >&2; exit 1; }
[[ -s "$PRACTICAL/model.safetensors" ]] || { echo "missing checkpoint: $PRACTICAL" >&2; exit 1; }

run_eval() {
  local name=$1 output=$2 max_per_shard=${3:-}
  mkdir -p "$output"
  CHECKPOINT="$PRACTICAL" \
  OUTPUT_DIR="$output" \
  PROTOCOL="$PROTOCOL" \
  POLICY_BACKEND=alphabrain \
  GPU_COUNT=2 \
  DSOL_GPU_DEVICES="$DEVICES" \
  BASE_PORT=19100 \
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
smoke_count=$(awk 'NF {n++} END {print n+0}' "$SMOKE"/episodes-shard-*.jsonl 2>/dev/null || true)
if [[ "$smoke_count" != 20 ]]; then
  if [[ "$smoke_count" != 0 ]]; then
    stale="$SMOKE.partial.$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$SMOKE" "$stale"
    printf 'moved_partial_smoke=%s episodes=%s\n' "$stale" "$smoke_count"
  fi
  printf 'early_protocol_smoke_start=%s\n' "$(date -u +%FT%TZ)"
  run_eval broad64-practical-smoke "$SMOKE" 10
fi
smoke_count=$(awk 'NF {n++} END {print n+0}' "$SMOKE"/episodes-shard-*.jsonl)
[[ "$smoke_count" == 20 ]] || { echo "protocol smoke expected 20 episodes, found $smoke_count" >&2; exit 1; }
printf 'early_protocol_smoke_complete=%s episodes=%s\n' "$(date -u +%FT%TZ)" "$smoke_count"

FULL=$OUTPUT_ROOT/broad64-practical
if [[ ! -s "$FULL/analysis/metrics.json" ]]; then
  partial_count=$(awk 'NF {n++} END {print n+0}' "$FULL"/episodes-shard-*.jsonl 2>/dev/null || true)
  if [[ "$partial_count" != 0 ]]; then
    stale="$FULL.partial.$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$FULL" "$stale"
    printf 'moved_partial_full=%s episodes=%s\n' "$stale" "$partial_count"
  fi
  printf 'early_practical_full_start=%s\n' "$(date -u +%FT%TZ)"
  run_eval broad64-practical "$FULL"
fi
[[ -s "$FULL/analysis/metrics.json" ]] || { echo "early practical analysis missing" >&2; exit 1; }
printf 'early_practical_complete=%s output=%s\n' \
  "$(date -u +%FT%TZ)" "$FULL/analysis/metrics.json"
