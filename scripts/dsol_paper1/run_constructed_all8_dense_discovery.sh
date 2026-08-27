#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${DSOL_CONSTRUCTED_ALL8_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-view-value-discovery-v1/constructed-all8-v1}
PROTOCOL=${DSOL_CONSTRUCTED_ALL8_DENSE_PROTOCOL:-$ROOT/dense-discovery-protocol.json}
CHECKPOINT=${DSOL_CONSTRUCTED_ALL8_CHECKPOINT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_broad64-quick-gate-v1_seed41_g8_gb32_steps2000/final_model}
RUN_ROOT=$ROOT/dense-discovery-run
EVALUATOR=$REPO_ROOT/scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh
ANALYZER=$REPO_ROOT/scripts/dsol_paper1/analyze_view_value_discovery.py

mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/dense-discovery-controller.log") 2>&1

jq -e '.status == "PASS" and .selected_state_count == 32 and
  .source_episode_count == 16 and .candidate_count == 97 and
  .episode_count == 3104 and .confirmatory_test_eligible == false' \
  "$PROTOCOL" >/dev/null
[[ -s "$CHECKPOINT/model.safetensors" ]]

printf 'constructed_all8_dense_start=%s git_commit=%s\n' \
  "$(date -u +%FT%TZ)" "$(git -C "$REPO_ROOT" rev-parse HEAD)"

CHECKPOINT="$CHECKPOINT" \
OUTPUT_DIR="$RUN_ROOT" \
PROTOCOL="$PROTOCOL" \
POLICY_BACKEND=alphabrain \
GPU_COUNT=8 \
DSOL_GPU_DEVICES=0,1,2,3,4,5,6,7 \
BASE_PORT=20300 \
REPLAN_STEPS=5 \
WAIT_STEPS=0 \
EVAL_SEED=20260834 \
VIDEO_EPISODES=64 \
RUN_ANALYSIS=0 \
  "$EVALUATOR"

python3 "$ANALYZER" "$RUN_ROOT"/episodes-shard-*.jsonl \
  --output-dir "$RUN_ROOT/analysis_discovery"

jq -n \
  --arg completed_at "$(date -u +%FT%TZ)" \
  --arg git_commit "$(git -C "$REPO_ROOT" rev-parse HEAD)" \
  '{schema:"dsol_constructed_all8_dense_completion_v1",status:"COMPLETE",completed_at:$completed_at,git_commit:$git_commit,selection_uses_policy_outcomes:true,confirmatory_test_eligible:false}' \
  > "$ROOT/dense-discovery-completion.json"

printf 'constructed_all8_dense_complete=%s\n' "$(date -u +%FT%TZ)"
