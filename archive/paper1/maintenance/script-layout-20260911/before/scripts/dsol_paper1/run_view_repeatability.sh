#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${DSOL_VIEW_REPEATABILITY_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-view-value-discovery-v1/constructed-repeatability-v1}
CHECKPOINT=${DSOL_VIEW_REPEATABILITY_CHECKPOINT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_broad64-quick-gate-v1_seed41_g8_gb32_steps2000/final_model}
EVALUATOR=$REPO_ROOT/scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh
SUMMARIZER=$REPO_ROOT/scripts/dsol_paper1/summarize_view_repeatability.py
SEEDS=(${DSOL_VIEW_REPEATABILITY_SEEDS:-20260831 20260832 20260833})

mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/controller.log") 2>&1

[[ -s "$CHECKPOINT/model.safetensors" ]]
printf 'view_repeatability_start=%s git_commit=%s\n' \
  "$(date -u +%FT%TZ)" "$(git -C "$REPO_ROOT" rev-parse HEAD)"

for index in "${!SEEDS[@]}"; do
  seed=${SEEDS[$index]}
  protocol=$ROOT/protocols/protocol-seed-$seed.json
  run_root=$ROOT/runs/seed-$seed
  jq -e \
    --argjson seed "$seed" \
    '.status == "PASS" and .eval_seed == $seed and .state_count == 4 and
     .candidate_count_per_state == 8 and .episode_count == 32' \
    "$protocol" >/dev/null

  if [[ $(find "$run_root" -maxdepth 1 -name 'episodes-shard-*.jsonl' -type f \
      -exec awk 'NF {n++} END {print n+0}' {} + 2>/dev/null | awk '{s+=$1} END {print s+0}') == 32 ]]; then
    printf 'view_repeatability_skip_complete seed=%s\n' "$seed"
    continue
  fi

  CHECKPOINT="$CHECKPOINT" \
  OUTPUT_DIR="$run_root" \
  PROTOCOL="$protocol" \
  POLICY_BACKEND=alphabrain \
  GPU_COUNT=8 \
  DSOL_GPU_DEVICES=0,1,2,3,4,5,6,7 \
  BASE_PORT=$((20200 + index * 20)) \
  REPLAN_STEPS=5 \
  WAIT_STEPS=0 \
  EVAL_SEED="$seed" \
  VIDEO_EPISODES=32 \
  RUN_ANALYSIS=0 \
    "$EVALUATOR"
done

python3 "$SUMMARIZER" "$ROOT"/runs/seed-*/episodes-shard-*.jsonl \
  --expected-seeds "${#SEEDS[@]}" \
  --output-dir "$ROOT/analysis"

jq -n \
  --arg completed_at "$(date -u +%FT%TZ)" \
  --arg git_commit "$(git -C "$REPO_ROOT" rev-parse HEAD)" \
  --argjson seed_count "${#SEEDS[@]}" \
  '{schema:"dsol_view_repeatability_completion_v1",status:"COMPLETE",completed_at:$completed_at,git_commit:$git_commit,seed_count:$seed_count}' \
  > "$ROOT/completion.json"

printf 'view_repeatability_complete=%s\n' "$(date -u +%FT%TZ)"
