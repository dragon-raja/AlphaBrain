#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${DSOL_CONSTRUCTED_ALL8_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-view-value-discovery-v1/constructed-all8-v1}
DENSE_ROOT=${DSOL_CONSTRUCTED_ALL8_DENSE_RUN_ROOT:-$ROOT/dense-discovery-run-w32}
REPEAT_ROOT=$ROOT/dense-repeatability
EVAL_WORKER_COUNT=${DSOL_CONSTRUCTED_ALL8_EVAL_WORKERS:-32}
BASE_PROTOCOL=$ROOT/dense-discovery-protocol.json
CHECKPOINT=${DSOL_CONSTRUCTED_ALL8_CHECKPOINT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_broad64-quick-gate-v1_seed41_g8_gb32_steps2000/final_model}
BUILDER=$REPO_ROOT/scripts/dsol_paper1/build_view_repeatability_protocols.py
EVALUATOR=$REPO_ROOT/scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh
SUMMARIZER=$REPO_ROOT/scripts/dsol_paper1/summarize_view_repeatability.py
PLOTTER=$REPO_ROOT/scripts/dsol_paper1/plot_view_repeatability.py
SEEDS=(20260835 20260836 20260837)

mkdir -p "$REPEAT_ROOT/logs" "$REPEAT_ROOT/protocols"
exec > >(tee -a "$REPEAT_ROOT/logs/controller.log") 2>&1

jq -e '.status == "COMPLETE" and .confirmatory_test_eligible == false' \
  "$ROOT/dense-discovery-completion.json" >/dev/null

python3 "$BUILDER" "$DENSE_ROOT"/episodes-shard-*.jsonl \
  --base-protocol "$BASE_PROTOCOL" \
  --eval-seeds "$(IFS=,; echo "${SEEDS[*]}")" \
  --shortlist-size 8 \
  --selection-mode all_states \
  --output-dir "$REPEAT_ROOT/protocols"

printf 'constructed_all8_repeat_start=%s git_commit=%s\n' \
  "$(date -u +%FT%TZ)" "$(git -C "$REPO_ROOT" rev-parse HEAD)"

for index in "${!SEEDS[@]}"; do
  seed=${SEEDS[$index]}
  protocol=$REPEAT_ROOT/protocols/protocol-seed-$seed.json
  run_root=$REPEAT_ROOT/runs/seed-$seed
  mkdir -p "$run_root"
  jq -e \
    --argjson seed "$seed" \
    '.status == "PASS" and .eval_seed == $seed and
     .selection_mode == "all_states" and .state_count == 32 and
     .candidate_count_per_state == 8 and .episode_count == 256' \
    "$protocol" >/dev/null

  actual=$(find "$run_root" -maxdepth 1 -name 'episodes-shard-*.jsonl' -type f \
    -exec awk 'NF {n++} END {print n+0}' {} + 2>/dev/null | \
    awk '{sum += $1} END {print sum + 0}')
  if [[ "$actual" == 256 ]]; then
    printf 'constructed_all8_repeat_skip_complete seed=%s\n' "$seed"
    continue
  fi

  CHECKPOINT="$CHECKPOINT" \
  OUTPUT_DIR="$run_root" \
  PROTOCOL="$protocol" \
  POLICY_BACKEND=alphabrain \
  GPU_COUNT=8 \
  EVAL_WORKER_COUNT="$EVAL_WORKER_COUNT" \
  DSOL_GPU_DEVICES=0,1,2,3,4,5,6,7 \
  BASE_PORT=$((20400 + index * 20)) \
  REPLAN_STEPS=5 \
  WAIT_STEPS=0 \
  EVAL_SEED="$seed" \
  VIDEO_EPISODES=64 \
  RUN_ANALYSIS=0 \
    "$EVALUATOR"
done

python3 "$SUMMARIZER" "$REPEAT_ROOT"/runs/seed-*/episodes-shard-*.jsonl \
  --expected-seeds "${#SEEDS[@]}" \
  --output-dir "$REPEAT_ROOT/analysis"

/alphabrain/.venv/bin/python "$PLOTTER" \
  --analysis "$REPEAT_ROOT/analysis/analysis.json" \
  --candidates "$REPEAT_ROOT/analysis/candidate_repeatability.csv" \
  --output "$REPEAT_ROOT/analysis/view_repeatability.png"

jq -n \
  --arg completed_at "$(date -u +%FT%TZ)" \
  --arg git_commit "$(git -C "$REPO_ROOT" rev-parse HEAD)" \
  '{schema:"dsol_constructed_all8_repeatability_completion_v1",status:"COMPLETE",completed_at:$completed_at,git_commit:$git_commit,selection_uses_policy_outcomes:true,confirmatory_test_eligible:false}' \
  > "$REPEAT_ROOT/completion.json"

printf 'constructed_all8_repeat_complete=%s\n' "$(date -u +%FT%TZ)"
