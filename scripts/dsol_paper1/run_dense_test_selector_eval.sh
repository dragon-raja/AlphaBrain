#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${DSOL_CONSTRUCTED_ALL8_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-view-value-discovery-v1/constructed-all8-v1}
PROTOCOL=${DSOL_DENSE_TEST_SELECTOR_PROTOCOL:-$ROOT/dense-test-selector-protocol.json}
RUN_ROOT=${DSOL_DENSE_TEST_SELECTOR_RUN_ROOT:-$ROOT/dense-test-selector-eval}
CHECKPOINT=${DSOL_CONSTRUCTED_ALL8_CHECKPOINT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_broad64-quick-gate-v1_seed41_g8_gb32_steps2000/final_model}
EVALUATOR=$REPO_ROOT/scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh
SUMMARIZER=$REPO_ROOT/scripts/dsol_paper1/summarize_dense_test_selectors.py
PLOTTER=$REPO_ROOT/scripts/dsol_paper1/plot_dense_test_selectors.py
ANALYSIS_PYTHON=${DSOL_ANALYSIS_PYTHON:-/alphabrain/.venv/bin/python}
EVAL_WORKER_COUNT=${DSOL_CONSTRUCTED_ALL8_EVAL_WORKERS:-32}
SEEDS=(20260841 20260842 20260843 20260844 20260845)

mkdir -p "$RUN_ROOT/logs"
exec > >(tee -a "$RUN_ROOT/logs/controller.log") 2>&1

jq -e '.status == "PASS" and .split == "test" and
  .selection_uses_test_policy_outcomes == false and
  .selected_state_count == 48 and .source_episode_count == 24 and
  .selector_method_count == 6 and .episode_count == 288' "$PROTOCOL" >/dev/null
[[ -s "$CHECKPOINT/model.safetensors" ]]

printf 'dense_test_selector_start=%s git_commit=%s\n' \
  "$(date -u +%FT%TZ)" "$(git -C "$REPO_ROOT" rev-parse HEAD)"

for index in "${!SEEDS[@]}"; do
  seed=${SEEDS[$index]}
  output=$RUN_ROOT/runs/seed-$seed
  mkdir -p "$output"
  actual=$(find "$output" -maxdepth 1 -name 'episodes-shard-*.jsonl' -type f \
    -exec awk 'NF {n++} END {print n+0}' {} + 2>/dev/null | \
    awk '{sum += $1} END {print sum + 0}')
  if [[ "$actual" == 288 ]]; then
    printf 'dense_test_selector_skip_complete seed=%s\n' "$seed"
    continue
  fi
  CHECKPOINT="$CHECKPOINT" \
  OUTPUT_DIR="$output" \
  PROTOCOL="$PROTOCOL" \
  POLICY_BACKEND=alphabrain \
  GPU_COUNT=8 \
  EVAL_WORKER_COUNT="$EVAL_WORKER_COUNT" \
  DSOL_GPU_DEVICES=0,1,2,3,4,5,6,7 \
  BASE_PORT=$((20700 + index * 20)) \
  REPLAN_STEPS=5 \
  WAIT_STEPS=0 \
  EVAL_SEED="$seed" \
  VIDEO_EPISODES=1 \
  RUN_ANALYSIS=0 \
    "$EVALUATOR"
done

"$ANALYSIS_PYTHON" "$SUMMARIZER" \
  "$RUN_ROOT"/runs/seed-*/episodes-shard-*.jsonl \
  --expected-repeats "${#SEEDS[@]}" \
  --output-dir "$RUN_ROOT/analysis"

"$ANALYSIS_PYTHON" "$PLOTTER" \
  --analysis "$RUN_ROOT/analysis/analysis.json" \
  --output "$RUN_ROOT/analysis/dense_test_selectors.png"

jq -n \
  --arg completed_at "$(date -u +%FT%TZ)" \
  --arg git_commit "$(git -C "$REPO_ROOT" rev-parse HEAD)" \
  --argjson repeats "${#SEEDS[@]}" \
  --argjson episodes "$((288 * ${#SEEDS[@]}))" \
  '{schema:"dsol_dense_test_selector_completion_v1",status:"COMPLETE",completed_at:$completed_at,git_commit:$git_commit,selection_uses_test_policy_outcomes:false,policy_noise_repeats:$repeats,total_episodes:$episodes}' \
  > "$RUN_ROOT/completion.json"

printf 'dense_test_selector_complete=%s\n' "$(date -u +%FT%TZ)"
