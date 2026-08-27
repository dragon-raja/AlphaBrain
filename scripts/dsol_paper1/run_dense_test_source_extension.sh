#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BASE_ROOT=${DSOL_CONSTRUCTED_ALL8_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-view-value-discovery-v1/constructed-all8-v1}
ROOT=${DSOL_DENSE_TEST_EXTENSION_ROOT:-$BASE_ROOT/independent-source-extension}
PAIR_ROOT=${DSOL_LIBERO_PAIR_ROOT:-/share/longjunyu/alphabrain/datasets/dsol-libero-broad-pairs-v1/quick_gate_seed41_broad32_stride2}
HDF5_ROOT=${DSOL_LIBERO_HDF5_ROOT:-/share/longjunyu/alphabrain/datasets/libero-original-hdf5-v1}
CATALOG=$REPO_ROOT/configs/dsol_paper1/libero_view_catalog_v2_m1.json
COLLECTION_PLAN=$REPO_ROOT/configs/dsol_paper1/libero_pair_quick_gate_v1.json
OLD_SCAN_PLAN=$BASE_ROOT/scan-plan.json
SOURCE_PLAN=$ROOT/source-scan-plan.json
CONSTRUCTION_PLAN=$ROOT/construction-scan-plan.json
CONSTRUCTION_SCAN=$ROOT/construction-scan
DENSE_PROTOCOL=$ROOT/dense-test-protocol.json
FEATURE_PLAN=$ROOT/test-feature-scan-plan.json
FEATURE_SCAN=$ROOT/test-feature-scan
SELECTOR_PROTOCOL=$ROOT/dense-test-selector-protocol.json
RUN_ROOT=$ROOT/dense-test-selector-eval
COMBINED_ANALYSIS=$ROOT/combined-analysis
CHECKPOINT=${DSOL_CONSTRUCTED_ALL8_CHECKPOINT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_broad64-quick-gate-v1_seed41_g8_gb32_steps2000/final_model}
SIM_PYTHON=${DSOL_SIM_PYTHON:-/workspace/envs/fresh-libero/bin/python}
ANALYSIS_PYTHON=${DSOL_ANALYSIS_PYTHON:-/alphabrain/.venv/bin/python}
EVAL_WORKER_COUNT=${DSOL_CONSTRUCTED_ALL8_EVAL_WORKERS:-32}
SEEDS=(20260841 20260842 20260843 20260844 20260845)

mkdir -p "$ROOT/logs" "$RUN_ROOT/logs"
exec > >(tee -a "$ROOT/logs/controller.log") 2>&1

printf 'dense_test_source_extension_start=%s git_commit=%s\n' \
  "$(date -u +%FT%TZ)" "$(git -C "$REPO_ROOT" rev-parse HEAD)"

"$SIM_PYTHON" "$REPO_ROOT/scripts/dsol_paper1/build_libero_visibility_scan_plan.py" \
  --pair-root "$PAIR_ROOT" \
  --collection-plan "$COLLECTION_PLAN" \
  --hdf5-root "$HDF5_ROOT" \
  --output "$SOURCE_PLAN" \
  --seed 20260827 \
  --splits test \
  --all-test-episodes \
  --exclude-plan "$OLD_SCAN_PLAN" \
  --allow-empty-task-splits \
  --stage-fractions 0.05,0.15,0.25,0.35,0.45,0.55,0.65

"$SIM_PYTHON" "$REPO_ROOT/scripts/dsol_paper1/build_strong_information_gate_plan.py" \
  --source-plan "$SOURCE_PLAN" \
  --maximum-stage 0.65 \
  --task-spec goal_cream_cheese_bowl="$REPO_ROOT/configs/dsol_paper1/libero_constructed_goal_cream_cheese_bowl_v1.json" \
  --task-spec goal_top_drawer_bowl="$REPO_ROOT/configs/dsol_paper1/libero_constructed_goal_top_drawer_bowl_v1.json" \
  --task-spec goal_wine_rack="$REPO_ROOT/configs/dsol_paper1/libero_constructed_goal_wine_rack_v1.json" \
  --task-spec libero10_book_caddy="$REPO_ROOT/configs/dsol_paper1/libero_constructed_libero10_book_caddy_v1.json" \
  --task-spec libero10_bowl_bottom_drawer="$REPO_ROOT/configs/dsol_paper1/libero_constructed_libero10_bowl_bottom_drawer_v1.json" \
  --task-spec libero10_mug_microwave="$REPO_ROOT/configs/dsol_paper1/libero_constructed_libero10_mug_microwave_v1.json" \
  --task-spec object_cream_cheese_basket="$REPO_ROOT/configs/dsol_paper1/libero_constructed_object_cream_cheese_basket_v1.json" \
  --output "$CONSTRUCTION_PLAN"

jq -e '.status == "PASS" and .record_count == 114' "$CONSTRUCTION_PLAN" >/dev/null
if [[ ! -s "$CONSTRUCTION_SCAN/analysis/summary.json" ]] || \
  ! jq -e '.scan_count == 114' "$CONSTRUCTION_SCAN/analysis/summary.json" >/dev/null; then
  PLAN="$CONSTRUCTION_PLAN" \
  CATALOG="$CATALOG" \
  OUTPUT_ROOT="$CONSTRUCTION_SCAN" \
  CANDIDATE_GROUPS=canonical \
  GPU_COUNT=8 \
  DSOL_GPU_DEVICES=0,1,2,3,4,5,6,7 \
    "$REPO_ROOT/scripts/dsol_paper1/run_libero_visibility_scan_matrix.sh"
fi

"$SIM_PYTHON" "$REPO_ROOT/scripts/dsol_paper1/build_constructed_view_oracle_protocol.py" \
  "$CONSTRUCTION_SCAN"/shard-*.jsonl \
  --catalog "$CATALOG" \
  --split test \
  --stage-targets 0.20,0.55 \
  --output "$DENSE_PROTOCOL"
jq -e '.status == "PASS" and .selected_state_count == 36 and
  .source_episode_count == 18 and .candidate_count == 97 and
  .episode_count == 3492' "$DENSE_PROTOCOL" >/dev/null

"$SIM_PYTHON" "$REPO_ROOT/scripts/dsol_paper1/build_dense_test_scan_plan.py" \
  --source-plan "$CONSTRUCTION_PLAN" \
  --protocol "$DENSE_PROTOCOL" \
  --output "$FEATURE_PLAN"
jq -e '.status == "PASS" and .record_count == 36' "$FEATURE_PLAN" >/dev/null

POSE_IDS=$(jq -r \
  '[.broad_training_64[], .broad_heldout_32[]] | map(.pose_id) | join(",")' \
  "$CATALOG")
if [[ ! -s "$FEATURE_SCAN/analysis/summary.json" ]] || \
  ! jq -e '.scan_count == 36' "$FEATURE_SCAN/analysis/summary.json" >/dev/null; then
  PLAN="$FEATURE_PLAN" \
  CATALOG="$CATALOG" \
  OUTPUT_ROOT="$FEATURE_SCAN" \
  CANDIDATE_GROUPS=broad_training_64,broad_heldout_32 \
  POSE_IDS="$POSE_IDS" \
  GPU_COUNT=8 \
  DSOL_GPU_DEVICES=0,1,2,3,4,5,6,7 \
    "$REPO_ROOT/scripts/dsol_paper1/run_libero_visibility_scan_matrix.sh"
fi

"$ANALYSIS_PYTHON" "$REPO_ROOT/scripts/dsol_paper1/build_dense_test_selector_protocol.py" \
  "$FEATURE_SCAN"/shard-*.jsonl \
  --dense-test-protocol "$DENSE_PROTOCOL" \
  --global-fixed-candidate broad_heldout_014 \
  --visibility-gain-threshold 0.005 \
  --output "$SELECTOR_PROTOCOL"
jq -e '.status == "PASS" and .selection_uses_test_policy_outcomes == false and
  .selected_state_count == 36 and .source_episode_count == 18 and
  .selector_method_count == 6 and .episode_count == 216' \
  "$SELECTOR_PROTOCOL" >/dev/null

for index in "${!SEEDS[@]}"; do
  seed=${SEEDS[$index]}
  output=$RUN_ROOT/runs/seed-$seed
  mkdir -p "$output"
  actual=$(find "$output" -maxdepth 1 -name 'episodes-shard-*.jsonl' -type f \
    -exec awk 'NF {n++} END {print n+0}' {} + 2>/dev/null | \
    awk '{sum += $1} END {print sum + 0}')
  if [[ "$actual" == 216 ]]; then
    printf 'dense_test_source_extension_skip seed=%s\n' "$seed"
    continue
  fi
  CHECKPOINT="$CHECKPOINT" \
  OUTPUT_DIR="$output" \
  PROTOCOL="$SELECTOR_PROTOCOL" \
  POLICY_BACKEND=alphabrain \
  GPU_COUNT=8 \
  EVAL_WORKER_COUNT="$EVAL_WORKER_COUNT" \
  DSOL_GPU_DEVICES=0,1,2,3,4,5,6,7 \
  BASE_PORT=$((20900 + index * 20)) \
  REPLAN_STEPS=5 \
  WAIT_STEPS=0 \
  EVAL_SEED="$seed" \
  VIDEO_EPISODES=1 \
  RUN_ANALYSIS=0 \
    "$REPO_ROOT/scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh"
done

"$ANALYSIS_PYTHON" "$REPO_ROOT/scripts/dsol_paper1/summarize_dense_test_selectors.py" \
  "$RUN_ROOT"/runs/seed-*/episodes-shard-*.jsonl \
  --expected-repeats "${#SEEDS[@]}" \
  --output-dir "$RUN_ROOT/analysis"
"$ANALYSIS_PYTHON" "$REPO_ROOT/scripts/dsol_paper1/plot_dense_test_selectors.py" \
  --analysis "$RUN_ROOT/analysis/analysis.json" \
  --output "$RUN_ROOT/analysis/dense_test_selectors.png"

"$ANALYSIS_PYTHON" "$REPO_ROOT/scripts/dsol_paper1/summarize_dense_test_selectors.py" \
  "$BASE_ROOT"/dense-test-selector-eval/runs/seed-*/episodes-shard-*.jsonl \
  "$RUN_ROOT"/runs/seed-*/episodes-shard-*.jsonl \
  --expected-repeats "${#SEEDS[@]}" \
  --output-dir "$COMBINED_ANALYSIS"
"$ANALYSIS_PYTHON" "$REPO_ROOT/scripts/dsol_paper1/plot_dense_test_selectors.py" \
  --analysis "$COMBINED_ANALYSIS/analysis.json" \
  --output "$COMBINED_ANALYSIS/dense_test_selectors.png"

jq -n \
  --arg completed_at "$(date -u +%FT%TZ)" \
  --arg git_commit "$(git -C "$REPO_ROOT" rev-parse HEAD)" \
  --argjson repeats "${#SEEDS[@]}" \
  '{schema:"dsol_dense_test_source_extension_completion_v1",status:"COMPLETE",completed_at:$completed_at,git_commit:$git_commit,new_source_groups:18,new_states:36,new_episodes:1080,combined_source_groups:42,combined_states:84,combined_episodes:2520,policy_noise_repeats:$repeats,selection_uses_test_policy_outcomes:false}' \
  > "$ROOT/completion.json"

printf 'dense_test_source_extension_complete=%s\n' "$(date -u +%FT%TZ)"
