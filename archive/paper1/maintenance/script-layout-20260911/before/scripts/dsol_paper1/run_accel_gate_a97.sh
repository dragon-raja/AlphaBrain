#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUTPUT_ROOT=${DSOL_GATE_A97_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-accel-gate-a97-v1}
PROTOCOL_ROOT=${DSOL_GATE_A97_PROTOCOL_ROOT:-$OUTPUT_ROOT/protocols}
EVALUATOR=$REPO_ROOT/scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh
ANALYZER=$REPO_ROOT/scripts/dsol_paper1/summarize_accel_gate_a97.py
TRAIN_ROOT=/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs

declare -A CHECKPOINTS=(
  [broad64-practical]="$TRAIN_ROOT/dsol_broad_unpaired_practical_broad64-quick-gate-v1_seed41_g8_gb32_steps2000/final_model"
  [broad64-state-matched]="$TRAIN_ROOT/dsol_broad_unpaired_state_matched_broad64-parallel-formal-v1_seed41_g2_gb32_steps2000/final_model"
  [broad64-paired-fm]="$TRAIN_ROOT/dsol_broad_paired_fm_broad64-parallel-formal-v1_seed41_g2_gb32_steps2000/final_model"
  [broad64-paired-consistency]="$TRAIN_ROOT/dsol_broad_paired_consistency_broad64-parallel-formal-v1_seed41_g2_gb32_steps2000/final_model"
)
declare -A DEVICES=(
  [broad64-practical]="0,1"
  [broad64-state-matched]="2,3"
  [broad64-paired-fm]="4,5"
  [broad64-paired-consistency]="6,7"
)
declare -A PORTS=(
  [broad64-practical]=19800
  [broad64-state-matched]=19820
  [broad64-paired-fm]=19840
  [broad64-paired-consistency]=19860
)

mkdir -p "$OUTPUT_ROOT/logs"
exec > >(tee -a "$OUTPUT_ROOT/controller.log") 2>&1
printf 'gate_a97_start=%s git_commit=%s\n' "$(date -u +%FT%TZ)" "$(git -C "$REPO_ROOT" rev-parse HEAD)"

models=(broad64-practical broad64-state-matched broad64-paired-fm broad64-paired-consistency)
for model in "${models[@]}"; do
  protocol="$PROTOCOL_ROOT/$model-shortlist.json"
  checkpoint=${CHECKPOINTS[$model]}
  [[ -s "$protocol" ]] || { echo "missing protocol: $protocol" >&2; exit 2; }
  [[ -s "$checkpoint/model.safetensors" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 2; }
  jq -e '.status == "PASS" and .mode == "shortlist" and .selected_state_count == 96 and .episode_count == 576' "$protocol" >/dev/null
done

pids=()
for model in "${models[@]}"; do
  output="$OUTPUT_ROOT/shortlist/$model"
  protocol="$PROTOCOL_ROOT/$model-shortlist.json"
  if [[ -s "$output/analysis.json" ]]; then
    printf 'reuse_shortlist=%s\n' "$model"
    continue
  fi
  (
    CHECKPOINT="${CHECKPOINTS[$model]}" \
    OUTPUT_DIR="$output" \
    PROTOCOL="$protocol" \
    POLICY_BACKEND=alphabrain \
    GPU_COUNT=2 \
    DSOL_GPU_DEVICES="${DEVICES[$model]}" \
    BASE_PORT="${PORTS[$model]}" \
    REPLAN_STEPS=5 \
    WAIT_STEPS=0 \
    EVAL_SEED=20260825 \
    VIDEO_EPISODES=24 \
    RUN_ANALYSIS=0 \
      "$EVALUATOR"
    python "$ANALYZER" "$output"/episodes-shard-*.jsonl \
      --protocol "$protocol" --output "$output/analysis.json"
  ) > "$OUTPUT_ROOT/logs/$model-shortlist.log" 2>&1 &
  pids+=("$!")
  printf 'launched_shortlist=%s pid=%s devices=%s\n' "$model" "$!" "${DEVICES[$model]}"
done

failed=0
for pid in "${pids[@]:-}"; do
  [[ -n "$pid" ]] && wait "$pid" || failed=1
done
[[ "$failed" == 0 ]] || { echo "one or more Gate A97 shortlist runs failed" >&2; exit 1; }

oracle_protocol="$PROTOCOL_ROOT/broad64-practical-oracle.json"
oracle_output="$OUTPUT_ROOT/oracle97/broad64-practical"
jq -e '.status == "PASS" and .mode == "oracle" and .selected_state_count == 8 and .episode_count == 776' "$oracle_protocol" >/dev/null
if [[ ! -s "$oracle_output/analysis.json" ]]; then
  CHECKPOINT="${CHECKPOINTS[broad64-practical]}" \
  OUTPUT_DIR="$oracle_output" \
  PROTOCOL="$oracle_protocol" \
  POLICY_BACKEND=alphabrain \
  GPU_COUNT=8 \
  DSOL_GPU_DEVICES=0,1,2,3,4,5,6,7 \
  BASE_PORT=19900 \
  REPLAN_STEPS=5 \
  WAIT_STEPS=0 \
  EVAL_SEED=20260825 \
  VIDEO_EPISODES=16 \
  RUN_ANALYSIS=0 \
    "$EVALUATOR" > "$OUTPUT_ROOT/logs/broad64-practical-oracle97.log" 2>&1
  python "$ANALYZER" "$oracle_output"/episodes-shard-*.jsonl \
    --protocol "$oracle_protocol" --output "$oracle_output/analysis.json"
fi

jq -n \
  --arg completed_at "$(date -u +%FT%TZ)" \
  --arg git_commit "$(git -C "$REPO_ROOT" rev-parse HEAD)" \
  '{schema:"dsol_accel_gate_a97_completion_v1",status:"COMPLETE",completed_at:$completed_at,git_commit:$git_commit}' \
  > "$OUTPUT_ROOT/completion.json"
printf 'gate_a97_complete=%s\n' "$(date -u +%FT%TZ)"
