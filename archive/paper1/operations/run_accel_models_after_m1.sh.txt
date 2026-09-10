#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
M1_SESSION=${DSOL_M1_SESSION:-dsol-constructed-m1-matrix-v2}
POLL_SECONDS=${DSOL_WAIT_POLL_SECONDS:-30}
TRAIN_ROOT=${DSOL_PAIR_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs}
M1_ROOT=${DSOL_CONSTRUCTED_M1_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m1-v2}
ACCEL_ROOT=${DSOL_ACCEL_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-accel-constructed-v2}
RENDER_SOURCE=${DSOL_ACCEL_RENDER_SOURCE:-$ACCEL_ROOT/broad64-practical-seed41-full}
RUNNER=$REPO_ROOT/scripts/dsol_paper1/run_constructed_accel_bank.sh
JOINER=$REPO_ROOT/scripts/dsol_paper1/join_constructed_accel_m1.py

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid poll seconds" >&2; exit 2; }
mkdir -p "$ACCEL_ROOT/post-m1-logs"
exec > >(tee -a "$ACCEL_ROOT/post-m1-controller.log") 2>&1
printf 'post_m1_accel_controller_start=%s git_commit=%s\n' \
  "$(date -u +%FT%TZ)" "$(git -C "$REPO_ROOT" rev-parse HEAD)"

while tmux has-session -t "$M1_SESSION" 2>/dev/null; do
  printf 'waiting_for_m1=%s at=%s\n' "$M1_SESSION" "$(date -u +%FT%TZ)"
  sleep "$POLL_SECONDS"
done
while [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; do
  printf 'waiting_for_clean_worktree=%s at=%s\n' "$REPO_ROOT" "$(date -u +%FT%TZ)"
  sleep "$POLL_SECONDS"
done
[[ -s "$RENDER_SOURCE/render_summary.json" ]] || {
  echo "shared Accel render bank is incomplete: $RENDER_SOURCE" >&2
  exit 1
}
[[ -s "$M1_ROOT/cross-model-analysis/metrics.json" ]] || {
  echo "constructed M1 cross-model analysis is incomplete" >&2
  exit 1
}

declare -a names=(broad64-state-matched broad64-paired-fm broad64-paired-consistency)
declare -a checkpoints=(
  "$TRAIN_ROOT/dsol_broad_unpaired_state_matched_broad64-parallel-formal-v1_seed41_g2_gb32_steps2000/final_model"
  "$TRAIN_ROOT/dsol_broad_paired_fm_broad64-parallel-formal-v1_seed41_g2_gb32_steps2000/final_model"
  "$TRAIN_ROOT/dsol_broad_paired_consistency_broad64-parallel-formal-v1_seed41_g2_gb32_steps2000/final_model"
)
declare -a devices=(1 3 5)
declare -a pids=()
for index in "${!names[@]}"; do
  name=${names[$index]}
  checkpoint=${checkpoints[$index]}
  device=${devices[$index]}
  [[ -s "$checkpoint/model.safetensors" ]] || {
    echo "missing trained checkpoint: $checkpoint" >&2
    exit 1
  }
  (
    DSOL_ACCEL_CHECKPOINT="$checkpoint" \
    DSOL_ACCEL_OUTPUT_DIR="$ACCEL_ROOT/$name-seed41-full" \
    DSOL_ACCEL_RENDER_SOURCE_DIR="$RENDER_SOURCE" \
    DSOL_ACCEL_DEVICE="cuda:$device" \
    DSOL_ACCEL_RENDER_GPU="$device" \
      "$RUNNER" > "$ACCEL_ROOT/post-m1-logs/$name.log" 2>&1
  ) &
  pids+=("$!")
  printf 'launched_accel_rank=%s pid=%s device=%s\n' "$name" "$!" "$device"
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
[[ "$failed" == 0 ]] || { echo "post-M1 Accel ranking failed" >&2; exit 1; }

declare -a all_names=(broad64-practical "${names[@]}")
for name in "${all_names[@]}"; do
  accel_name=$name
  if [[ "$name" == "broad64-practical" ]]; then
    accel_dir=$RENDER_SOURCE
  else
    accel_dir=$ACCEL_ROOT/$name-seed41-full
  fi
  m1_dir=$M1_ROOT/$name
  output=$ACCEL_ROOT/m1-joins/$name
  /alphabrain/.venv/bin/python "$JOINER" \
    --accel-root "$accel_dir" \
    --m1-root "$m1_dir" \
    --output-dir "$output"
  printf 'completed_accel_m1_join=%s output=%s\n' "$name" "$output/metrics.json"
done

printf 'post_m1_accel_controller_complete=%s\n' "$(date -u +%FT%TZ)"
