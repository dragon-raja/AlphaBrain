#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
EVAL_ROOT=${KYC_CAMERA_EVAL_ROOT:-/share/longjunyu/cabi-vla/kyc-camera-eval-v1}
PYTHON=${KYC_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
PRIMARY_EDGES=red-left,red-right
ALL_EDGES=red-left,red-right,white-left,yellow_white-right
STATE_INDICES=40,41,42,43,44,45,46,47,48,49
EXPECTED_HALF=260
EXPECTED_FULL=520

jobs=(
  "base 41 1"
  "poseaug_rgb 41 2"
  "poseaug_control 41 3"
  "kyc 41 4"
  "poseaug_control 42 5"
  "kyc 42 6"
  "poseaug_control 43 7"
  "kyc 43 1"
  "pm_fixed 41 0"
)

json_row_count() {
  local path=$1
  if [[ -s "$path" ]]; then
    jq -r '.rows | length' "$path"
  else
    printf '0\n'
  fi
}

wait_for_exact_complete() {
  local path=$1
  local expected=$2
  local session=$3
  while ! [[ -s "$path" ]] || ! jq -e \
    --argjson expected "$expected" \
    '.status == "complete" and (.rows | length) == $expected' \
    "$path" >/dev/null; do
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "shard session exited without a valid result: $session" >&2
      return 1
    fi
    sleep 5
  done
}

stop_primary_at_half() {
  local method=$1
  local seed=$2
  local session="kyc-eval-gate-${method}-s${seed}"
  local run_dir="$EVAL_ROOT/gate/${method}_s${seed}_gate"
  local partial="$run_dir/camera_sweep_test.partial.json"
  local final="$run_dir/camera_sweep_test.json"

  while [[ $(json_row_count "$partial") -lt "$EXPECTED_HALF" ]]; do
    if [[ -s "$final" ]]; then
      return
    fi
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "primary session exited before its first half completed: $session" >&2
      return 1
    fi
    sleep 1
  done

  tmux send-keys -t "$session" C-c
  for _ in $(seq 1 120); do
    if ! tmux has-session -t "$session" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "primary session did not stop cleanly: $session" >&2
    return 1
  fi

  local count
  count=$(json_row_count "$partial")
  if [[ "$count" != "$EXPECTED_HALF" ]]; then
    echo "primary half has $count rows instead of $EXPECTED_HALF: $partial" >&2
    return 1
  fi
  jq -e \
    --argjson expected "$EXPECTED_HALF" \
    --arg edges "$PRIMARY_EDGES" \
    '(.rows | length) == $expected
     and ([.rows[].edge_id] | unique | sort)
       == ($edges | split(",") | sort)' \
    "$partial" >/dev/null
}

merge_job() {
  local method=$1
  local seed=$2
  local primary_dir="$EVAL_ROOT/gate/${method}_s${seed}_gate"
  local tail_dir="$EVAL_ROOT/gate_tail/${method}_s${seed}_tail"
  local output="$primary_dir/camera_sweep_test.json"
  local partial="$primary_dir/camera_sweep_test.partial.json"
  local tail="$tail_dir/camera_sweep_test.json"
  local tail_session="kyc-eval-tail-${method}-s${seed}"

  stop_primary_at_half "$method" "$seed"
  if [[ -s "$output" ]]; then
    return
  fi
  wait_for_exact_complete "$tail" "$EXPECTED_HALF" "$tail_session"

  if [[ -d "$tail_dir/frames" ]]; then
    mkdir -p "$primary_dir/frames"
    cp -a "$tail_dir/frames/." "$primary_dir/frames/"
  fi

  "$PYTHON" "$REPO_ROOT/scripts/cabi_vla/merge_kyc_camera_eval_shards.py" \
    --fragment "$partial" \
    --fragment "$tail" \
    --output "$output" \
    --expected-edges "$ALL_EDGES" \
    --expected-state-indices "$STATE_INDICES" \
    --expected-execution-horizons 3
  jq -e \
    --argjson expected "$EXPECTED_FULL" \
    '.status == "complete" and (.rows | length) == $expected' \
    "$output" >/dev/null
  rm -f "$partial"
}

for specification in "${jobs[@]}"; do
  read -r method seed gpu <<<"$specification"
  session="kyc-eval-tail-${method}-s${seed}"
  result="$EVAL_ROOT/gate_tail/${method}_s${seed}_tail/camera_sweep_test.json"
  run_dir=${result%/camera_sweep_test.json}
  if [[ -s "$result" ]]; then
    continue
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    continue
  fi
  if [[ -e "$run_dir" ]]; then
    echo "incomplete tail shard requires inspection: $run_dir" >&2
    exit 1
  fi
  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "exec bash scripts/cabi_vla/run_kyc_gate_tail_shard.sh $method $seed $gpu"
done

pids=()
for specification in "${jobs[@]}"; do
  read -r method seed _gpu <<<"$specification"
  merge_job "$method" "$seed" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
