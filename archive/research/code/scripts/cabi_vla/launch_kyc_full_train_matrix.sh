#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUTPUT_ROOT=${KYC_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/kyc-runs}
WAIT_ROOT=${KYC_WAIT_FOR_FOV_ROOT:-}

if [[ -n "$WAIT_ROOT" ]]; then
  edges=(
    red-left
    red-right
    white-left
    white-right
    yellow_white-left
    yellow_white-right
  )
  while true; do
    complete=1
    for edge in "${edges[@]}"; do
      result="$WAIT_ROOT/$edge.json"
      if [[ ! -s "$result" ]] || ! jq -e \
        '.status == "complete" and (.rows | length) == 780' \
        "$result" >/dev/null; then
        complete=0
        break
      fi
    done
    [[ "$complete" == 1 ]] && break
    sleep 30
  done
fi

jobs=(
  "poseaug_control 41 6"
  "kyc 41 7"
  "poseaug_control 42 0"
  "kyc 42 1"
  "poseaug_control 43 2"
  "kyc 43 3"
  "poseaug_rgb 41 4"
  "pm_fixed 41 5"
)

for specification in "${jobs[@]}"; do
  read -r arm seed gpu_id <<<"$specification"
  session="kyc-full-s${seed}-${arm}"
  final_model="$OUTPUT_ROOT/kyc_${arm}_h20_seed${seed}_steps33000/final_model/model.safetensors"
  run_dir=${final_model%/final_model/model.safetensors}

  if [[ -s "$final_model" ]]; then
    echo "already complete: $arm seed=$seed"
    continue
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "already running: $session"
    continue
  fi
  if [[ -e "$run_dir" ]]; then
    echo "incomplete run directory requires inspection: $run_dir" >&2
    exit 1
  fi

  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "exec bash scripts/cabi_vla/run_kyc_train.sh $arm $seed $gpu_id 33000"
  echo "started: $session gpu=$gpu_id"
done
