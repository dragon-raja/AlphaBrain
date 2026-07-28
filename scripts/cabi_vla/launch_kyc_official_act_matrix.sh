#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUTPUT_ROOT=${KYC_OFFICIAL_OUTPUT_ROOT:-/share/longjunyu/kyc-official-data/runs}

jobs=(
  "image 0 0"
  "kyc 0 1"
  "image 1 2"
  "kyc 1 3"
  "image 2 4"
  "kyc 2 5"
)

for specification in "${jobs[@]}"; do
  read -r arm seed gpu_id <<<"$specification"
  session="kyc-official-${arm}-s${seed}"
  final_checkpoint="$OUTPUT_ROOT/official_act_lift_randomized_${arm}_seed${seed}/epoch_20000.pth"
  if [[ -s "$final_checkpoint" ]]; then
    echo "already complete: $arm seed=$seed"
    continue
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "already running: $session"
    continue
  fi
  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "exec bash scripts/cabi_vla/run_kyc_official_act.sh '$arm' '$seed' '$gpu_id'"
  echo "started: $session gpu=$gpu_id"
done

