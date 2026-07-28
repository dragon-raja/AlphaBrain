#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUTPUT_ROOT=${KYC_OFFICIAL_OUTPUT_ROOT:-/share/longjunyu/kyc-official-data/runs}
GPU_IDS_CSV=${KYC_OFFICIAL_GPU_IDS:-0,1,2,3,4,5}

IFS=, read -r -a gpu_ids <<<"$GPU_IDS_CSV"
if [[ ${#gpu_ids[@]} -ne 6 ]]; then
  echo "KYC_OFFICIAL_GPU_IDS must contain exactly six comma-separated GPUs" >&2
  exit 2
fi
for gpu_id in "${gpu_ids[@]}"; do
  if [[ ! "$gpu_id" =~ ^[0-7]$ ]]; then
    echo "KYC_OFFICIAL_GPU_IDS entries must be in [0, 7]" >&2
    exit 2
  fi
done

arms=(
  "image 0"
  "kyc 0"
  "image 1"
  "kyc 1"
  "image 2"
  "kyc 2"
)

for index in "${!arms[@]}"; do
  read -r arm seed <<<"${arms[$index]}"
  gpu_id=${gpu_ids[$index]}
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
