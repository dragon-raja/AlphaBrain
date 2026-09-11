#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BOUNDARY_ROOT=${KYC_BOUNDARY_FOV_ROOT:-/share/longjunyu/cabi-vla/camera-viewpoint-study-v2/fov_guard_test40-49_v5}
DENSE_ROOT=${KYC_DENSE_FOV_ROOT:-/share/longjunyu/cabi-vla/camera-viewpoint-study-v2/fov_guard_policy_dense_state40_v6}
DENSE_CONFIG=${KYC_DENSE_FOV_CONFIG:-$REPO_ROOT/docs/cabi_vla/configs/camera_pose_policy_dense_v2.json}

edges=(
  red-left
  red-right
  white-left
  white-right
  yellow_white-left
  yellow_white-right
)

wait_for_results() {
  local root=$1
  local expected=$2
  while true; do
    local complete=1
    for edge in "${edges[@]}"; do
      result="$root/$edge.json"
      if [[ ! -s "$result" ]] || ! jq -e \
        --argjson expected "$expected" \
        '.status == "complete" and (.rows | length) == $expected' \
        "$result" >/dev/null; then
        complete=0
        break
      fi
    done
    [[ "$complete" == 1 ]] && return
    sleep 30
  done
}

wait_for_results "$BOUNDARY_ROOT" 780

for index in "${!edges[@]}"; do
  edge=${edges[$index]}
  session="kyc-fov-dense-$edge"
  result="$DENSE_ROOT/$edge.json"
  partial="$DENSE_ROOT/$edge.partial.json"
  if [[ -s "$result" ]] && jq -e \
    '.status == "complete" and (.rows | length) == 35' \
    "$result" >/dev/null; then
    continue
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    continue
  fi
  if [[ -e "$result" || -e "$partial" ]]; then
    echo "incomplete dense FOV output requires inspection: $edge" >&2
    exit 1
  fi
  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "exec env KYC_FOV_CONFIG=$DENSE_CONFIG KYC_FOV_OUTPUT_ROOT=$DENSE_ROOT KYC_FOV_SPLIT=test KYC_FOV_STATE_INDICES=40 bash scripts/cabi_vla/run_kyc_fov_fragment.sh $edge $index"
done

wait_for_results "$DENSE_ROOT" 35
exec bash "$REPO_ROOT/scripts/cabi_vla/launch_kyc_full_train_matrix.sh"
