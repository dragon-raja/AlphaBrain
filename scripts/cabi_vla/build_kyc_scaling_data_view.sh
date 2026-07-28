#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SOURCE_VIEW=${KYC_SOURCE_TRAINING_VIEW:-/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v15-decision-observed-edge-phase-loss-balanced-h20}
CAMERA_CONFIG=${KYC_CAMERA_CONFIG:-$REPO_ROOT/docs/cabi_vla/configs/camera_pose_train_global_scaling_v3.json}
DATA_ROOT=${KYC_SCALING_DATA_ROOT:-/share/longjunyu/cabi-vla/kyc-scaling-v3}
PYTHON=${KYC_MERGE_PYTHON:-$REPO_ROOT/.venv/bin/python}

CELL=${1:?usage: build_kyc_scaling_data_view.sh CELL CATALOG_SIZE SCENE_MODE [GPU_OFFSET]}
CATALOG_SIZE=${2:?usage: build_kyc_scaling_data_view.sh CELL CATALOG_SIZE SCENE_MODE [GPU_OFFSET]}
SCENE_MODE=${3:?usage: build_kyc_scaling_data_view.sh CELL CATALOG_SIZE SCENE_MODE [GPU_OFFSET]}
GPU_OFFSET=${4:-0}

if [[ ! "$CELL" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "CELL must contain only safe filename characters" >&2
  exit 2
fi
if [[ ! "$CATALOG_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "CATALOG_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ "$SCENE_MODE" != fixed && "$SCENE_MODE" != cue_randomized ]]; then
  echo "SCENE_MODE must be fixed or cue_randomized" >&2
  exit 2
fi
if [[ ! "$GPU_OFFSET" =~ ^[0-7]$ ]]; then
  echo "GPU_OFFSET must be in [0, 7]" >&2
  exit 2
fi

FRAGMENT_ROOT="$DATA_ROOT/fragments/$CELL"
OUTPUT="$DATA_ROOT/views/libero-bind-kyc-${CELL}-h20"
LOG="$DATA_ROOT/logs/build-${CELL}.log"
EDGES=(red-left red-right white-left yellow_white-right)

if [[ -s "$OUTPUT/manifest.json" ]]; then
  echo "already complete: $OUTPUT"
  exit 0
fi
if [[ -e "$OUTPUT" ]]; then
  echo "incomplete output requires inspection: $OUTPUT" >&2
  exit 1
fi

mkdir -p "$DATA_ROOT/logs" "$FRAGMENT_ROOT"
exec > >(tee -a "$LOG") 2>&1

for edge_index in "${!EDGES[@]}"; do
  edge=${EDGES[$edge_index]}
  fragment="$FRAGMENT_ROOT/$edge"
  session="kyc-data-${CELL}-${edge}"
  gpu_id=$(((GPU_OFFSET + edge_index) % 8))
  if [[ -s "$fragment/manifest.json" ]]; then
    echo "fragment already complete: $edge"
    continue
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "fragment already running: $session"
    continue
  fi
  if [[ -e "$fragment" ]]; then
    echo "incomplete fragment requires inspection: $fragment" >&2
    exit 1
  fi
  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "export KYC_SOURCE_TRAINING_VIEW='$SOURCE_VIEW'; \
     export KYC_CAMERA_CONFIG='$CAMERA_CONFIG'; \
     export KYC_FRAGMENT_ROOT='$FRAGMENT_ROOT'; \
     export KYC_CAMERA_CATALOG_SIZE='$CATALOG_SIZE'; \
     export KYC_EPOCH_REPLICAS=3; \
     export KYC_SCENE_CUE_MODE='$SCENE_MODE'; \
     exec bash scripts/cabi_vla/run_kyc_camera_fragment.sh '$edge' '$gpu_id'"
  echo "started fragment: $session gpu=$gpu_id"
done

while true; do
  complete=1
  for edge in "${EDGES[@]}"; do
    if [[ ! -s "$FRAGMENT_ROOT/$edge/manifest.json" ]]; then
      complete=0
      break
    fi
  done
  [[ "$complete" == 1 ]] && break
  sleep 30
done

fragments=()
for edge in "${EDGES[@]}"; do
  fragments+=("$FRAGMENT_ROOT/$edge")
done

"$PYTHON" scripts/cabi_vla/merge_libero_bind_camera_training_fragments.py \
  --training-view "$SOURCE_VIEW" \
  --output "$OUTPUT" \
  --fragments "${fragments[@]}"

echo "complete: $OUTPUT"

