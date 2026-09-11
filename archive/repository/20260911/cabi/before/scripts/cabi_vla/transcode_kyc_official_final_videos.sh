#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN_ROOT=${KYC_OFFICIAL_OUTPUT_ROOT:-/share/longjunyu/kyc-official-data/runs}
OUTPUT_ROOT=${KYC_OFFICIAL_AV1_ROOT:-/share/longjunyu/kyc-official-data/videos_av1_final}
PYTHON=${KYC_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
MANIFEST="$OUTPUT_ROOT/manifest.json"

if [[ -s "$MANIFEST" ]]; then
  echo "official KYC AV1 videos already complete: $MANIFEST"
  exit 0
fi

mapfile -t inputs < <(
  find "$RUN_ROOT" \
    -path '*/official_act_lift_randomized_*_seed*/eval_epoch_20000_*/*.mp4' \
    -type f -print | sort
)
if [[ ${#inputs[@]} -eq 0 ]]; then
  echo "no official epoch-20000 KYC videos found under $RUN_ROOT" >&2
  exit 1
fi

"$PYTHON" scripts/cabi_vla/transcode_to_av1_webm.py \
  --input "${inputs[@]}" \
  --relative-root "$RUN_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --manifest "$MANIFEST"

echo "official KYC AV1 video copies complete: $MANIFEST"
