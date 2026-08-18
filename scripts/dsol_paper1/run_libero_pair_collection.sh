#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${DSOL_RENDER_PYTHON:-/workspace/envs/fresh-libero/bin/python}
OUTPUT=${DSOL_PAIR_COLLECTION_OUTPUT:-/share/longjunyu/alphabrain/datasets/dsol-libero-broad-pairs-v1/quick_gate_seed41_broad32_stride2}
WORKERS=${DSOL_PAIR_COLLECTION_WORKERS:-8}
PLAN=${DSOL_PAIR_COLLECTION_PLAN:-$REPO_ROOT/configs/dsol_paper1/libero_pair_quick_gate_v1.json}
CATALOG=${DSOL_PAIR_COLLECTION_CATALOG:-$REPO_ROOT/configs/dsol_paper1/libero_view_catalog_v2.json}

[[ "$WORKERS" =~ ^[1-8]$ ]] || { echo "WORKERS must be in [1,8]" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "missing renderer Python: $PYTHON" >&2; exit 1; }

STOPPED_KEEPALIVES=()
restore_keepalive() {
  for gpu in "${STOPPED_KEEPALIVES[@]:-}"; do
    [[ -n "$gpu" ]] || continue
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      "${AI2R_KEEPALIVE_EXTRA_GIB:-1}" "${AI2R_KEEPALIVE_N:-8192}" \
      "gpu-keepalive-${gpu}" "$gpu" >/dev/null || true
  done
}
trap restore_keepalive EXIT

for ((gpu=0; gpu<WORKERS; gpu++)); do
  if tmux has-session -t "gpu-keepalive-${gpu}" 2>/dev/null; then
    tmux kill-session -t "gpu-keepalive-${gpu}"
    STOPPED_KEEPALIVES+=("$gpu")
  fi
done

cd "$REPO_ROOT"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" scripts/dsol_paper1/generate_libero_pair_collection.py \
  --plan "$PLAN" \
  --hdf5-root /share/longjunyu/alphabrain/datasets/libero-original-hdf5-v1 \
  --runtime /share/longjunyu/alphabrain/datasets/libero-plus/runtime/LIBERO-plus \
  --catalog "$CATALOG" \
  --acquisition /workspace/ai2r/debug/libero_plus_revalidation_v1/receipts/libero_original_hdf5_v1/acquisition.json \
  --config-root /workspace/ai2r/debug/libero_plus_revalidation_v1/pair_collection_runtime_config \
  --output "$OUTPUT" \
  --generator scripts/dsol_paper1/generate_libero_hdf5_view_pairs.py \
  --python "$PYTHON" \
  --workers "$WORKERS"
