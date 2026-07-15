#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SEED=${1:?usage: run_post_feedback_mode_probe.sh SEED GPU_ID}
GPU_ID=${2:?usage: run_post_feedback_mode_probe.sh SEED GPU_ID}
RUN_ROOT=${FRESH_RESEARCH_RESET_ROOT:-/share/longjunyu/fresh-vla/research-reset}
BASELINE_ROOT=${FRESH_CLOSED_LOOP_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2}
WINDOWS_ROOT=${FRESH_WINDOWS_ROOT:-/share/longjunyu/fresh-vla/libero-full-episode-windows-v2-128}
EPISODE_ROOT=${FRESH_EPISODE_ROOT:-/share/longjunyu/fresh-vla/libero-full-episode-v2-128}
SESSION="gpu-keepalive-${GPU_ID}"
WAS_RUNNING=0

case "$SEED" in
  41|42|43) ;;
  *) echo "seed must be one of 41, 42, 43" >&2; exit 2 ;;
esac
if [[ ! "$GPU_ID" =~ ^[0-7]$ ]]; then
  echo "GPU_ID must be in [0, 7]" >&2
  exit 2
fi

restore_keepalive() {
  if [ "$WAS_RUNNING" = 1 ]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      "${AI2R_KEEPALIVE_EXTRA_GIB:-1}" "${AI2R_KEEPALIVE_N:-8192}" "$SESSION" "$GPU_ID" >/dev/null || true
  fi
}
trap restore_keepalive EXIT

if tmux has-session -t "$SESSION" 2>/dev/null; then
  WAS_RUNNING=1
  tmux kill-session -t "$SESSION"
fi

mkdir -p "$RUN_ROOT"
cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PRETRAINED_MODELS_DIR=/share/longjunyu/alphabrain/pretrained_models \
PYTHONPATH="$REPO_ROOT" \
"$REPO_ROOT/.venv/bin/python" scripts/fresh_vla/probe_post_feedback_modes.py \
  --checkpoint "$BASELINE_ROOT/fresh_closed_loop_full_h_seed${SEED}/final_model" \
  --windows-root "$WINDOWS_ROOT" \
  --episode-root "$EPISODE_ROOT" \
  --output "$RUN_ROOT/post_feedback_modes_seed${SEED}.json" \
  --seed "$SEED" \
  --device cuda:0
