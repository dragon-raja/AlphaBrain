#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/share/longjunyu/alphabrain/experiments/libero-original-full-retention-v1}
PROTOCOL=${PROTOCOL:-/share/longjunyu/alphabrain/experiments/libero-plus-camera-full-v1/protocol.json}
BROAD_RUN=${BROAD_RUN:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_broad64-quick-gate-v1_seed41_g8_gb32_steps2000}
OFFICIAL_CHECKPOINT=${OFFICIAL_CHECKPOINT:-/share/longjunyu/alphabrain/pretrained_models/openpi/pi05_libero_pytorch}
GPU_COUNT=${GPU_COUNT:-8}
TOOLS_PYTHON=${TOOLS_PYTHON:-/alphabrain/.venv/bin/python}

OFFICIAL_SMOKE=$EXPERIMENT_ROOT/official-pi05-frozen-smoke10
OFFICIAL_FULL=$EXPERIMENT_ROOT/official-pi05-frozen-original2000
BROAD_FULL=$EXPERIMENT_ROOT/broad64-seed41-original2000

mkdir -p "$EXPERIMENT_ROOT"

if [[ ! -s "$OFFICIAL_SMOKE/metrics.json" ]]; then
  CHECKPOINT="$OFFICIAL_CHECKPOINT" \
  PROTOCOL="$PROTOCOL" \
  OUTPUT_DIR="$OFFICIAL_SMOKE" \
  GPU_COUNT="$GPU_COUNT" \
  BASE_PORT=18900 \
  INIT_STATE_COUNT=1 \
  EVAL_MODES=original_full \
  EVAL_SUITES=libero_spatial \
  PROBE_SAMPLES=0 \
  VIDEO_EPISODES=1 \
    bash "$REPO_ROOT/scripts/cabi_vla/run_pi05_libero_plus_view_study.sh"
fi

if [[ ! -s "$OFFICIAL_FULL/metrics.json" ]]; then
  CHECKPOINT="$OFFICIAL_CHECKPOINT" \
  PROTOCOL="$PROTOCOL" \
  OUTPUT_DIR="$OFFICIAL_FULL" \
  GPU_COUNT="$GPU_COUNT" \
  BASE_PORT=18920 \
  INIT_STATE_COUNT=50 \
  EVAL_MODES=original_full \
  PROBE_SAMPLES=0 \
  VIDEO_EPISODES=2 \
    bash "$REPO_ROOT/scripts/cabi_vla/run_pi05_libero_plus_view_study.sh"
fi

if [[ ! -s "$BROAD_FULL/metrics.json" ]]; then
  CHECKPOINT="$BROAD_RUN/final_model" \
  PROTOCOL="$PROTOCOL" \
  OUTPUT_DIR="$BROAD_FULL" \
  GPU_COUNT="$GPU_COUNT" \
  BASE_PORT=18940 \
  INIT_STATE_COUNT=50 \
  EVAL_MODES=original_full \
  PROBE_SAMPLES=0 \
  VIDEO_EPISODES=2 \
  POLICY_PYTHON=/alphabrain/.venv/bin/python \
    bash "$REPO_ROOT/scripts/cabi_vla/run_alphabrain_pi05_libero_plus_view_eval.sh"
fi

mapfile -t official_episodes < <(
  find "$OFFICIAL_FULL" -maxdepth 1 -type f -name 'episodes-shard-*.jsonl' | sort
)
mapfile -t broad_episodes < <(
  find "$BROAD_FULL" -maxdepth 1 -type f -name 'episodes-shard-*.jsonl' | sort
)

"$TOOLS_PYTHON" "$REPO_ROOT/scripts/cabi_vla/analyze_libero_original_full.py" \
  --episodes "${broad_episodes[@]}" \
  --baseline-episodes "${official_episodes[@]}" \
  --expected-count 2000 \
  --expected-trials-per-task 50 \
  --output-json "$EXPERIMENT_ROOT/broad64-seed41-vs-official.json" \
  --output-report "$EXPERIMENT_ROOT/broad64-seed41-vs-official.md"

echo "original_libero_retention_gate_complete=$EXPERIMENT_ROOT"
