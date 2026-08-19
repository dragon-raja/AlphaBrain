#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/share/longjunyu/alphabrain/experiments/libero-plus-camera-full-v1}
FULL_PROTOCOL=${FULL_PROTOCOL:-$EXPERIMENT_ROOT/protocol.json}
SMOKE_PROTOCOL=${SMOKE_PROTOCOL:-$EXPERIMENT_ROOT/protocol-smoke8.json}
BROAD_RUN=${BROAD_RUN:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_broad64-quick-gate-v1_seed41_g8_gb32_steps2000}
PRIOR_GATE=${PRIOR_GATE:-/share/longjunyu/alphabrain/experiments/libero-plus-view-gap-v1/pi05-libero-broad64-quick-gate-v1/metrics.json}
OFFICIAL_CHECKPOINT=${OFFICIAL_CHECKPOINT:-/share/longjunyu/alphabrain/pretrained_models/openpi/pi05_libero_pytorch}
GPU_COUNT=${GPU_COUNT:-8}
WAIT_TIMEOUT_SECONDS=${WAIT_TIMEOUT_SECONDS:-86400}
TOOLS_PYTHON=${TOOLS_PYTHON:-/alphabrain/.venv/bin/python}

OFFICIAL_SMOKE=$EXPERIMENT_ROOT/official-pi05-frozen-smoke8
OFFICIAL_FULL=$EXPERIMENT_ROOT/official-pi05-frozen-camera1599
BROAD_FULL=$EXPERIMENT_ROOT/broad64-seed41-camera1599

mkdir -p "$EXPERIMENT_ROOT/pipeline_logs"

if [[ ! -s "$FULL_PROTOCOL" ]]; then
  echo "missing full camera protocol: $FULL_PROTOCOL" >&2
  exit 2
fi

if [[ ! -s "$SMOKE_PROTOCOL" ]]; then
  FULL_PROTOCOL="$FULL_PROTOCOL" SMOKE_PROTOCOL="$SMOKE_PROTOCOL" \
    "$TOOLS_PYTHON" - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

source = Path(os.environ["FULL_PROTOCOL"])
target = Path(os.environ["SMOKE_PROTOCOL"])
protocol = json.loads(source.read_text())
counts = Counter()
selected = []
for row in protocol["official_camera_tasks"]:
    suite = str(row["suite"])
    if counts[suite] < 2:
        selected.append(row)
        counts[suite] += 1
protocol["protocol_scope"] = "libero_plus_camera_smoke8"
protocol["official_camera_tasks"] = selected
protocol["summary"] = {
    "camera_population_count": len(selected),
    "selected_count": len(selected),
    "selected_unique_base_task_count": len(
        {(row["suite"], row["base_task"]) for row in selected}
    ),
    "smoke_only": True,
}
temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
os.replace(temporary, target)
PY
fi

deadline=$(( $(date +%s) + WAIT_TIMEOUT_SECONDS ))
while [[ ! -s "$PRIOR_GATE" ]]; do
  if (( $(date +%s) >= deadline )); then
    echo "timed out waiting for prior quick gate: $PRIOR_GATE" >&2
    exit 1
  fi
  sleep 30
done

if [[ ! -s "$OFFICIAL_SMOKE/metrics.json" ]]; then
  CHECKPOINT="$OFFICIAL_CHECKPOINT" \
  PROTOCOL="$SMOKE_PROTOCOL" \
  OUTPUT_DIR="$OFFICIAL_SMOKE" \
  GPU_COUNT="$GPU_COUNT" \
  BASE_PORT=18800 \
  INIT_STATE_COUNT=1 \
  EVAL_MODES=camera_full \
  PROBE_SAMPLES=0 \
  VIDEO_EPISODES=1 \
    bash "$REPO_ROOT/scripts/cabi_vla/run_pi05_libero_plus_view_study.sh"
fi

if [[ ! -s "$OFFICIAL_FULL/metrics.json" ]]; then
  CHECKPOINT="$OFFICIAL_CHECKPOINT" \
  PROTOCOL="$FULL_PROTOCOL" \
  OUTPUT_DIR="$OFFICIAL_FULL" \
  GPU_COUNT="$GPU_COUNT" \
  BASE_PORT=18820 \
  INIT_STATE_COUNT=1 \
  EVAL_MODES=camera_full \
  PROBE_SAMPLES=0 \
  VIDEO_EPISODES=2 \
    bash "$REPO_ROOT/scripts/cabi_vla/run_pi05_libero_plus_view_study.sh"
fi

if [[ ! -s "$BROAD_FULL/metrics.json" ]]; then
  CHECKPOINT="$BROAD_RUN/final_model" \
  PROTOCOL="$FULL_PROTOCOL" \
  OUTPUT_DIR="$BROAD_FULL" \
  GPU_COUNT="$GPU_COUNT" \
  BASE_PORT=18840 \
  INIT_STATE_COUNT=1 \
  EVAL_MODES=camera_full \
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

"$TOOLS_PYTHON" \
  "$REPO_ROOT/scripts/cabi_vla/analyze_libero_plus_camera_full.py" \
  --episodes "${broad_episodes[@]}" \
  --baseline-episodes "${official_episodes[@]}" \
  --expected-count 1599 \
  --output-json "$EXPERIMENT_ROOT/broad64-seed41-vs-official.json" \
  --output-report "$EXPERIMENT_ROOT/broad64-seed41-vs-official.md"

echo "camera_full_gate_complete=$EXPERIMENT_ROOT"
