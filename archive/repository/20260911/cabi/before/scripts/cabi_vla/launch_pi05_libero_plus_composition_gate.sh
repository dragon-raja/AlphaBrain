#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SCRIPT=$(realpath "$0")
COMMAND=${1:-status}
SESSION=${PLUS_COMPOSITION_SESSION:-plus-composition-gate}
ROOT=${PLUS_COMPOSITION_ROOT:-/share/longjunyu/alphabrain/experiments/libero-plus-camera-background-v1}
PROTOCOL=${PLUS_COMPOSITION_PROTOCOL:-$ROOT/protocol-v1.json}
VIEW_PROTOCOL=${PLUS_VIEW_PROTOCOL:-/share/longjunyu/alphabrain/experiments/libero-plus-view-gap-v1/protocol-v3.json}
CLASSIFICATION=${PLUS_CLASSIFICATION:-/share/longjunyu/alphabrain/datasets/libero-plus/runtime/LIBERO-plus/libero/libero/benchmark/task_classification.json}
PLUS_ROOT=${PLUS_ROOT:-/share/longjunyu/alphabrain/datasets/libero-plus/runtime/LIBERO-plus}
BDDL_ROOT=${BDDL_ROOT:-$PLUS_ROOT/libero/libero/bddl_files}
BEST_CHECKPOINT=${BEST_CHECKPOINT:-/share/longjunyu/alphabrain/experiments/libero-plus-mv-rgb-v1/runs/pi05_plus_mv_visual_lora_gate-v1-b025_seed41_steps33000/final_model}
BEST_OUTPUT=$ROOT/visual_b025
OFFICIAL_OUTPUT=$ROOT/official_pi05
FINAL_OUTPUT=$ROOT/final
LOG=$ROOT/orchestrator.log

expected_count() {
  PYTHONPATH="$REPO_ROOT/scripts/cabi_vla" PROTOCOL="$PROTOCOL" \
    "$REPO_ROOT/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path
from evaluate_pi05_libero_plus_views import build_episode_specs

protocol = json.loads(Path(os.environ["PROTOCOL"]).read_text())
print(len(build_episode_specs(protocol, modes=["composition"], init_state_count=2)))
PY
}

episode_count() {
  local output=$1
  find "$output" -maxdepth 1 -name 'episodes-shard-*.jsonl' -type f \
    -exec awk 'NF {count += 1} END {print count + 0}' {} + 2>/dev/null \
    | awk '{total += $1} END {print total + 0}'
}

print_status() {
  mkdir -p "$ROOT"
  local expected=unknown
  if [[ -s "$PROTOCOL" ]]; then
    expected=$(expected_count)
  fi
  echo "visual_b025=$(episode_count "$BEST_OUTPUT")/$expected"
  echo "official_pi05=$(episode_count "$OFFICIAL_OUTPUT")/$expected"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "orchestrator=RUNNING session=$SESSION"
  elif [[ -s "$FINAL_OUTPUT/metrics.json" ]]; then
    echo "orchestrator=COMPLETE output=$FINAL_OUTPUT"
  else
    echo "orchestrator=NOT_RUNNING"
  fi
}

prepare_protocol() {
  mkdir -p "$ROOT"
  if [[ ! -s "$PROTOCOL" ]]; then
    PYTHONPATH="$REPO_ROOT/scripts/cabi_vla" \
      "$REPO_ROOT/.venv/bin/python" \
      "$REPO_ROOT/scripts/cabi_vla/build_libero_plus_composition_protocol.py" \
      --view-protocol "$VIEW_PROTOCOL" \
      --classification "$CLASSIFICATION" \
      --output "$PROTOCOL"
  fi
  local protocol_sha
  protocol_sha=$(sha256sum "$PROTOCOL" | awk '{print $1}')
  local smoke_sha=missing
  if [[ -s "$ROOT/protocol-smoke.json" ]]; then
    smoke_sha=$(SMOKE="$ROOT/protocol-smoke.json" "$REPO_ROOT/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path
print(json.loads(Path(os.environ["SMOKE"]).read_text()).get("protocol_sha256", "missing"))
PY
)
  fi
  if [[ "$smoke_sha" != "$protocol_sha" ]]; then
    LIBERO_CONFIG_PATH=/share/longjunyu/alphabrain/envs/libero-plus-runtime-config-v1 \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    PYTHONPATH="$PLUS_ROOT:/share/longjunyu/alphabrain/envs/libero-plus-runtime-overlay-v1:$REPO_ROOT/scripts/cabi_vla" \
      /share/longjunyu/capt-vla/envs/libero/bin/python \
      "$REPO_ROOT/scripts/cabi_vla/validate_libero_plus_composition_protocol.py" \
      --protocol "$PROTOCOL" \
      --bddl-root "$BDDL_ROOT" \
      --output-json "$ROOT/protocol-smoke.json" \
      --output-figure "$ROOT/protocol-smoke.png" \
      --task-count 2 \
      --render-gpu 0
  fi
}

run_worker() {
  mkdir -p "$ROOT"
  exec > >(tee -a "$LOG") 2>&1
  echo "composition_gate_started=$(date -u +%FT%TZ)"
  prepare_protocol
  local expected
  expected=$(expected_count)
  echo "expected_episodes_per_model=$expected"

  if [[ "$(episode_count "$BEST_OUTPUT")" -ne "$expected" ]]; then
    CHECKPOINT="$BEST_CHECKPOINT" \
    OUTPUT_DIR="$BEST_OUTPUT" \
    PROTOCOL="$PROTOCOL" \
    EVAL_MODES=composition \
    PROBE_SAMPLES=0 \
    VIDEO_EPISODES=16 \
    SKIP_ANALYSIS=1 \
    BASE_PORT=18500 \
      "$REPO_ROOT/scripts/cabi_vla/run_alphabrain_pi05_libero_plus_view_eval.sh"
  fi

  if [[ "$(episode_count "$OFFICIAL_OUTPUT")" -ne "$expected" ]]; then
    OUTPUT_DIR="$OFFICIAL_OUTPUT" \
    PROTOCOL="$PROTOCOL" \
    EVAL_MODES=composition \
    PROBE_SAMPLES=0 \
    VIDEO_EPISODES=16 \
    SKIP_ANALYSIS=1 \
    BASE_PORT=18600 \
      "$REPO_ROOT/scripts/cabi_vla/run_pi05_libero_plus_view_study.sh"
  fi

  mkdir -p "$FINAL_OUTPUT"
  "$REPO_ROOT/.venv/bin/python" \
    "$REPO_ROOT/scripts/cabi_vla/analyze_pi05_libero_plus_composition.py" \
    --run "official_pi05=$OFFICIAL_OUTPUT" \
    --run "visual_b025=$BEST_OUTPUT" \
    --reference official_pi05 \
    --candidate visual_b025 \
    --output-json "$FINAL_OUTPUT/metrics.json" \
    --output-report "$FINAL_OUTPUT/report_zh.md" \
    --output-figure "$FINAL_OUTPUT/summary.png"
  echo "composition_gate_complete=$(date -u +%FT%TZ) output=$FINAL_OUTPUT"
}

case "$COMMAND" in
  start)
    mkdir -p "$ROOT"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "orchestrator already running: $SESSION"
      exit 0
    fi
    if [[ -s "$FINAL_OUTPUT/metrics.json" ]]; then
      echo "composition gate already complete: $FINAL_OUTPUT"
      exit 0
    fi
    tmux new-session -d -s "$SESSION" "$SCRIPT" worker
    echo "orchestrator started: $SESSION log=$LOG"
    ;;
  status)
    print_status
    ;;
  worker)
    run_worker
    ;;
  *)
    echo "usage: $0 {start|status}" >&2
    exit 2
    ;;
esac
