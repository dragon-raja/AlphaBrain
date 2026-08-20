#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OPENPI_ROOT=${OPENPI_ROOT:-/projects/openpi}
CHECKPOINT=${CHECKPOINT:-/share/longjunyu/alphabrain/pretrained_models/openpi/pi05_libero_pytorch}
DATA_ROOT=${DATA_ROOT:-/share/longjunyu/alphabrain/datasets/libero-plus}
PROTOCOL=${PROTOCOL:-/share/longjunyu/alphabrain/experiments/libero-plus-view-gap-v1/protocol-v3.json}
OUTPUT_DIR=${OUTPUT_DIR:-/share/longjunyu/alphabrain/experiments/libero-plus-view-gap-v1/pi05-libero-official-v1}
OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/share/longjunyu/alphabrain/cache/openpi}
GPU_COUNT=${GPU_COUNT:-8}
BASE_PORT=${BASE_PORT:-18100}
INIT_STATE_COUNT=${INIT_STATE_COUNT:-2}
REPLAN_STEPS=${REPLAN_STEPS:-5}
EVAL_SEED=${EVAL_SEED:-20260804}
PROBE_SAMPLES=${PROBE_SAMPLES:-3}
PROBE_HORIZON=${PROBE_HORIZON:-5}
VIDEO_EPISODES=${VIDEO_EPISODES:-4}
EVAL_MODES=${EVAL_MODES:-"gap candidates"}
EVAL_SUITES=${EVAL_SUITES:-}
SKIP_ANALYSIS=${SKIP_ANALYSIS:-0}
EXPECTED_CHECKPOINT_SHA256=${EXPECTED_CHECKPOINT_SHA256:-\
0f8c489e37b01c72251c45f2e73595894f3933fc6297f4f1cf95fc8737db4c74}

OPENPI_PYTHON=${OPENPI_PYTHON:-${OPENPI_ROOT}/.venv/bin/python}
TOOLS_PYTHON=${TOOLS_PYTHON:-/alphabrain/.venv/bin/python}
SIM_PYTHON=${SIM_PYTHON:-/share/longjunyu/capt-vla/envs/libero/bin/python}
SIM_OVERLAY=${SIM_OVERLAY:-/share/longjunyu/alphabrain/envs/libero-plus-runtime-overlay-v1}
SIM_CONFIG=${SIM_CONFIG:-/share/longjunyu/alphabrain/envs/libero-plus-runtime-config-v1}
PLUS_ROOT=${PLUS_ROOT:-${DATA_ROOT}/runtime/LIBERO-plus}
BDDL_ROOT=${BDDL_ROOT:-${PLUS_ROOT}/libero/libero/bddl_files}

read -r -a MODE_ARGS <<<"$EVAL_MODES"
SUITE_ARGS=()
if [[ -n "$EVAL_SUITES" ]]; then
  read -r -a SUITE_VALUES <<<"$EVAL_SUITES"
  SUITE_ARGS=(--suites "${SUITE_VALUES[@]}")
fi

if [[ "$SKIP_ANALYSIS" != 0 && "$SKIP_ANALYSIS" != 1 ]]; then
  echo "SKIP_ANALYSIS must be 0 or 1" >&2
  exit 2
fi
if [[ " $EVAL_MODES " == *" camera_full "* && "$INIT_STATE_COUNT" != 1 ]]; then
  echo "camera_full requires INIT_STATE_COUNT=1 for official compatibility" >&2
  exit 2
fi

for required in \
  "$OPENPI_PYTHON" \
  "$SIM_PYTHON" \
  "$CHECKPOINT/model.safetensors" \
  "$CHECKPOINT/config.json" \
  "$CHECKPOINT/assets/physical-intelligence/libero/norm_stats.json" \
  "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model" \
  "$PROTOCOL" \
  "$BDDL_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

actual_checkpoint_sha256=$(sha256sum "$CHECKPOINT/model.safetensors" | awk '{print $1}')
if [ "$actual_checkpoint_sha256" != "$EXPECTED_CHECKPOINT_SHA256" ]; then
  echo "checkpoint SHA-256 mismatch" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR/logs"
protocol_sha256=$(sha256sum "$PROTOCOL" | awk '{print $1}')
tokenizer_sha256=$(sha256sum "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model" | awk '{print $1}')
alphabrain_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)
openpi_commit=$(git -C "$OPENPI_ROOT" rev-parse HEAD)
study_code_sha256=$(sha256sum \
  "$REPO_ROOT/scripts/cabi_vla/build_libero_plus_view_protocol.py" \
  "$REPO_ROOT/scripts/cabi_vla/serve_openpi_deterministic.py" \
  "$REPO_ROOT/scripts/cabi_vla/evaluate_pi05_libero_plus_views.py" \
  "$REPO_ROOT/scripts/cabi_vla/analyze_pi05_libero_plus_views.py" \
  "$REPO_ROOT/scripts/cabi_vla/analyze_libero_plus_camera_full.py" \
  "$REPO_ROOT/scripts/cabi_vla/analyze_libero_original_full.py" \
  "$REPO_ROOT/scripts/cabi_vla/run_pi05_libero_plus_view_study.sh" \
  | sha256sum | awk '{print $1}')
manifest="$OUTPUT_DIR/run_manifest.json"
manifest_temporary="$OUTPUT_DIR/.run_manifest.$$.tmp"
CHECKPOINT="$CHECKPOINT" CHECKPOINT_SHA256="$actual_checkpoint_sha256" \
PROTOCOL="$PROTOCOL" PROTOCOL_SHA256="$protocol_sha256" \
TOKENIZER_SHA256="$tokenizer_sha256" ALPHABRAIN_COMMIT="$alphabrain_commit" \
OPENPI_COMMIT="$openpi_commit" STUDY_CODE_SHA256="$study_code_sha256" \
GPU_COUNT="$GPU_COUNT" INIT_STATE_COUNT="$INIT_STATE_COUNT" \
REPLAN_STEPS="$REPLAN_STEPS" EVAL_SEED="$EVAL_SEED" \
PROBE_SAMPLES="$PROBE_SAMPLES" PROBE_HORIZON="$PROBE_HORIZON" \
EVAL_MODES="$EVAL_MODES" EVAL_SUITES="$EVAL_SUITES" \
  "$TOOLS_PYTHON" - <<'PY' >"$manifest_temporary"
import json
import os

manifest = {
    "schema_version": 1,
    "model_config": "pi05_libero",
    "checkpoint": os.environ["CHECKPOINT"],
    "checkpoint_sha256": os.environ["CHECKPOINT_SHA256"],
    "protocol": os.environ["PROTOCOL"],
    "protocol_sha256": os.environ["PROTOCOL_SHA256"],
    "tokenizer_sha256": os.environ["TOKENIZER_SHA256"],
    "alphabrain_commit": os.environ["ALPHABRAIN_COMMIT"],
    "openpi_commit": os.environ["OPENPI_COMMIT"],
    "study_code_sha256": os.environ["STUDY_CODE_SHA256"],
    "gpu_count": int(os.environ["GPU_COUNT"]),
    "initial_state_count": int(os.environ["INIT_STATE_COUNT"]),
    "replan_steps": int(os.environ["REPLAN_STEPS"]),
    "evaluation_seed": int(os.environ["EVAL_SEED"]),
    "probe_samples": int(os.environ["PROBE_SAMPLES"]),
    "probe_horizon": int(os.environ["PROBE_HORIZON"]),
    "modes": os.environ["EVAL_MODES"].split(),
    "suites": os.environ["EVAL_SUITES"].split(),
}
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
if [ -e "$manifest" ]; then
  if ! cmp -s "$manifest" "$manifest_temporary"; then
    echo "run manifest differs from existing output; use a new OUTPUT_DIR" >&2
    rm -f "$manifest_temporary"
    exit 2
  fi
  rm -f "$manifest_temporary"
else
  mv "$manifest_temporary" "$manifest"
fi

policy_pids=()
eval_pids=()
keepalive_stopped=0

cleanup() {
  for pid in "${eval_pids[@]:-}" "${policy_pids[@]:-}"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  if [ "$keepalive_stopped" = 1 ]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start_all.sh 1 8192 gpu-keepalive || true
  fi
}
trap cleanup EXIT INT TERM

bash /workspace/ai2r/gpu_compute_keepalive/stop_all.sh gpu-keepalive
keepalive_stopped=1

for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
  port=$((BASE_PORT + gpu))
  CUDA_VISIBLE_DEVICES=$gpu OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
    PYTHONPATH="${REPO_ROOT}/scripts/cabi_vla" \
    "$OPENPI_PYTHON" "${REPO_ROOT}/scripts/cabi_vla/serve_openpi_deterministic.py" \
      --checkpoint "$CHECKPOINT" \
      --config pi05_libero \
      --port "$port" \
      --device cuda:0 \
      >"$OUTPUT_DIR/logs/policy-gpu-${gpu}.log" 2>&1 &
  policy_pids+=("$!")
done

BASE_PORT="$BASE_PORT" GPU_COUNT="$GPU_COUNT" "$OPENPI_PYTHON" - <<'PY'
import os
import socket
import time

base_port = int(os.environ["BASE_PORT"])
gpu_count = int(os.environ["GPU_COUNT"])
deadline = time.monotonic() + 600
pending = set(range(gpu_count))
while pending and time.monotonic() < deadline:
    for index in list(pending):
        try:
            with socket.create_connection(("127.0.0.1", base_port + index), timeout=1):
                pending.remove(index)
        except OSError:
            pass
    if pending:
        time.sleep(2)
if pending:
    raise SystemExit(f"policy servers did not become ready: {sorted(pending)}")
print(f"policy_servers_ready={gpu_count}")
PY

for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
  port=$((BASE_PORT + gpu))
  LIBERO_CONFIG_PATH="$SIM_CONFIG" \
  PYTHONPATH="${OPENPI_ROOT}/packages/openpi-client/src:${PLUS_ROOT}:${SIM_OVERLAY}:${REPO_ROOT}/scripts/cabi_vla" \
    "$SIM_PYTHON" "${REPO_ROOT}/scripts/cabi_vla/evaluate_pi05_libero_plus_views.py" \
      --protocol "$PROTOCOL" \
      --output-dir "$OUTPUT_DIR" \
      --bddl-root "$BDDL_ROOT" \
      --host 127.0.0.1 \
      --port "$port" \
      --modes "${MODE_ARGS[@]}" \
      "${SUITE_ARGS[@]}" \
      --init-state-count "$INIT_STATE_COUNT" \
      --replan-steps "$REPLAN_STEPS" \
      --seed "$EVAL_SEED" \
      --probe-samples "$PROBE_SAMPLES" \
      --probe-horizon "$PROBE_HORIZON" \
      --num-shards "$GPU_COUNT" \
      --shard-index "$gpu" \
      --render-gpu "$gpu" \
      --video-episodes "$VIDEO_EPISODES" \
      >"$OUTPUT_DIR/logs/eval-shard-${gpu}.log" 2>&1 &
  eval_pids+=("$!")
done

failed=0
for pid in "${eval_pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [ "$failed" = 1 ]; then
  echo "one or more evaluation shards failed; inspect $OUTPUT_DIR/logs" >&2
  exit 1
fi

mapfile -t episode_files < <(find "$OUTPUT_DIR" -maxdepth 1 -name 'episodes-shard-*.jsonl' -type f | sort)
if [ "${#episode_files[@]}" -ne "$GPU_COUNT" ]; then
  echo "expected $GPU_COUNT episode shards, found ${#episode_files[@]}" >&2
  exit 1
fi

expected_episode_count=$(PYTHONPATH="$REPO_ROOT/scripts/cabi_vla" \
  PROTOCOL="$PROTOCOL" MODES="$EVAL_MODES" SUITES="$EVAL_SUITES" \
  INIT_STATE_COUNT="$INIT_STATE_COUNT" "$TOOLS_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
from evaluate_pi05_libero_plus_views import build_episode_specs

protocol = json.loads(Path(os.environ["PROTOCOL"]).read_text())
suites = os.environ["SUITES"].split() or None
specs = build_episode_specs(
    protocol,
    modes=os.environ["MODES"].split(),
    suites=suites,
    init_state_count=int(os.environ["INIT_STATE_COUNT"]),
)
print(len(specs))
PY
)
actual_episode_count=$(awk 'NF {count += 1} END {print count + 0}' "${episode_files[@]}")
if [ "$actual_episode_count" -ne "$expected_episode_count" ]; then
  echo "expected $expected_episode_count episodes, found $actual_episode_count" >&2
  exit 1
fi

if [[ "$SKIP_ANALYSIS" == 0 ]]; then
  if [[ "$EVAL_MODES" == "camera_full" ]]; then
    "$TOOLS_PYTHON" "${REPO_ROOT}/scripts/cabi_vla/analyze_libero_plus_camera_full.py" \
      --episodes "${episode_files[@]}" \
      --expected-count "$expected_episode_count" \
      --output-json "$OUTPUT_DIR/metrics.json" \
      --output-report "$OUTPUT_DIR/report.md"
  elif [[ "$EVAL_MODES" == "original_full" ]]; then
    "$TOOLS_PYTHON" "${REPO_ROOT}/scripts/cabi_vla/analyze_libero_original_full.py" \
      --episodes "${episode_files[@]}" \
      --expected-count "$expected_episode_count" \
      --expected-trials-per-task "$INIT_STATE_COUNT" \
      --output-json "$OUTPUT_DIR/metrics.json" \
      --output-report "$OUTPUT_DIR/report.md"
  else
    "$TOOLS_PYTHON" "${REPO_ROOT}/scripts/cabi_vla/analyze_pi05_libero_plus_views.py" \
      --episodes "${episode_files[@]}" \
      --output-json "$OUTPUT_DIR/metrics.json" \
      --output-figure "$OUTPUT_DIR/summary.png" \
      --output-report "$OUTPUT_DIR/report_zh.md"
  fi
fi

echo "study_complete=$OUTPUT_DIR"
