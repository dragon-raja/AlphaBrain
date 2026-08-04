#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to an AlphaBrain final_model directory}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a new or resumable evaluation directory}
PROTOCOL=${PROTOCOL:-/share/longjunyu/alphabrain/experiments/libero-plus-view-gap-v1/protocol-v3.json}
GPU_COUNT=${GPU_COUNT:-8}
BASE_PORT=${BASE_PORT:-18300}
INIT_STATE_COUNT=${INIT_STATE_COUNT:-2}
REPLAN_STEPS=${REPLAN_STEPS:-5}
EVAL_SEED=${EVAL_SEED:-20260804}
PROBE_SAMPLES=${PROBE_SAMPLES:-0}
PROBE_HORIZON=${PROBE_HORIZON:-5}
VIDEO_EPISODES=${VIDEO_EPISODES:-8}
EVAL_MODES=${EVAL_MODES:-gap}
EVAL_SUITES=${EVAL_SUITES:-}

POLICY_PYTHON=${POLICY_PYTHON:-$REPO_ROOT/.venv/bin/python}
SIM_PYTHON=${SIM_PYTHON:-/share/longjunyu/capt-vla/envs/libero/bin/python}
PLUS_ROOT=${PLUS_ROOT:-/share/longjunyu/alphabrain/datasets/libero-plus/runtime/LIBERO-plus}
SIM_OVERLAY=${SIM_OVERLAY:-/share/longjunyu/alphabrain/envs/libero-plus-runtime-overlay-v1}
SIM_CONFIG=${SIM_CONFIG:-/share/longjunyu/alphabrain/envs/libero-plus-runtime-config-v1}
BDDL_ROOT=${BDDL_ROOT:-$PLUS_ROOT/libero/libero/bddl_files}

read -r -a MODE_ARGS <<<"$EVAL_MODES"
SUITE_ARGS=()
if [[ -n "$EVAL_SUITES" ]]; then
  read -r -a SUITE_VALUES <<<"$EVAL_SUITES"
  SUITE_ARGS=(--suites "${SUITE_VALUES[@]}")
fi

if [[ ! "$GPU_COUNT" =~ ^[1-8]$ ]]; then
  echo "GPU_COUNT must be in [1,8]" >&2
  exit 2
fi
for required in \
  "$POLICY_PYTHON" \
  "$SIM_PYTHON" \
  "$CHECKPOINT/model.safetensors" \
  "$CHECKPOINT/framework_config.yaml" \
  "$PROTOCOL" \
  "$BDDL_ROOT"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR/logs"
checkpoint_sha256=$(sha256sum "$CHECKPOINT/model.safetensors" | awk '{print $1}')
protocol_sha256=$(sha256sum "$PROTOCOL" | awk '{print $1}')
code_sha256=$(sha256sum \
  "$REPO_ROOT/scripts/cabi_vla/serve_alphabrain_pi05_websocket.py" \
  "$REPO_ROOT/scripts/cabi_vla/evaluate_pi05_libero_plus_views.py" \
  "$REPO_ROOT/scripts/cabi_vla/analyze_pi05_libero_plus_views.py" \
  "$REPO_ROOT/scripts/cabi_vla/run_alphabrain_pi05_libero_plus_view_eval.sh" \
  | sha256sum | awk '{print $1}')
manifest="$OUTPUT_DIR/run_manifest.json"
temporary_manifest="$OUTPUT_DIR/.run_manifest.$$.tmp"

CHECKPOINT="$CHECKPOINT" CHECKPOINT_SHA256="$checkpoint_sha256" \
PROTOCOL="$PROTOCOL" PROTOCOL_SHA256="$protocol_sha256" CODE_SHA256="$code_sha256" \
GPU_COUNT="$GPU_COUNT" INIT_STATE_COUNT="$INIT_STATE_COUNT" \
REPLAN_STEPS="$REPLAN_STEPS" EVAL_SEED="$EVAL_SEED" \
PROBE_SAMPLES="$PROBE_SAMPLES" PROBE_HORIZON="$PROBE_HORIZON" \
EVAL_MODES="$EVAL_MODES" EVAL_SUITES="$EVAL_SUITES" \
  "$POLICY_PYTHON" - <<'PY' >"$temporary_manifest"
import json
import os

print(json.dumps({
    "schema_version": 1,
    "framework": "AlphaBrain",
    "model": "Pi0.5",
    "checkpoint": os.environ["CHECKPOINT"],
    "checkpoint_sha256": os.environ["CHECKPOINT_SHA256"],
    "protocol": os.environ["PROTOCOL"],
    "protocol_sha256": os.environ["PROTOCOL_SHA256"],
    "evaluation_code_sha256": os.environ["CODE_SHA256"],
    "gpu_count": int(os.environ["GPU_COUNT"]),
    "initial_state_count": int(os.environ["INIT_STATE_COUNT"]),
    "replan_steps": int(os.environ["REPLAN_STEPS"]),
    "evaluation_seed": int(os.environ["EVAL_SEED"]),
    "probe_samples": int(os.environ["PROBE_SAMPLES"]),
    "probe_horizon": int(os.environ["PROBE_HORIZON"]),
    "modes": os.environ["EVAL_MODES"].split(),
    "suites": os.environ["EVAL_SUITES"].split(),
}, indent=2, sort_keys=True))
PY
if [[ -e "$manifest" ]]; then
  if ! cmp -s "$manifest" "$temporary_manifest"; then
    echo "run manifest differs from existing output; use a new OUTPUT_DIR" >&2
    rm -f "$temporary_manifest"
    exit 2
  fi
  rm -f "$temporary_manifest"
else
  mv "$temporary_manifest" "$manifest"
fi

policy_pids=()
eval_pids=()
keepalive_stopped=0
cleanup() {
  for pid in "${eval_pids[@]:-}" "${policy_pids[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  if [[ "$keepalive_stopped" == 1 ]]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start_all.sh 1 8192 gpu-keepalive >/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

bash /workspace/ai2r/gpu_compute_keepalive/stop_all.sh gpu-keepalive >/dev/null
keepalive_stopped=1

for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
  port=$((BASE_PORT + gpu))
  CUDA_VISIBLE_DEVICES=$gpu \
  PRETRAINED_MODELS_DIR=/share/longjunyu/alphabrain/pretrained_models \
  ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1 \
  PYTHONPATH="/projects/openpi/src:/projects/openpi/packages/openpi-client/src" \
    "$POLICY_PYTHON" "$REPO_ROOT/scripts/cabi_vla/serve_alphabrain_pi05_websocket.py" \
      --checkpoint "$CHECKPOINT" \
      --port "$port" \
      --device cuda:0 \
      >"$OUTPUT_DIR/logs/policy-gpu-${gpu}.log" 2>&1 &
  policy_pids+=("$!")
done

BASE_PORT="$BASE_PORT" GPU_COUNT="$GPU_COUNT" "$POLICY_PYTHON" - <<'PY'
import os
import urllib.request
import time

base_port = int(os.environ["BASE_PORT"])
gpu_count = int(os.environ["GPU_COUNT"])
deadline = time.monotonic() + 900
pending = set(range(gpu_count))
while pending and time.monotonic() < deadline:
    for index in list(pending):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{base_port + index}/healthz", timeout=1) as response:
                if response.status == 200:
                    pending.remove(index)
        except Exception:
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
  PYTHONPATH="/projects/openpi/packages/openpi-client/src:${PLUS_ROOT}:${SIM_OVERLAY}:${REPO_ROOT}/scripts/cabi_vla" \
    "$SIM_PYTHON" "$REPO_ROOT/scripts/cabi_vla/evaluate_pi05_libero_plus_views.py" \
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
if [[ "$failed" == 1 ]]; then
  echo "one or more evaluation shards failed; inspect $OUTPUT_DIR/logs" >&2
  exit 1
fi

mapfile -t episode_files < <(find "$OUTPUT_DIR" -maxdepth 1 -name 'episodes-shard-*.jsonl' -type f | sort)
if [[ "${#episode_files[@]}" -ne "$GPU_COUNT" ]]; then
  echo "expected $GPU_COUNT episode shards, found ${#episode_files[@]}" >&2
  exit 1
fi

actual_episode_count=$(awk 'NF {count += 1} END {print count + 0}' "${episode_files[@]}")
expected_episode_count=$(PYTHONPATH="$REPO_ROOT/scripts/cabi_vla" \
  PROTOCOL="$PROTOCOL" MODES="$EVAL_MODES" SUITES="$EVAL_SUITES" \
  INIT_STATE_COUNT="$INIT_STATE_COUNT" "$POLICY_PYTHON" - <<'PY'
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
if [[ "$actual_episode_count" -ne "$expected_episode_count" ]]; then
  echo "expected $expected_episode_count episodes, found $actual_episode_count" >&2
  exit 1
fi

"$POLICY_PYTHON" "$REPO_ROOT/scripts/cabi_vla/analyze_pi05_libero_plus_views.py" \
  --episodes "${episode_files[@]}" \
  --output-json "$OUTPUT_DIR/metrics.json" \
  --output-figure "$OUTPUT_DIR/summary.png" \
  --output-report "$OUTPUT_DIR/report_zh.md"

echo "alphabrain_plus_eval_complete=$OUTPUT_DIR episodes=$actual_episode_count"
