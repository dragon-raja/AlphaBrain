#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to a self-contained AlphaBrain or OpenPI checkpoint directory}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a new or resumable evaluation directory}
PROTOCOL=${PROTOCOL:-$REPO_ROOT/configs/dsol_paper1/libero_hdf5_closed_loop_quick_gate_v1.json}
GPU_COUNT=${GPU_COUNT:-8}
EVAL_WORKER_COUNT=${EVAL_WORKER_COUNT:-$GPU_COUNT}
GPU_DEVICES=${DSOL_GPU_DEVICES:-}
BASE_PORT=${BASE_PORT:-18600}
REPLAN_STEPS=${REPLAN_STEPS:-5}
WAIT_STEPS=${WAIT_STEPS:-10}
EVAL_SEED=${EVAL_SEED:-20260818}
VIDEO_EPISODES=${VIDEO_EPISODES:-8}
MAX_EPISODES_PER_SHARD=${MAX_EPISODES_PER_SHARD:-}
RUN_ANALYSIS=${RUN_ANALYSIS:-1}
KEEPALIVE_MODE=${KEEPALIVE_MODE:-managed}
ANALYZER=${ANALYZER:-$REPO_ROOT/scripts/dsol_paper1/summarize_dsol_libero_hdf5_closed_loop.py}
POLICY_PYTHON=${POLICY_PYTHON:-/alphabrain/.venv/bin/python}
SIM_PYTHON=${SIM_PYTHON:-/workspace/envs/fresh-libero/bin/python}
RUNTIME=${RUNTIME:-/share/longjunyu/alphabrain/datasets/libero-plus/runtime/LIBERO-plus}
SIM_CONFIG=${SIM_CONFIG:-/share/longjunyu/alphabrain/envs/libero-plus-runtime-config-v1}
FFMPEG_EXE=${FFMPEG_EXE:-/usr/bin/ffmpeg}
POLICY_BACKEND=${POLICY_BACKEND:-alphabrain}
OPENPI_ROOT=${OPENPI_ROOT:-/projects/openpi}
OPENPI_PYTHON=${OPENPI_PYTHON:-$OPENPI_ROOT/.venv/bin/python}
OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/share/longjunyu/alphabrain/cache/openpi}
OPENPI_CONFIG=${OPENPI_CONFIG:-pi05_libero}
NOISE_BANK_MANIFEST=${NOISE_BANK_MANIFEST:-}
REQUIRE_EXPLICIT_NOISE=${REQUIRE_EXPLICIT_NOISE:-0}

for required in \
  "$CHECKPOINT/model.safetensors" \
  "$PROTOCOL" \
  "$ANALYZER" \
  "$SIM_PYTHON" \
  "$FFMPEG_EXE" \
  "$RUNTIME/libero/libero/bddl_files"; do
  [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 2; }
done
case "$POLICY_BACKEND" in
  alphabrain)
    for required in "$CHECKPOINT/framework_config.yaml" "$POLICY_PYTHON"; do
      [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 2; }
    done
    READY_PYTHON=$POLICY_PYTHON
    ;;
  openpi)
    for required in \
      "$CHECKPOINT/config.json" \
      "$CHECKPOINT/assets/physical-intelligence/libero/norm_stats.json" \
      "$OPENPI_PYTHON" \
      "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"; do
      [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 2; }
    done
    READY_PYTHON=$OPENPI_PYTHON
    ;;
  *)
    echo "POLICY_BACKEND must be alphabrain or openpi" >&2
    exit 2
    ;;
esac
[[ "$GPU_COUNT" =~ ^[1-8]$ ]] || { echo "GPU_COUNT must be in [1,8]" >&2; exit 2; }
[[ "$EVAL_WORKER_COUNT" =~ ^[1-9][0-9]*$ ]] && (( EVAL_WORKER_COUNT <= 64 )) || {
  echo "EVAL_WORKER_COUNT must be in [1,64]" >&2
  exit 2
}
if [[ -z "$GPU_DEVICES" ]]; then
  physical_gpus=()
  for ((gpu=0; gpu<GPU_COUNT; gpu++)); do physical_gpus+=("$gpu"); done
  GPU_DEVICES=$(IFS=,; echo "${physical_gpus[*]}")
else
  IFS=, read -r -a physical_gpus <<< "$GPU_DEVICES"
fi
[[ "${#physical_gpus[@]}" -eq "$GPU_COUNT" ]] || {
  echo "DSOL_GPU_DEVICES must list exactly GPU_COUNT=$GPU_COUNT devices" >&2
  exit 2
}
declare -A seen_gpus=()
for gpu in "${physical_gpus[@]}"; do
  [[ "$gpu" =~ ^[0-7]$ ]] || { echo "invalid physical GPU index: $gpu" >&2; exit 2; }
  [[ -z "${seen_gpus[$gpu]:-}" ]] || { echo "duplicate physical GPU index: $gpu" >&2; exit 2; }
  seen_gpus[$gpu]=1
done
[[ -z "$MAX_EPISODES_PER_SHARD" || "$MAX_EPISODES_PER_SHARD" =~ ^[1-9][0-9]*$ ]] || {
  echo "MAX_EPISODES_PER_SHARD must be empty or a positive integer" >&2
  exit 2
}
[[ "$WAIT_STEPS" =~ ^[0-9]+$ ]] || { echo "WAIT_STEPS must be a nonnegative integer" >&2; exit 2; }
[[ "$VIDEO_EPISODES" =~ ^[0-9]+$ ]] || { echo "VIDEO_EPISODES must be a nonnegative integer" >&2; exit 2; }
[[ "$RUN_ANALYSIS" =~ ^[01]$ ]] || { echo "RUN_ANALYSIS must be 0 or 1" >&2; exit 2; }
[[ "$REQUIRE_EXPLICIT_NOISE" =~ ^[01]$ ]] || { echo "REQUIRE_EXPLICIT_NOISE must be 0 or 1" >&2; exit 2; }
if [[ "$REQUIRE_EXPLICIT_NOISE" == 1 ]]; then
  [[ -n "$NOISE_BANK_MANIFEST" && -f "$NOISE_BANK_MANIFEST" ]] || {
    echo "formal evaluation requires NOISE_BANK_MANIFEST" >&2; exit 2;
  }
fi
[[ "$KEEPALIVE_MODE" == managed || "$KEEPALIVE_MODE" == preserve ]] || {
  echo "KEEPALIVE_MODE must be managed or preserve" >&2; exit 2;
}

mkdir -p "$OUTPUT_DIR/logs"
checkpoint_sha256=$(sha256sum "$CHECKPOINT/model.safetensors" | awk '{print $1}')
protocol_sha256=$(sha256sum "$PROTOCOL" | awk '{print $1}')
noise_bank_sha256=""
if [[ -n "$NOISE_BANK_MANIFEST" ]]; then
  noise_bank_sha256=$(sha256sum "$NOISE_BANK_MANIFEST" | awk '{print $1}')
fi
code_sha256=$(sha256sum \
  "$REPO_ROOT/scripts/cabi_vla/serve_alphabrain_pi05_websocket.py" \
  "$REPO_ROOT/scripts/cabi_vla/serve_openpi_deterministic.py" \
  "$REPO_ROOT/scripts/dsol_paper1/evaluate_dsol_libero_hdf5_views.py" \
  "$ANALYZER" \
  "$REPO_ROOT/scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh" \
  | sha256sum | awk '{print $1}')
jq -n \
  --arg checkpoint "$CHECKPOINT" \
  --arg checkpoint_sha256 "$checkpoint_sha256" \
  --arg policy_backend "$POLICY_BACKEND" \
  --arg openpi_config "$OPENPI_CONFIG" \
  --arg protocol "$PROTOCOL" \
  --arg protocol_sha256 "$protocol_sha256" \
  --arg noise_bank_manifest "$NOISE_BANK_MANIFEST" \
  --arg noise_bank_manifest_sha256 "$noise_bank_sha256" \
  --argjson require_explicit_noise "$REQUIRE_EXPLICIT_NOISE" \
  --arg analyzer "$ANALYZER" \
  --arg code_sha256 "$code_sha256" \
  --arg gpu_devices "$GPU_DEVICES" \
  --argjson gpu_count "$GPU_COUNT" \
  --argjson eval_worker_count "$EVAL_WORKER_COUNT" \
  --argjson replan_steps "$REPLAN_STEPS" \
  --argjson wait_steps "$WAIT_STEPS" \
  --argjson eval_seed "$EVAL_SEED" \
  --argjson video_episodes_per_worker "$VIDEO_EPISODES" \
  --arg max_episodes_per_shard "$MAX_EPISODES_PER_SHARD" \
  --argjson run_analysis "$RUN_ANALYSIS" \
  --arg keepalive_mode "$KEEPALIVE_MODE" \
  '{schema:"dsol_libero_hdf5_closed_loop_run_v1",checkpoint:$checkpoint,checkpoint_sha256:$checkpoint_sha256,policy_backend:$policy_backend,openpi_config:(if $policy_backend == "openpi" then $openpi_config else null end),protocol:$protocol,protocol_sha256:$protocol_sha256,noise_bank_manifest:(if $noise_bank_manifest == "" then null else $noise_bank_manifest end),noise_bank_manifest_sha256:(if $noise_bank_manifest_sha256 == "" then null else $noise_bank_manifest_sha256 end),require_explicit_noise:($require_explicit_noise == 1),analyzer:$analyzer,code_sha256:$code_sha256,gpu_count:$gpu_count,eval_worker_count:$eval_worker_count,gpu_devices:($gpu_devices|split(",")|map(tonumber)),replan_steps:$replan_steps,wait_steps:$wait_steps,eval_seed:$eval_seed,video_episodes_per_worker:$video_episodes_per_worker,max_episodes_per_shard:(if $max_episodes_per_shard == "" then null else ($max_episodes_per_shard | tonumber) end),run_analysis:($run_analysis == 1),keepalive_mode:$keepalive_mode}' \
  > "$OUTPUT_DIR/run_manifest.json"

policy_pids=()
eval_pids=()
stopped_keepalives=()
cleanup() {
  for pid in "${eval_pids[@]:-}" "${policy_pids[@]:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  for gpu in "${stopped_keepalives[@]:-}"; do
    [[ -n "$gpu" ]] || continue
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      1 8192 "gpu-keepalive-${gpu}" "$gpu" >/dev/null || true
  done
}
trap cleanup EXIT INT TERM
for gpu in "${physical_gpus[@]}"; do
  if [[ "$KEEPALIVE_MODE" == managed ]] && tmux has-session -t "gpu-keepalive-${gpu}" 2>/dev/null; then
    tmux kill-session -t "gpu-keepalive-${gpu}"
    stopped_keepalives+=("$gpu")
  fi
done

max_episode_args=()
if [[ -n "$MAX_EPISODES_PER_SHARD" ]]; then
  max_episode_args=(--max-episodes "$MAX_EPISODES_PER_SHARD")
fi
noise_bank_args=()
if [[ -n "$NOISE_BANK_MANIFEST" ]]; then
  noise_bank_args=(--noise-bank-manifest "$NOISE_BANK_MANIFEST")
fi
if [[ "$REQUIRE_EXPLICIT_NOISE" == 1 ]]; then
  noise_bank_args+=(--require-explicit-noise)
fi

for ((shard=0; shard<GPU_COUNT; shard++)); do
  gpu=${physical_gpus[$shard]}
  port=$((BASE_PORT + shard))
  if [[ "$POLICY_BACKEND" == alphabrain ]]; then
    CUDA_VISIBLE_DEVICES=$gpu \
    PRETRAINED_MODELS_DIR=/share/longjunyu/alphabrain/pretrained_models \
    ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1 \
    PYTHONPATH="$REPO_ROOT:/projects/openpi/src:/projects/openpi/packages/openpi-client/src" \
      "$POLICY_PYTHON" "$REPO_ROOT/scripts/cabi_vla/serve_alphabrain_pi05_websocket.py" \
        --checkpoint "$CHECKPOINT" --port "$port" --device cuda:0 \
        > "$OUTPUT_DIR/logs/policy-shard-${shard}-gpu-${gpu}.log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$gpu \
    OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
    PYTHONPATH="$REPO_ROOT/scripts/cabi_vla" \
      "$OPENPI_PYTHON" "$REPO_ROOT/scripts/cabi_vla/serve_openpi_deterministic.py" \
        --checkpoint "$CHECKPOINT" --config "$OPENPI_CONFIG" \
        --port "$port" --device cuda:0 \
        > "$OUTPUT_DIR/logs/policy-shard-${shard}-gpu-${gpu}.log" 2>&1 &
  fi
  policy_pids+=("$!")
done

BASE_PORT="$BASE_PORT" GPU_COUNT="$GPU_COUNT" "$READY_PYTHON" - <<'PY'
import os, socket, time
base = int(os.environ["BASE_PORT"]); count = int(os.environ["GPU_COUNT"])
pending = set(range(count)); deadline = time.monotonic() + 900
while pending and time.monotonic() < deadline:
    for index in list(pending):
        try:
            with socket.create_connection(("127.0.0.1", base + index), timeout=1):
                pending.remove(index)
        except OSError:
            pass
    if pending: time.sleep(2)
if pending: raise SystemExit(f"policy servers did not become ready: {sorted(pending)}")
PY

for ((shard=0; shard<EVAL_WORKER_COUNT; shard++)); do
  server_index=$((shard % GPU_COUNT))
  gpu=${physical_gpus[$server_index]}
  port=$((BASE_PORT + server_index))
  LIBERO_CONFIG_PATH="$SIM_CONFIG" \
  IMAGEIO_FFMPEG_EXE="$FFMPEG_EXE" \
  PYTHONPATH="$REPO_ROOT:/projects/openpi/packages/openpi-client/src:$REPO_ROOT/scripts/cabi_vla:$REPO_ROOT/scripts/dsol_paper1" \
    "$SIM_PYTHON" "$REPO_ROOT/scripts/dsol_paper1/evaluate_dsol_libero_hdf5_views.py" \
      --protocol "$PROTOCOL" --output-dir "$OUTPUT_DIR" \
      --runtime "$RUNTIME" --config-root "$SIM_CONFIG" \
      --host 127.0.0.1 --port "$port" \
      --replan-steps "$REPLAN_STEPS" --wait-steps "$WAIT_STEPS" --seed "$EVAL_SEED" \
      --num-shards "$EVAL_WORKER_COUNT" --shard-index "$shard" --render-gpu "$gpu" \
      --video-episodes "$VIDEO_EPISODES" \
      "${noise_bank_args[@]}" \
      "${max_episode_args[@]}" \
      > "$OUTPUT_DIR/logs/eval-shard-${shard}-gpu-${gpu}.log" 2>&1 &
  eval_pids+=("$!")
done

failed=0
for pid in "${eval_pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]] || { echo "one or more evaluation shards failed" >&2; exit 1; }
actual=$(awk 'NF {n++} END {print n+0}' "$OUTPUT_DIR"/episodes-shard-*.jsonl)
expected=$(PROTOCOL="$PROTOCOL" EVAL_WORKER_COUNT="$EVAL_WORKER_COUNT" MAX_EPISODES_PER_SHARD="$MAX_EPISODES_PER_SHARD" \
  "$READY_PYTHON" - <<'PY'
import json
import os

with open(os.environ["PROTOCOL"], encoding="utf-8") as handle:
    protocol = json.load(handle)
worker_count = int(os.environ["EVAL_WORKER_COUNT"])
limit_text = os.environ["MAX_EPISODES_PER_SHARD"]
limit = int(limit_text) if limit_text else None
count = 0
for shard in range(worker_count):
    shard_count = sum(index % worker_count == shard for index in range(len(protocol["specs"])))
    count += min(shard_count, limit) if limit is not None else shard_count
print(count)
PY
)
[[ "$actual" == "$expected" ]] || { echo "expected $expected episodes, found $actual" >&2; exit 1; }
if [[ "$RUN_ANALYSIS" == 1 ]]; then
  "$POLICY_PYTHON" "$ANALYZER" \
    "$OUTPUT_DIR"/episodes-shard-*.jsonl \
    --output-dir "$OUTPUT_DIR/analysis" \
    > "$OUTPUT_DIR/logs/analysis.log"
fi
echo "dsol_hdf5_closed_loop_complete=$OUTPUT_DIR episodes=$actual"
