#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PLAN=${PLAN:-$REPO_ROOT/configs/dsol_paper1/libero_visibility_scan_quick_gate_v1.json}
CATALOG=${CATALOG:-$REPO_ROOT/configs/dsol_paper1/libero_view_catalog_v2.json}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT to a new or resumable scan directory}
GPU_COUNT=${GPU_COUNT:-8}
GPU_DEVICES=${DSOL_GPU_DEVICES:-}
MAX_STATES_PER_SHARD=${MAX_STATES_PER_SHARD:-}
CANDIDATE_GROUPS=${CANDIDATE_GROUPS:-broad_heldout_32,wide_extrapolation_24,diagnostic_extreme_orbit,diagnostic_look_away}
POSE_IDS=${POSE_IDS:-}
SIM_PYTHON=${SIM_PYTHON:-/workspace/envs/fresh-libero/bin/python}
RUNTIME=${RUNTIME:-/share/longjunyu/alphabrain/datasets/libero-plus/runtime/LIBERO-plus}
SIM_CONFIG=${SIM_CONFIG:-/share/longjunyu/alphabrain/envs/libero-plus-runtime-config-v1}

for required in "$PLAN" "$CATALOG" "$SIM_PYTHON" "$RUNTIME/libero/libero/bddl_files"; do
  [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 2; }
done
[[ "$GPU_COUNT" =~ ^[1-8]$ ]] || { echo "GPU_COUNT must be in [1,8]" >&2; exit 2; }
[[ -z "$MAX_STATES_PER_SHARD" || "$MAX_STATES_PER_SHARD" =~ ^[1-9][0-9]*$ ]] || {
  echo "MAX_STATES_PER_SHARD must be empty or positive" >&2
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

mkdir -p "$OUTPUT_ROOT/logs"
plan_sha256=$(sha256sum "$PLAN" | awk '{print $1}')
catalog_sha256=$(sha256sum "$CATALOG" | awk '{print $1}')
code_sha256=$(sha256sum \
  "$REPO_ROOT/scripts/dsol_paper1/libero_visibility.py" \
  "$REPO_ROOT/scripts/dsol_paper1/libero_constructed_view.py" \
  "$REPO_ROOT/scripts/dsol_paper1/scan_libero_hdf5_views.py" \
  "$REPO_ROOT/scripts/dsol_paper1/run_libero_visibility_scan_plan.py" \
  "$REPO_ROOT/scripts/dsol_paper1/summarize_libero_visibility_scan.py" \
  "$REPO_ROOT/scripts/dsol_paper1/run_libero_visibility_scan_matrix.sh" \
  | sha256sum | awk '{print $1}')
jq -n \
  --arg plan "$PLAN" --arg plan_sha256 "$plan_sha256" \
  --arg catalog "$CATALOG" --arg catalog_sha256 "$catalog_sha256" \
  --arg code_sha256 "$code_sha256" \
  --arg gpu_devices "$GPU_DEVICES" \
  --arg groups "$CANDIDATE_GROUPS" --arg pose_ids "$POSE_IDS" --arg max_states_per_shard "$MAX_STATES_PER_SHARD" \
  --argjson gpu_count "$GPU_COUNT" \
  '{schema:"dsol_libero_visibility_scan_run_v1",plan:$plan,plan_sha256:$plan_sha256,catalog:$catalog,catalog_sha256:$catalog_sha256,code_sha256:$code_sha256,groups:$groups,pose_ids:(if $pose_ids == "" then null else ($pose_ids|split(",")) end),gpu_count:$gpu_count,gpu_devices:($gpu_devices|split(",")|map(tonumber)),max_states_per_shard:(if $max_states_per_shard == "" then null else ($max_states_per_shard|tonumber) end)}' \
  > "$OUTPUT_ROOT/run_manifest.json"

pids=()
stopped_keepalives=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  for gpu in "${stopped_keepalives[@]:-}"; do
    [[ -n "$gpu" ]] || continue
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh 1 8192 "gpu-keepalive-${gpu}" "$gpu" >/dev/null || true
  done
}
trap cleanup EXIT INT TERM

max_state_args=()
if [[ -n "$MAX_STATES_PER_SHARD" ]]; then
  max_state_args=(--max-states "$MAX_STATES_PER_SHARD")
fi
pose_args=()
if [[ -n "$POSE_IDS" ]]; then
  pose_args=(--pose-ids "$POSE_IDS")
fi
for ((shard=0; shard<GPU_COUNT; shard++)); do
  gpu=${physical_gpus[$shard]}
  if tmux has-session -t "gpu-keepalive-${gpu}" 2>/dev/null; then
    tmux kill-session -t "gpu-keepalive-${gpu}"
    stopped_keepalives+=("$gpu")
  fi
  LIBERO_CONFIG_PATH="$SIM_CONFIG" \
  PYTHONPATH="$REPO_ROOT:$REPO_ROOT/scripts/dsol_paper1:$REPO_ROOT/scripts/cabi_vla" \
    "$SIM_PYTHON" "$REPO_ROOT/scripts/dsol_paper1/run_libero_visibility_scan_plan.py" \
      --plan "$PLAN" --output-root "$OUTPUT_ROOT" \
      --runtime "$RUNTIME" --catalog "$CATALOG" --config-root "$SIM_CONFIG" \
      --groups "$CANDIDATE_GROUPS" --num-shards "$GPU_COUNT" --shard-index "$shard" \
      --render-gpu "$gpu" "${max_state_args[@]}" "${pose_args[@]}" \
      > "$OUTPUT_ROOT/logs/shard-${shard}-gpu-${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]] || { echo "one or more visibility shards failed" >&2; exit 1; }
actual=$(OUTPUT_ROOT="$OUTPUT_ROOT" "$SIM_PYTHON" - <<'PY'
import glob
import json
import os

passed = set()
for path in glob.glob(os.path.join(os.environ["OUTPUT_ROOT"], "shard-*.jsonl")):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "PASS":
                    passed.add(row["scan_id"])
print(len(passed))
PY
)
expected=$(PLAN="$PLAN" GPU_COUNT="$GPU_COUNT" MAX_STATES_PER_SHARD="$MAX_STATES_PER_SHARD" \
  "$SIM_PYTHON" - <<'PY'
import json
import os

with open(os.environ["PLAN"], encoding="utf-8") as handle:
    count = len(json.load(handle)["records"])
gpu_count = int(os.environ["GPU_COUNT"])
limit_text = os.environ["MAX_STATES_PER_SHARD"]
limit = int(limit_text) if limit_text else None
expected = 0
for shard in range(gpu_count):
    shard_count = sum(index % gpu_count == shard for index in range(count))
    expected += min(shard_count, limit) if limit is not None else shard_count
print(expected)
PY
)
[[ "$actual" == "$expected" ]] || { echo "expected $expected states, found $actual" >&2; exit 1; }
"$SIM_PYTHON" "$REPO_ROOT/scripts/dsol_paper1/summarize_libero_visibility_scan.py" \
  "$OUTPUT_ROOT"/shard-*.jsonl --output-dir "$OUTPUT_ROOT/analysis" \
  > "$OUTPUT_ROOT/logs/analysis.log"
echo "visibility_scan_complete=$OUTPUT_ROOT states=$actual"
