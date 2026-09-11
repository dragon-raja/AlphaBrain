#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
POLL_SECONDS=${H20_HANDOFF_POLL_SECONDS:-30}
MASTER_SESSION=${H20_MASTER_SESSION:-cabi-h20-migration-gate-v15-s41}
RUN_ROOT=${CABI_TRAIN_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/runs}
HANDOFF_ROOT=${CABI_HANDOFF_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/target-handoff-diagnostics}
COMPARISON_ROOT=${CABI_COMPARISON_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/comparisons}

BC_GPU=${H20_BC_GPU:-0}
BRIDGE_GPU=${H20_BRIDGE_GPU:-1}
PLAIN_GPU=${H20_PLAIN_GPU:-4}
GROUNDED_GPU=${H20_GROUNDED_GPU:-5}
BC_CHECKPOINT=$RUN_ROOT/cabi_bind_pi05_bc_h20_smoke_seed41_steps33000_h20-edge-balanced-3epoch-v15/final_model
BRIDGE_CHECKPOINT=$RUN_ROOT/cabi_bind_pi05_action_bridge_h20_smoke_seed41_steps33000_h20-bridge-edge-balanced-3epoch-v15/final_model
PLAIN_CHECKPOINT=$RUN_ROOT/cabi_bind_pi05_action_completion_h20_smoke_seed41_steps33000_h20-cafc-edge-balanced-3epoch-v15/final_model
GROUNDED_CHECKPOINT=$RUN_ROOT/cabi_bind_pi05_action_bridge_completion_h20_smoke_seed41_steps33000_h20-bridge-cafc-edge-balanced-3epoch-v15/final_model

BC_RUN=bc_h20_target_handoff_s41_v15
BRIDGE_RUN=bridge_h20_target_handoff_s41_v15
PLAIN_RUN=cafc_h20_target_handoff_s41_v15
GROUNDED_RUN=bridge_cafc_h20_target_handoff_s41_v15
SUMMARY=$COMPARISON_ROOT/cafc_h20_target_handoff_seed41_v15.json

if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "H20_HANDOFF_POLL_SECONDS must be positive" >&2
  exit 2
fi
echo "waiting formal H20 master: $MASTER_SESSION"
while tmux has-session -t "$MASTER_SESSION" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done

for checkpoint in \
  "$BC_CHECKPOINT" "$BRIDGE_CHECKPOINT" "$PLAIN_CHECKPOINT" "$GROUNDED_CHECKPOINT"; do
  if [[ ! -s "$checkpoint/model.safetensors" ]]; then
    echo "missing H20 checkpoint: $checkpoint" >&2
    exit 1
  fi
done

run_handoff() {
  local checkpoint=$1
  local run_name=$2
  local gpu=$3
  local output=$HANDOFF_ROOT/$run_name/target_handoff.json
  if [[ -s "$output" ]]; then
    return 0
  fi
  mkdir -p "$HANDOFF_ROOT/$run_name"
  CABI_HANDOFF_STATE_INDICES=0 \
  CABI_HANDOFF_K=3 \
  CABI_HANDOFF_TOTAL_BUDGET=320 \
    "$REPO_ROOT/scripts/cabi_vla/run_libero_bind_target_handoff.sh" \
      "$checkpoint" "$run_name" "$gpu" \
      >"$HANDOFF_ROOT/$run_name/launcher.log" 2>&1
}

run_handoff "$BC_CHECKPOINT" "$BC_RUN" "$BC_GPU" & bc_pid=$!
run_handoff "$BRIDGE_CHECKPOINT" "$BRIDGE_RUN" "$BRIDGE_GPU" & bridge_pid=$!
run_handoff "$PLAIN_CHECKPOINT" "$PLAIN_RUN" "$PLAIN_GPU" & plain_pid=$!
run_handoff "$GROUNDED_CHECKPOINT" "$GROUNDED_RUN" "$GROUNDED_GPU" & grounded_pid=$!

failed=0
for item in \
  "$bc_pid:bc" "$bridge_pid:bridge" "$plain_pid:plain" "$grounded_pid:grounded"; do
  pid=${item%%:*}
  status=0
  wait "$pid" || status=$?
  if (( status != 0 )); then
    echo "target handoff arm failed: ${item#*:} status=$status" >&2
    failed=1
  fi
done
if (( failed != 0 )); then exit 1; fi

render_one() {
  local run_name=$1
  local run_dir=$HANDOFF_ROOT/$run_name
  local output_dir=$run_dir/videos_h264_av1
  if [[ -s "$output_dir/manifest.json" ]] && \
     jq -e '.codecs | (index("h264") != null and index("av1") != null)' "$output_dir/manifest.json" >/dev/null; then
    return 0
  fi
  if [[ -e "$output_dir" ]]; then
    echo "partial handoff render requires audit: $output_dir" >&2
    return 1
  fi
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/cabi_vla/render_libero_bind_eval_frames.py" \
    --evaluation "$run_dir/target_handoff.json" \
    --frame-dir "$run_dir/frames" \
    --output-dir "$output_dir" \
    --codecs h264,av1 \
    --fps 20 >"$run_dir/render_h264_av1.log" 2>&1
}

render_pair() {
  local label=$1
  local baseline_name=$2
  local method_name=$3
  local baseline_run=$4
  local method_run=$5
  local output_dir=$COMPARISON_ROOT/cafc_h20_target_handoff_seed41_v15_${label}_paired_h264_av1
  if [[ -s "$output_dir/manifest.json" ]] && \
     jq -e '.codecs | (index("h264") != null and index("av1") != null)' "$output_dir/manifest.json" >/dev/null; then
    return 0
  fi
  if [[ -e "$output_dir" ]]; then
    echo "partial paired handoff render requires audit: $output_dir" >&2
    return 1
  fi
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/cabi_vla/render_libero_bind_paired_videos.py" \
    --baseline-evaluation "$HANDOFF_ROOT/$baseline_run/target_handoff.json" \
    --baseline-frame-dir "$HANDOFF_ROOT/$baseline_run/frames" \
    --method-evaluation "$HANDOFF_ROOT/$method_run/target_handoff.json" \
    --method-frame-dir "$HANDOFF_ROOT/$method_run/frames" \
    --output-dir "$output_dir" \
    --baseline-name "$baseline_name" \
    --method-name "$method_name" \
    --codecs h264,av1 \
    --fps 20 >"$COMPARISON_ROOT/cafc_h20_target_handoff_seed41_v15_${label}_render.log" 2>&1
}

render_one "$BC_RUN"
render_one "$BRIDGE_RUN"
render_one "$PLAIN_RUN"
render_one "$GROUNDED_RUN"
mkdir -p "$COMPARISON_ROOT"
render_pair plain BC-H20 CAFC-H20 "$BC_RUN" "$PLAIN_RUN"
render_pair grounded Bridge-H20 Bridge+CAFC-H20 "$BRIDGE_RUN" "$GROUNDED_RUN"

if [[ -e "$SUMMARY" ]]; then
  echo "refusing to overwrite target handoff summary: $SUMMARY" >&2
  exit 1
fi
"$REPO_ROOT/.venv/bin/python" - \
  "$HANDOFF_ROOT/$BC_RUN/target_handoff.json" \
  "$HANDOFF_ROOT/$BRIDGE_RUN/target_handoff.json" \
  "$HANDOFF_ROOT/$PLAIN_RUN/target_handoff.json" \
  "$HANDOFF_ROOT/$GROUNDED_RUN/target_handoff.json" \
  "$SUMMARY" <<'PY'
import json
import os
import sys
from pathlib import Path

names = ("bc_h20", "bridge_h20", "cafc_h20", "bridge_cafc_h20")
paths = list(map(Path, sys.argv[1:5]))
output = Path(sys.argv[5])
arms = {}
for name, path in zip(names, paths):
    payload = json.loads(path.read_text())
    rows = payload["rows"]
    arms[name] = {
        "output": str(path),
        "task_success": sum(bool(row["success"]) for row in rows),
        "transport_success": sum(bool(row["transport_success"]) for row in rows),
        "edge_count": len(rows),
        "mean_final_xy_distance": sum(
            float(row["final_source_target_xy_distance"]) for row in rows
        ) / len(rows),
        "by_edge": {
            row["edge_id"]: {
                "success": bool(row["success"]),
                "transport_success": bool(row["transport_success"]),
                "final_xy_distance": float(row["final_source_target_xy_distance"]),
            }
            for row in rows
        },
    }
result = {
    "schema_version": 1,
    "status": "complete",
    "formal_gate_metric": False,
    "diagnostic": "exact_teacher_prefix_target_handoff",
    "training_action_horizon": 20,
    "execution_horizon": 3,
    "arms": arms,
    "note": "This post-hoc mechanism diagnostic cannot establish end-to-end migration.",
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output)
print(json.dumps({"status": "complete", "output": str(output)}))
PY

echo "h20_target_handoff_complete summary=$SUMMARY"

