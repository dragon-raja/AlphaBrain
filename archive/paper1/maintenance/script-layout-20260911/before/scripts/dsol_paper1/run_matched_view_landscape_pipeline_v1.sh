#!/usr/bin/env bash
# Bounded stages 1--3 only. This launcher never trains or moves a live camera.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LANDSCAPE_ROOT=/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908
ACCEL_ROOT=$LANDSCAPE_ROOT/canonical-accel
mkdir -p "$LANDSCAPE_ROOT/logs" "$ACCEL_ROOT/logs"
exec > >(tee -a "$LANDSCAPE_ROOT/logs/pipeline.log") 2>&1
exec 8>"$LANDSCAPE_ROOT/logs/pipeline-gpu.lock"
flock -n 8 || { echo 'another matched-landscape pipeline holds the GPUs' >&2; exit 3; }

[[ -f "$ACCEL_ROOT/manifest.json" ]] || { echo 'frozen Accel manifest missing' >&2; exit 2; }
[[ -f "$LANDSCAPE_ROOT/protocols/protocol_manifest.json" ]] || exit 2
[[ -f "$REPO_ROOT/scripts/dsol_paper1/run_matched_view_landscape_v1.sh" ]] || exit 2
bash "$REPO_ROOT/scripts/dsol_paper1/run_matched_view_landscape_v1.sh" --check-only
/alphabrain/.venv/bin/python - <<'PY'
import subprocess
from pathlib import Path
rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                '--format=csv,noheader,nounits'], text=True)
occupied = []
for value in set(rows.split()):
    if not value.isdigit():
        continue
    try:
        command = (Path('/proc') / value / 'cmdline').read_bytes().replace(b'\0', b' ')
    except FileNotFoundError:
        continue
    if b'/workspace/ai2r/gpu_compute_keepalive/gpu_compute_keepalive.py' not in command:
        occupied.append(value)
if occupied:
    raise SystemExit(f'GPU allocation conflict; no job changed: {occupied}')
PY

active_pids=()
stopped_keepalives=()
restore_keepalives() {
  local gpu
  for gpu in "${stopped_keepalives[@]}"; do
    if ! tmux has-session -t "gpu-keepalive-$gpu" 2>/dev/null; then
      bash /workspace/ai2r/gpu_compute_keepalive/start.sh 1 8192 "gpu-keepalive-$gpu" "$gpu" || true
    fi
  done
  stopped_keepalives=()
}
cleanup() {
  local pid
  for pid in "${active_pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${active_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  restore_keepalives
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "accel_start $(date -u +%FT%TZ) budget=8_states_x_97_views_x_8_noise max_worker_seconds=5400"
for gpu in 0 1 2 3 4 5 6 7; do
  if tmux has-session -t "gpu-keepalive-$gpu" 2>/dev/null; then
    tmux kill-session -t "gpu-keepalive-$gpu"
    stopped_keepalives+=("$gpu")
  fi
done
for gpu in 0 1 2 3 4 5 6 7; do
  timeout --signal=TERM --kill-after=120s 5400s env \
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    PRETRAINED_MODELS_DIR=/share/longjunyu/alphabrain/pretrained_models \
    ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1 \
    PYTHONPATH="$REPO_ROOT:/projects/openpi/src:/projects/openpi/packages/openpi-client/src" \
    /alphabrain/.venv/bin/python "$REPO_ROOT/scripts/dsol_paper1/run_matched_view_accel_v1.py" \
      --manifest "$ACCEL_ROOT/manifest.json" --num-shards 8 --shard-index "$gpu" --device cuda:0 \
      > "$ACCEL_ROOT/logs/shard-$gpu.log" 2>&1 &
  active_pids+=("$!")
done
failed=0
for pid in "${active_pids[@]}"; do wait "$pid" || failed=1; done
active_pids=()
restore_keepalives
[[ "$failed" == 0 ]] || { echo 'Accel failed; closed-loop stages were not launched' >&2; exit 1; }

PYTHONPATH="$REPO_ROOT/scripts/dsol_paper1" /alphabrain/.venv/bin/python - "$ACCEL_ROOT" <<'PY'
import json, sys
from pathlib import Path
from run_matched_view_accel_v1 import validate_result
from run_view_value_expectation_accel_ensemble import atomic_json, sha256_file
root = Path(sys.argv[1])
manifest_path = root / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
digest = sha256_file(manifest_path)
states = {state['asset_source_pair_key']: state for state in manifest['states']}
rows = [json.loads(line) for file in sorted(root.glob('rank-shard-*.jsonl'))
        for line in file.read_text().splitlines() if line.strip()]
if len(rows) != 8 or {row['pair_key'] for row in rows} != set(states):
    raise SystemExit('canonical Accel output is not exactly the frozen eight states')
for row in rows:
    validate_result(row, states[row['pair_key']], digest)
receipt = {'status': 'PASS_COMPLETE', 'manifest_sha256': digest,
           'state_count': 8, 'candidate_count': 97, 'ensemble_size': 8,
           'checkpoint_sha256': manifest['checkpoint_sha256'],
           'state_inference_seconds': [row['inference_seconds'] for row in rows]}
path = root / 'completion.json'
if path.exists() and json.loads(path.read_text()) != receipt:
    raise SystemExit('existing completion differs; no overwrite')
if not path.exists():
    atomic_json(path, receipt)
print(json.dumps(receipt), flush=True)
PY
echo "accel_complete $(date -u +%FT%TZ)"
echo "closed_loop_start $(date -u +%FT%TZ) budget=24832_unique_episodes"
bash "$REPO_ROOT/scripts/dsol_paper1/run_matched_view_landscape_v1.sh" &
active_pids=("$!")
wait "${active_pids[0]}"
active_pids=()
echo "cpu_reports_start $(date -u +%FT%TZ)"
bash "$REPO_ROOT/scripts/dsol_paper1/finalize_matched_view_landscape_reports_v1.sh" &
active_pids=("$!")
wait "${active_pids[0]}"
active_pids=()
echo "pipeline_complete $(date -u +%FT%TZ)"
