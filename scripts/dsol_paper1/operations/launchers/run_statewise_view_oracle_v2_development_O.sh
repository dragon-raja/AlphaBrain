#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
ROOT=${DSOL_ORACLE_V2_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2}
CHECKPOINT=${CHECKPOINT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_m-b-formal-v1_seed41_g2_gb32_steps2000/final_model}
NOISE_MANIFEST=$ROOT/noise-banks/bank_O.manifest.json
GPU_COUNT=${GPU_COUNT:-8}
EVAL_WORKER_COUNT=${EVAL_WORKER_COUNT:-32}
POLICY_SERVER_COPIES_PER_GPU=${POLICY_SERVER_COPIES_PER_GPU:-2}
POLICY_CPU_THREADS=${POLICY_CPU_THREADS:-2}
SIM_CPU_THREADS=${SIM_CPU_THREADS:-1}
GPU_DEVICES=${DSOL_GPU_DEVICES:-0,1,2,3,4,5,6,7}
BASE_PORT=${BASE_PORT:-23200}

mkdir -p "$ROOT/logs" "$ROOT/development/O"
exec > >(tee -a "$ROOT/logs/development-O-controller.log") 2>&1
exec 9>"$ROOT/logs/development-O-controller.lock"
flock -n 9 || {
  echo "another development-O controller already owns $ROOT" >&2
  exit 3
}

for required in \
  "$ROOT/preparation/receipt.json" \
  "$ROOT/smoke/S/audit.json" \
  "$NOISE_MANIFEST" \
  "$CHECKPOINT/model.safetensors"; do
  [[ -f "$required" ]] || { echo "missing required frozen input: $required" >&2; exit 2; }
done

PREPARATION="$ROOT/preparation/receipt.json" SMOKE="$ROOT/smoke/S/audit.json" python - <<'PY'
import json, os
preparation = json.load(open(os.environ["PREPARATION"], encoding="utf-8"))
smoke = json.load(open(os.environ["SMOKE"], encoding="utf-8"))
if preparation.get("status") != "PASS_READY_FOR_SMOKE":
    raise SystemExit("preparation receipt is not PASS_READY_FOR_SMOKE")
if smoke.get("status") != "PASS_COMPLETE":
    raise SystemExit("smoke audit is not PASS_COMPLETE")
if not smoke.get("every_policy_call_matches_frozen_noise_bank"):
    raise SystemExit("smoke did not prove exact explicit-noise injection")
PY

count_rows() {
  local directory=$1
  if [[ ! -d "$directory" ]]; then
    printf '0\n'
    return
  fi
  find "$directory" -maxdepth 1 -name 'episodes-shard-*.jsonl' -type f \
    -exec awk 'NF{n++} END{print n+0}' {} \; 2>/dev/null | \
    awk '{sum += $1} END {print sum + 0}'
}

for wave_index in $(seq 0 7); do
  wave=$(printf '%02d' "$wave_index")
  protocol=$ROOT/protocols/dense-O-development-wave-$wave.json
  output=$ROOT/development/O/wave-$wave
  audit=$output/audit.json
  [[ -f "$protocol" ]] || { echo "missing frozen wave protocol: $protocol" >&2; exit 2; }
  expected=$(PROTOCOL="$protocol" python - <<'PY'
import json, os
print(json.load(open(os.environ["PROTOCOL"], encoding="utf-8"))["episode_count"])
PY
)
  mkdir -p "$output"
  actual=$(count_rows "$output")
  if [[ "$actual" != "$expected" ]]; then
    echo "development_O_wave_start wave=$wave complete=$actual expected=$expected time=$(date -u +%FT%TZ)"
    CHECKPOINT="$CHECKPOINT" \
    OUTPUT_DIR="$output" \
    PROTOCOL="$protocol" \
    NOISE_BANK_MANIFEST="$NOISE_MANIFEST" \
    REQUIRE_EXPLICIT_NOISE=1 \
    GPU_COUNT="$GPU_COUNT" \
    EVAL_WORKER_COUNT="$EVAL_WORKER_COUNT" \
    POLICY_SERVER_COPIES_PER_GPU="$POLICY_SERVER_COPIES_PER_GPU" \
    POLICY_CPU_THREADS="$POLICY_CPU_THREADS" \
    SIM_CPU_THREADS="$SIM_CPU_THREADS" \
    DSOL_GPU_DEVICES="$GPU_DEVICES" \
    BASE_PORT="$((BASE_PORT + wave_index * 20))" \
    REPLAN_STEPS=5 \
    WAIT_STEPS=0 \
    VIDEO_EPISODES=0 \
    RUN_ANALYSIS=0 \
    KEEPALIVE_MODE=managed \
      "$REPO_ROOT/scripts/dsol_paper1/operations/launchers/run_dsol_libero_hdf5_closed_loop_eval.sh"
  else
    echo "development_O_wave_skip_complete wave=$wave episodes=$actual"
  fi
  /alphabrain/.venv/bin/python \
    "$REPO_ROOT/scripts/dsol_paper1/diagnostics/audit_statewise_view_oracle_v2_run.py" \
    --protocol "$protocol" \
    --noise-bank-manifest "$NOISE_MANIFEST" \
    --run-manifest "$output/run_manifest.json" \
    --episode-ledgers "$output/episodes-shard-*.jsonl" \
    --output "$audit"
  echo "development_O_wave_complete wave=$wave episodes=$expected time=$(date -u +%FT%TZ)"
done

echo "development_O_all_complete time=$(date -u +%FT%TZ)"
