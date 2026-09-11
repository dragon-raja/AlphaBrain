#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${DSOL_ORACLE_V2_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2}
CHECKPOINT_41=${CHECKPOINT_41:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_m-b-formal-v1_seed41_g2_gb32_steps2000/final_model}
CHECKPOINT_42=${CHECKPOINT_42:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_m-b-formal-v1_seed42_g2_gb32_steps2000/final_model}
CHECKPOINT_43=${CHECKPOINT_43:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_m-b-formal-v1_seed43_g2_gb32_steps2000/final_model}
POPULATION=$ROOT/population/population-v2.json
CATALOG=$REPO_ROOT/configs/dsol_paper1/libero_view_catalog_v2_m1.json
ORACLE_CONFIG=$REPO_ROOT/configs/dsol_paper1/statewise_view_oracle_v2.json
LEARNER_CONFIG=$REPO_ROOT/configs/dsol_paper1/statewise_view_selector_learner_v1.json
LEGACY_ROOT=/share/longjunyu/alphabrain/experiments/dsol-view-value-expectation-v1
GPU_COUNT=${GPU_COUNT:-8}
EVAL_WORKER_COUNT=${EVAL_WORKER_COUNT:-32}
POLICY_SERVER_COPIES_PER_GPU=${POLICY_SERVER_COPIES_PER_GPU:-2}
POLICY_CPU_THREADS=${POLICY_CPU_THREADS:-2}
SIM_CPU_THREADS=${SIM_CPU_THREADS:-1}
GPU_DEVICES=${DSOL_GPU_DEVICES:-0,1,2,3,4,5,6,7}

mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/v2-tail-controller.log") 2>&1
exec 9>"$ROOT/logs/v2-tail-controller.lock"
flock -n 9 || { echo "another oracle-v2 tail already owns $ROOT" >&2; exit 3; }

count_rows() {
  local directory=$1
  if [[ ! -d "$directory" ]]; then printf '0\n'; return; fi
  find "$directory" -maxdepth 1 -name 'episodes-shard-*.jsonl' -type f \
    -exec awk 'NF{n++} END{print n+0}' {} \; 2>/dev/null | \
    awk '{sum += $1} END {print sum + 0}'
}

run_matrix() {
  local label=$1
  local protocol=$2
  local bank=$3
  local output=$4
  local checkpoint=$5
  local port=$6
  local expected actual
  expected=$(PROTOCOL="$protocol" python - <<'PY'
import json, os
print(json.load(open(os.environ["PROTOCOL"], encoding="utf-8"))["episode_count"])
PY
)
  mkdir -p "$output"
  actual=$(count_rows "$output")
  if [[ "$actual" != "$expected" ]]; then
    echo "matrix_start label=$label complete=$actual expected=$expected time=$(date -u +%FT%TZ)"
    CHECKPOINT="$checkpoint" OUTPUT_DIR="$output" PROTOCOL="$protocol" \
    NOISE_BANK_MANIFEST="$bank" REQUIRE_EXPLICIT_NOISE=1 \
    GPU_COUNT="$GPU_COUNT" EVAL_WORKER_COUNT="$EVAL_WORKER_COUNT" \
    POLICY_SERVER_COPIES_PER_GPU="$POLICY_SERVER_COPIES_PER_GPU" \
    POLICY_CPU_THREADS="$POLICY_CPU_THREADS" SIM_CPU_THREADS="$SIM_CPU_THREADS" \
    DSOL_GPU_DEVICES="$GPU_DEVICES" BASE_PORT="$port" \
    REPLAN_STEPS=5 WAIT_STEPS=0 VIDEO_EPISODES=0 RUN_ANALYSIS=0 \
    KEEPALIVE_MODE=managed \
      "$REPO_ROOT/scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh"
  else
    echo "matrix_skip_complete label=$label episodes=$actual"
  fi
  /alphabrain/.venv/bin/python \
    "$REPO_ROOT/scripts/dsol_paper1/audit_statewise_view_oracle_v2_run.py" \
    --protocol "$protocol" --noise-bank-manifest "$bank" \
    --run-manifest "$output/run_manifest.json" \
    --episode-ledgers "$output/episodes-shard-*.jsonl" \
    --output "$output/audit.json"
  echo "matrix_complete label=$label episodes=$expected time=$(date -u +%FT%TZ)"
}

while :; do
  complete=0
  for wave_index in $(seq 0 7); do
    wave=$(printf '%02d' "$wave_index")
    audit=$ROOT/development/O/wave-$wave/audit.json
    if [[ -f "$audit" ]] && AUDIT="$audit" python - <<'PY'
import json, os
raise SystemExit(0 if json.load(open(os.environ["AUDIT"], encoding="utf-8")).get("status") == "PASS_COMPLETE" else 1)
PY
    then
      complete=$((complete + 1))
    fi
  done
  if [[ "$complete" == 8 ]]; then break; fi
  if ! tmux has-session -t dsol-oracle-v2-dev-O 2>/dev/null; then
    echo "development O owner exited before all audited waves completed: $complete/8" >&2
    exit 4
  fi
  echo "waiting_for_development_O audited_waves=$complete/8 time=$(date -u +%FT%TZ)"
  sleep 60
done

P_DEV_PROTOCOL=$ROOT/protocols/P-development.json
P_DEV_FREEZE=$ROOT/protocols/P-development-freeze.json
/alphabrain/.venv/bin/python "$REPO_ROOT/scripts/dsol_paper1/build_statewise_view_oracle_v2_stage.py" \
  --stage P --role development --population "$POPULATION" --catalog "$CATALOG" \
  --protocol-config "$ORACLE_CONFIG" \
  --input-ledgers "$ROOT/development/O/wave-*/episodes-shard-*.jsonl" \
  --input-audits "$ROOT/development/O/wave-*/audit.json" \
  --output-protocol "$P_DEV_PROTOCOL" --output-freeze-receipt "$P_DEV_FREEZE"
run_matrix development-P "$P_DEV_PROTOCOL" "$ROOT/noise-banks/bank_P.manifest.json" \
  "$ROOT/development/P" "$CHECKPOINT_41" 23500

Q_DEV_PROTOCOL=$ROOT/protocols/Q-development.json
Q_DEV_FREEZE=$ROOT/protocols/Q-development-freeze.json
/alphabrain/.venv/bin/python "$REPO_ROOT/scripts/dsol_paper1/build_statewise_view_oracle_v2_stage.py" \
  --stage Q --role development --population "$POPULATION" --catalog "$CATALOG" \
  --protocol-config "$ORACLE_CONFIG" \
  --input-ledgers "$ROOT/development/P/episodes-shard-*.jsonl" \
  --input-audits "$ROOT/development/P/audit.json" \
  --output-protocol "$Q_DEV_PROTOCOL" --output-freeze-receipt "$Q_DEV_FREEZE"
run_matrix development-Q "$Q_DEV_PROTOCOL" "$ROOT/noise-banks/bank_Q.manifest.json" \
  "$ROOT/development/Q" "$CHECKPOINT_41" 23600

SELECTOR_DIR=$ROOT/selector
/alphabrain/.venv/bin/python "$REPO_ROOT/scripts/dsol_paper1/train_statewise_view_selector_v1.py" \
  --population "$POPULATION" --learner-config "$LEARNER_CONFIG" \
  --legacy-root "$LEGACY_ROOT" \
  --input-ledgers "$ROOT/development/O/wave-*/episodes-shard-*.jsonl" \
  --input-audits "$ROOT/development/O/wave-*/audit.json" \
  --output-dir "$SELECTOR_DIR" --device cuda:0 --batch-size 128
SELECTOR_FREEZE=$SELECTOR_DIR/selector-freeze-receipt.json
[[ -f "$SELECTOR_FREEZE" ]] || { echo "selector freeze receipt was not produced" >&2; exit 2; }

for wave_index in $(seq 0 7); do
  wave=$(printf '%02d' "$wave_index")
  run_matrix test-O-wave-$wave \
    "$ROOT/protocols/dense-O-test-wave-$wave.json" \
    "$ROOT/noise-banks/bank_O.manifest.json" \
    "$ROOT/test/O/wave-$wave" "$CHECKPOINT_41" "$((23700 + wave_index * 20))"
done

P_TEST_PROTOCOL=$ROOT/protocols/P-test.json
P_TEST_FREEZE=$ROOT/protocols/P-test-freeze.json
/alphabrain/.venv/bin/python "$REPO_ROOT/scripts/dsol_paper1/build_statewise_view_oracle_v2_stage.py" \
  --stage P --role test --population "$POPULATION" --catalog "$CATALOG" \
  --protocol-config "$ORACLE_CONFIG" --selector-freeze-receipt "$SELECTOR_FREEZE" \
  --input-ledgers "$ROOT/test/O/wave-*/episodes-shard-*.jsonl" \
  --input-audits "$ROOT/test/O/wave-*/audit.json" \
  --output-protocol "$P_TEST_PROTOCOL" --output-freeze-receipt "$P_TEST_FREEZE"
run_matrix test-P "$P_TEST_PROTOCOL" "$ROOT/noise-banks/bank_P.manifest.json" \
  "$ROOT/test/P" "$CHECKPOINT_41" 23900

Q_TEST_PROTOCOL=$ROOT/protocols/Q-test.json
Q_TEST_FREEZE=$ROOT/protocols/Q-test-freeze.json
/alphabrain/.venv/bin/python "$REPO_ROOT/scripts/dsol_paper1/build_statewise_view_oracle_v2_stage.py" \
  --stage Q --role test --population "$POPULATION" --catalog "$CATALOG" \
  --protocol-config "$ORACLE_CONFIG" --selector-freeze-receipt "$SELECTOR_FREEZE" \
  --input-ledgers "$ROOT/test/P/episodes-shard-*.jsonl" \
  --input-audits "$ROOT/test/P/audit.json" \
  --output-protocol "$Q_TEST_PROTOCOL" --output-freeze-receipt "$Q_TEST_FREEZE"
run_matrix test-Q-seed41 "$Q_TEST_PROTOCOL" "$ROOT/noise-banks/bank_Q.manifest.json" \
  "$ROOT/test/Q-seed41" "$CHECKPOINT_41" 24000
run_matrix test-Q-seed42 "$Q_TEST_PROTOCOL" "$ROOT/noise-banks/bank_Q.manifest.json" \
  "$ROOT/transfer/Q-seed42" "$CHECKPOINT_42" 24100
run_matrix test-Q-seed43 "$Q_TEST_PROTOCOL" "$ROOT/noise-banks/bank_Q.manifest.json" \
  "$ROOT/transfer/Q-seed43" "$CHECKPOINT_43" 24200

/alphabrain/.venv/bin/python "$REPO_ROOT/scripts/dsol_paper1/analyze_statewise_view_oracle_v2.py" \
  --protocol "$Q_TEST_PROTOCOL" --population "$POPULATION" \
  --seed41-ledgers "$ROOT/test/Q-seed41/episodes-shard-*.jsonl" \
  --seed42-ledgers "$ROOT/transfer/Q-seed42/episodes-shard-*.jsonl" \
  --seed43-ledgers "$ROOT/transfer/Q-seed43/episodes-shard-*.jsonl" \
  --audits "$ROOT/test/Q-seed41/audit.json" "$ROOT/transfer/Q-seed42/audit.json" "$ROOT/transfer/Q-seed43/audit.json" \
  --output-dir "$ROOT/final-analysis"

reserve_status=$(python - <<PY
import json
print(json.load(open("$ROOT/final-analysis/reserve-decision.json", encoding="utf-8"))["status"])
PY
)
if [[ "$reserve_status" == "ACTIVATE_R" ]]; then
  R_TEST_PROTOCOL=$ROOT/protocols/R-test.json
  /alphabrain/.venv/bin/python \
    "$REPO_ROOT/scripts/dsol_paper1/clone_statewise_view_oracle_v2_reserve.py" \
    --input-protocol "$Q_TEST_PROTOCOL" --output-protocol "$R_TEST_PROTOCOL"
  run_matrix test-R-seed41 "$R_TEST_PROTOCOL" "$ROOT/noise-banks/bank_R.manifest.json" \
    "$ROOT/test/R-seed41" "$CHECKPOINT_41" 24300
  /alphabrain/.venv/bin/python "$REPO_ROOT/scripts/dsol_paper1/analyze_statewise_view_oracle_v2.py" \
    --protocol "$Q_TEST_PROTOCOL" --population "$POPULATION" \
    --seed41-ledgers "$ROOT/test/Q-seed41/episodes-shard-*.jsonl" \
    --seed41-reserve-ledgers "$ROOT/test/R-seed41/episodes-shard-*.jsonl" \
    --seed42-ledgers "$ROOT/transfer/Q-seed42/episodes-shard-*.jsonl" \
    --seed43-ledgers "$ROOT/transfer/Q-seed43/episodes-shard-*.jsonl" \
    --audits "$ROOT/test/Q-seed41/audit.json" "$ROOT/test/R-seed41/audit.json" \
      "$ROOT/transfer/Q-seed42/audit.json" "$ROOT/transfer/Q-seed43/audit.json" \
    --output-dir "$ROOT/final-analysis"
fi

echo "statewise_view_oracle_v2_complete time=$(date -u +%FT%TZ)"
