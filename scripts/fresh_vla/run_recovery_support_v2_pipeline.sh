#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
RUN_ROOT=${FRESH_RECOVERY_SUPPORT_ROOT:-/share/longjunyu/fresh-vla/runs/recovery-support-repaired-v2-step10353}
BASELINE_ROOT=${FRESH_BASELINE_VIEW_ROOT:-/share/longjunyu/fresh-vla/runs/baseline-repair-v1/eval_views}
BASELINE_GATE=${FRESH_BASELINE_GATE:-/share/longjunyu/fresh-vla/runs/baseline-repair-v1/baseline_repair_v2_three_seed_gate.json}
EPISODE_ROOT=${FRESH_EPISODE_ROOT:-/share/longjunyu/fresh-vla/libero-full-episode-v2-128}
WINDOW_ROOT=${FRESH_WINDOW_ROOT:-/share/longjunyu/fresh-vla/libero-full-episode-windows-v2-128}
BASELINE_STEPS=10353
SUPPORT_STEPS=6902
EVAL_TAG=val_support_v2
BASELINE_TAG=val_gate_v2
DRY_RUN=${FRESH_RECOVERY_V2_DRY_RUN:-0}
SEEDS=(41 42 43)
ARMS=(base_continuation clean_recovery_replay policy_state_recovery)

validate_gate() {
  "$PYTHON" - "$BASELINE_GATE" <<'PY'
import json, pathlib, sys
p = json.loads(pathlib.Path(sys.argv[1]).read_text())
if p.get("decision") != "BASELINE_VALID_PROCEED_TO_RECOVERY_CONTROLS":
    raise SystemExit(f"baseline gate did not pass: {p.get('decision')}")
if p.get("uniform_training_budget_steps") != 10353 or p.get("test_split_opened") is not False:
    raise SystemExit("baseline gate identity mismatch or test was opened")
PY
}

run_parallel_three() {
  local label=$1
  shift
  local pids=() status=0 index=0
  for seed in "${SEEDS[@]}"; do
    "$@" "$seed" "$index" &
    pids+=("$!")
    index=$((index + 1))
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
  done
  if [ "$status" != 0 ]; then
    echo "$label failed" >&2
    exit 1
  fi
}

collect_one() {
  local seed=$1 gpu=$2 output="$RUN_ROOT/corrections/policy_state_repaired_seed${seed}"
  if [ -f "$output/quality_report.json" ]; then
    "$PYTHON" - "$output/quality_report.json" <<'PY'
import json, pathlib, sys
if json.loads(pathlib.Path(sys.argv[1]).read_text()).get("passed") is not True:
    raise SystemExit(f"existing correction output failed quality gate: {sys.argv[1]}")
PY
    return
  fi
  if [ -e "$output" ]; then
    echo "refusing incomplete correction output: $output" >&2
    exit 1
  fi
  env FRESH_RECOVERY_SUPPORT_ROOT="$RUN_ROOT" \
    FRESH_BASELINE_GATE="$BASELINE_GATE" FRESH_BASELINE_STEPS="$BASELINE_STEPS" \
    bash "$REPO_ROOT/scripts/fresh_vla/run_policy_state_recovery_collection.sh" "$seed" "$gpu"
}

build_one() {
  local seed=$1 output="$RUN_ROOT/data/seed${seed}"
  if [ -f "$output/quality_report.json" ]; then
    "$PYTHON" - "$output/quality_report.json" <<'PY'
import json, pathlib, sys
if json.loads(pathlib.Path(sys.argv[1]).read_text()).get("passed") is not True:
    raise SystemExit(f"existing matched view failed quality gate: {sys.argv[1]}")
PY
    return
  fi
  if [ -e "$output" ]; then
    echo "refusing incomplete matched view: $output" >&2
    exit 1
  fi
  "$PYTHON" "$REPO_ROOT/scripts/fresh_vla/build_recovery_support_views.py" \
    --episode-root "$EPISODE_ROOT" --window-root "$WINDOW_ROOT" \
    --correction-root "$RUN_ROOT/corrections/policy_state_repaired_seed${seed}" \
    --output-root "$output" --seed "$seed" --steps "$SUPPORT_STEPS"
}

train_one() {
  local arm=$1 seed=$2 gpu=$3
  local run="$RUN_ROOT/recovery_support_${arm}_seed${seed}_steps${SUPPORT_STEPS}"
  if [ -f "$run/final_model/model.safetensors" ]; then return; fi
  if [ -e "$run" ]; then
    echo "refusing incomplete training run: $run" >&2
    exit 1
  fi
  env FRESH_RECOVERY_SUPPORT_ROOT="$RUN_ROOT" \
    FRESH_BASELINE_GATE="$BASELINE_GATE" FRESH_BASELINE_STEPS="$BASELINE_STEPS" \
    bash "$REPO_ROOT/scripts/fresh_vla/run_recovery_support_train.sh" \
      "$arm" "$seed" "$gpu" "$SUPPORT_STEPS"
}

train_arm() {
  local arm=$1 pids=() status=0 index=0
  for seed in "${SEEDS[@]}"; do
    train_one "$arm" "$seed" "$index" &
    pids+=("$!")
    index=$((index + 1))
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
  done
  if [ "$status" != 0 ]; then
    echo "training arm $arm failed" >&2
    exit 1
  fi
}

eval_baseline_one() {
  local seed=$1 gpu=$2 run_id="fresh_closed_loop_repair_step${BASELINE_STEPS}_seed${seed}"
  env FRESH_CLOSED_LOOP_OUTPUT_ROOT="$BASELINE_ROOT" FRESH_RUN_ID="$run_id" \
    FRESH_EVAL_SPLIT=val FRESH_EVAL_ONLY=all FRESH_EVAL_OUTPUT_TAG="$BASELINE_TAG" \
    FRESH_SAVE_EVAL_VIDEOS=1 FRESH_EVAL_VIDEO_GROUPS=13 \
    bash "$REPO_ROOT/scripts/fresh_vla/run_libero_closed_loop_eval.sh" baseline "$seed" "$gpu"
}

eval_candidate_one() {
  local arm=$1 seed=$2 gpu=$3 run_id="recovery_support_${arm}_seed${seed}_steps${SUPPORT_STEPS}"
  env FRESH_CLOSED_LOOP_OUTPUT_ROOT="$RUN_ROOT" FRESH_RUN_ID="$run_id" \
    FRESH_EVAL_SPLIT=val FRESH_EVAL_ONLY=all FRESH_EVAL_OUTPUT_TAG="$EVAL_TAG" \
    FRESH_SAVE_EVAL_VIDEOS=1 FRESH_EVAL_VIDEO_GROUPS=13 \
    bash "$REPO_ROOT/scripts/fresh_vla/run_libero_closed_loop_eval.sh" "$arm" "$seed" "$gpu"
}

eval_arm() {
  local arm=$1 pids=() status=0 index=0
  for seed in "${SEEDS[@]}"; do
    eval_candidate_one "$arm" "$seed" "$index" &
    pids+=("$!")
    index=$((index + 1))
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
  done
  if [ "$status" != 0 ]; then
    echo "evaluation arm $arm failed" >&2
    exit 1
  fi
}

audit_candidates() {
  local runs=() arm seed
  for arm in "${ARMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      runs+=("$RUN_ROOT/recovery_support_${arm}_seed${seed}_steps${SUPPORT_STEPS}")
    done
  done
  "$PYTHON" "$REPO_ROOT/scripts/fresh_vla/audit_libero_closed_loop_videos.py" \
    "${runs[@]}" --tag "$EVAL_TAG" --video-groups 13 \
    --output "$RUN_ROOT/recovery_support_v2_video_artifact_audit.json"
}

validate_gate
if [ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]; then
  echo "formal recovery pipeline requires a clean Git worktree" >&2
  exit 1
fi
if [ "$DRY_RUN" = 1 ]; then
  printf '{"status":"dry_run_passed","baseline_steps":%s,"support_steps":%s,"split":"val","test_split_opened":false}\n' \
    "$BASELINE_STEPS" "$SUPPORT_STEPS"
  exit 0
fi

mkdir -p "$RUN_ROOT"
run_parallel_three collection collect_one
for seed in "${SEEDS[@]}"; do build_one "$seed"; done
for arm in "${ARMS[@]}"; do train_arm "$arm"; done
run_parallel_three baseline_evaluation eval_baseline_one
for arm in "${ARMS[@]}"; do eval_arm "$arm"; done
audit_candidates
"$PYTHON" "$REPO_ROOT/scripts/fresh_vla/summarize_recovery_support_closed_loop.py" \
  --output-root "$RUN_ROOT" --baseline-root "$BASELINE_ROOT" \
  --baseline-steps "$BASELINE_STEPS" --steps "$SUPPORT_STEPS" \
  --tag "$EVAL_TAG" --baseline-tag "$BASELINE_TAG" --split val
