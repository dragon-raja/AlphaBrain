#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TRAIN_WORKTREE=${FRESH_V2_TRAIN_WORKTREE:-/tmp/alphabrain-baseline-ce552}
TRAIN_SHA=ce552faf64f1cea994d10899ef500380ab02f2b5
TRAIN_PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
RUN_ROOT=${FRESH_BASELINE_REPAIR_ROOT:-/share/longjunyu/fresh-vla/runs/baseline-repair-v1}
EVAL_ROOT="$RUN_ROOT/eval_views"
STEPS=10353
TAG=val_gate_v2
AMENDMENT=docs/embodied_research_reset/baseline_validity_repair_v2_amendment.md
SEED42_SESSION=${FRESH_V2_SEED42_SESSION:-fresh-baseline-repair-v2-s42}
DRY_RUN=${FRESH_V2_DRY_RUN:-0}

run_dir() {
  local seed=$1
  if [ "$seed" = 41 ]; then
    printf '%s\n' "$RUN_ROOT/baseline_repair_full_h_ddp8_seed41_steps13804_formal-v2"
  else
    printf '%s\n' "$RUN_ROOT/baseline_repair_full_h_ddp8_seed${seed}_steps${STEPS}_formal-budget-v2"
  fi
}

checkpoint_dir() {
  local seed=$1
  printf '%s/checkpoints/steps_%s\n' "$(run_dir "$seed")" "$STEPS"
}

validate_training_run() {
  local seed=$1
  local directory
  directory=$(run_dir "$seed")
  "$TRAIN_PYTHON" - "$directory" "$seed" "$STEPS" "$TRAIN_SHA" <<'PY'
import json
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
seed = int(sys.argv[2])
steps = int(sys.argv[3])
expected_sha = sys.argv[4]
identity = json.loads((directory / "run_identity.json").read_text())
resume = json.loads((directory / "checkpoints" / f"steps_{steps}" / "resume_meta.json").read_text())
if identity.get("seed") != seed or identity.get("git_sha") != expected_sha:
    raise SystemExit(f"training identity mismatch: {directory}")
if identity.get("git_dirty_at_launch") is not False or identity.get("test_split_opened") is not False:
    raise SystemExit(f"invalid training provenance: {directory}")
if identity.get("effective_batch_size") != 8 or identity.get("optimizer_steps", 0) < steps:
    raise SystemExit(f"training budget mismatch: {directory}")
if resume.get("completed_steps") != steps or resume.get("effective_batch_size") != 8:
    raise SystemExit(f"checkpoint metadata mismatch: {directory}")
model = directory / "checkpoints" / f"steps_{steps}" / "model.safetensors"
if model.stat().st_size != 17_591_583_484:
    raise SystemExit(f"checkpoint size mismatch: {model}")
if seed != 41:
    complete = json.loads((directory / "run_complete.json").read_text())
    if complete.get("completed") is not True or complete.get("optimizer_steps") != steps:
        raise SystemExit(f"run did not complete: {directory}")
PY
}

wait_for_seed42() {
  local complete
  complete="$(run_dir 42)/run_complete.json"
  while [ ! -f "$complete" ]; do
    if ! tmux has-session -t "$SEED42_SESSION" 2>/dev/null; then
      echo "seed-42 training exited without run_complete.json" >&2
      exit 1
    fi
    sleep 30
  done
  while tmux has-session -t "$SEED42_SESSION" 2>/dev/null; do
    sleep 2
  done
  validate_training_run 42
}

train_seed43() {
  local directory
  directory=$(run_dir 43)
  if [ -f "$directory/run_complete.json" ]; then
    validate_training_run 43
    return
  fi
  if [ -e "$directory" ]; then
    echo "refusing incomplete pre-existing seed-43 run: $directory" >&2
    exit 1
  fi
  env \
    FRESH_TRAIN_PYTHON="$TRAIN_PYTHON" \
    FRESH_BASELINE_REPAIR_PORT=30743 \
    bash "$TRAIN_WORKTREE/scripts/fresh_vla/run_libero_baseline_repair_train.sh" \
      43 "$STEPS" "$STEPS" formal-budget-v2
  validate_training_run 43
}

prepare_eval_view() {
  local seed=$1
  local directory checkpoint view identity
  directory=$(run_dir "$seed")
  checkpoint=$(checkpoint_dir "$seed")
  identity="$directory/run_identity.json"
  view="$EVAL_ROOT/fresh_closed_loop_repair_step${STEPS}_seed${seed}"
  mkdir -p "$view"
  if [ ! -e "$view/final_model" ]; then
    ln -s "$checkpoint" "$view/final_model"
  fi
  if [ ! -e "$view/training_run_identity.json" ]; then
    ln -s "$identity" "$view/training_run_identity.json"
  fi
  if [ "$(readlink -f "$view/final_model")" != "$(readlink -f "$checkpoint")" ]; then
    echo "eval checkpoint link mismatch: $view" >&2
    exit 1
  fi
  if [ "$(readlink -f "$view/training_run_identity.json")" != "$(readlink -f "$identity")" ]; then
    echo "eval identity link mismatch: $view" >&2
    exit 1
  fi
}

run_eval() {
  local seed=$1
  local gpu=$2
  local run_id="fresh_closed_loop_repair_step${STEPS}_seed${seed}"
  local view="$EVAL_ROOT/$run_id"
  env \
    FRESH_CLOSED_LOOP_OUTPUT_ROOT="$EVAL_ROOT" \
    FRESH_RUN_ID="$run_id" \
    FRESH_EVAL_SPLIT=val \
    FRESH_EVAL_ONLY=end_to_end \
    FRESH_EVAL_OUTPUT_TAG="$TAG" \
    FRESH_SAVE_EVAL_VIDEOS=1 \
    FRESH_EVAL_VIDEO_GROUPS=13 \
    FRESH_TRAIN_PYTHON="$TRAIN_PYTHON" \
    bash "$REPO_ROOT/scripts/fresh_vla/run_libero_closed_loop_eval.sh" \
      repair_budget_v2 "$seed" "$gpu" >"$view/eval_driver_${TAG}.log" 2>&1
}

evaluate_all_seeds() {
  local pids=()
  local status=0
  local seed
  for seed in 41 42 43; do
    prepare_eval_view "$seed"
    run_eval "$seed" "$((seed - 41))" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [ "$status" != 0 ]; then
    echo "one or more validation evaluations failed" >&2
    exit 1
  fi
}

audit_and_gate() {
  local audit="$RUN_ROOT/baseline_repair_v2_three_seed_video_artifact_audit.json"
  local gate="$RUN_ROOT/baseline_repair_v2_three_seed_gate.json"
  local report="$RUN_ROOT/baseline_repair_v2_three_seed_gate.md"
  if [ ! -f "$audit" ]; then
    "$TRAIN_PYTHON" "$REPO_ROOT/scripts/fresh_vla/audit_libero_closed_loop_videos.py" \
      "$EVAL_ROOT/fresh_closed_loop_repair_step${STEPS}_seed41" \
      "$EVAL_ROOT/fresh_closed_loop_repair_step${STEPS}_seed42" \
      "$EVAL_ROOT/fresh_closed_loop_repair_step${STEPS}_seed43" \
      --tag "$TAG" \
      --video-groups 13 \
      --output "$audit"
  fi
  "$TRAIN_PYTHON" - "$audit" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
if payload.get("passed") is not True or payload.get("actual_video_count") != 117:
    raise SystemExit("formal video audit did not pass")
PY
  if [ ! -f "$gate" ] && [ ! -f "$report" ]; then
    "$TRAIN_PYTHON" "$REPO_ROOT/scripts/fresh_vla/summarize_libero_baseline_repair_gate.py" \
      --eval-root "$EVAL_ROOT" \
      --steps "$STEPS" \
      --seeds 41 42 43 \
      --tag "$TAG" \
      --output "$gate" \
      --report "$report"
  fi
  "$TRAIN_PYTHON" - "$gate" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
if payload.get("status") != "complete" or payload.get("test_split_opened") is not False:
    raise SystemExit("formal baseline v2 gate is invalid")
print(json.dumps({"decision": payload["decision"], "gate": payload["gate"]}, sort_keys=True))
PY
}

if [ ! -f "$REPO_ROOT/$AMENDMENT" ]; then
  echo "missing frozen amendment: $REPO_ROOT/$AMENDMENT" >&2
  exit 1
fi
if [ "$(git -C "$TRAIN_WORKTREE" rev-parse HEAD)" != "$TRAIN_SHA" ]; then
  echo "training worktree SHA mismatch" >&2
  exit 1
fi
if [ -n "$(git -C "$TRAIN_WORKTREE" status --porcelain)" ]; then
  echo "training worktree is dirty" >&2
  exit 1
fi
validate_training_run 41

if [ "$DRY_RUN" = 1 ]; then
  printf '{"status":"dry_run_passed","steps":%s,"test_split_opened":false,"training_sha":"%s"}\n' \
    "$STEPS" "$TRAIN_SHA"
  exit 0
fi

wait_for_seed42
train_seed43
evaluate_all_seeds
audit_and_gate
