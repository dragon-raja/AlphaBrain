#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG=${FRESH_RECOVERY_SUPPORT_CONFIG:-$REPO_ROOT/configs/experiments/fresh_vla_libero_closed_loop.yaml}
PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
BASELINE_VIEW_ROOT=${FRESH_BASELINE_VIEW_ROOT:-/share/longjunyu/fresh-vla/runs/baseline-repair-v1/eval_views}
BASELINE_GATE=${FRESH_BASELINE_GATE:-/share/longjunyu/fresh-vla/runs/baseline-repair-v1/baseline_repair_v2_three_seed_gate.json}
BASELINE_STEPS=${FRESH_BASELINE_STEPS:-10353}
OUTPUT_ROOT=${FRESH_RECOVERY_SUPPORT_ROOT:-/share/longjunyu/fresh-vla/runs/recovery-support-repaired-v2-step10353}
PRETRAINED_MODELS_DIR=${PRETRAINED_MODELS_DIR:-/share/longjunyu/alphabrain/pretrained_models}

ARM=${1:?usage: run_recovery_support_train.sh ARM SEED GPU_ID STEPS}
SEED=${2:?usage: run_recovery_support_train.sh ARM SEED GPU_ID STEPS}
GPU_ID=${3:?usage: run_recovery_support_train.sh ARM SEED GPU_ID STEPS}
STEPS=${4:?usage: run_recovery_support_train.sh ARM SEED GPU_ID STEPS}

case "$ARM" in
  base_continuation|clean_recovery_replay|policy_state_recovery) ;;
  *) echo "ARM must be base_continuation, clean_recovery_replay, or policy_state_recovery" >&2; exit 2 ;;
esac
case "$SEED" in
  41) EXPECTED_SHA256=732da869fe5aab23ae83f6b517bb33a83bb0b5e7cea9c2535edc9388f07d61c4 ;;
  42) EXPECTED_SHA256=73d23cc8659ab7510eecdd013b1ffdc48c2ea97304ec14b3cf886906fc4da90a ;;
  43) EXPECTED_SHA256=cfd9547bde803ca83430bae37675aeb61e534f0d490f2b1d233ab3289baec4c4 ;;
  *) echo "seed must be one of 41, 42, or 43" >&2; exit 2 ;;
esac
if [[ ! "$GPU_ID" =~ ^[0-7]$ ]] || [[ ! "$STEPS" =~ ^[1-9][0-9]*$ ]] || (( STEPS % 2 != 0 )); then
  echo "GPU_ID must be in [0,7] and STEPS must be positive and even" >&2
  exit 2
fi

CHECKPOINT="$BASELINE_VIEW_ROOT/fresh_closed_loop_repair_step${BASELINE_STEPS}_seed${SEED}/final_model"
RUN_ID="recovery_support_${ARM}_seed${SEED}_steps${STEPS}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
MODE=fresh_recovery_support_full_h
DATASET="$OUTPUT_ROOT/data/seed${SEED}/$ARM"
export FRESH_RECOVERY_SUPPORT_DATA_ROOT="$DATASET"
if [ "${FRESH_ALLOW_UNGATED_SUPPORT_SMOKE:-0}" != 1 ]; then
  "$PYTHON" - "$BASELINE_GATE" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing repaired baseline gate: {path}")
payload = json.loads(path.read_text())
if payload.get("decision") != "BASELINE_VALID_PROCEED_TO_RECOVERY_CONTROLS":
    raise SystemExit(f"repaired baseline gate did not pass: {payload.get('decision')}")
if payload.get("test_split_opened", False):
    raise SystemExit("repaired baseline gate unexpectedly opened test")
PY
fi
if [ ! -f "$DATASET/quality_report.json" ]; then
  echo "missing recovery-support data view: $DATASET/quality_report.json" >&2
  exit 1
fi
"$PYTHON" - "$DATASET" "$STEPS" "$ARM" "$SEED" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
steps = int(sys.argv[2])
arm = sys.argv[3]
seed = int(sys.argv[4])
quality = json.loads((root / "quality_report.json").read_text())
manifest = json.loads((root / "manifest.json").read_text())
records = sum(1 for line in (root / "records.jsonl").read_text().splitlines() if line.strip())
if not quality.get("passed") or records != steps:
    raise SystemExit(f"invalid recovery-support view: passed={quality.get('passed')} records={records} steps={steps}")
if manifest.get("arm") != arm or manifest.get("seed") != seed or manifest.get("steps") != steps:
    raise SystemExit(f"recovery-support manifest identity mismatch: {manifest}")
if manifest.get("shuffle") is not False:
    raise SystemExit("formal recovery-support view must use its frozen slot order")
PY
KEEPALIVE_SESSION="gpu-keepalive-${GPU_ID}"
KEEPALIVE_WAS_RUNNING=0

if [ ! -f "$CHECKPOINT/model.safetensors" ]; then
  echo "missing baseline checkpoint: $CHECKPOINT/model.safetensors" >&2
  exit 1
fi
ACTUAL_SHA256=$(sha256sum "$CHECKPOINT/model.safetensors" | awk '{print $1}')
if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
  echo "checkpoint SHA256 mismatch for seed $SEED" >&2
  exit 1
fi
if [ -e "$OUTPUT_DIR" ]; then
  echo "refusing to overwrite existing run: $OUTPUT_DIR" >&2
  exit 1
fi

cd "$REPO_ROOT"
GIT_SHA=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "formal recovery-support training requires a clean Git worktree" >&2
  exit 1
fi

restore_keepalive() {
  if [ "$KEEPALIVE_WAS_RUNNING" = 1 ]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      "${AI2R_KEEPALIVE_EXTRA_GIB:-1}" "${AI2R_KEEPALIVE_N:-8192}" \
      "$KEEPALIVE_SESSION" "$GPU_ID" >/dev/null || true
  fi
}
trap restore_keepalive EXIT

if tmux has-session -t "$KEEPALIVE_SESSION" 2>/dev/null; then
  KEEPALIVE_WAS_RUNNING=1
  tmux kill-session -t "$KEEPALIVE_SESSION"
fi

mkdir -p "$OUTPUT_DIR"
"$PYTHON" - "$OUTPUT_DIR/run_identity.json" <<PY
import json, pathlib
path = pathlib.Path(__import__('sys').argv[1])
path.write_text(json.dumps({
    "schema_version": 1,
    "arm": "$ARM",
    "seed": $SEED,
    "steps": $STEPS,
    "git_sha": "$GIT_SHA",
    "git_dirty_at_launch": False,
    "initial_checkpoint": "$CHECKPOINT",
    "initial_checkpoint_sha256": "$ACTUAL_SHA256",
    "checkpoint_load_format_required": "alphabrain_native",
    "optimizer_state": "reset_equally_for_all_arms",
    "learning_rate": 1.0e-5,
    "minimum_learning_rate": 2.0e-6,
    "warmup_steps": 100,
    "baseline_optimizer_steps": $BASELINE_STEPS,
    "baseline_gate": "$BASELINE_GATE",
    "dataset": "$DATASET",
    "dataset_shuffle": False,
}, indent=2, sort_keys=True) + "\n")
PY

PORT=$((30600 + (SEED % 100) * 10 + GPU_ID))
export PRETRAINED_MODELS_DIR
export ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" -m accelerate.commands.launch \
  --config_file configs/deepspeed/accelerate_1gpu_simple.yaml \
  --num_processes 1 \
  --main_process_port "$PORT" \
  AlphaBrain/training/train_alphabrain.py \
  --config_yaml "$CONFIG" \
  --mode "$MODE" \
  "run_id=$RUN_ID" \
  "seed=$SEED" \
  "output_root_dir=$OUTPUT_ROOT" \
  "trainer.pretrained_checkpoint=$CHECKPOINT" \
  "trainer.learning_rate.base=1.0e-5" \
  "trainer.learning_rate.action_model=1.0e-5" \
  "trainer.learning_rate.paligemma_vl_interface=1.0e-5" \
  "trainer.num_warmup_steps=100" \
  "trainer.scheduler_specific_kwargs.min_lr=2.0e-6" \
  "trainer.max_train_steps=$STEPS" \
  "trainer.save_interval=$((STEPS + 1))" \
  "trainer.eval_interval=$((STEPS + 1))" \
  2>&1 | tee "$OUTPUT_DIR/launcher.log"

if ! grep -q 'Source format:  alphabrain_native' "$OUTPUT_DIR/launcher.log"; then
  echo "training did not confirm native checkpoint loading" >&2
  exit 1
fi
if ! grep -q 'Matched:        827/827' "$OUTPUT_DIR/launcher.log"; then
  echo "training did not load all 827 checkpoint keys" >&2
  exit 1
fi
test -f "$OUTPUT_DIR/final_model/model.safetensors"
