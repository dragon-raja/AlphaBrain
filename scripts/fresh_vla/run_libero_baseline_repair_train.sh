#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG=${FRESH_BASELINE_REPAIR_CONFIG:-$REPO_ROOT/configs/experiments/fresh_vla_libero_closed_loop.yaml}
PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
OUTPUT_ROOT=${FRESH_BASELINE_REPAIR_ROOT:-/share/longjunyu/fresh-vla/runs/baseline-repair-v1}
PRETRAINED_MODELS_DIR=${PRETRAINED_MODELS_DIR:-/share/longjunyu/alphabrain/pretrained_models}
DATASET=/share/longjunyu/fresh-vla/libero-full-episode-windows-v2-128

SEED=${1:?usage: run_libero_baseline_repair_train.sh SEED [STEPS] [SAVE_INTERVAL] [RUN_TAG]}
STEPS=${2:-13804}
SAVE_INTERVAL=${3:-3451}
RUN_TAG=${4:-formal}

case "$SEED" in
  41|42|43) ;;
  *) echo "seed must be one of 41, 42, or 43" >&2; exit 2 ;;
esac
if [[ ! "$STEPS" =~ ^[1-9][0-9]*$ ]] || [[ ! "$SAVE_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEPS and SAVE_INTERVAL must be positive integers" >&2
  exit 2
fi
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_TAG may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
fi

RUN_ID="baseline_repair_full_h_ddp8_seed${SEED}_steps${STEPS}_${RUN_TAG}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
MODE=fresh_closed_loop_full_h
CHECKPOINT="$PRETRAINED_MODELS_DIR/pi05_base"
PREREG=docs/embodied_research_reset/baseline_validity_repair_preregistration.md

for required in \
  "$CONFIG" \
  "$CHECKPOINT/model.safetensors" \
  "$PRETRAINED_MODELS_DIR/paligemma-3b-pt-224/config.json" \
  "$DATASET/quality_report.json" \
  "$DATASET/records.jsonl" \
  "$REPO_ROOT/$PREREG"; do
  if [ ! -e "$required" ]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done
if [ -e "$OUTPUT_DIR" ]; then
  echo "refusing to overwrite existing run: $OUTPUT_DIR" >&2
  exit 1
fi

cd "$REPO_ROOT"
GIT_SHA=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "baseline-repair training requires a clean Git worktree" >&2
  exit 1
fi

"$PYTHON" - "$DATASET/quality_report.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
quality = json.loads(path.read_text())
if not quality.get("passed"):
    raise SystemExit(f"dataset quality gate failed: {path}")
checks = quality.get("checks", {})
for key in ("group_preserving_split", "source_initial_state_disjoint"):
    if not checks.get(key):
        raise SystemExit(f"dataset split gate failed: {key}")
PY

KEEPALIVE_SESSIONS=()
restore_keepalives() {
  local session gpu
  for session in "${KEEPALIVE_SESSIONS[@]}"; do
    gpu=${session##*-}
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      "${AI2R_KEEPALIVE_EXTRA_GIB:-0.25}" "${AI2R_KEEPALIVE_N:-4096}" \
      "$session" "$gpu" >/dev/null || true
  done
}
trap restore_keepalives EXIT

for gpu in 0 1 2 3 4 5 6 7; do
  session="gpu-keepalive-$gpu"
  if tmux has-session -t "$session" 2>/dev/null; then
    KEEPALIVE_SESSIONS+=("$session")
    tmux kill-session -t "$session"
  fi
done

mkdir -p "$OUTPUT_DIR"
"$PYTHON" - "$OUTPUT_DIR/run_identity.json" <<PY
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "schema_version": 1,
    "purpose": "baseline_validity_repair_only",
    "method": "full_h",
    "seed": $SEED,
    "model_initialization_seed": $SEED,
    "training_rank_seed_rule": "seed + global_rank",
    "optimizer_steps": $STEPS,
    "save_interval": $SAVE_INTERVAL,
    "git_sha": "$GIT_SHA",
    "git_dirty_at_launch": False,
    "preregistration": "$PREREG",
    "initial_checkpoint": "$CHECKPOINT",
    "dataset": "$DATASET",
    "dataset_split": "train",
    "test_split_opened": False,
    "distributed_type": "DDP",
    "num_processes": 8,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "effective_batch_size": 8,
    "learning_rate": 5.0e-5,
    "warmup_steps": 1000,
    "minimum_learning_rate": 5.0e-5,
    "ema_enabled": False,
    "freeze_modules": "vlm_interface",
    "run_tag": "$RUN_TAG",
}, indent=2, sort_keys=True) + "\n")
PY

PORT=${FRESH_BASELINE_REPAIR_PORT:-30741}
export PRETRAINED_MODELS_DIR
export ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export USE_DDP=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PYTHON" -m accelerate.commands.launch \
  --config_file configs/deepspeed/accelerate_ddp.yaml \
  --num_processes 8 \
  --main_process_port "$PORT" \
  AlphaBrain/training/train_alphabrain.py \
  --config_yaml "$CONFIG" \
  --mode "$MODE" \
  "run_id=$RUN_ID" \
  "seed=$SEED" \
  "output_root_dir=$OUTPUT_ROOT" \
  "trainer.pretrained_checkpoint=$CHECKPOINT" \
  "trainer.learning_rate.base=5.0e-5" \
  "trainer.learning_rate.action_model=5.0e-5" \
  "trainer.learning_rate.paligemma_vl_interface=5.0e-5" \
  "trainer.num_warmup_steps=1000" \
  "trainer.scheduler_specific_kwargs.min_lr=5.0e-5" \
  "trainer.max_train_steps=$STEPS" \
  "trainer.save_interval=$SAVE_INTERVAL" \
  "trainer.eval_interval=$((STEPS + 1))" \
  "trainer.gradient_accumulation_steps=1" \
  "datasets.vla_data.per_device_batch_size=1" \
  "trainer.ema.enabled=false" \
  2>&1 | tee "$OUTPUT_DIR/launcher.log"

grep -q 'Distributed environment: DistributedType.MULTI_GPU' "$OUTPUT_DIR/launcher.log"
grep -q 'Model initialization seed: '$SEED "$OUTPUT_DIR/launcher.log"
grep -Eq 'Matched:[[:space:]]+814/827' "$OUTPUT_DIR/launcher.log"
grep -Eq 'Total batch size:[[:space:]]+8' "$OUTPUT_DIR/launcher.log"
test -s "$OUTPUT_DIR/final_model/model.safetensors"

"$PYTHON" - "$OUTPUT_DIR/run_complete.json" <<PY
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "completed": True,
    "optimizer_steps": $STEPS,
    "effective_batch_size": 8,
    "final_model": "$OUTPUT_DIR/final_model/model.safetensors",
}, indent=2, sort_keys=True) + "\n")
PY
