#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG=${DSOL_PAIR_CONFIG:-$REPO_ROOT/configs/experiments/dsol_libero_broad_pairing.yaml}
if [[ -n "${DSOL_PAIR_PYTHON:-}" ]]; then
  PYTHON=$DSOL_PAIR_PYTHON
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON=$REPO_ROOT/.venv/bin/python
else
  PYTHON=/alphabrain/.venv/bin/python
fi
DATA_ROOT=${DSOL_PAIR_DATA_ROOT:-/workspace/ai2r/debug/libero_plus_revalidation_v1/pair_generator_smoke/scene3_broad32_seed41_10episodes}
OUTPUT_ROOT=${DSOL_PAIR_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs}
GLOBAL_EXAMPLES=${DSOL_GLOBAL_EXAMPLES:-32}
WANDB_MODE_VALUE=${WANDB_MODE:-offline}
CALIBRATION=${DSOL_CALIBRATION:-0}
CALIBRATION_INTERVAL=${DSOL_CALIBRATION_INTERVAL:-250}
CALIBRATION_ITEMS=${DSOL_CALIBRATION_ITEMS:-256}
SKIP_FINAL_SAVE=${DSOL_SKIP_FINAL_SAVE:-1}
BUDGET_DECISION=${DSOL_BUDGET_DECISION:-}

ARM=${1:?usage: run_libero_pair_train.sh ARM SEED NUM_GPUS STEPS [RUN_TAG]}
SEED=${2:?usage: run_libero_pair_train.sh ARM SEED NUM_GPUS STEPS [RUN_TAG]}
NUM_GPUS=${3:?usage: run_libero_pair_train.sh ARM SEED NUM_GPUS STEPS [RUN_TAG]}
STEPS=${4:?usage: run_libero_pair_train.sh ARM SEED NUM_GPUS STEPS [RUN_TAG]}
RUN_TAG=${5:-smoke}
SCHEDULER_STEPS=${DSOL_SCHEDULER_STEPS:-$STEPS}
GPU_DEVICES=${DSOL_GPU_DEVICES:-}
MAIN_PROCESS_PORT=${DSOL_MAIN_PROCESS_PORT:-$((31800 + SEED % 100))}

case "$ARM" in
  canonical_unique|image_augmentation_unique|broad_unpaired_practical)
    EXAMPLES_PER_ITEM=1
    ;;
  canonical_repeat|broad_unpaired_state_matched|broad_paired_fm|broad_paired_consistency)
    EXAMPLES_PER_ITEM=2
    ;;
  *)
    echo "unknown DSOL arm: $ARM" >&2
    exit 2
    ;;
esac

for value in "$SEED" "$NUM_GPUS" "$STEPS" "$SCHEDULER_STEPS" "$GLOBAL_EXAMPLES"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "numeric arguments must be positive integers" >&2; exit 2; }
done
(( SCHEDULER_STEPS >= STEPS )) || {
  echo "DSOL_SCHEDULER_STEPS must be >= training STEPS" >&2
  exit 2
}
[[ "$NUM_GPUS" -le 8 ]] || { echo "NUM_GPUS must be <= 8" >&2; exit 2; }
[[ "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "unsafe RUN_TAG" >&2; exit 2; }
[[ "$MAIN_PROCESS_PORT" =~ ^[0-9]+$ ]] || { echo "invalid DSOL_MAIN_PROCESS_PORT" >&2; exit 2; }
(( MAIN_PROCESS_PORT >= 1024 && MAIN_PROCESS_PORT <= 65535 )) || {
  echo "DSOL_MAIN_PROCESS_PORT must be in [1024, 65535]" >&2
  exit 2
}
[[ "$CALIBRATION" =~ ^[01]$ ]] || { echo "DSOL_CALIBRATION must be 0 or 1" >&2; exit 2; }
[[ "$CALIBRATION_INTERVAL" =~ ^[1-9][0-9]*$ ]] || { echo "invalid DSOL_CALIBRATION_INTERVAL" >&2; exit 2; }
[[ "$CALIBRATION_ITEMS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid DSOL_CALIBRATION_ITEMS" >&2; exit 2; }
[[ "$SKIP_FINAL_SAVE" =~ ^[01]$ ]] || { echo "DSOL_SKIP_FINAL_SAVE must be 0 or 1" >&2; exit 2; }
[[ -z "$BUDGET_DECISION" || -s "$BUDGET_DECISION" ]] || {
  echo "missing DSOL_BUDGET_DECISION: $BUDGET_DECISION" >&2
  exit 2
}

DENOMINATOR=$((NUM_GPUS * EXAMPLES_PER_ITEM))
if (( GLOBAL_EXAMPLES % DENOMINATOR != 0 )); then
  echo "GLOBAL_EXAMPLES=$GLOBAL_EXAMPLES is not divisible by NUM_GPUS*EXAMPLES_PER_ITEM=$DENOMINATOR" >&2
  exit 2
fi
GRAD_ACC=$((GLOBAL_EXAMPLES / DENOMINATOR))
(( GRAD_ACC >= 1 )) || { echo "gradient accumulation would be zero" >&2; exit 2; }

if [[ -z "$GPU_DEVICES" ]]; then
  physical_gpus=()
  for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    physical_gpus+=("$gpu")
  done
  GPU_DEVICES=$(IFS=,; echo "${physical_gpus[*]}")
else
  IFS=, read -r -a physical_gpus <<< "$GPU_DEVICES"
fi
[[ "${#physical_gpus[@]}" -eq "$NUM_GPUS" ]] || {
  echo "DSOL_GPU_DEVICES must list exactly NUM_GPUS=$NUM_GPUS devices" >&2
  exit 2
}
declare -A seen_gpus=()
for gpu in "${physical_gpus[@]}"; do
  [[ "$gpu" =~ ^[0-7]$ ]] || { echo "invalid physical GPU index: $gpu" >&2; exit 2; }
  [[ -z "${seen_gpus[$gpu]:-}" ]] || { echo "duplicate physical GPU index: $gpu" >&2; exit 2; }
  seen_gpus[$gpu]=1
done

RUN_ID="dsol_${ARM}_${RUN_TAG}_seed${SEED}_g${NUM_GPUS}_gb${GLOBAL_EXAMPLES}_steps${STEPS}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
[[ ! -e "$OUTPUT_DIR" ]] || { echo "refusing to overwrite $OUTPUT_DIR" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 1; }
[[ -s "$DATA_ROOT/manifest.json" ]] || { echo "missing pair dataset: $DATA_ROOT" >&2; exit 1; }

ACCELERATE_CONFIG=configs/deepspeed/accelerate_1gpu_simple.yaml
if (( NUM_GPUS > 1 )); then
  ACCELERATE_CONFIG=configs/deepspeed/accelerate_ddp.yaml
fi

STOPPED_KEEPALIVES=()
restore_keepalive() {
  for gpu in "${STOPPED_KEEPALIVES[@]:-}"; do
    [[ -n "$gpu" ]] || continue
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      "${AI2R_KEEPALIVE_EXTRA_GIB:-1}" "${AI2R_KEEPALIVE_N:-8192}" \
      "gpu-keepalive-${gpu}" "$gpu" >/dev/null || true
  done
}
trap restore_keepalive EXIT

for gpu in "${physical_gpus[@]}"; do
  if tmux has-session -t "gpu-keepalive-${gpu}" 2>/dev/null; then
    tmux kill-session -t "gpu-keepalive-${gpu}"
    STOPPED_KEEPALIVES+=("$gpu")
  fi
done

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"
export PRETRAINED_MODELS_DIR=${PRETRAINED_MODELS_DIR:-/share/longjunyu/alphabrain/pretrained_models}
export ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export WANDB_MODE="$WANDB_MODE_VALUE"
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
# Visual-LoRA leaves a small number of conditional trainable parameters unused
# on some microbatches. Dynamic DDP plus unused-parameter detection is required
# for the strict global-batch gradient accumulation used by this experiment.
export USE_DDP=1
export ALPHABRAIN_DDP_STATIC_GRAPH=0
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

IMPORTED_ROOT=$("$PYTHON" -c 'import pathlib, AlphaBrain; print(pathlib.Path(AlphaBrain.__file__).resolve().parents[1])')
if [[ "$IMPORTED_ROOT" != "$REPO_ROOT" ]]; then
  echo "AlphaBrain import provenance mismatch: expected $REPO_ROOT, got $IMPORTED_ROOT" >&2
  exit 1
fi

budget_manifest_args=()
if [[ -n "$BUDGET_DECISION" ]]; then
  budget_manifest_args=(--budget-decision "$BUDGET_DECISION")
fi

"$PYTHON" scripts/dsol_paper1/write_dsol_training_manifest.py \
  --repo-root "$REPO_ROOT" \
  --output "$OUTPUT_DIR/run_manifest.json" \
  --data-root "$DATA_ROOT" \
  --arm "$ARM" \
  --seed "$SEED" \
  --num-gpus "$NUM_GPUS" \
  --gpu-devices "$GPU_DEVICES" \
  --main-process-port "$MAIN_PROCESS_PORT" \
  --steps "$STEPS" \
  --scheduler-steps "$SCHEDULER_STEPS" \
  --global-examples "$GLOBAL_EXAMPLES" \
  --examples-per-item "$EXAMPLES_PER_ITEM" \
  --gradient-accumulation "$GRAD_ACC" \
  --calibration "$CALIBRATION" \
  --calibration-interval "$CALIBRATION_INTERVAL" \
  --calibration-items "$CALIBRATION_ITEMS" \
  --skip-final-save "$SKIP_FINAL_SAVE" \
  --wandb-mode "$WANDB_MODE_VALUE" \
  "${budget_manifest_args[@]}"

printf 'arm=%s seed=%s num_gpus=%s gpu_devices=%s main_process_port=%s examples_per_item=%s grad_acc=%s global_model_examples=%s\n' \
  "$ARM" "$SEED" "$NUM_GPUS" "$GPU_DEVICES" "$MAIN_PROCESS_PORT" "$EXAMPLES_PER_ITEM" "$GRAD_ACC" "$GLOBAL_EXAMPLES" \
  | tee "$OUTPUT_DIR/batch_accounting.txt"

"$PYTHON" -m accelerate.commands.launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes "$NUM_GPUS" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  AlphaBrain/training/train_alphabrain.py \
  --config_yaml "$CONFIG" \
  --mode "dsol_${ARM}" \
  "run_id=$RUN_ID" \
  "seed=$SEED" \
  "output_root_dir=$OUTPUT_ROOT" \
  "datasets.vla_data.data_root_dir=$DATA_ROOT" \
  "datasets.vla_data.examples_per_item=$EXAMPLES_PER_ITEM" \
  "trainer.gradient_accumulation_steps=$GRAD_ACC" \
  "trainer.max_train_steps=$STEPS" \
  "trainer.scheduler_total_steps=$SCHEDULER_STEPS" \
  "trainer.save_interval=$((STEPS + 1))" \
  "trainer.eval_interval=$([[ "$CALIBRATION" == 1 ]] && echo "$CALIBRATION_INTERVAL" || echo "$((STEPS + 1))")" \
  "trainer.dsol_validation.enabled=$([[ "$CALIBRATION" == 1 ]] && echo true || echo false)" \
  "trainer.dsol_validation.max_data_items=$CALIBRATION_ITEMS" \
  "trainer.skip_final_save=$([[ "$SKIP_FINAL_SAVE" == 1 ]] && echo true || echo false)" \
  2>&1 | tee "$OUTPUT_DIR/launcher.log"
