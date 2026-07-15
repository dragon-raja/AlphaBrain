#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
SIM_PYTHON=${FRESH_LIBERO_PYTHON:-/workspace/envs/fresh-libero/bin/python}
LIBERO_SOURCE=${FRESH_LIBERO_SOURCE:-/projects/openpi/third_party/libero}
BASELINE_ROOT=${FRESH_CLOSED_LOOP_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2}
OUTPUT_ROOT=${FRESH_ORACLE_COMMIT_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-oracle-commit-final-v1}
EPISODE_ROOT=${FRESH_EPISODE_ROOT:-/share/longjunyu/fresh-vla/libero-full-episode-v2-128}
PRETRAINED_MODELS_DIR=${PRETRAINED_MODELS_DIR:-/share/longjunyu/alphabrain/pretrained_models}
EVAL_SPLIT=${FRESH_EVAL_SPLIT:-test}
EVAL_MAX_STEPS=${FRESH_EVAL_MAX_STEPS:-320}
MAX_GROUPS=${FRESH_EVAL_MAX_GROUPS:-}
REACH_MAX_STEPS=${FRESH_REACH_MAX_STEPS:-20}
REACH_TARGET_STEP=${FRESH_REACH_TARGET_STEP:-20}
SELF_SAMPLES=${FRESH_SELF_CONSISTENCY_SAMPLES:-8}
SELF_THRESHOLD=${FRESH_SELF_CONSISTENCY_THRESHOLD:-0.15}
SAVE_VIDEOS=${FRESH_SAVE_EVAL_VIDEOS:-1}
EVAL_ONLY=${FRESH_ORACLE_EVAL_ONLY:-all}
SERVER_START_TIMEOUT=${FRESH_POLICY_SERVER_TIMEOUT:-600}

METHOD=${1:?usage: run_libero_oracle_commit_eval.sh METHOD SEED GPU_ID}
SEED=${2:?usage: run_libero_oracle_commit_eval.sh METHOD SEED GPU_ID}
GPU_ID=${3:?usage: run_libero_oracle_commit_eval.sh METHOD SEED GPU_ID}
RUN_DIR="$OUTPUT_ROOT/oracle_commit_${METHOD}_seed${SEED}"
CHECKPOINT="$BASELINE_ROOT/fresh_closed_loop_full_h_seed${SEED}/final_model"
SESSION="gpu-keepalive-${GPU_ID}"
SOCKET_PATH="/tmp/fresh-oracle-commit-${METHOD}-${SEED}-$$.sock"
SERVER_PID=""
WAS_RUNNING=0

case "$METHOD" in
  fixed_k1|fixed_k2|fixed_k3|oracle_branch_safe_commit|oracle_feedback_reveal_commit|gripper_commit|random_matched_commit|self_consistency_commit) ;;
  *) echo "unknown commit method: $METHOD" >&2; exit 2 ;;
esac
case "$SEED" in
  41) EXPECTED_CHECKPOINT_SHA256=144a3b3d3dcc8421418564a62059a1038c9a7ef3196ac157f5f9ea1997a31f30 ;;
  42) EXPECTED_CHECKPOINT_SHA256=98dc52d2ed1983776d218fee7666f3131053d1a55296e93e9f521b1c088ce875 ;;
  43) EXPECTED_CHECKPOINT_SHA256=5db16350d9835c1f28d01b660dd6e9234bcab3da79abbce1f092e92b08ac9149 ;;
  *) echo "unsupported Full-H seed: $SEED" >&2; exit 2 ;;
esac
case "$EVAL_ONLY" in
  all|isolated|end_to_end|reach|isolated_reach) ;;
  *) echo "unknown FRESH_ORACLE_EVAL_ONLY: $EVAL_ONLY" >&2; exit 2 ;;
esac
if [ ! -f "$CHECKPOINT/model.safetensors" ]; then
  echo "missing frozen Full-H checkpoint: $CHECKPOINT/model.safetensors" >&2
  exit 1
fi
if ! command -v flock >/dev/null || ! command -v sha256sum >/dev/null; then
  echo "required command missing: flock and sha256sum are mandatory" >&2
  exit 1
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]] || [[ ! "$GPU_ID" =~ ^[0-7]$ ]]; then
  echo "seed must be numeric and GPU_ID must be in [0, 7]" >&2
  exit 2
fi

restore_runtime() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$SOCKET_PATH"
  if [ "$WAS_RUNNING" = 1 ]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      "${AI2R_KEEPALIVE_EXTRA_GIB:-1}" "${AI2R_KEEPALIVE_N:-8192}" "$SESSION" "$GPU_ID" >/dev/null || true
  fi
}
trap restore_runtime EXIT

if tmux has-session -t "$SESSION" 2>/dev/null; then
  WAS_RUNNING=1
  tmux kill-session -t "$SESSION"
fi

mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/.${EVAL_ONLY}.lock"
if ! flock -n 9; then
  echo "evaluation already running: method=$METHOD seed=$SEED mode=$EVAL_ONLY" >&2
  exit 1
fi
cd "$REPO_ROOT"
if [ -n "$(git status --porcelain)" ]; then
  echo "refusing to evaluate a dirty Git worktree" >&2
  exit 1
fi
export FRESH_GIT_SHA
FRESH_GIT_SHA=$(git rev-parse HEAD)
export FRESH_CHECKPOINT_SHA256
FRESH_CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT/model.safetensors" | awk '{print $1}')
if [ "$FRESH_CHECKPOINT_SHA256" != "$EXPECTED_CHECKPOINT_SHA256" ]; then
  echo "frozen checkpoint SHA256 mismatch for seed $SEED" >&2
  exit 1
fi
GROUP_COUNT=$("$SIM_PYTHON" - "$EPISODE_ROOT/manifest.json" "$EVAL_SPLIT" "$MAX_GROUPS" <<'PY'
import json
import sys

manifest_path, split, maximum = sys.argv[1:]
groups = [group for group in json.load(open(manifest_path))["groups"] if group["split"] == split]
if maximum:
    groups = groups[: int(maximum)]
print(len(groups))
PY
)
if [ "$GROUP_COUNT" -le 0 ]; then
  echo "no evaluation groups for split=$EVAL_SPLIT" >&2
  exit 1
fi

validate_existing_output() {
  local path=$1
  local expected_rows=$2
  local evaluation=$3
  "$SIM_PYTHON" - "$path" "$expected_rows" "$FRESH_GIT_SHA" \
    "$FRESH_CHECKPOINT_SHA256" "$METHOD" "$evaluation" <<'PY'
import json
import sys

path, expected_rows, git_sha, checkpoint_sha, method, evaluation = sys.argv[1:]
payload = json.load(open(path))
valid = (
    payload.get("status") == "complete"
    and len(payload.get("rows", ())) == int(expected_rows)
    and payload.get("git_sha") == git_sha
    and payload.get("policy_checkpoint_sha256") == checkpoint_sha
    and payload.get("commit_method") == method
    and payload.get("evaluation") == evaluation
)
raise SystemExit(0 if valid else 1)
PY
}

validate_existing_videos() {
  local directory=$1
  local expected=$2
  if [ "$SAVE_VIDEOS" != 1 ]; then
    return 0
  fi
  [ -d "$directory" ] && [ "$(find "$directory" -maxdepth 1 -type f -name '*.mp4' | wc -l)" -eq "$expected" ]
}
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PRETRAINED_MODELS_DIR="$PRETRAINED_MODELS_DIR" \
PYTHONDONTWRITEBYTECODE=1 \
"$PYTHON" scripts/fresh_vla/pi05_policy_server.py \
  --checkpoint "$CHECKPOINT" \
  --socket "$SOCKET_PATH" \
  --device cuda:0 >"$RUN_DIR/policy_server_${EVAL_ONLY}_gpu${GPU_ID}.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 "$SERVER_START_TIMEOUT"); do
  if [ -S "$SOCKET_PATH" ]; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Pi0.5 policy server exited before becoming ready" >&2
    exit 1
  fi
  sleep 1
done
if [ ! -S "$SOCKET_PATH" ]; then
  echo "timed out waiting for Pi0.5 policy server" >&2
  exit 1
fi

max_group_args=()
if [ -n "$MAX_GROUPS" ]; then
  max_group_args=(--max-groups "$MAX_GROUPS")
fi
random_boundary_args=()
if [ "$METHOD" = random_matched_commit ]; then
  boundary_map="$OUTPUT_ROOT/control_maps/random_boundary_seed${SEED}.json"
  if [ ! -f "$boundary_map" ]; then
    "$SIM_PYTHON" scripts/fresh_vla/build_random_commit_boundaries.py \
      --manifest "$EPISODE_ROOT/manifest.json" \
      --fixed-k3-results "$BASELINE_ROOT/fresh_closed_loop_full_h_seed${SEED}/closed_loop_end_to_end.json" \
      --output "$boundary_map" \
      --seed "$((161803 + SEED))"
  fi
  random_boundary_args=(--random-boundary-map "$boundary_map")
fi
common_args=(
  --policy-socket "$SOCKET_PATH"
  --episode-root "$EPISODE_ROOT"
  --commit-method "$METHOD"
  --max-commit 3
  --self-consistency-samples "$SELF_SAMPLES"
  --self-consistency-threshold "$SELF_THRESHOLD"
  --split "$EVAL_SPLIT"
  "${max_group_args[@]}"
  "${random_boundary_args[@]}"
)

for evaluation in isolated end_to_end; do
  if [ "$EVAL_ONLY" != all ] && [ "$EVAL_ONLY" != "$evaluation" ] \
    && ! { [ "$EVAL_ONLY" = isolated_reach ] && [ "$evaluation" = isolated ]; }; then
    continue
  fi
  output="$RUN_DIR/closed_loop_${evaluation}.json"
  if [ -f "$output" ]; then
    if validate_existing_output "$output" "$((GROUP_COUNT * 2))" "$evaluation" \
      && validate_existing_videos "$RUN_DIR/videos/$evaluation" "$GROUP_COUNT"; then
      echo "skip verified $output"
      continue
    fi
    echo "refusing to reuse invalid or incomplete output: $output" >&2
    exit 1
  fi
  video_args=()
  if [ "$SAVE_VIDEOS" = 1 ]; then
    video_args=(--video-dir "$RUN_DIR/videos/$evaluation")
  fi
  PYTHONPATH="$REPO_ROOT/scripts/fresh_vla:$LIBERO_SOURCE${PYTHONPATH:+:$PYTHONPATH}" \
  LIBERO_CONFIG_PATH="$REPO_ROOT/scripts/fresh_vla/libero_config" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  PRETRAINED_MODELS_DIR="$PRETRAINED_MODELS_DIR" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$SIM_PYTHON" scripts/fresh_vla/evaluate_libero_oracle_commit.py \
    "${common_args[@]}" \
    --output "$output" \
    --evaluation "$evaluation" \
    --max-steps "$EVAL_MAX_STEPS" \
    --seed "$((314159 + SEED))" \
    "${video_args[@]}"
done

reach_output="$RUN_DIR/deterministic_reach.json"
if [ "$EVAL_ONLY" = all ] || [ "$EVAL_ONLY" = reach ] || [ "$EVAL_ONLY" = isolated_reach ]; then
if [ ! -f "$reach_output" ]; then
  reach_video_args=()
  if [ "$SAVE_VIDEOS" = 1 ]; then
    reach_video_args=(--video-dir "$RUN_DIR/videos/reach")
  fi
  PYTHONPATH="$REPO_ROOT/scripts/fresh_vla:$LIBERO_SOURCE${PYTHONPATH:+:$PYTHONPATH}" \
  LIBERO_CONFIG_PATH="$REPO_ROOT/scripts/fresh_vla/libero_config" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  PRETRAINED_MODELS_DIR="$PRETRAINED_MODELS_DIR" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$SIM_PYTHON" scripts/fresh_vla/evaluate_libero_oracle_commit_reach.py \
    "${common_args[@]}" \
    --output "$reach_output" \
    --max-steps "$REACH_MAX_STEPS" \
    --reference-target-step "$REACH_TARGET_STEP" \
    --seed "$((271828 + SEED))" \
    "${reach_video_args[@]}"
else
  if validate_existing_output "$reach_output" "$GROUP_COUNT" deterministic_reach \
    && validate_existing_videos "$RUN_DIR/videos/reach" "$GROUP_COUNT"; then
    echo "skip verified $reach_output"
  else
    echo "refusing to reuse invalid or incomplete output: $reach_output" >&2
    exit 1
  fi
fi
fi
