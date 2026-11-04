# RoboCasa Tabletop

Evaluation + training for the **RoboCasa-GR1-Tabletop-Tasks** benchmark — pick-and-place tasks on a Fourier GR1 humanoid (arms + waist + two Fourier hands). Entry points live in this directory; training & eval launchers live at repo root.

---

## GR1 active-head preparation

The active-head search integration is separate from the legacy QwenOFT launchers
below and is not admitted for task execution yet. CPU-only modules are:

- `head_observation.py`: injected observation/control port, fresh-frame identities,
  MOVE/STOP and authority checks; no implicit simulator or model construction.
- `head_layout.py`: explicit yaw/roll/pitch targets and named qpos/qvel mapping;
  restores removed head joints/motors from an identically named reference XML,
  rejecting changed mounts, duplicate restoration and coupled topology;
  mechanical joint limits are not a collision-safe search domain.
- `gr1_package.py`: explicit, bounded acquisition of a pinned public inference
  package; verifies the independent catalog and manifest, preserves interrupted
  transfers and never executes downloaded code.
- `head_servo.py`: injected three-joint motor tracking with explicit gains,
  stable position/velocity samples, authority checks, step/time limits and HOLD;
  its mock tests do not establish physical controller stability.
- `state_replay.py`: same-model MuJoCo integration-state replay with mandatory
  task/RNG/cache hooks. Rejects changed model/topology and never resumes dispatch.
- `task_runtime.py`: explicit registered-task construction and layout/style
  binding before object registries import. Empty enabled asset categories require
  exact declared exceptions; missing optional assets cannot silently change a run.
- `task_state.py`: real task/controller/observable/RNG snapshots and monotonic
  observation-cache invalidation. Replay never restores a budget or dispatch state.
- `head_geometry.py` / `head_io.py`: actual articulated-head FK in scratch data,
  and named MuJoCo motor IO. Full-scene movement and collision admission remain
  separate from the isolated CPU motor fixture.
- `model_runtime.py`: local checkpoint/tokenizer/norm loading, original native
  visual tap and velocity trace. The scoring/executed action prefixes must match.
  Production inference is closed; CPU loading/abstract tracing is not admission.
- `operation_io.py`: upstream ego padding, task instruction and absolute 29D
  action/state mapping. `operation_rollout.py` composes an explicitly bounded
  operation with per-call grants and durable unknown-outcome receipts.
- `search_runtime.py`: actual-FK geometry plus a single real ego stream into the
  existing native collector and shared Actor/Critic/PPO, with paired query noise.
- `model_process.py`: isolated synthetic-call worker or real CPU checkpoint /
  processor worker, per-call parent authorization, no reconnect after failure.
  Real inference remains gated; loading/preprocessing is not model admission.
- `snapshot_io.py`: data-only replay serialization and exact task/hash checking.
  Upstream object generation uses Python sets: freeze the interpreter/hash
  fingerprint as well as the task seed. Changing hash seeds can change scenes.
- `head_control.py`: native substep head holding and the explicitly named
  `PROJECT_NONHEAD_INTEGRATION_V1` simulation candidate. It freezes non-head
  integration during search, so must not be described as unmodified physics.
- `task_composition.py`, `operation_backend.py`, `trial_runtime.py`: actual task,
  measured selected-view handoff, synchronous capture, shared search and bounded
  operation composition. Synthetic frame mode is explicit. Native capture
  requires its own render admission; task/model dispatch remains disabled.
- `policy_binding.py`: configurable shared U/T core, exact native/text contracts,
  semantic checkpoint provenance and the common collector; no GR1-only network.
- `task_backend.py`: measured native head settlement and synchronous ego frame
  freshness checks around injected native IO/renderer; never silently freezes bodies.
- `episode_runtime.py`: bounded search entry with durable intent/unknown/complete
  receipts, no implicit retry, refund or fallback STOP on timeout.
- `continuous_authority.py`: registered CPU-only GR1 worker adapter using shared
  `execution/model_agent.py` and `model_plan.py`; no second controller. The
  explicit `cpu_contract.py` fixture uses synthetic images/features and the
  actual native-query path under live center grants. Real blocks remain closed.

Tests are in `tests/dsol_paper1/runtime/test_gr1_*.py`. Preparation evidence belongs
in the persistent project's `runs/gr1-cpu-preparation-20261104/`. The standalone
MuJoCo geometry probe is explicit CPU preparation, outside default unit tests.
The r11 explicit probes additionally construct all 24 target tasks at one
declared layout/style/seed, replay full integration and Python state, and check
native action/state/FK mappings. This is not coverage of every randomized scene.
The upstream pinned assets contain named empty optional categories, recorded
in the preparation receipt; those exceptions require explicit binding.
Real model numerical equivalence, rendering, measured-return tolerances,
full-scene head control, real operation trajectories and GR1 production authority
registration remain admission requirements. Keep the
operation-model environment separate from the task simulator: the inspected
upstream model lock pins MuJoCo 2.3.7 while the task package requires 3.2.6.

## 1. Environment Setup

Two conda environments are involved:

- **AlphaBrain** —  Install per the top-level repo README.
- **robocasa_tabletop** — Install by following the [official robocasa-gr1-tabletop-tasks repository](https://github.com/robocasa/robocasa-gr1-tabletop-tasks) 

---

## 2. Training

### 2.1 Dataset preparation

Download the Isaac GR00T X-Embodiment-Sim dataset from HuggingFace: [`nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim). Only the `gr1_unified.*_1000` subfolders are needed for this benchmark — 24 tasks × 1000 episodes each.



### 2.2 `.env` setup

Append to repo-root `.env`:

```bash
# RoboCasa Tabletop
ROBOCASA_TABLETOP_PYTHON=<your_path_to_robocasa_tabletop_conda_env>/bin/python
ROBOCASA_TABLETOP_DATA_ROOT=<your_path_to_PhysicalAI-Robotics-GR00T-X-Embodiment-Sim>
```

### 2.3 Run training

```bash
bash scripts/run_finetune.sh qwen_oft_robocasa_tabletop
```

The validated framework for this benchmark is **QwenOFT** (Qwen3-VL-4B + OFT action head). All settings come from `modes.qwen_oft_robocasa_tabletop` in `configs/finetune_config.yaml`.

**Knobs you will likely change:**

| Setting | Default | Note |
|---|---|---|
| `dataset_mix` | `fourier_gr1_unified_1000` | The mixture to train on. **Edit in two places:** top-level `dataset_mix` *and* `datasets.vla_data.dataset_mix`. |
| `base_vlm` | `./playground/Pretrained_models/Qwen3-VL-4B-Instruct` | Path to your local Qwen3-VL-4B weights. |
| `run_id` | `qwen_oft_robocasa_tabletop` | Output directory name. |

---

## 3. Evaluation

```bash
bash scripts/run_eval.sh robocasa_tabletop_eval
```

Reads `modes.robocasa_tabletop_eval` in `configs/finetune_config.yaml`. Edit there to change:

| Key | Default | Meaning |
|---|---|---|
| `checkpoint` | `./results/training/qwen_oft_robocasa_tabletop/checkpoints/steps_80000` | Path to trained model |
| `n_episodes` | 50 | Rollouts per task |
| `n_action_steps` | 12 | Action chunk size per policy query |
| `port` | 5694 | WebSocket port |
| `gpu_id` | 0 | CUDA device for server |

Results + aggregate stats land under `results/evaluation/robocasa_tabletop/<env_slug>/`.
