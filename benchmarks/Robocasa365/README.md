# RoboCasa365

## Paper 1实际后端（2026-09-29核验）

本目录同时包含框架通用适配与论文专用适配，两者不能混称。**当前论文的RoboCasa
π0.5训练/评测使用外部OpenPI实现（JAX/Orbax），不是AlphaBrain的PyTorch π0.5。**
`evaluation_policy.py`、`openpi_training.py` 负责接入；核心指标与分析复用研究包。
下面历史通用指南中QwenGR00T/QwenOFT的支持声明，不构成RoboCasa π0.5迁移验收。

实际后端、数据与权重导航见[完整项目地图](../../docs/dsol_paper1/protocols/project_inventory_zh.md)。
不能把现有Orbax检查点直接当作AlphaBrain权重，也不能通过换一份不同训练来源的权重
声称“仅统一了框架”。后续迁移先完成转换覆盖、预处理/状态/动作/噪声与闭环一致性验证。

Paper 1 第二阶段新增适配：`camera_rig.py` 将左右外部相机视作刚性组件；
`training_render.py` 在原始示范帧上重渲染规范／多视角输入。安装偏置在第0帧
定义并沿整段示范保持不变，相机随后随底盘及躯干支撑结构自然运动。
两组共用该帧原始腕部RGB，原状态、动作和语言不变。显式入口见
[Paper 1脚本导航](../../scripts/dsol_paper1/README.md)。这些数据构造不属于在线主动感知评测。

Evaluation + training for the RoboCasa **365-task** suite (atomic + composite pick-and-place, Panda-Omron). Entry points live in this directory; training & eval launchers live at repo root.

---

## 1. Environment Setup

Two conda environments are involved:

- **AlphaBrain** — Install per the top-level repo README.
- **robocasa365** — Install by following the [official RoboCasa365 repository](https://github.com/robocasa/robocasa).

---

## 2. Training

### 2.1 Dataset preparation

Download the RoboCasa365 dataset following the [robocasa.ai dataset docs](https://robocasa.ai/docs/build/html/datasets/using_datasets.html).

After download, arrange files so `${ROBOCASA365_DATA_ROOT}` matches the layout below — this is the shape expected by the `robocasa365_*` mixtures defined in `AlphaBrain/dataloader/gr00t_lerobot/mixtures.py`:

```
${ROBOCASA365_DATA_ROOT}/
├── pretrain/
│   ├── atomic/<task>/<date>/     
│   │   ├── README.md
│   │   └── lerobot/              
│   └── composite/<task>/<date>/
└── target/
    ├── atomic/<task>/<date>/      
    └── composite/<task>/<date>/
```

Available mixtures (`dataset_mix` names in the training config). RoboCasa365 ships data in two trees — `pretrain/` (broad multi-task data used for pretraining) and `target/` (the specific tasks you evaluate on) — and the mixture name tells you which tree it draws from:

**Pretrain dataset** — draws from `pretrain/`:

| Mixture | Glob |
|---|---|
| `robocasa365_pretrain300` | `pretrain/*/*/*/lerobot` |

**Target-task dataset** — draws from `target/`:

| Mixture | Glob / contents |
|---|---|
| `robocasa365_target_atomic` | `target/atomic/*/*/lerobot` |
| `robocasa365_target_composite_seen` | 16 composite tasks — seen split (DeliverStraw, KettleBoiling, PrepareCoffee, …) |
| `robocasa365_target_composite_unseen` | 16 composite tasks — unseen split (ArrangeBreadBasket, ArrangeTea, BreadSelection, …) |

### 2.2 `.env` setup

Append to repo-root `.env`:

```bash
# RoboCasa365
ROBOCASA365_PYTHON=<your_path_to_robocasa_conda_env>/bin/python
ROBOCASA365_DATA_ROOT=<your_path_to_robocasa365_dataset>
```

### 2.3 Run training

```bash
bash scripts/run_finetune.sh robocasa365_train
```

Two model frameworks are currently validated on RoboCasa365 — **QwenGR00T** (Qwen3-VL-4B + GR00T-style diffusion action head) and **QwenOFT** (Qwen3-VL-4B + OFT action head). The example below uses QwenGR00T via `modes.robocasa365_train` in `configs/finetune_config.yaml`.

**Knobs you will likely change:**

| Setting | Default | Note |
|---|---|---|
| `dataset_mix` | `robocasa365_target_composite_unseen` | The split to train on (see §2.1 mixture table). **Edit in two places:** top-level `dataset_mix` *and* `datasets.vla_data.dataset_mix`. |
| `base_vlm` | `./playground/Pretrained_models/Qwen3-VL-4B-Instruct` | Path to your local Qwen3-VL-4B weights. |
| `run_id` | `qwen_gr00t_robocasa365_composite_unseen` | Output directory name. |

**Recipes:**

**A. Pretrain.**
1. First pass — set `dataset_mix: robocasa365_pretrain300` and run `bash scripts/run_finetune.sh robocasa365_train`.
2. When the pretrain run finishes, edit the config: under `trainer:` add `pretrained_checkpoint: <path/to/pretrain/checkpoint>`.
3. Switch `dataset_mix` to the target mixture you want — one of `robocasa365_target_atomic`, `robocasa365_target_composite_seen`, or `robocasa365_target_composite_unseen` — and re-launch.

**B. Target-only.** Skip steps 1–2; set `dataset_mix` directly to a target mixture.

---

## 3. Evaluation

Eval runs as two processes communicating over WebSocket:
- a **policy server** (AlphaBrain env) serving actions;
- a **simulation client** (robocasa365 env) stepping MuJoCo envs.

```bash
bash scripts/run_eval.sh robocasa365_eval
```

Reads `modes.robocasa365_eval` in `configs/finetune_config.yaml`. Edit there to change:

| Key | Default | Meaning |
|---|---|---|
| `checkpoint` | `results/training/.../final_model` | Path to trained model |
| `task_set` / `task_suite` | `composite_seen` | Options: `composite_seen` / `composite_unseen` / `atomic_seen`. |
| `n_episodes` | 50 | Rollouts per task |
| `n_action_steps` | 16 | Action chunk size per policy query |
| `port` | 5694 | WebSocket port |
| `gpu_id` | 0 | CUDA device for server |

Results + aggregate stats land under `results/evaluation/robocasa365/<split>/<task_slug>/`.

---

## 4. Results

| Suite | GR00T w/o pretrain | GR00T w/ pretrain | QwenGR00T w/o state | QwenOFT |
|-------|:---------------------:|:--------------------:|:---------:|:-------:|
| Atomic-Seen (18) | 59.1% | 68.6% | 75.1% | 66.4% |
| Composite-Seen (16) | 34.6% | 40.6% | 36.1% | 14.4% |
| Composite-Unseen (16) | 30.8% | 42.1% | 37.6% | 17.8% |
| **Average** | **43.7%** | **51.1%** | **50.6%** | **34.2%** |

> **QwenGR00T w/o state** and **QwenOFT** columns report **Target Task Learning Only** (no pretrain stage; trained directly on the target mixture).

> **GR00T w/o pretrain** and **GR00T w/ pretrain** numbers are taken from the official source.
# Paper 1 持久化初态适配

`snapshot.py`与`snapshot_bundle.py`在2026-09-16从只读恢复源码提升为本仓库维护模块。
原始文件保留在`/workspace/ai2r/recovery/paper1_20260814/materialized/ai2r/benchmarks/robocasa/`：

- `robocasa_snapshot.py` SHA256：`f7d0c8a9b49fd605a3cc3348bddd3da8d0c0ec48f1c37db180e996cea1200860`。
- `robocasa_snapshot_archive.py` SHA256：`06ddd5d9abffaf1738f0a5db075fb68f51b0ec072e42d3dd1f80f7562b9d3d5d`。

保留capture/restore和读取／模型签名／数值指纹语义；不引入旧写入器。旧pickle中的`robocasa_snapshot.EnvSnapshot`通过显式类名映射读取，不导入恢复目录。这里只允许可信本地校验过的pickle；文件哈希不构成不可信pickle的安全沙箱。导入不会启动仿真或GPU。

`evaluation_installation.py`负责本轮已发布候选安装的首帧像素核查、起点恢复和整轨迹安装偏置检查；只允许左右外部相机改变安装，腕部保持原安装。`evaluation_render.py`仍是静态缓存渲染，不将其当作闭环成功率评测。数据发布准入位于研究包`data/evaluation_admission.py`；CLI只显式组合这些适配器。
# P0 evaluation diagnostics

`evaluation_policy.py` owns the resident OpenPI inference/Accel adapter;
`policy_replay.py` isolates fixed-input numerical reproducibility without an environment.
`render_schedule.py` is an opt-in optimization requiring physical replay validation.
`multisample_diagnostic.py` and render-pipeline ablations are diagnostic-only, not approved
replacement renderers. Changing MSAA requires an explicit evaluation protocol release and
new image references; existing training and candidate releases remain immutable.

`prefix_replay.py` is a bounded XR-1 scoring-prefix diagnostic. It restores the
original released physical state/camera installation and original history/noise
addresses, checks the stored initial inputs/actions and all five flow/Accel
traces exactly, then retains per-step observations, issued commands and realized
simulator arrays. It does not add complete-task success labels or relax renderer
admission. Missing historical later action/state records are not claimed to have
been compared. Shared telemetry, analysis and plots live in the research package;
execution requires explicit configuration and `--execute`.

## Same-state restoration diagnostic (2026-10-15)

`same_state_restore.py` is an explicit, no-action observation replay gate.
`forward_raw_state` retains the original exact independent-state/geometry check.
The separately authorized `separate_recorded_render_pose` mode retains exact raw
qpos/qvel/ctrl/time/warmstart, installs the recorded body world poses, and derives
sites, geoms, cameras and lights with native MuJoCo operations. Sites are compared
against independent historical records; rendered images must pass the unchanged
native pixel gate. This is historical observation replay, **not** a dynamically
complete state for continuing physics. No step/reset/action is permitted.

The split is necessary because historical Robosuite telemetry pairs integrated
qpos with rendering caches from before the last 2 ms integration. Calling forward
on raw qpos cannot reproduce that earlier geometry. Missing actuator activation
remains an error. Mocap inputs come from each recorded body's world pose through
`body_mocapid`; the pure mapping in `data/recorded_mocap.py` checks identities and
complete slot coverage, including moving targets and nonconsecutive body IDs.

Direct reconstruction uses the original `model.xml.gz` and the original
`Kitchen.edit_model_xml` with saved camera metadata, without constructing an
environment. The exported compiled XML remains checksummed audit evidence, but
its rounded constants are unsuitable for exact reconstruction. Full archived
model signatures and candidate array shapes are checked before rendering.
Generative textures and unrecorded deformable/tracking render state fail closed.

Renderer ownership remains on the simulator; resources are freed with their own
GL context current before another scene is constructed. A renderer local retained
across scenes can otherwise be destroyed under the next scene's GL context.
`render_preflight.py` checks Robosuite's numeric EGL GPU convention before imports.

Use a new `--output` and `--previous-attempt` only for an authorized attempt.
Previous reserved model calls carry forward against the same cap. Existing
outputs and historical failures are never resumed or overwritten. The authorized
split-pose attempt passed all 192 cabinet frames / 576 images with zero pixel
difference, then stopped at the first sink image with corrupted pixels. After
explicit retry authorization and the context-lifecycle correction, both cases
passed: 384 frames / 1152 images were pixel-exact, and raw state and independently
reconstructed sites were exact. Earlier failures remain archived.

`same_state_diagnostic.py` then completed 192 fixed cells, reusing 32 historical
outputs. New calls total exactly 172: 160 diagnostic calls (including four cached
control baselines), four historical fixed-input controls and eight rerender
controls. All model controls were exact. Full seven-frame history is retained for
the original four-frame sampler; cross-view intervention changes only sampled
external images. No generated action is executed. Native complete velocity traces,
input hashes, seeds and a reserve-before-dispatch ledger are retained. The ledger
rejects automatic resume and carries prior attempts against the cumulative cap.

Pure factorial contrasts live in `analysis/same_state.py`; channel metrics retain
the original score definition. `reports/paper1/same_state_report.py` consumes the
completed artifacts. State/history, view and interaction effects are descriptive
for these two selected cases, not new closed-loop labels or proof of an improved
selector. Cabinet late-stage effects strongly involve the gripper; sink effects
are mostly translational and view preferences depend on the origin state/history.
Do not turn these findings into universal channel weights.
