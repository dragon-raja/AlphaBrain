# LIBERO 维护入口

当前 Paper 1 代码归属、可复用范围与开源前欠账见[框架维护审查](../../docs/architecture/framework_readiness_zh.md)；结构合同见[共享搜索核心](../../docs/architecture/shared_search_core_zh.md)。

- 当前受控链：`continuous_authority.py` → 模型/任务适配；实验组合见 `prefix_campaign.py`、`prefix_continuation.py`、`prefix_evaluation.py`。
- 共享 Actor/Critic 与 PPO 在 `AlphaBrain/research/dsol/selectors/`；新结构使用显式统一 schema，历史模型结构仍保留，不能混标结果。
- RLDX 已有 `rldx_search_runtime.py` 与前缀 PPO 接入。下面“尚未接入”的文字是当时的历史记录，不是当前能力表。
- 当前源码不等于在途冻结源码；准入及机器状态以每次运行的绑定和活进程检查为准，不能复用下方历史主机状态。
- 旧 `baseline_dispatch.py` 在当前开发树明确关闭；不将历史静态派发作为新实验入口。

## 历史接入记录与上游使用说明

以下保留原说明的来源和时间边界，不作为当前派发许可或最新状态。

<!-- FastWAM candidate status: numerical admission is separate from formal evaluation. -->

前缀对照入口：`prefix_pilot.py`通过既有受控采集/恢复评测适配器接入，使用完整on-policy批次和显式预算；`score_prefix_runtime.py`保持操作输出不变。RLDX的真实时序与联合query特征由`rldx_search_runtime.py`提供，序列布局必须显式传入会话/Gym，不能冒充图像网格。真实入口准入见`prefix_admission.py`，其采集smoke不产生优化更新。历史入口仍需各自冻结许可，不因新接口存在而全局隔离。

# 🚀 LIBERO Evaluation

RLDX候选入口：`rldx_runtime.py`保留原生四帧双相机、16×64噪声和4步Euler，评分仅前8动作×7维；联合cognition tokens不能冒充独立相机特征。`controlled_rldx.py`负责数值一致性，`controlled_rldx_operation.py`负责完整工程轨迹及有限开发选位诊断，均复用center实时授权/账本。它们尚未接入PPO采集器；数值通过不代表选位有效、跨模型权重可直接迁移或GR1通过。持久证据：`runs/rldx-basic-validation-20261102/`。


2026-10-30原生特征入口：`model_adapters.py`注册FastWAM／GR00T，`native_feature_runtime.py`定义真实捕获合同，`native_view_source.py`接入公共采集；新schema复用受控collector／learner／recovery。旧源视觉桥保留在其历史协议中。扩展步骤与真实全轨迹验收边界见[原生模型接口](../../docs/architecture/distributed_execution_zh.md#原生模型特征接入与搜索训练接口)。

FastWAM 新入口：`fastwam_runtime.py`保留官方双相机像素、float64归一化、32步动作/10步重规划/旧权重shift=5；独立源视觉塔供PPO使用。`controlled_admission.py`通过现有center授权链执行最多两初态的数值对照，CPU模拟不构造真实模型。默认入口与直接加载器仍关闭；真实数值检查、视觉跨环境一致性和完整轨迹准入分别验收，不把CPU通过写成生产可用。证据在`runs/fastwam-fresh-only-20261028/runtime-adapter/`。

小批域适配：`controlled_training.py`、`controlled_learner.py`接入独立8回合/64回合schema，复用原center授权与完整批次发布；source/random actor与首批固定/窗口尺度均有CPU合成入口测试。FastWAM资产/独立环境准备在持久项目`runs/fastwam-production-20261028/`，其真实模型/图像/动作/噪声适配尚未准入，不能套用原生模型运行许可。

`controlled_campaign.py`支持显式GPU列表和每卡worker数，新模型不得继承旧原生8×2容量。它只连接任务实际使用的节点；`excluded_hosts`中的节点若出现在任务中直接拒绝，不转派。当前GNU/Video均暂停，新配置同时排除`gnu`和`video`，旧模板可保留但不触发其巡检。该机制不终止旧进程、不修改冻结绑定，不是全局隔离服务。

`coordinate_probe.py` 组合安装参考机制检查的既有受控队列，按schema支持历史小检查和完整12任务验证；显式 `--execute` 只接受冻结配置，先CPU队列、再计入预算的首块、最后完整配对面板。它不改URDF或相机运动学，不训练。搜索到操作执行通过 `continuous_evaluation.move_to_search_selection` 保持同一参考系下的位姿矩阵计算，不能以数学等价放宽逐值输入验收。完整验证还要求初始查询使用同一安装参考重放；机械回零独立于Accel最低值选位。已核清的历史调用仍计入全局上限；未知消耗不释放预留，不自动重试。

Paper 1 的当前受控入口：`controlled_campaign.py`复用标准center ledger、实时授权、GPU身份检查与驻留worker，配置提供路径；`reward_adaptation.py`组合已批准B/G/L短适配的完整批次屏障、CPU优化和配对恢复评测。启动需显式`--execute-authorized`与冻结配置，失败不自动重放。它们不是通用DDP/FSDP或批推理实现。历史命令不能代替当前运行许可。

## Paper 1 bounded prefix diagnostics

`prefix_replay.py` reuses the `window_probe.py` private batch16 policy bridge for
explicitly frozen initial/view/noise subsets. It verifies all saved flow, action
and Accel prefixes exactly, retaining failed checks; it records observations and
realized simulator motion without additional renders. It is not a new full-task
evaluation or an online camera-control method. Shared read-only recording lives
in `AlphaBrain/research/dsol/data/prefix_telemetry.py`; analysis and plotting live
in the corresponding `analysis/prefix_replay.py` and `plotting/prefix_replay.py`.
Imports do not start execution; the runtime requires an explicit config and `--execute`.

This document provides instructions for reproducing our **experimental results** with LIBERO.  
The evaluation process consists of two main parts:  

1. Setting up the `LIBERO` environment and dependencies.  
2. Running the evaluation by launching services in both `AlphaBrain` and `LIBERO` environments.  

We have verified that this workflow runs successfully on both **NVIDIA A100** and **RTX 4090** GPUs.  

---


## ⬇️ 0. Download Checkpoints


We provide a collection of pretrained checkpoints on Hugging Face to make community evaluation easier: [🤗 AlphaBrain/bench-libero](https://huggingface.co/collections/AlphaBrain/bench-libero). Their corresponding results on LIBERO are summarized in the table below.

### 📊 Experimental Results

| Model               | Steps | Epochs | Spatial | Object | Goal | Long  | Avg   |
|---------------------|-------|--------|---------|--------|------|-------|-------|
| $\pi_0$+FAST | -     | -      | 96.4    | 96.8   | 88.6 | 60.2  | 85.5  |
| $\pi_0$             | -     | -      | 96.8    | 98.8   | 95.8 | 85.2  | 94.1  |
| GR00T-N1.5 | 20K   | 203    | 92.0    | 92.0   | 86.0 | 76.0  | 86.5  |
| **Qwen2.5-VL-FAST** | 30K   | 9.54   | 97.3    | 97.2   | 96.1 | 90.2  | 95.2  |
| **Qwen2.5-VL-OFT**  | 30K   | 9.54   | 97.4    | 98.0   | 96.8 | 92.0  | 96.1  |
| **Qwen2.5-VL-GR00T**| 30K   | 9.54   | 97.8    | 98.2   | 94.6 | 90.8  | 95.4  |
| **Qwen3-VL-FAST**   | 30K   | 9.54   | 97.3    | 97.4   | 96.3 | 90.6  | 95.4  |
| **Qwen3-VL-OFT**    | 30K   | 9.54   | 97.8    | 98.6   | 96.2 | 93.8  | 96.6  |
| **Qwen3-VL-GR00T**  | 30K   | 9.54   | 97.8    | 98.8   | 97.4 | 92.0  | 96.5  |

We train one policy for all 4 suites. All
scores are averaged over 500 trials for each task suite (10 tasks × 50 episodes).

---


## 📦 1. Environment Setup

To set up the environment, please first follow the official [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO) to install the base `LIBERO` environment.  

⚠️ **Common issue:** LIBERO defaults to Python 3.8, but the syntax updates between 3.8 and 3.10 are substantial. **We verified that using Python 3.10 avoids many issues**. 


Afterwards, inside the `LIBERO` environment, install the following dependencies:  

```bash
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4
```

---

## 🚀 2. Evaluation Workflow

The evaluation should be run **from the repository root** using **two separate terminals**, one for each environment:  

- **AlphaBrain environment**: runs the inference server.  
- **LIBERO environment**: runs the simulation.  

### Step 1. Start the server (AlphaBrain environment)

In the first terminal, activate the `AlphaBrain` conda environment and run:  

```bash
bash benchmarks/LIBERO/eval/run_policy_server.sh
```

⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `benchmarks/LIBERO/eval/run_policy_server.sh`  


---

### Step 2. Start the simulation (LIBERO environment)

In the second terminal, activate the `LIBERO` conda environment and run:  

```bash
bash benchmarks/LIBERO/eval/eval_libero.sh
```
⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `eval_libero.sh` to load action unnormalization stats. 

Also ensure the environment variables at the top of `eval_libero.sh` are correctly set.

Finally, each result will also save a video for visualization, as shown below:

![Example](example.gif)

---


# 🚀 LIBERO Training

## 📦 Step 0: Download the training dataset
Download the datasets to the playground/Datasets/LEROBOT_LIBERO_DATA directory:
- [LIBERO-spatial](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot)
- [LIBERO-object](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot)
- [LIBERO-goal](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot)
- [LIBERO-10](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot)

And move `modality.json` to each `$LEROBOT_LIBERO_DATA/subset/meta/modality.json`.

You could quickly prepare these by running:
```bash
# Set DEST to the directory where you want to store the data
export DEST=/path/to/your/data/directory
bash benchmarks/LIBERO/data_preparation.sh
```


## 🚀 Step1: Start Training

Most of the required training files have been organized in [train_files](train_files).  

Please run the following command to start training:

```bash
bash benchmarks/LIBERO/train/run_libero_train.sh
```
⚠️ **Note:** Please ensure that you specify the correct path in `benchmarks/LIBERO/train/run_libero_train.sh`


2026-10-23：本工作树的连续评测/训练、驻留worker、共享VLA runtime与fleet启动边界已关闭生产执行，仅保留CPU准备/验证；旧冻结副本不受此限制，禁止将其当作恢复后启动许可。当前入口状态见 `docs/dsol_paper1/history/center_dispatch_hardening_20261023_zh.md`。

## Center live-authority adapter (2026-10-23)

`continuous_authority.py` binds one frozen paired/nominal block to the current
center session. It preserves the original scientific loops and checks the live
channel before model calls. Per-task locks and process-group identities prevent
replay while workers or simulator children remain. CPU contract mode follows the
same wire lifecycle without model imports. `continuous_authority_plan.py` composes
the standard queue, immutable bindings and explicit stage permits; its thin CLI
is `scripts/dsol_paper1/operations/manage_model_queue.py`.

Direct historical fleet/training commands remain closed in this release. Only
the bound worker scope admits the seven required paired/nominal/runtime/simulator
boundaries. A missing or draft permit does not start a model. Frozen copies are
preserved and remain outside OS-level enforcement.

`native_view_panel.py` 是原生固定状态响应面的新科学适配器，复用既有原生模型、相机移动和完整任务执行；仅经 `continuous_authority` 的显式 `native_view_panel_v1` 绑定与实时授权进入。旧672条执行及冻结版本不变。

## Center resident execution candidate (CPU verified, production off)

`continuous_authority.py` supports opt-in `resident_workers` for native panel blocks. It retains a single immutable policy through `execution/resident_runtime.py`, but every task receives a new ledger reservation, GO, live per-call authorization, result validation and settlement. A disconnected or failed session cannot reconnect or replay automatically.

Optional `resident_simulator` uses `simulator_session.py` and `search_simulator.serve_resident` to retain the interpreter only: each trajectory creates/closes a fresh environment and renderer. Persistent logs live outside immutable block results. Physics-state caching, GPU batching and PPO learner changes are not enabled. Three slots per GPU are configurable, not a capacity certificate. New frozen source/host bindings and numerical/trajectory admission remain necessary before production.

`native_view_panel` 同时支持显式 `NATIVE_FULL40_BASELINE_V1`：40任务、400初态、27视角、2执行种子，独立于旧768协议。全状态共用一个明确sim配置，驻留时仍逐轨迹新建环境。`LiveEpisodes` 可接收独立reward_scale供新的校准训练控制器使用；旧入口默认保留原分母，尚未因此开放新的PPO生产派发。

当前原生扩展适配：`controlled_training.py`负责账本内校准/采样，`controlled_learner.py`只做完整批次CPU更新，`controlled_recovery.py`负责最终策略配对评测。均由continuous_authority绑定；历史直接入口不作为新许可。

`reward_adaptation.py`通过显式协议处理短期暖启动与完整G从零训练；完整G为64个32回合屏障，3种子/48块，随后完整9600搜索。实际CPU历史B第一更新复现、首批真实采样配对与首块操作门槛均保留；旧pilot恢复证明不得用于新完整运行。

`controlled_target_evaluation.py` 接入 FastWAM 32/10 完整操作与严格配对轨迹验证；`local_search_rollout.py` 共享操作循环，旧 π0.5 10/5 不变。

`controlled_adaptation.py` 组合8回合正式采集、CPU学习器及不可变端点；`controlled_target_recovery.py` 执行目标模型的预定少量适配开发评测，使用同一授权/预算入口。

`controlled_adaptation_evaluation.py` 组装预定0/8/32/64回合端点和共享基线，首批精确比较通过后才复用重合端点；运行结果按配对留存。

目标操作/恢复入口的 `execution_hosts` 显式绑定资源，默认Fresh；端点组装器允许1–8搜索起点的短块和预先绑定的主机轮换，不改变科学行、噪声或预算。临时Video最初以两机各7卡接续；后续GPU7须独立核实保活身份、实测峰值和显存余量，不能套用旧并发配置。断联沿用全阶段停止和显式审计，不提供自动迁移/重放。独立运行快照与准入证据见持久 `runs/fastwam-small-validation-20261029/video-reactivation/`。

`target_recovery_resume.py` 只接受哈希绑定的显式停止/审计证明。完整轨迹复用，部分轨迹以保存的动作预测重建同一逻辑回合；逐次核对输入、噪声与已知终止，缺失输出才新增计费，历史不明消耗不退款。随机选位对照重新连接相机时，`RenderedViewSource(current_angles=...)` 必须与实际回执位置吻合，不能假设已回零。当前修复证据和16卡分区执行入口位于持久 `runs/fastwam-small-validation-20261029/evaluation-repair/`。CPU实际入口按主机/后端机制验证，完整科学清单仍全量校验，无需对每一科学行重复纯模拟任务。
# 借用计算节点的资源绑定

`controlled_target_evaluation.execution_hosts` 接受显式节点名，节点必须同时存在于运行清单、节点身份模板和 SSH 映射中，且不在排除列表。节点名本身不是运行许可；新节点仍需解释器/源码/共享输入/保活/实际轨迹验收。科学任务身份不随节点改变；在途任务不得自动迁移。旧冻结版本和结果保持不变。


`controlled_target_panel.py` 是固定候选空间诊断入口：预登记缺失位置的两成员评分与可选两次完整操作，独立于PPO访问轨迹；共享驻留模型、动作循环、实时单点许可与预算账本。复用结果另附哈希来源，未测标签不作失败，预算与旧恢复实验隔离。

`groot_flow_capture.py` 为固定 FluxVLA N1.5 源码生成独立 CUDA Graph 记录候选，保留原生计算并以预分配缓冲区记录每步状态/速度和 replay 代次。旧 Python hook 不等于 graph replay 记录。CPU 原语测试和源码还原检查不代表真实模型输出/轨迹准入；不提供直接推理或派发入口。原始第三方源码、候选补丁、哈希及验证证据位于持久 `runs/groot-spatial-readiness-20261029/graph-capture-candidate/`。

RLDX完整空间补齐：`rldx_spatial_panel.py`仅执行冻结missing清单，复用审计后的旧操作与P3/P4轨迹；每块不超过4条操作，模型通过标准resident lease复用，仿真解释器复用但每条操作新建物理环境。失败lease不可续用；CPU与原生完整轨迹都需验收。`data/spatial_completion.py`按科学身份去重，重复结果不一致拒绝复用，不按成功筛选。证据在 `runs/rldx-spatial-completion-20261102/`。


`prefix_resumption.py`组合完整前缀主线的显式PPO边界接续：核对已接受权重、原owner退出、零派发审计与旧预算，再调用冻结采集版本和原CPU learner。原优化器/RNG/G状态保留；资源交接等待旧评测完整结算与实际GPU空闲。它不重连、不重放未知阶段，不替代模型/主机/suite实际验收。
