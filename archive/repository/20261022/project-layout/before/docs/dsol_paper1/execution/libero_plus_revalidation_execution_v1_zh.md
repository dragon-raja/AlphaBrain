# LIBERO-Plus 视角重验证执行协议 v1

状态：`PARTIAL_QUICK_GATE_EVIDENCE_ONLY`
日期：`2026-08-18`
职责：在 RoboCasa Target 迁移期间，先用 LIBERO-Plus 跑通修正后的 Phase B/M0/M1 证据链。

> 2026-08-19 审计：七组 seed-41 Broad32 快速训练和被动闭环已完成；自然
> M0/M1 只完成了两任务、三模型的位姿 support 子门。Broad64、曝光量、正式
> 多 seed、构造型 Blind-Reveal、全方法 M1 与新版 Accel 尚未完成。完整状态见
> [view_revalidation_full_program_status_20260819_zh.md](view_revalidation_full_program_status_20260819_zh.md)。

## 2026-08-18 七组训练链路验收

基于官方固定 revision HDF5 恢复的 400 个 episode、38,193 个 exact-state 训练窗口，七个预算匹配训练组均完成 8 卡、全局 32 图像样本、20 optimizer-step smoke：Canonical-unique、Canonical-repeat、ImageAug-unique、Broad-unpaired-practical、Broad-unpaired-state-matched、Broad-paired-FM 与 Broad-paired-consistency。

全部运行达到 20/20 step、`examples_seen=640`，无 traceback。该结果只证明数据、Visual-LoRA、DDP、严格 batch 记账与 paired objective 链路可运行，不构成方法优劣结论。统一预算校准、exact-state 闭环工程 smoke 与 M0 可见性工程 smoke 均已通过，下一门为七组统一预算训练与 paired 闭环比较。

## 2026-08-18 统一训练预算校准

校准只运行 `Canonical-unique, seed=41`，用于冻结所有训练 arm 共用的 optimizer-step 预算，不用于比较算法。验证集固定为 256 条 `val` 记录：8 个任务各 32 条，覆盖 29 个独立 episode，并在每个 episode 内均匀抽取时间位置。训练前和每 250 optimizer steps 使用相同 sample ledger、相同 flow-noise seed 评估 held-out FM loss；验证过程恢复 Python、NumPy、CPU Torch 和 CUDA RNG 状态，不改变训练顺序。

预注册的 3,000-step 校准已完整结束。固定 256-example held-out loss 在 step 2,000 达到最低值 `0.220598`；step 1,500 已进入最低值的 1% 范围；step 3,000 为 `0.230063`。quick-gate 对所有训练 arm 统一冻结为 2,000 optimizer steps，并保持 scheduler 总长度为 3,000 steps，使正式训练严格复用校准曲线的前 2,000-step 学习率轨迹。测试结果不得反向修改该预算。机器可读决策见 `configs/dsol_paper1/libero_training_budget_quick_gate_v1.json`。

## 2026-08-18 数据等价性边界

本轮不是用动作重放近似官方数据。四个 LIBERO suite 的 40 个 HDF5 来自官方仓库固定 revision，逐文件核对官方 LFS SHA-256；其中 2,000 条 demonstration、338,575 个 transition 的 simulator state 与 action 是正式真值。四套件 exact-state restore 的最大绝对误差均为 0。

当前 MuJoCo 3.2.3 与源数据生成 renderer 存在像素漂移，因此不能声称重渲染 RGB 与下载文件中的 RGB 逐 bit 相同。正式对照采用同一冻结 runtime，从相同官方 state 为 canonical 与所有变化视角统一重渲染；源 RGB 只用于来源与 renderer drift 审计。该设计保持物理状态和动作真值不变，并将 renderer 差异固定为所有训练臂共同条件，适合严格 same-state camera intervention。禁止混用源 RGB 与重渲染 RGB 比较方法。

## 2026-08-18 评测基础设施状态

- 完整闭环 quick-gate 协议：8 tasks、24 个独立 test episode、每组 7 个 camera/sensor 条件，共 168 条 episode；
- 每个 paired group 使用相同官方 XML、初始 simulator state、policy flow-noise seed 与任务 horizon；
- 已实现 AV1/WebM 视频、断点续跑、逐 episode JSONL、Wilson 区间和 paired group bootstrap；
- 已实现 160-state M0 visibility scan plan，覆盖 8 tasks、val/test 隔离和 4 个任务阶段；
- M0 首轮仍使用任务实体在实际启用相机中的等权可见像素，不输入决策信息或 Accel；
- `Broad-unpaired-state-matched` 与两个 paired arm 现在共享完全相同的 pair-level flow noise/time，保证 paired-FM 对比只改变第二张 RGB；只有 consistency arm 增加一致性损失。
- LIBERO Python 3.8.20 环境补充 `websockets==13.1`、`dm-tree==0.1.8`、`msgpack==1.1.0` 和 `imageio-ffmpeg==0.5.1`；后者固定使用系统 `/usr/bin/ffmpeg` 编码 AV1/WebM。
- exact-state 完整闭环工程 smoke 已完成 1 条 520-step episode、104 次正常重复规划并成功保存 AV1/WebM；该单条轨迹失败只用于链路验收，不参与方法结论。
- M0 可见性工程 smoke 已完成 8 个状态、576 条候选记录，所有候选均有效。Look-away 全部降低可见性，极端视角产生更大的正负可见性差；该小样本只证明测量链路与强干预可用。

## 1. 结论边界

本轮仍然重做旧 Phase B/M0/M1 的原研究问题，但修正窄视角、重复样本、配对未进入目标、信息差异弱和短闭环等设计问题。旧结果只作为窄视角 Legacy Anchor。

LIBERO-Plus 可以先回答：

1. 大范围相机覆盖是否优于旧的局部视角覆盖；
2. 在状态、曝光与 pose support 受控时，same-state pairing 是否有额外价值；
3. pairing 被显式一致性目标利用后是否继续提升；
4. 模型能否利用可见性更高的候选视角，而不只是容忍相机变化；
5. Blind、Look-away 与 Reveal 是否形成可进入完整闭环的强信息差异。

LIBERO-Plus 不替代 RoboCasa 的 Human300/Target、wrist shortcut、场景构造和最终跨分布结论。

## 2. 已就绪资源

| 资源 | 当前状态 | 数量 |
|---|---|---:|
| LIBERO-Plus runtime | 可渲染 | commit `4976dc30028e805ff8094b55501d532c48fec182` |
| Broad-unpaired RLDS | 可直接训练 | 2,876 episodes / 449,053 steps / 40 tasks |
| Broad-unpaired train split | 可直接训练 | 2,303 episodes / 361,006 steps |
| 外部相机 pose support | 已显著宽于 Legacy-8 | 1,178 个 rounded poses；train 为 948 groups |
| Factor-separated 子集 | 可作历史诊断 | 1,417 episodes / 172,459 steps / 9 tasks |
| Pi0.5 JAX / PyTorch checkpoint | 已就绪 | 两种格式均存在 |
| Renderer smoke | 通过 | external 与 wrist 均为 224x224 RGB |
| Broad-unpaired GPU 反传 smoke | 通过 | 2 optimizer steps，未保存权重 |
| 本地 state-action bootstrap | 可立即重渲染 | 7 LIBERO-Goal tasks / 35 episodes / 453 windows |
| 共享 canonical LeRobot v1 | 可直接复用 | 4 suites / 1,693 episodes / 约 1.8 GiB |
| 共享 canonical LeRobot MuJoCo 3.3.2 | 可直接复用 | 4 suites / 约 8.8 GiB |
| longjunyu canonical LeRobot v1 | 已复制并校验 | 15,353 files / 1,890,962,767 bytes / SHA-256 VERIFIED |

现有 broad-unpaired 数据不是旧 RoboCasa 的 8-cell 小扰动，不需要重新下载或生成。

本地 bootstrap 位于：

```text
/share/longjunyu/capt-vla/tapaq/chunk_world_tasks1_7_states25_29_v1.npz
```

它保存 `current_qpos`、`next_qpos`、`qpos_sequence`、`action_chunk`、当前 external/wrist RGB 和任务/episode 标识。35 条 episode 中 30 条成功，足以完成 pair renderer、DataLoader、consistency objective、可见性候选与小规模闭环的工程验证。它来自既有策略 rollout，独立 episode 和任务覆盖有限，因此只标记为 `BOOTSTRAP_DEBUG`，不能替代官方 demonstration 的 rapid/formal 结论。

共享 canonical 数据位于：

```text
/share/sunguoying/datasets/LEROBOT_LIBERO_DATA/
/share/zhaoguoyang/data/libero_mujoco3.3.2/
```

较小的完整副本已非破坏性复制到：

```text
/share/longjunyu/alphabrain/datasets/libero-canonical-lerobot-v1/
```

源/目标逐文件 SHA-256 manifest 一致；复制没有使用 `--delete`。正式代码应优先引用 longjunyu 副本，其他用户目录只保留为来源证据。

两者均覆盖 `libero_10`、`libero_goal`、`libero_object`、`libero_spatial`，可只读复用为 Canonical-unique、Canonical-repeat 和 ImageAug-unique 的数据来源。其公开特征包含 RGB、proprioception、action 和 episode/task identity，但没有 object-level full MuJoCo state，因此不能直接修改相机后重渲染同一帧。

## 3. 同状态重渲染数据的获取方式

当前 RLDS 只包含 RGB、action、state、language 和 episode camera extrinsics，不包含可恢复的完整 MuJoCo simulator state，因此不能从中反推 exact same-state pairs。

我们不缺 LIBERO-Plus 训练数据，也不缺原始 LIBERO 的 canonical RGB/action 数据。新增需求仅来自 strict same-state pairing：需要在同一个物理状态下改变相机并重新渲染。

### 3.1 立即 bootstrap：无需下载

先用本地 453 个 state-action windows 生成 canonical、broad、paired、Blind、Reveal 和 matched-control 视图，跑通 WP1-WP6 的缩小版。该阶段负责发现实现错误、确定显存、冻结格式和筛选候选，不负责论文结论。

### 3.2 正式数据源：官方原始 HDF5

正式数据生成只使用官方 `yifengzhu-hf/LIBERO-datasets` 原始 demonstration HDF5。下载固定到 revision：

```text
f13aa24a3da8c43c7225569f28c562979fa0e35a
```

范围固定为 `libero_10`、`libero_goal`、`libero_object`、`libero_spatial` 四套件，共 40 个 HDF5、`33,784,856,577` bytes（约 31.47 GiB），不下载与当前研究无关的 `libero_90`。目标目录为：

```text
/share/longjunyu/alphabrain/datasets/libero-original-hdf5-v1/
```

下载器采用匿名访问、断点续传、固定 revision，并逐文件核对预期字节数和官方 LFS SHA-256。只有全部 40 个文件通过校验后，才允许作为 formal exact-state pair 的来源。

### 3.3 诊断方案：从共享 LeRobot 确定性重放

对共享 LeRobot episode：

1. 从 task identity 和 episode-local index 恢复官方 canonical init state；
2. 在冻结的 LIBERO runtime、controller 与 control frequency 下重放 action；
3. 保存每步完整 simulator state/qpos；
4. 将重放的 canonical RGB 与共享视频逐帧比较；
5. 只有通过图像、proprioception、长度和终态成功一致性门的 episode 才可生成 same-state views。

确定性重放即使通过，也只证明当前 runtime/controller 的轨迹恢复能力，不替代正式 HDF5。重放不一致的 episode 必须标记失败，不能用近似 qpos 冒充 exact pair。以下 6-task 列表保留为恢复 smoke 与快速可视化审计子集：

| Suite | 任务 | 大小 |
|---|---|---:|
| libero_goal | open the top drawer and put the bowl inside | 0.948 GiB |
| libero_goal | put the cream cheese in the bowl | 0.499 GiB |
| libero_object | pick up the cream cheese and place it in the basket | 0.670 GiB |
| libero_spatial | pick up the black bowl in the top drawer and place it on the plate | 0.697 GiB |
| libero_10 | put the yellow and white mug in the microwave and close it | 1.407 GiB |
| libero_10 | pick up the book and place it in the back compartment of the caddy | 0.876 GiB |

合计约 `5.10 GiB`。这些任务覆盖容器内部、抽屉、微波炉、篮子和隔间，比随机选择更适合构造 Blind/Reveal 与可见性差异。下载后仍需按 episode/demonstration 划分 train/calibration/test。

完整 40-task 四套件已下载并逐文件校验：`libero_10=13,730,608,904` bytes、`libero_goal=6,373,112,875` bytes、`libero_object=7,444,084,034` bytes、`libero_spatial=6,237,050,764` bytes。全量包含 2,000 demos 和 338,575 transitions；40 文件 schema 全部通过，四个 suite 的 exact-state restore 最大误差均为 0。

源数据 RGB 与当前 MuJoCo 3.2.3 渲染栈存在小幅、可测的像素漂移，尤其是近景 wrist 相机。因此正式训练不混用源 RGB 与新渲染 RGB：canonical、unpaired、paired 和信息视角全部由同一冻结 runtime 从官方 state 重渲染，源 RGB 仅用于来源身份与渲染漂移审计。

### 3.4 Mac 备选

Mac 迁移仅作为官方直连持续失败时的备用通路，并须保持相同 revision 与逐文件 SHA。候选源为：

```text
/datasets/L2_Simulation/LIBERO
```

迁移前先输出目录树、文件数、总字节和扩展名统计；优先只迁移包含原始 demonstration、`states`、`actions`、任务元数据和必要 BDDL 的部分。正式目标目录固定为：

```text
/share/longjunyu/alphabrain/datasets/libero-original-hdf5-v1/
```

所有路径都必须生成逐文件 SHA-256 manifest 和 acquisition/replay receipt。不得读取认证信息，也不得覆盖现有 LIBERO-Plus 数据。

## 4. 新数据生成范围

### 4.1 不重新生成

- broad-unpaired practical；
- 现有 nominal/camera track；
- 已有 factor-separated 历史结果。

### 4.2 必须生成

1. `broad_state_matched_repeat`：同一状态、同一 broad RGB 精确重复；
2. `broad_paired_fm`：同一状态、同一动作、两个不同 broad 视角；
3. `broad_paired_consistency`：与 paired-FM 完全相同的数据，只改变训练目标；
4. `visibility_candidate_bank`：同状态下 canonical、matched-control、Reveal、Blind、Look-away；
5. `info_pose_support_train`：训练侧独立 episode/state 的正常 Reveal pose，不包含测试图像或测试状态。

黑相机、全黑和 Look-away 只用于评测与候选筛选，不进入普通动作监督训练。

## 5. 新计划的完整执行链

| WP | 核心问题 | 主要输出 |
|---|---|---|
| WP0 Legacy Anchor | 迁移后的旧结论是否可复现 | Official、旧 MV8、旧扰动数值锚点 |
| WP1 Coverage | 扩大 pose support 后，普通多视角数据能解决多少被动视角泛化 | Canonical-unique、ImageAug-unique、Broad-unpaired |
| WP2 Pairing | 在状态、pose 和曝光预算受控时，same-state pairing 本身是否有价值 | State-matched repeat vs Paired-FM |
| WP3 Objective | pairing 被显式利用后是否产生额外收益 | Paired-consistency vs Paired-FM |
| WP4 Visibility/M0 | 可见性分数能否稳定区分 Reveal、matched-control、Blind 和 Look-away | 同状态多视角图像、segmentation 与可见像素统计 |
| WP5 Full M1 | 信息视角是否改善完整闭环，而非只改变单步动作 | success、progress、Rescue、Harm、steps |
| WP6 Active-ready analysis | 候选选择是否偏向训练熟悉度或真正有信息/闭环更优的视角 | Accel/selector 与 visibility、train support、oracle 的关系 |

Accel 不参与 WP4 的信息视角定义。第一轮信息视角仍由任务实体在部署相机中的等权可见像素增量定义。黑相机、Look-away 和人工遮挡只用于评测、信息差异构造与候选筛选，不作为普通动作监督训练样本。

## 6. 快速门

目的：在一个连续运行日内得到方向性信号，不作为论文最终数字。

### 6.1 数据

- 6-10 tasks；
- 20k-50k exact same-state pairs；
- 每任务至少 50 个 M0 paired states；
- 训练、校准和测试按 demonstration/episode 隔离。

### 6.2 训练 arms

| Arm | 用途 |
|---|---|
| Official-frozen | 不继续训练的能力锚点 |
| Canonical-unique | 固定视角 continuation，排除 exact-repeat 混杂 |
| ImageAug-unique | 无重复样本的图像增强基线 |
| Broad-unpaired-practical | 大范围普通数据覆盖强基线 |
| Broad-state-matched-repeat | pairing 的严格状态/曝光/flow draw 控制 |
| Broad-paired-FM | 与 state-matched 共享 flow draw，只测第二视角不同 |
| Broad-paired-consistency | 与 paired-FM 同数据和 flow draw，只增加显式跨视角目标 |

快速门先用 seed 41 和统一更新步。所有训练 arms 使用相同 Pi0.5 初始化、LoRA 模块、optimizer、global batch、action horizon 与 checkpoint 选择规则。

### 6.3 评测

1. nominal ID；
2. LIBERO-Plus camera track；
3. Reveal / matched-control / Blind / Look-away；
4. wrist-on 与 wrist-masked 分开报告；
5. 小规模完整闭环，不使用短 action chunk 代替任务成功率。

## 7. 正式门

快速门通过后执行：

- 扩展到约 338k same-state pairs 或由冻结功效分析确定规模；
- seeds 41/42/43；
- paired-FM 与 paired-consistency 采用完全相同 pair ledger；
- 每个相机任务实例至少 3 rollouts；
- M0 覆盖 6-10 tasks、每任务 50-100 states；
- M1 从相同初态执行完整 horizon，并报告 success、progress、Rescue、Harm 和 completion steps；
- 统计单位为 task x initial-state/episode group，使用 paired group bootstrap 95% CI。

## 8. 算力与时间

以下是墙钟时间，不包含外部传输等待：

| 阶段 | 预计时间 | 说明 |
|---|---:|---|
| 官方 40-task HDF5 下载与验收 | 依网络吞吐 | 31.47 GiB；固定 revision，4 workers，可续传 |
| HDF5 state restore + 重放诊断 | 2-6 小时 | 先测 2 tasks x 5 episodes；重放不替代 HDF5 |
| pair generator、loader、consistency loss | 6-12 小时 | 含单元审计与 20-50 step smoke |
| 20k-50k pilot pairs | 2-6 小时 | 取决于渲染和共享盘吞吐 |
| seed-41 快速训练矩阵 | 3-6 小时 | 多 GPU 并行；先冻结共同 global batch |
| 快速闭环与统计 | 3-6 小时 | reduced camera track + M0/M1 pilot |
| 首个可信方向信号 | 18-30 小时 | source 及时到位时 |
| 约 338k 正式 pair 数据 | 6-18 小时 | 预计新增约 35-70 GB，编码决定大小 |
| 三 seed 正式训练 | 8-16 小时 | 约两波并行，consistency arm 更慢 |
| 正式 camera track + ID | 12-18 小时 | 依评测并行度变化 |
| Blind/Reveal M0 + 完整 M1 | 1-3 天 | 任务筛选、候选审计和完整闭环占主导 |
| 完整 LIBERO-Plus 证据链 | 4-7 天 | 连续运行且无数据/renderer blocker |

单次旧 visual-LoRA 33k-step 训练在本机历史耗时约 5.7-6.0 小时；一次 8-shard 全模型评测历史耗时约 54 分钟。上表为包含新 pair 数据和严格闭环后的保守预算。

## 9. 历史执行快照

以下列表保留 2026-08-18 启动时状态，不代表当前完整计划已经完成。当前权威状态
以 2026-08-19 完整审计文档为准。

- `PASS`：LIBERO-Plus EGL/render runtime；
- `PASS`：现有 broad-unpaired 数据索引与分布审计；
- `PASS`：Pi0.5 PyTorch + visual LoRA 的 2-step GPU backward；
- `READY`：本地 7-task / 453-window state-action bootstrap，可立即实现和测试 pairs；
- `READY`：共享盘 40-task canonical LeRobot 数据，可直接作为训练基线并尝试确定性 state recovery；
- `PASS`：paired-consistency 的 8 卡、global image batch 32、20-step smoke；
- `PASS`：40-task 官方 HDF5 下载与逐文件 SHA-256 验收；
- `PASS`：40 文件 schema 与四 suite exact-state restore；
- `RECORDED`：源 RGB 与当前 runtime 的 render drift，正式 arms 统一重渲染；
- `PASS`：38,193 条 exact-state quick-gate pairs，覆盖 8 tasks / 400 episodes / 32 Broad poses；
- `PASS`：pair collection 二次审计，128 条 state/action/robot-state 回指官方 HDF5 全部一致；
- `PASS`：六组 DataLoader 与 paired consistency objective；
- `PASS`：paired-FM 与 paired+consistency 的基础 FM loss 隔离检查（均为 `0.64474`）；
- `PASS`：七组 8 卡、global image batch 32、20-step smoke matrix；
- `RUNNING`：Canonical-unique 3,000-step 固定 held-out ledger 统一预算校准；
- `PENDING`：强 Blind/Reveal 可见性候选和完整闭环。

因此 LIBERO-Plus quick gate 已不再依赖 Mac 或新增下载。统一预算冻结后进入七组正式训练、扩大 Phase A、强 Blind/Reveal M0 和完整闭环 M1。Mac 可继续专注 Target 迁移。
