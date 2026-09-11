# LIBERO Expanded A：构造式 Strong-info 第一轮闭环 Gate

## 1. 本轮回答的问题

旧自然场景中的视角变化没有形成足够强的信息差，因此本轮只改变评测场景和相机观察，构造可审计的 Blind–Reveal 条件，回答两个彼此独立的问题：

1. 能否在不使用策略输出的前提下，按任务实体可见性预先选出 Strong-info 与等幅 Matched-control？
2. 当前 Broad-64 Pi0.5 是否已经能把更高可见性转化为更高闭环成功率？

Strong-info 的选择不使用闭环成功结果。验证集冻结视角方向，测试集只复用冻结规则；train、validation、test episode 相互独立。

## 2. 构造与数据定义

- 任务：`put_the_cream_cheese_in_the_bowl` 与抽屉取碗放盘任务。
- 物理状态：原始 LIBERO HDF5 simulator state。
- 场景干预：加入静态、无碰撞、无关节的视觉遮挡板；同一状态的所有相机条件使用完全相同的场景 XML。
- 相机候选：围绕当前任务实体中心生成左右镜像的 task-centric orbit pose；一对候选具有相同平移幅度、俯仰角和半径。
- Strong-info：验证集上可见性增量较高的一侧。
- Matched-control：同一镜像 pair 中位移等幅、可见性增量接近零或为负的一侧。
- 黑屏、盲视角、遮挡场景只用于评测，不进入训练。

可见性保持原定义：对任务实体和相机等权平均其实例分割可见像素占整幅图像的比例。

## 3. 可见性 Gate

| 任务 | 冻结 pair | Strong-info 方向 | Strong 增量 | Control 增量 | 信息特异性 |
|---|---|---|---:|---:|---:|
| cream cheese → bowl | side30_near | negative | +0.555 pp | -0.037 pp | +0.592 pp |
| drawer bowl → plate | side45_near | positive | +0.905 pp | -0.304 pp | +1.209 pp |

独立测试集共 12 个状态，其中 10/12 同时满足 Strong 增量阈值与 Control 容差。该结果证明本轮成功构造了方向特异的可见性干预，但尚不证明策略会利用新增证据。

人工审计图：

![Canonical、Strong-info 与 Matched-control 对照](/share/longjunyu/alphabrain/experiments/dsol-libero-expanded-a-v1/constructed_taskcentric_test_v1/manual_audit/selected_roles_contact_sheet.png)

## 4. 第一轮完整闭环结果

策略为已完成 2,000 step 训练的 Broad-64 practical Pi0.5。每个状态从同一 simulator state 开始，使用相同控制预算，只改变外部相机位姿或相机通道掩码。

| 条件 | 成功 | 成功率 | 严格可见性子集 |
|---|---:|---:|---:|
| Canonical，外部 + wrist | 10/12 | 83.3% | 90.0% |
| Strong-info，外部 + wrist | 9/12 | 75.0% | 80.0% |
| Matched-control，外部 + wrist | 8/12 | 66.7% | 70.0% |
| Canonical，wrist-only | 7/12 | 58.3% | 60.0% |
| Canonical/Strong/Control，external-only | 0/12 | 0.0% | 0.0% |
| 全相机黑屏 | 0/12 | 0.0% | 0.0% |

配对 source-episode bootstrap：

| 对比 | 点估计 | 95% CI | Rescue / Harm |
|---|---:|---:|---:|
| Strong-info − Canonical | -8.3 pp | [-33.3, +16.7] | 1 / 2 |
| Strong-info − Matched-control | +8.3 pp | [-33.3, +41.7] | 3 / 2 |
| Wrist-only − Canonical | -25.0 pp | [-41.7, -8.3] | 1 / 4 |

结果图：

![构造式视角闭环成功率](/share/longjunyu/alphabrain/experiments/dsol-libero-expanded-a-v1/constructed_taskcentric_test_v1/closed_loop_broad64_practical_v1/analysis_constructed/condition_success.png)

## 5. 当前结论

1. **数据构造成立。** Strong-info 在独立测试状态上确实比 canonical 看见更多任务实体，且 matched-control 控制了相机移动幅度。
2. **当前策略尚未把新增可见性转化为稳定收益。** Strong-info 没有超过 canonical；相对 control 的 +8.3 pp 仅是方向性信号，区间很宽。
3. **差异不能归因于“加入某种训练数据导致 Strong-info 变差”。** 本轮所有条件使用同一个既有 Broad-64 checkpoint，没有新增训练。
4. **存在明显 wrist shortcut / 双视角依赖。** external-only 全部失败，wrist-only 仍有 58.3%，而双视角达到 83.3%。当前模型既不具备独立外部视角能力，也不能据此否定外部信息价值。
5. **不能按测试成功率重新挑 Strong-info。** 这会把待检验结论写入测试集，产生结果选择偏差。

## 6. 下一 Gate：Info-pose-support

下一步使用相同初始化、相同 train episode、相同 batch、step、优化器和 LoRA 配置训练两个匹配数据臂：

| 数据臂 | 新增训练观察 | 作用 |
|---|---|---|
| Matched continuation | Broad-64 companion + 冻结 pair 的 control 方向 | 控制继续训练、任务分布和曝光量 |
| Info-pose-support | 同一个 Broad-64 companion + 冻结 pair 的 Strong-info 方向 | 只增加信息视角 pose support |

训练不包含测试遮挡板、Blind、Look-away 或黑屏。训练后复用当前冻结测试协议，主要检验：

1. Strong-info 是否超过 matched-control；
2. Strong-info 是否达到或超过 canonical；
3. canonical 是否明显退化；
4. external-only 是否从 0% 恢复；
5. 收益是否跨任务、跨 seed 稳定。

若 Info-pose-support 仍不能改善 Strong-info，则应进一步构造具有**决策分支差异**的任务，而不是继续按可见性增量筛选同类状态；若其显著改善，则说明第一轮失败主要来自 informative pose OOD，而不是新增视觉证据本身无用。

## 7. Info-pose-support 正式训练

两组数据均由相同的 6,441 个 simulator-state/action window 构成：

- train/val/test：5,503 / 503 / 435；
- 每条记录共享 canonical、wrist、Broad-64 companion、状态和动作；
- `pose_a` 分别为验证冻结的 Strong-info 或 Matched-control 方向；
- 6,441/6,441 条 task-centric 图像不同；
- canonical 与 Broad companion 图像全部 bit-exact；
- wrist 有 2/6,441 条出现 EGL 栅格化差异，最多 9 个像素、最大强度差 1/255，均在预注册容差内；
- 未注入遮挡板、Blind、Look-away 或黑屏训练样本。

两臂都从同一个 Broad-64 checkpoint 继续训练 600 steps，global batch 32，约 1.75 exposure epochs；Visual-LoRA、action expert、优化器、seed 和数据顺序一致。

数据配对审计：`/share/longjunyu/alphabrain/datasets/dsol-libero-taskcentric-support-v1/pair_audit.json`

## 8. 冻结闭环复评

| 模型 | Canonical | Strong-info | Matched-control | Wrist-only | External-only | Blackout |
|---|---:|---:|---:|---:|---:|---:|
| 训练前 Broad-64 | 83.3% | 75.0% | 66.7% | 58.3% | 0.0% | 0.0% |
| Control-support | 75.0% | 75.0% | 58.3% | 58.3% | 0.0% | 0.0% |
| Info-pose-support | 91.7% | 75.0% | 66.7% | 75.0% | 0.0% | 0.0% |

![三模型冻结协议对比](/share/longjunyu/alphabrain/experiments/dsol-libero-taskcentric-support-v1/cross_model_analysis/main_conditions.png)

关键配对效应：

| 效应 | 点估计 | 95% CI |
|---|---:|---:|
| Info − Control：Strong-info | 0.0 pp | [0.0, 0.0] |
| Info − Baseline：Strong-info | 0.0 pp | [-25.0, +33.3] |
| (Strong−Canonical) Info − Control | -16.7 pp | [-25.0, 0.0] |
| (Strong−Control-view) Info − Control | -8.3 pp | [-25.0, 0.0] |

任务拆分显示：

- cream-cheese 任务在两个 continuation 模型上三个双相机条件均为 6/6，已经饱和，不能提供信息视角辨识力；
- drawer-bowl 任务中 Info 模型为 Canonical 5/6、Strong 3/6、Control 2/6；Strong 高于 Control，但仍低于 Canonical；
- 两个模型的 external-only 均为 0/12，说明 exact external pose support 没有消除固定 wrist / 双视角依赖。

### Gate 结论

`INFO_POSE_SUPPORT_GATE = FAIL`

本轮排除了“Strong-info 仅因相机位姿 OOD 而失败”这一解释。当前可见性提升虽然真实存在，但任务状态没有形成足够强的动作分支需求；一个任务已饱和，另一个任务仍可由 canonical+wrist 完成。第一轮闭环和本轮训练均不能作为 Strong-info 行为收益的正证据。

## 9. 下一轮构造要求

下一轮不按策略成功结果挑样本，而按物理状态和可见性预先构造更强的 A-style 数据：

1. canonical 对关键任务实体的可见性必须显著低于当前阈值；
2. Strong-info 必须产生更大的绝对与相对可见性增量；
3. Matched-control 保持等幅相机移动，但不暴露关键实体；
4. 同一任务包含多个隐藏目标位置或容器状态，使后续正确动作确实不同；
5. 优先使用初始阶段和开环前状态，避免模型已从历史动作推断目标位置；
6. wrist 仍保留为主设置，但必须选择 wrist 同样无法提前看见关键证据的任务；wrist-off 仅作机制消融；
7. Blackout、Look-away 和极端盲视角继续只用于评测，不进入训练；
8. 先用可见性和人工审计冻结数据，再运行策略，禁止以 `Strong > Canonical` 作为样本入选条件。

只有在这一更强数据 Gate 上出现可复现的 `Strong > Canonical` 与 `Strong > Matched-control`，才进入更大任务规模和多 seed；否则结论应收缩为“实体可见性不是足够的 view-value 定义”。

## 10. 强可见性 A2 Gate：Book--Caddy

为排除“第一轮信息增量太小”，新增不参与碰撞的静态遮挡板，并将外部候选相机移近任务区域。状态和候选仅按 split、轨迹阶段及可见性阈值筛选，不使用任何策略输出：

- validation：3 个合格状态、2 个 source episodes；
- test：7 个合格状态、4 个 source episodes；
- canonical 外部相机任务实体可见性不超过 0.5%；
- canonical wrist 可见性不超过 2%；
- Strong-info 相对 canonical 至少增加 5%；
- 同一镜像 pair 的 Matched-control 相对 canonical 变化不超过 2%。

实际入选 test 状态中，Strong 增量约为 `+13.9` 至 `+15.4 pp`。人工审计确认 7/7 渲染有效，遮挡板不改变 MuJoCo 状态维度，也不参与碰撞。

![A2 Gate 人工审核图](/share/longjunyu/alphabrain/experiments/dsol-libero-expanded-a-v1/strong_information_gate_v4/manual_audit_contact_sheet.png)

使用同一个 Broad-64 checkpoint 运行 56 条完整闭环，`K=5`，每个状态的八个条件共享相同 simulator state：

| 条件 | 状态成功率 | source-episode 宏平均 |
|---|---:|---:|
| Canonical，外部 + wrist | 85.7% | 87.5% |
| Strong-info，外部 + wrist | 85.7% | 87.5% |
| Matched-control，外部 + wrist | 100.0% | 100.0% |
| Canonical，wrist-only | 100.0% | 100.0% |
| Canonical/Strong/Control，external-only | 0.0% | 0.0% |
| 全相机黑屏 | 0.0% | 0.0% |

配对 source-episode bootstrap：

| 对比 | 点估计 | 95% CI | State Rescue / Harm |
|---|---:|---:|---:|
| Strong-info − Canonical | 0.0 pp | [-37.5, +37.5] | 1 / 1 |
| Strong-info − Matched-control | -12.5 pp | [-37.5, 0.0] | 0 / 1 |
| Wrist-only − Canonical | +12.5 pp | [0.0, +37.5] | 1 / 0 |
| Blackout − Canonical | -87.5 pp | [-100.0, -62.5] | 0 / 6 |

![A2 Gate 完整闭环](/share/longjunyu/alphabrain/experiments/dsol-libero-expanded-a-v1/strong_information_gate_v4/closed_loop_broad64_v1/analysis_constructed/condition_success.png)

### A2 结论

`STRONG_VISIBILITY_A2_GATE = FAIL`

本轮已把 Strong 可见性增量从第一轮约 `+0.5--1.6 pp` 提高到约 `+14 pp`，因此不能再把负结果归因于视角变化或像素增量过小。但该任务仍可由 wrist-only 以 100% 完成，说明策略可以在闭环运动中等待腕部相机获得证据；外部 Strong 视角没有成为必要条件。Matched-control 的高成功率也表明，相机位姿本身仍可能影响行为，实体像素增量不是充分的 view value。

下一轮停止继续调 Book--Caddy 相机参数。新构造必须同时具备：

1. 同一任务包含至少两个隐藏目标位置或容器状态；
2. 两个隐藏状态要求不同的早期正确动作；
3. canonical 与初始 wrist 均无法区分隐藏状态；
4. Strong-info 能在动作执行前区分，Matched-control 仍不能；
5. 错误早期分支产生可测时间或成功代价，但评测仍保持完整闭环；
6. 候选选择继续使用当前实体可见性定义，不使用策略成功率或决策信息打分。

只有该双状态构造的可见性、人工审核和 scripted expert 门通过后，才运行下一次策略闭环。
