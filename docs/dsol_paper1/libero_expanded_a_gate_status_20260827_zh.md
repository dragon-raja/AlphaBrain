# LIBERO 扩大 A 实验准入状态（2026-08-27）

## 结论

扩大 A 实验已经具备两种不同状态，必须分开解释：

1. **自然场景 4 任务 pilot：可以进入完整闭环。** 自动筛选、20 组平衡人工图像审计和协议冻结均已通过。
2. **正式 8 任务扩大 A：仍为 HOLD。** 只有 4/8 个任务在冻结阈值下产生了完整的 Strong-info、Matched-control、Blind 和 Look-away 条件，未达到至少 6 个任务的正式门槛。

不得降低可见性阈值或放宽 matched-control 约束来补齐任务数。正式实验需要继续构造更强的 Blind-Reveal 物理状态或场景。

## 扩大扫描

| 项目 | 数值 |
|---|---:|
| 任务数 | 8 |
| 每任务 validation episode | 2 |
| 每任务 test episode | 3 |
| 每条 episode 的阶段采样 | 10 |
| 状态总数 | 400 |
| 每状态候选视角 | 88 |
| 无效渲染 | 0 |
| validation/test episode 重叠 | 0 |

每个状态包含 canonical、32 个 broad held-out、24 个 wide extrapolation、24 个诊断轨道/极端/回避视角以及 3 个传感器控制条件。阈值只在 validation episode 上冻结，并仅在 test episode 上应用一次。

## 正式 8 任务门

| 任务 | 通过 test episode | 严格候选状态 | 状态 |
|---|---:|---:|---|
| wine rack | 3/3 | 7 | PASS |
| bowl → bottom drawer | 3/3 | 14 | PASS |
| mug → microwave | 3/3 | 12 | PASS |
| bowl → plate（spatial） | 3/3 | 7 | PASS |
| cream cheese → bowl | 0/3 | 0 | HOLD |
| bowl → top drawer | 0/3 | 0 | HOLD |
| book → caddy | 0/3 | 0 | HOLD |
| cream cheese → basket | 0/3 | 0 | HOLD |

阶段采样从 4 个增加到 10 个后，严格候选由 20 增至 40，但通过任务仍为 4 个。这说明失败原因不是帧采样不足，而是自然任务缺少足够强且可配对控制的可见性变化。

## Natural-4 Pilot

Natural-4 pilot 使用完全相同的冻结阈值，只显式限制在四个自然通过任务；它不是对正式 8 任务门的替代。

| 检查 | 结果 |
|---|---|
| 自动候选筛选 | PASS，40 个候选状态 |
| test episode 覆盖 | 4 任务 × 3 episode |
| 平衡人工审计 | PASS，4 任务 × 5 状态 |
| 渲染有效性 | 20/20 组通过 |
| Strong-info 可见增益 | 20/20 组人工确认 |
| Matched-control | 20/20 组人工确认 |
| Blind / Look-away | 20/20 组人工确认 |
| 闭环协议 | PASS，20 状态 × 10 条件 = 200 episodes |

酒架任务的自然可见性增益约为 1.4 个百分点，虽然肉眼可辨，但仍属于弱自然机会。该任务可用于 pilot，不应被描述为强因果 Blind-Reveal 证据。

## 接下来

1. 先运行 Broad64 practical primary model 的 20-episode 协议 smoke，验证状态恢复、十条件配对和输出链路。
2. smoke 通过后运行 200-episode natural-4 完整闭环，报告任务成功率、阶段进度、Rescue/Harm 与 information specificity；统计单位为 source episode，不把同 episode 的帧当作独立样本。
3. 并行针对四个失败任务构造更强的评测状态：保持训练数据不变，仅增强评测时的 Blind-Reveal 信息差；全黑相机只作为负对照，不进入正常训练。
4. 至少新增两个独立任务通过相同冻结门后，才启动正式多模型、跨 seed 扩大 A 比较。

## 关键产物

- 正式扫描计划：`configs/dsol_paper1/libero_expanded_a_scan_plan_v1.json`
- 正式筛选规则：`configs/dsol_paper1/libero_expanded_a_selection_v1.json`
- Natural-4 pilot 规则：`configs/dsol_paper1/libero_expanded_a_natural4_pilot_selection_v1.json`
- 正式任务构造计划：`configs/dsol_paper1/libero_expanded_a_constructed_extension_v1.json`
- 正式筛选结果：`/share/longjunyu/alphabrain/experiments/dsol-libero-expanded-a-v1/selection/automated_selection_400_v1.json`
- Natural-4 筛选结果：`/share/longjunyu/alphabrain/experiments/dsol-libero-expanded-a-v1/selection/natural4_pilot_selection_400_v1.json`
- 人工审计：`/share/longjunyu/alphabrain/experiments/dsol-libero-expanded-a-v1/manual_audit_natural4_v1/manual_visual_audit_v1.json`
- 冻结闭环协议：`/share/longjunyu/alphabrain/experiments/dsol-libero-expanded-a-v1/constructed_m1_natural4_pilot_protocol_v1.json`
