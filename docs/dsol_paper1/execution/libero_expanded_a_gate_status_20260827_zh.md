# LIBERO Expanded A 准入状态（2026-08-27）

## 当前结论

Expanded A 已分成两个不可混用的层级：

1. **Natural-4 prefix pilot：PASS，可进入开发性完整闭环。**
2. **Formal expanded A：HOLD。** 目前只有 4/8 个自然任务通过，未达到至少 6 个任务的正式门槛。

旧 Natural-4 的 200-episode 运行包含轨迹 75%--95% 的晚期状态，其中 spatial 任务即使全黑相机也可在 1--4 步成功。该运行仅用于发现协议问题，所有成功率和 information-specificity 数值均不得作为科学结果。

## 修复后的 eligibility

候选状态必须同时满足：

- 恢复物理状态后，LIBERO 正式成功条件为 false；
- `stage_fraction <= 0.70`，即只保留预设 5%、15%、...、65% 采样档，不使用 75%、85%、95% 晚期状态。

阶段上限在 validation 阈值冻结和 test 候选应用之前执行。它用于排除接近终点、无需可靠视觉也可能自然成功的状态，不根据 test 闭环结果调参。

| 排除原因 | 状态数 |
|---|---:|
| 初始状态已经成功 | 29 |
| 超过前 70% 轨迹阶段 | 91 |
| 合计 | 120 / 400 |

## 扩大扫描身份

| 项目 | 数值 |
|---|---:|
| 任务数 | 8 |
| validation / test episode | 每任务 2 / 3 |
| 每条 episode 阶段采样 | 10 |
| 状态总数 | 400 |
| 每状态候选视角 | 88 |
| 无效渲染 | 0 |
| validation/test episode 重叠 | 0 |

每个状态包含 canonical、32 个 broad held-out、24 个 wide extrapolation、诊断性极端/回避视角和传感器控制。可见性阈值只在合格的 validation 状态上冻结，并仅在合格的 test 状态上应用一次。

## Prefix-filtered 正式筛选

| 任务 | 有完整候选的 test episode | 候选状态 | 状态 |
|---|---:|---:|---|
| wine bottle -> rack | 2/3 | 8 | PASS |
| book -> caddy | 3/3 | 6 | PASS |
| bowl -> bottom drawer | 3/3 | 9 | PASS |
| mug -> microwave | 2/3 | 8 | PASS |
| cream cheese -> bowl | 0/3 | 0 | HOLD |
| bowl -> top drawer | 0/3 | 0 | HOLD |
| cream cheese -> basket | 0/3 | 0 | HOLD |
| bowl -> plate (spatial) | 0/3 | 0 | HOLD |

总计 31 个 test 候选状态。spatial 任务仅在 75% 之后形成候选，已被完整排除。正式门要求至少 6 个任务，因此不得通过降低可见性阈值或重新纳入晚期状态补齐任务数。

## Natural-4 人工审计

从上述 31 个状态中按任务平衡渲染 20 组，每组同时展示 canonical、strong-info、matched-control、blind、look-away、wrist/external blackout 和全黑控制。

| 任务 | 审计状态 | source episode | 可见性增量特征 |
|---|---:|---:|---|
| wine bottle -> rack | 5 | 2 | 约 1.3--1.6pp，弱自然机会 |
| book -> caddy | 5 | 3 | 约 0.5--1.4pp，弱自然机会 |
| bowl -> bottom drawer | 5 | 3 | 约 5pp，差异清晰 |
| mug -> microwave | 5 | 2 | 约 5pp，差异清晰 |

20/20 组通过人工渲染审计；未发现图像损坏，matched-control 位移可比，Blind/Look-away 降低任务实体可见性。wine 和 book 只能称为自然机会机制样本，不得称为强因果 Blind--Reveal 构造。

冻结闭环协议包含 20 个状态、10 个条件，共 200 episodes。统计单位为 source episode；同一 episode 的多个阶段状态必须聚类统计。

## Natural-4 prefix 闭环结果

Broad64 practical primary model 已完成 200/200 episodes。20 个状态均有完整 10 条条件，组内初始物理哈希不一致为 0，初始成功状态为 0；视频为 448 x 224 AV1/WebM。

| 条件 | 状态成功数 | 状态成功率 | source-episode 宏平均 |
|---|---:|---:|---:|
| canonical，外部 + wrist | 16/20 | 80% | 85.0% |
| strong-info，外部 + wrist | 14/20 | 70% | 76.7% |
| matched-control，外部 + wrist | 11/20 | 55% | 58.3% |
| blind，外部 + wrist | 12/20 | 60% | 60.0% |
| canonical，wrist-only | 10/20 | 50% | 51.7% |
| all-camera blackout | 0/20 | 0% | 0% |

配对 source-episode bootstrap 的主要差值：

- strong-info - canonical：-8.3pp，95% CI [-20.0, 0.0]；
- strong-info - matched-control：+18.3pp，95% CI [-11.7, +50.0]；
- strong-info - blind：+16.7pp，95% CI [-13.3, +45.0]；
- blackout - canonical：-85.0pp，95% CI [-100.0, -68.3]。

全黑成功率从旧污染运行的 30% 降为 0%，证明 prefix eligibility 修复有效。strong-info 相对 matched-control 有正点估计，但区间较宽且相对 canonical 未提升；当前自然场景结果只能支持“信息视角可能缓解部分普通视角位移损失”，不能支持“信息视角提高标准闭环成功率”。正式裁决必须依赖更强、独立构造的 Blind--Reveal 任务。

## 数据是否还需扩展

### 已完成

- Natural-4 prefix 的 Broad64 practical 完整闭环；
- full-task success、Rescue/Harm 与 `Info - Matched-control` 配对统计；
- 全黑、Blind 和 Look-away 诊断负对照。

### 正式 A 仍需构造

- 在 4 个失败任务中构造独立的强 Blind--Reveal 评测状态；
- 至少新增 2 个任务通过同一 validation 冻结门，使正式覆盖达到 6 个任务；
- 构造状态与现有 test episode 独立，不能用 test 闭环结果筛选；
- 极端/全黑视角只用于评测和信息视角筛选，不进入正常训练。

### 暂不需要重造

- 400-state x 88-view 自然扫描无需重跑；
- Natural-4 的 20 个 prefix 状态无需再次筛选；
- Wide-MV 训练数据扩展属于 Phase B，不能替代 Expanded A 的强信息评测构造。

## 权威产物

- 扫描计划：`configs/dsol_paper1/libero_expanded_a_scan_plan_v1.json`
- 正式筛选规则：`configs/dsol_paper1/libero_expanded_a_selection_v1.json`
- Natural-4 规则：`configs/dsol_paper1/libero_expanded_a_natural4_pilot_selection_v1.json`
- 正式筛选：`/share/longjunyu/alphabrain/experiments/dsol-libero-expanded-a-v1/selection/automated_selection_400_v3_prefix70.json`
- Natural-4 筛选：`/share/longjunyu/alphabrain/experiments/dsol-libero-expanded-a-v1/selection/natural4_pilot_selection_400_v3_prefix70.json`
- 人工审计：`/share/longjunyu/alphabrain/experiments/dsol-libero-expanded-a-v1/manual_audit_natural4_v3_prefix70/manual_visual_audit_v3.json`
- 冻结协议：`/share/longjunyu/alphabrain/experiments/dsol-libero-expanded-a-v1/constructed_m1_natural4_prefix70_protocol_v3.json`
- 闭环指标：`/share/longjunyu/alphabrain/experiments/dsol-libero-expanded-a-v1/closed_loop_natural4_prefix70_v3/broad64-practical/analysis/metrics.json`
