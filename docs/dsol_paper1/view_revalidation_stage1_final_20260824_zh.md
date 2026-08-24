# VLA 视角泛化与 Active-Ready 感知重验证：第一阶段最终状态

状态：`STAGE1_M_A_M_B_COMPLETE`

日期：2026-08-24

## 1. 完成范围

约定的加速版第一阶段 M-A / M-B 已完成，当前没有训练或评测进程运行。正式控制器回执为
`COMPLETE`，覆盖 seeds 41/42/43、6 个 Camera Full 模型和 6 个 Original Full 模型。

第一阶段已经完成：

- 38,193 条 Broad64 same-state 记录及 episode 级无泄漏划分；
- 七个训练组织机制臂的 seed-41 exact-state 开发门；
- Broad practical 与 Broad paired consistency 的三 seed 正式训练；
- LIBERO-Plus Camera Full 与 Original LIBERO Full；
- 180 states / 15,840 candidates 的 M0 可见性扫描；
- 5 models / 1,050 episodes 的 M1 完整闭环开发门；
- Accel fixed-state 候选关系分析。

以下属于长期研究计划的后置阶段，尚未运行：

- LIBERO-Plus Full 10,030 条非相机扰动副作用审计；
- 更大任务分布和更多独立 source demonstrations 的 Blind-Reveal 确认；
- RoboCasa 跨 benchmark、动态主动视角和真机验证。

因此准确表述是“第一阶段完成”，不是“整个长期研究计划全部完成”。

## 2. 正式结果

| 模型 | Camera Full | 相对 Official | Original Full | 相对 Official |
|---|---:|---:|---:|---:|
| Official Pi0.5 frozen | 75.86% | - | 96.45% | - |
| Broad64 practical，三 seed 均值 | 82.16% | +5.15pp `[+0.99,+9.62]` | 93.92% | -2.53pp `[-4.05,-1.15]` |
| Broad64 paired consistency，三 seed 均值 | 77.78% | +0.40pp `[-5.15,+6.16]` | 84.67% | -11.78pp `[-17.07,-7.28]` |

Paired consistency 相对 Broad practical：

| Benchmark | 差值 | 95% CI | Seed 方向 |
|---|---:|---:|---|
| Camera Full | -4.75pp | `[-7.85,-1.73]` | 3/3 为负 |
| Original Full | -9.25pp | `[-13.32,-5.80]` | 3/3 为负 |

裁决：Broad64 practical 通过预注册的相机增益门和 5pp retention 门；当前 paired-consistency
组合方案没有增量价值，并明显损害基础能力。

## 3. 机制结果

- Exact-state 中 Broad practical 在 Broad held-out / Wide extrapolation 为 87.5% / 91.7%，
  Canonical unique 为 33.3% / 25.0%，Image-Aug 为 25.0% / 16.7%。
- Accel 每 21 个状态有 15-18 个选择 canonical，相对 canonical 成功率为 -2.5pp 到 +3.3pp；
  它没有捕获已有 oracle headroom，不能作为当前主动视角选择器。

### 3.1 M0：先构造可控的观测差异

M0 不是策略效果实验。它在同一冻结 MuJoCo state 下渲染候选视角，通过 simulator
instance segmentation 计算任务实体的可见像素比例：

\[
I_{task}(v)=\operatorname{mean}_{camera,entity}
\frac{\text{visible pixels}}{224\times224},\qquad
\Delta I(v)=I_{task}(v)-I_{task}(canonical).
\]

每个通过筛选的状态冻结四个角色：

| 角色 | 定义 | 后续用途 |
|---|---|---|
| Canonical | 原规范外部视角，`ΔI=0` | 基准观测 |
| Strong-info | 相对 canonical 明显增加任务实体可见像素 | 测试新增可见信息 |
| Matched-control | 与 Strong-info 的相机平移/旋转相近，但 `ΔI≈0` | 控制普通相机移动影响 |
| Blind | 任务实体可见性明显降低的极端视角 | 信息缺失负对照 |

扫描覆盖 180 states / 15,840 candidates。Crossed-orbit 的最大 `ΔI` 为 +20.85pp；
Look-away 与 sensor blackout 的正增量比例均为 0%。最终 21 个 test states 通过人工视觉审计进入
M1。这里能得出的结论是“候选池中确实存在稀疏的信息上尾，并且可以构造位姿匹配对照”，不能据此
宣称模型已经利用了信息。

### 3.2 M1：再检验新增可见信息能否改善完整闭环

M1 从上述筛选后的中间状态出发，只替换外部相机角色；腕部相机保持启用并随机器人正常动态更新。
Pi0.5 持续闭环重规划，直到达到官方任务成功条件或超时。它是 selected-state closed-loop continuation，
不是从任务初始状态开始的标准 benchmark 成功率。

| 观测条件 | 21 frame states 原始率 | 6 demonstrations 等权率 |
|---|---:|---:|
| Canonical | 57.1% | 42.5% |
| Strong-info | 71.4% | 52.5% |
| Matched-control | 52.4% | 39.2% |
| Blind | 23.8% | 18.3% |

Matched-control 用于扣除普通相机移动的影响：

\[
\text{Information Specificity}
=(SR_{info}-SR_{canonical})-(SR_{control}-SR_{canonical})
=SR_{info}-SR_{control}.
\]

按 21 个相关 frame states 直接汇总时，信息特异性为 `71.4%-52.4%=+19.0pp`，仅作为描述性数字。
预注册主统计先在每条 source HDF5 demonstration 内聚合，再对 6 条独立演示等权，得到
`52.5%-39.2%=+13.3pp`，cluster 95% CI
`[+3.3,+26.7]`。这种口径避免包含更多 frame states 的单条演示主导结果。

因此当前可以写成：Broad practical 出现了“新增任务可见信息优于等幅普通换视角”的方向性闭环证据。
仍不能写成最终普遍结论，因为独立统计单位只有 6 条演示、任务分布不均，而且五个模型的
external-only 条件均为 0%；该信号依赖正常的 external + wrist 双相机输入。

## 4. KYC / CVC 方法边界

- KYC matched Pi0.5：Control 44.49%，KYC 43.33%，差 -1.16pp；
- 双相机 KYC：Control 20.71%，Dual-KYC 15.71%，差 -5.00pp；
- 当前 CVC-style recipe：Camera Full 相对 practical -4.75pp，Original Full -9.25pp。

正式决定：停止把 KYC raw-ray 和无条件跨视角一致性作为标准 external + wrist Pi0.5 的研究主线；
保留它们作为单外部相机受控协议的强基线。研究重点应转向区分冗余、缺失与互补视角，并只在新
观测增加任务相关证据时改变决策。

当前正式 CVC 比较同时改变了独立状态数量和数据组织，因此否定的是当前组合 recipe，而不是所有
paired consistency 的普遍可能性。该限制不妨碍停止继续投入当前实现。

## 5. 入口

- 第一阶段 PDF：`view_revalidation_stage1_final_brief_20260824_zh.pdf`
- 正式完成回执：`/share/longjunyu/alphabrain/experiments/dsol-view-revalidation-m-b-v1/formal_evaluation_completion.json`
- Camera Full：`/share/longjunyu/alphabrain/experiments/dsol-view-revalidation-m-b-v1/camera_full/multiseed_metrics.json`
- Original Full：`/share/longjunyu/alphabrain/experiments/dsol-view-revalidation-m-b-v1/original_full/multiseed_metrics.json`
- M0：`/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m0-v1/operational-three-task-scan-v2/analysis/summary.json`
- M1：`/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m1-v2/cross-model-analysis/metrics.json`
