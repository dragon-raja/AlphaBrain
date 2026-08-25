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
- Accel 核心曲率公式与单次 velocity trace 实现正确，但本地任务是原文之外的跨视角选择扩展，
  不是原论文失败检测数值的完整复现。预注册 `accel_3` 每 21 个状态有 15-18 个选择 canonical，
  source-demonstration 等权差值为 -2.5pp 到 +3.3pp；它没有捕获已有候选 oracle headroom。

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

### 3.3 Accel：不确定性分数能否直接充当视角价值

原文 Accel 对 10-step flow-matching 轨迹的前 `p` 步计算：

\[
\mathrm{accel}_p
=
\frac{p\sum_{t=1}^{p-1}\lVert v_t-v_{t-1}\rVert_2}
{\sum_{t=0}^{p-1}\lVert v_t\rVert_2}.
\]

低分表示当前动作去噪轨迹更直、模型内部更确定，不直接表示该视角包含更多任务信息。本地在同一
物理状态的候选间共享完全相同的 `x0`，保存 `accel_2...10`，再扩展测试
`argmin_v accel_3(v)` 能否选择闭环更好的视角。

| 模型 | 选择 Canonical | Accel 所选成功率 | Canonical | 任一候选可成功 | 状态级差值 |
|---|---:|---:|---:|---:|---:|
| Broad practical | 18/21 | 57.1% | 57.1% | 85.7% | 0.0pp |
| Broad state-matched | 16/21 | 57.1% | 52.4% | 76.2% | +4.8pp |
| Broad paired FM | 15/21 | 47.6% | 52.4% | 71.4% | -4.8pp |
| Broad paired consistency | 18/21 | 57.1% | 57.1% | 76.2% | 0.0pp |

Broad practical 的任一候选成功率比 Canonical 高 28.6pp，但 Accel 所选成功率没有提高。前缀
`p=2...10` 与官方参考式路径坐标缩放的事后敏感性分析也没有跨模型稳定正增益。因此当前结果
否定的是“直接最小化 Accel 即可得到高价值视角”，不否定原文将 Accel 用作 uncertainty proxy
和在线 failure detector 的主张。完整审计见 `accel_reproduction_audit_20260824_zh.md`。

### 3.4 Accel 扩大诊断：训练支持、留出视角与跨噪声稳定性

旧表和扩大表必须分开理解：旧表在 21 个状态的四个角色视角上已经运行闭环，因此能查询
Accel 所选视角的成功或失败；扩大表覆盖 8 个任务、96 个 test 状态、97 个 operational 视角和
6 个共享 flow-noise seed，只分析固定状态动作流排名，不包含这些 97 个视角的闭环成功率。

97 个 operational 候选由以下三类组成：

| 区域 | 候选数 | 精确定义 |
|---|---:|---|
| Canonical | 1 | 原始规范相机位姿 |
| Broad64 training support | 64 | 具体位姿进入过 Broad64 全局训练 catalog；不表示同一 test 状态见过该位姿 |
| Broad32 held-out | 32 | 与训练位姿处于同一宽范围，但具体位姿未参与训练；是同范围插值留出，不是极端 OOD |

对每个状态先对 6 个 flow-noise seed 的 `accel_3` 取均值，再在 97 个视角中选择最低值。结果为：

| 模型 | Canonical | 训练支持 | 同范围留出 | 六 seed 精确 Top-1 全同 | 跨噪声排名 Spearman |
|---|---:|---:|---:|---:|---:|
| Broad practical | 21.9% | 37.5% | 40.6% | 2.1% | 0.412 |
| Broad state-matched | 32.3% | 32.3% | 35.4% | 3.1% | 0.416 |
| Broad paired FM | 33.3% | 39.6% | 27.1% | 4.2% | 0.430 |
| Broad paired consistency | 29.2% | 45.8% | 25.0% | 2.1% | 0.287 |

这些百分比是“96 个状态中，ensemble Top-1 落入该区域的比例”，不是成功率。候选数也不相等，
因此必须与均匀候选基准 `1/97、64/97、32/97` 比较。以 Broad practical 为例：Canonical
为 `21/96=21.9%`，相对 `1/97` 富集 21.2 倍；训练支持为 `36/96=37.5%`，相对其候选占比
仅 0.57 倍；留出为 `39/96=40.6%`，相对候选占比为 1.23 倍。这说明模型仍有很强的
Canonical attractor，但低-Accel 兼容域并未锁死在 64 个训练精确位姿。

扩大诊断还得到：

- 单次 flow noise 的精确 Top-1 高度不稳定，不能把一次运行的 `10/21` 等计数当作模型属性；
- State-matched 与 Paired-FM 的 ensemble 视角排名相关性为 0.815，说明普通 FM 下 same-state
  pairing 没有形成明显不同的兼容域结构；
- consistency 模型的跨视角 Accel 相对离散度仅 2.67%，ensemble Top-1 间隔仅 1.68%，主要作用
  是压平视角差异，而非对齐任务可见性；
- 全黑相机平均位于第 101/104，Look-away 位于第 86–92，说明 Accel 能稳定排斥明显无效观测；
- 最高可见性视角平均只位于第 46–51，且相对 matched-control 的偏好随任务阶段改变符号，没有
  形成稳定的“可见信息更多即 Accel 更低”关系。

因此扩大结果支持把 Accel 用作模型视角兼容域和坏观测诊断器，不支持把单次 `argmin Accel`
直接部署为主动视角价值函数。行为价值仍须在任务均衡的闭环 shortlist 上单独验证。

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

- 第一阶段统一 PDF：`view_revalidation_stage1_integrated_20260825_zh.pdf`
- 旧版第一阶段 PDF：`view_revalidation_stage1_final_brief_20260824_zh.pdf`
- Accel 扩大诊断原始汇总：`/share/longjunyu/alphabrain/experiments/dsol-accel-expanded-diagnostic-v1/analysis/summary.json`
- 正式完成回执：`/share/longjunyu/alphabrain/experiments/dsol-view-revalidation-m-b-v1/formal_evaluation_completion.json`
- Camera Full：`/share/longjunyu/alphabrain/experiments/dsol-view-revalidation-m-b-v1/camera_full/multiseed_metrics.json`
- Original Full：`/share/longjunyu/alphabrain/experiments/dsol-view-revalidation-m-b-v1/original_full/multiseed_metrics.json`
- M0：`/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m0-v1/operational-three-task-scan-v2/analysis/summary.json`
- M1：`/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m1-v2/cross-model-analysis/metrics.json`
- Accel 审计：`accel_reproduction_audit_20260824_zh.md`
- Accel 机器可读敏感性：`accel_view_selector_audit.json`
