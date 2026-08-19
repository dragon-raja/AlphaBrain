# 视角泛化与 Active-Ready 完整计划状态审计

日期：2026-08-19  
总体状态：`PARTIAL_QUICK_GATE_EVIDENCE_ONLY`

## 一句话结论

完整研究计划**尚未跑完**。目前完成的是 LIBERO 上的 Broad32、seed-41 快速门，
以及自然场景可见性 M0/M1 的一个受控子门。Legacy Anchor、Broad64/数据规模、
正式多 seed、构造型 Blind-Reveal、全方法 M1 和新版 Accel 均未完成。

本状态审计中的 quick gate 应准确称为“LIBERO official HDF5 exact-state
view intervention in a frozen LIBERO-Plus runtime”，不是完整 LIBERO-Plus 官方
benchmark。修正后的总协议见
[view_revalidation_master_protocol_v3_zh.md](view_revalidation_master_protocol_v3_zh.md)。

## 分阶段状态

| 阶段 | 状态 | 已完成 | 主要缺口 |
|---|---|---|---|
| WP0 Legacy Anchor | PARTIAL | 旧 Official 640-episode 指标和旧 checkpoint 均存在 | 未在当前 exact-state 协议重跑 Official 与 Legacy-MV8 |
| WP1 View Catalog | ENGINEERING PASS | 160-entry catalog：Legacy8、Broad64、held-out32、wide24、crossed16、extreme8、look-away4、sensor3；Broad64 exact-state 数据审计通过 | catalog 仍需在 M0 前冻结信息阈值与 constructed-scene 候选 |
| Phase A 被动鲁棒性 | QUICK GATE PARTIAL | 7 模型 × 24 groups × 7 条件，共 1,176 闭环 episodes | 缺 Official、Legacy-MV8、三 seed、Reveal/Blind 全模型矩阵、新场景 × Wide |
| Phase B 训练组织 | QUICK GATE PARTIAL | Broad32 七模型完成；Broad64 exact-state 数据完成并开始 seed-41、2,000-step、GB32 训练 | 缺 Broad64 结果、1x/2x/4x exposure、正式约 338k pairs、seeds 42/43 |
| M0 可见性 | NATURAL-SCENE PARTIAL | 8 tasks、160 states、14,080 candidates、160 montages；极端/Look-away/blackout 链路可用 | 严格门仅 2 tasks 可形成 Info-Control 对；未构造遮挡板或专门 Blind-Reveal 场景 |
| M1 完整闭环 | SUB-GATE COMPLETE | Broad32、Info-only、Info+Control 三模型；每模型 210 episodes | 未覆盖 7 个训练 arms、6-10 tasks、三 seed；Rescue/Harm/progress 未完整汇总 |
| Accel | NOT IMPLEMENTED | 只有旧版不确定性/静态候选 Legacy 结果 | 无 accel2-10、共享 flow noise、dense search、dynamic search、train/info/reveal/oracle 关系分析 |
| RoboCasa 扩展 | ASSETS READY | assets、Objaverse、Human300、checkpoint、48/50 Target tasks 已到位 | 尚未运行新计划的构造场景、M0、M1 或 Accel |
| 论文正式门 | NOT STARTED | 无 | 功效分析规模、三 seed、正式任务族、确认性统计均未运行 |

## 已完成的渲染与数据

### Broad32 训练数据

- 8 个任务；
- 400 个 source episodes；
- 38,193 exact-state records；
- train/val/test 为 31,440 / 2,701 / 4,052；
- 32 个 pose IDs；
- state、action、robot state、图片尺寸、pose 区分和 split leakage 审计均通过；
- 数据大小约 2.7GB。

训练相机范围已经明显大于旧 Legacy8：

| Catalog | 方位角 | 仰俯角 | 距离比例 |
|---|---:|---:|---:|
| Broad64 候选 | `[-59.5°, +59.1°]` | `[-24.8°, +24.4°]` | `[0.90, 1.25]` |
| Wide held-out | `[-84.4°, +78.8°]` | `[-37.2°, +38.9°]` | `[0.75, 1.50]` |
| Extreme diagnostic | `[-180°, +180°]` | `[-60°, +45°]` | 诊断用 |

当前实际训练只使用 Broad32；Broad64 只是 catalog 中已设计，尚未生成对应正式数据。

### 极端视角与 M0 渲染

- 160 个 val/test states；
- 每个 state 扫描 88 个候选，共 14,080 条记录；
- 保存 160 张 `visibility_extremes.png` montage；
- Look-away 的可见性增量全部为负；
- crossed/extreme orbit 能产生约 `+0.21` 的最大可见性增量；
- 已包含 external、wrist 与 all-camera blackout 控制。

这些结果证明“极端相机干预与可见性测量链路可工作”，但没有证明自然任务中存在
足够多的行为相关 Reveal。严格几何匹配后，只有 `goal_wine_rack` 与
`libero10_mug_microwave` 两个任务满足当前快速门。

尚未完成：

- RoboCasa 遮挡板或 fixture 改造；
- 专门构造的左/右目标、容器内部或抽屉内部 Blind-Reveal；
- 与信息无关但位移匹配的场景级 Matched-control；
- constructed-evidence demonstrations。

## Phase A / Phase B 已有结果

当前被动闭环矩阵使用 24 个 exact-state groups，每组 7 个相机/通道条件。

| 模型 | Canonical | Broad held-out | Wide extrapolation |
|---|---:|---:|---:|
| Canonical-unique | 91.7% | 33.3% | 25.0% |
| Canonical-repeat | 83.3% | 33.3% | 16.7% |
| ImageAug-unique | 91.7% | 25.0% | 16.7% |
| Broad-unpaired practical | 91.7% | 87.5% | 91.7% |
| Broad-state-matched | 95.8% | 83.3% | 83.3% |
| Broad-paired FM | 95.8% | 79.2% | 79.2% |
| Broad-paired consistency | 91.7% | 83.3% | 83.3% |

在该 seed-41 快速门中：

- Broad32 相对 Canonical-unique 的 held-out 提升约 50-54pp；
- ImageAug 没有复现 Broad32 的收益；
- Paired-FM 没有稳定超过 state-matched；
- consistency 在当前 24 groups 下也没有稳定超过 Paired-FM。

这些只支持“宽相机覆盖是强快速基线”。样本、seed 和 pose-density 均不足以发布
“pairing 无效”或“consistency 无效”的正式结论。

## M0 / M1 子门结果

严格 M1 协议包含 21 个 frame states、6 个 source demonstrations、2 个任务和
10 个条件。三个模型各执行 210 条完整闭环：

| 模型 | Canonical | Strong-info | Matched-control | Blind |
|---|---:|---:|---:|---:|
| Broad32 | 71.4% | 42.9% | 52.4% | 33.3% |
| Info-only | 66.7% | 71.4% | 47.6% | 61.9% |
| Info+Control | 76.2% | 66.7% | 66.7% | 57.1% |

该子门只回答一个问题：当 Info 和 Control 的精确位姿都进入训练 support 后，
二者成功率相同，因此 Info-only 的正差主要来自 support 不对称。它没有完成完整
M1 方法矩阵，也没有否定在更强构造场景中的信息视角价值。

## Accel 当前真实状态

新版 Accel 尚未实现。仓库当前没有以下产物：

- 每个候选的 `accel_2 ... accel_10`；
- 主指标 `accel_3`；
- 同一状态候选共享 flow noise `x0`；
- 64-96 view dense ranking；
- 8-16 view dynamic closed-loop search；
- Accel 到 canonical、最近 train pose、Info、Reveal 和 oracle@shortlist 的距离；
- 对当前 Broad32/Info-support/Wide 模型的关系迁移分析。

历史 Official Pi0.5 的 640-episode 主动候选结果只能作为 Legacy：旧 uncertainty
selector 与 canonical 持平，六视角 oracle 仅约 +3.1pp。它不是本计划指定的
flow-matching Accel，也没有使用当前扩大的候选池和训练策略。

## 当前资源状态

- LIBERO 原始 HDF5：40/40、33.78GB、2,000 demos、338,575 transitions，已验证；
- RoboCasa Human300：300 datasets、32,043 episodes、约 112.96GB，已验证；
- RoboCasa Target：48/50 registry tasks、24,300 episodes、约 60.55GB，逐文件验证；
- Target 缺少 `CloseToasterOvenDoor` 与 `CoffeeSetupMug` 两项，源端本身不存在，
  不阻塞先行任务筛选；
- official_75k 与旧 Phase B checkpoint 已到位；
- RoboCasa assets 与 Objaverse 引用完整。

因此当前主要缺口不是文件迁移，而是实验实现和运行。

## 恢复执行顺序

1. 用现有旧 checkpoint 完成 WP0 Official / Legacy-MV8 数值锚点；
2. 生成 Broad64 和固定 pose-support 的 1x/2x/4x exposure 数据；
3. 完成 seed-41 扩大 Phase A，包含 Reveal、Blind、Look-away 和通道消融；
4. 在 RoboCasa Target/Pretrain 中筛选并构造 6-10 个强 Blind-Reveal 任务；
5. 先通过可见性、人工 montage、Matched-control 和 wrist-off M0 gate；
6. 再运行完整 M1 方法矩阵，并补齐 progress、Rescue、Harm 和 steps；
7. 实现 Accel2-10、dense ranking、dynamic shortlist 和 oracle 关系分析；
8. 快速门成立后，才扩展 seeds 42/43 与正式数据规模。

在第 5 步前直接扩大自然 LIBERO M1，或在候选信息空间未成立前直接跑 Accel，
都会重复旧实验“干预太弱、结果不可解释”的问题。
