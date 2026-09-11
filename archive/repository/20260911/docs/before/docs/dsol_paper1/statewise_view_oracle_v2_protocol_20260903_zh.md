# 状态条件视角 Oracle 与选择器实验 v2

冻结日期：2026-09-03

## 这次到底验证什么

这次不再问“97 个视角里是否存在一个全局固定的最好视角”。主问题是：对同一个物理状态 `s` 和候选外部视角 `v`，在完整 Pi0.5 flow-noise 轨迹上取期望后，策略价值

`J(s,v) = E_Xi[success(rollout(s,v;Xi))]`

是否存在稳定、非平凡、并且随状态变化的提升空间。后续两个问题才是：这种状态条件的价值结构能否由任务语言、图像、本体状态和相机几何预测；冻结后的选择器能否泛化到 source-disjoint 状态。

有限样本永远不能证明无限样本意义下的精确 `argmax`。所以正式结论使用“独立噪声确认的搜索 oracle 下界”：O bank 对全部 97 个视角做 dense discovery，P bank 对每个状态独立复筛 Top-8，Q bank 只评测已经冻结的 Top-1、canonical 和部署基线。O 上的直接最大值只画作诊断上限，不作为效应量。

## 为什么旧实验不足、又为什么没有作废

旧 A→B→C→D 已经是逐状态筛选，并不是全局固定视角；但 A 在 97 个视角上每个只有 4 条二值 rollout，过早 Top-K 裁剪容易漏掉真实好视角。D 的独立确认因此能确认“漏斗找到的候选”，却不能充分排除“好候选在 A 被误删”。

旧 D 的 16 个状态中，12 个状态为正、4 个持平、0 个为负，source-equal 平均增益为 `+4.785 pp`。事后层次 bootstrap 为 `[+1.270,+8.691] pp`。这说明值得增加密度，但它是设计依据，不进入 v2 主估计。旧报告中的 `VIEW_HEADROOM_NOT_CONFIRMED` 只否定了“至少 25% 状态同时达到成功率 80% 和增益 20pp”的强门槛，并不等于完全没有较小视角价值。

## 人群与盲测

沿用上一轮 64 个经过物理状态哈希审计的状态和完整 97 视角静态资产，以免无意义地重新渲染相同物理状态。旧 rollout 结果不并入 v2。

- 8 个任务，每个任务 8 个互不相同的 source demonstration。
- 每个任务从上一轮 held-out source 中以冻结哈希选 2 个作为最终 test，共 16 个。
- 其余每任务 6 个作为 development，共 48 个，并构成 6 个 source-grouped CV fold。
- test dense 输出在选择器、超参数、全局固定视角和逐任务固定视角全部冻结前不分析。

复用的是物理状态、相机几何扫描和无 outcome 的 97 视角图像；不复用正式效应所需的 policy-noise rollout。

## 三层实验

| 层 | 直接问题 | 正式数据 | 可支持的结论 |
|---|---|---|---|
| 1. Oracle 存在性 | 对状态变化的好视角空间是否真的存在 | 全 97×32 的 O；Top-8×32 的 P；冻结 Top-1 与 canonical×64 的 Q | 独立噪声确认的、状态条件搜索下界；不是精确无限样本 oracle |
| 2. 可预测性 | 哪些可观察输入能预测高价值候选 | development dense map，按 source 做 6-fold CV | 几何、可见性、任务、图像、本体信息各自能解释多少视角价值 |
| 3. 泛化 | 冻结选择器能否在新 source 上兑现收益 | 选择器冻结后才打开 16-state test；Q 上与 canonical/global/task-fixed 比较 | 是否存在可部署的选择规律，而不只是事后挑最好视角 |

图像查询型 selector 与无需候选图像的 selector 必须分开报告。前者假设可以先移动相机或虚拟渲染候选图像；后者只允许 canonical 图像、任务语言、本体状态和候选位姿。因此本实验可以为后续 pan-tilt、头部相机或虚拟视角获取提供依据，但本身不声称已经完成 active perception。

## 噪声与时间对齐

每个 bank 在正式执行前完整落盘。键为 `(bank, state, repeat, replan_index)`，每次 replanning 注入一个固定的 `10×7` 标准高斯 tensor。同一个状态、repeat 和 replan index 下，所有视角读取完全相同 tensor。每次推理都执行相同的 10 步 flow Euler 时间网格；不同视角造成中间 velocity、动作和终止时间不同是被测效应，不应该强制轨迹动作相同。提前成功的视角不再消费后续噪声，也不会让其他视角的随机流错位。

S 只用于 smoke。O、P、Q、R 相互独立；R 在执行前也预生成，但只有 Q 精度不够且命中冻结规则时才打开。

## 规模和预计时间

- Development O：`48×97×32 = 148,992` episodes。
- Test O：`16×97×32 = 49,664` episodes。
- Development P：`48×9×32 = 13,824` episodes。
- Test P：`16×9×32 = 4,608` episodes。
- Development Q 最少 `48×2×64 = 6,144` episodes。
- Test Q 会加入冻结的 global、task-fixed 和 learned selector；相同视角去重后再运行。

上一轮实测约 1,000 episodes/hour。主 checkpoint 的 dense、复筛和确认约 9–11 天；选择器冻结、test 开封和分析约 1–2 天；seed-42/43 只做冻结候选的 transfer confirmation，再需约 1–2 天。稳定运行时三层完整闭环预计 10–14 天，即大约 2026-09-13 至 2026-09-17 完成。硬件失败会顺延，但所有 wave 均可从 episode ledger 续跑。

## 成功和失败都怎样解释

- Oracle 对 canonical 的 Q-test source-cluster bootstrap CI 排除 0，说明存在稳定的搜索下界；点增益达到 5pp 才称为实际显著。
- Oracle 同时优于 development 冻结的 global fixed view，且差值至少 3pp，才支持“必须随状态变化”，候选 ID 多样本身不够。
- Learned selector 在 test Q 上优于 canonical、增益至少 3pp、捕获至少一半正 oracle headroom 且 harm 增加不超过 5pp，才支持可泛化选择规律。
- 如果只有 oracle 成立、learner 不成立，结论是“有价值空间，但现有可观察量/学习器还找不到”。
- 如果 oracle 也不成立，才说明在当前 checkpoint、固定外部相机、97 视角 bank 和这些状态上，没有得到可信的可利用空间；不能外推到腕部相机、头部主动相机或其他 checkpoint。

机器可读冻结协议为 `configs/dsol_paper1/statewise_view_oracle_v2.json`。
