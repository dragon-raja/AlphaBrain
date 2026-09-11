# 视角价值期望验证冻结协议 v1

## 1. 协议目的

本协议属于 `How VLAs Should Use Views` 系统分析中的随机性与视角偏好层。它不预设必须提出方法，
也不把视角选择单独升级为整篇论文的中心。目标是在讨论可见性、Accel、视角偏好、融合或主动获取前，
先可靠估计：

> 在固定物理状态和 checkpoint 下，一个候选视角相对规范视角的完整闭环期望收益是否存在。

完整机器可读协议为
`configs/dsol_paper1/view_value_expectation_protocol_v1.json`。状态固定为
`FROZEN_PREREGISTRATION_RUNNER_HOLD`：统计设计已冻结，但显式噪声 runner 和执行 release 尚未完成，
因此本次不会启动正式 episode。

## 2. 此前“三个噪声”的准确含义

此前 `20260861/62/63` 是三个完整 episode 的根 seed。每条 episode 在每次 replanning 时派生一个
新 seed，重新采样一个与被评 checkpoint 一致的高斯初始 action tensor，经 10 步 flow 去噪后生成 action chunk，执行
前 `K=5` 步，再次观察和采样。因此它们不是三个训练 seed，也不是仅在 episode 开头采三次噪声，
而是三条各自包含多次随机 action generation 的闭环轨迹。

旧 evaluator 对同一状态、同一 repeat 的不同视角共享 replanning seed，这是合理的 common-random-number
配对；但它同时用 evaluation seed 设置环境 RNG，且只记录根 seed，没有逐次记录实际噪声和 action hash。
本协议要求全部拆分和显式化。

## 3. 旧证据的保留边界

| 旧证据 | 本协议中的地位 |
|---|---|
| Camera Full 1,599 episodes、三 checkpoint seeds | 保留为被动视角泛化总体证据 |
| 21 状态 M1、每条件一条根噪声轨迹 | 探索性机制信号 |
| 单噪声 Oracle@97 | 候选空间 pilot |
| 三噪声 Discovery Best | 粗筛稳定性 pilot |
| Visibility 与 Accel | 待正式复验的选择基线 |

旧数据不进入新协议的主检验，也不能继续支持“已找到稳定好视角”或“当前 selector 有效”。

## 4. 需要估计的期望

对固定状态 `s`、视角 `v`、checkpoint `pi`，定义：

\[
p(s,v,\pi)=\mathbb E_{\epsilon}[\mathrm{Success}(s,v,\pi,\epsilon)]
\]

其中 `epsilon` 是完整 episode 的 replanning flow-noise 序列。视角收益为：

\[
U(s,v,\pi)=p(s,v,\pi)-p(s,v_{canonical},\pi)
\]

本协议不要求穷尽无限高斯噪声，只要求从部署一致的 `N(0,I)` 中抽取独立、配对、可审计的样本，
并在预注册精度下估计期望。Noise repeat 只用于估计同一状态的随机策略分布，不能被当作新的任务样本。

## 5. 随机性隔离

正式 runner 必须同时满足：

1. HDF5/MuJoCo 物理状态精确恢复，所有视角 SHA-256 一致；
2. environment seed 在同一状态内固定，不随 policy noise 改变；
3. 每次 replanning 注入显式 `10 x 7` 标准高斯 tensor；该形状来自本实验 Broad64 AlphaBrain checkpoint 的 `action_horizon=10`，不是 LeRobot base 配置的 50；
4. 同一 state/repeat/replan 下所有候选使用完全相同 tensor；
5. repeat 与 replan 之间独立；
6. 每次记录 noise seed、noise SHA-256 和 action chunk SHA-256；
7. A–F 六个噪声银行执行前全部物化并写 manifest，银行之间不得重叠。

单独设置 `torch.manual_seed()` 不足以 release，因为模型内部任何额外随机调用都可能改变真正用于 action
generation 的 tensor。正式路径必须显式向 sampler 传入 noise。

### 执行前修订 A1

首次真实 checkpoint smoke 在任何正式 episode 运行前发现：所选 AlphaBrain Broad64 checkpoint 声明
`action_horizon=10`，原文中的 `50 x 7` 来自 LeRobot Pi0.5 base 配置，并非该 checkpoint 的实际采样形状。
因此显式噪声形状修订为 `10 x 7`。episode 预算、`K=5`、去噪步数、候选选择、统计门槛均未改变；
原错误噪声银行不进入任何正式运行。

## 6. Calibration 密集候选实验

Calibration 使用至少 8 个任务、16 个 source-disjoint 物理状态，每任务至少 2 个状态。97 候选保持
canonical + 64 train-support + 32 held-out 的 E0 operational bank，不外推为真机 pan–tilt 可达域。

| 阶段 | 候选数 | 每候选新增完整噪声序列 | 作用 |
|---|---:|---:|---|
| A | 97 | 4 | 粗筛，不作效应结论 |
| B | 24 | 8 | 收缩候选并保留几何分层 |
| C | 6 | 16 | 冻结状态级候选 |
| D | candidate + canonical | 64 | 独立估计状态级期望 |

阶段 D 打开前必须冻结候选，看到 D 的结果后禁止更换。一个状态被计为强候选信号，要求候选成功率
至少 80%，相对 canonical 提升至少 20pp。当前候选空间通过稳定 headroom 门，还要求至少 25% 的
calibration 状态满足上述双门，并覆盖至少 4 个 source groups 和 3 个任务。该结论只表示一部分状态
稳定存在 headroom，不等同于所有状态都应换视角，也不证明 selector 能部署或跨状态泛化。

## 7. Held-out 系统分析

Test 使用 48 个物理状态、至少 24 个 source groups、至少 8 个任务，与 calibration 的 source 和 state
完全不相交。Seed41 完整比较：

- canonical；
- 确定性随机候选；
- calibration 冻结的全局固定 pose；
- visibility increment；
- entity visibility harmonic mean；
- Accel ensemble。

不允许使用 test 闭环结果、未来状态、oracle rank 或人工 test routing。Seed42/43 只确认 canonical 与
不看 test outcome 冻结的最佳非规范规则，避免将三 checkpoint 预算浪费在已明显失败的 baseline 上。

主银行 E 每条件固定 32 条完整噪声序列。只有机器审计在打开 reserve 前确认以下任一条件成立，才启用
预生成银行 F 的额外 32 条：

- source-cluster 95% CI 半宽大于 5pp；
- 前 16 与前 32 条估计漂移大于 5pp；
- Harm 的 Wilson 95% CI 半宽大于 5pp。

若 64 条后仍不满足精度，只能报告 `INCONCLUSIVE_NO_DIRECTIONAL_CLAIM`，不得继续追加到显著为止。

## 8. 统计单位与主张门

计算顺序固定为：先在 state 内对 noise 求均值，再在 source 内对 state 等权，最后按 source group 和
task strata 推断。主区间为 paired source-cluster bootstrap 95%，单成功率同时报告 Wilson 95%。

Selector 正式收益要求：

- held-out population gain 至少 +5pp；
- paired source-cluster CI 排除 0；
- Harm 增量不超过 +5pp；
- checkpoint seeds 41/42/43 方向一致。

没有通过该门仍然可以形成系统分析结果，但不能包装为有效 view selector。只有稳定 candidate headroom、
可复现的 provided/selected-view 信号，以及明确的训练或融合缺口同时出现，才释放可选的综合方法实验；
否则论文停在 controlled system study 和设计律，不强加方法。

## 9. 计算预算与收敛解释

固定候选全暴力 `20 x 97 x 20` 需要 38,800 episodes，且仍无法解决 selection/test leakage。本协议用
多阶段独立银行把大量预算集中在仍可能有收益的候选上，同时用 64 条确认序列估计状态级期望，用更多
held-out source states 估计总体规律。

“噪声够不够”不再由一句固定次数判断，而由以下证据共同决定：

- 4/8/16/32/64 前缀估计曲线；
- paired CI 精度；
- reserve 启用记录；
- checkpoint seed 复现；
- source-disjoint test。

这不等于覆盖完整高斯空间，但足以在预注册误差范围内验证期望是否存在。若期望没有收敛，协议要求
报告不确定，而不是通过无限追加或事后换候选制造结论。

## 10. 执行前阻断项

在新的 execution release receipt 生成前，以下工作仍为 HOLD：

- 修改 runner 支持显式 per-replan noise；
- environment/policy seed 解耦；
- noise/action hash 审计；
- calibration/test source-disjoint manifest；
- 噪声银行 A–F manifest；
- runner 单元测试与最小 paired smoke。

协议冻结不等于批准正式运行。下一步应先实现和审计底座，再由单独 receipt 释放 calibration 阶段。
