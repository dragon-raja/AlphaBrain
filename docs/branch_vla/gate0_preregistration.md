# Branch-VLA Gate 0 预注册：Contingent Chunk 表示是否可学

状态：结果前冻结

冻结时间：2026-07-18 UTC

## 1. 唯一问题

在结果尚未揭示的同一 LIBERO conditioning 下，能否从 train outcome twins 学到 attached 与 slip
两条后续 action branches，并在 validation 上由 feedback 后图像正确路由，使 suffix 风险显著低于
只能输出一条未来的 deterministic chunk？

本 Gate 不训练 Pi0.5、不运行闭环、不声称 Branch-VLA 已有效。

## 2. 数据边界

- 数据根：`/share/longjunyu/fresh-vla/libero-full-episode-v2-128`；
- 只打开 manifest 中 `train` 的 102 groups 与 `val` 的 13 groups；
- original test 与所有 confirmation/sealed 路径 fail closed；
- 统计单位为 source initial state；lead offsets 和同组两 branch 不能当独立样本；
- 不读取 branch outcome 作为 predictor 或 guard 的部署输入；outcome 只用于构造监督与计算指标。

## 3. 固定样本构造

对每个 group 的 `feedback_reveal_time = r`，固定使用 `lead = 1..5`：

- predictor 输入：`t=r-lead` 的 agent-view、wrist-view、robot state，以及 lead one-hot；
- 图像特征：每个 view 做固定 `8x8` RGB block average；
- target：从 `t` 开始长度 `H=16` 的 attached 与 slipped teacher action chunks；
- 强制审计 `t<r` 的 paired images/state/actions 完全一致；
- suffix 指标只使用 chunk 内 `lead..H-1`，即结果揭示后的动作；
- guard 输入：`r` 的两视角图像与 robot state；pre-feedback control 使用 `r-1`。

动作只用 train targets 的逐维 mean/std 标准化。所有 ridge 正则从
`{0.01, 0.1, 1, 10, 100}` 中按 train source-grouped 5-fold CV 选择；validation 不调参。

## 4. 固定方法

1. `per_lead_constant`：每个 lead、每个 branch 的 train action 均值；检验 branch targets 是否仅是模板。
2. `linear_chunk`：相同 pre-feedback 输入只预测 attached/slipped targets 的平均 chunk。
3. `two_branch_oracle_route`：两个 branch regressors；仅在评测时用真实 outcome 选择，作为表示上界。
4. `two_branch_learned_route`：同一两个 regressors；使用 feedback frame 的 learned guard 选择。
5. `random_precommit`：结果揭示前以 50/50 选择某个 predicted branch，表示普通 latent mixture 的
   随机提前提交风险。

所有 action predictors 使用完全相同的输入特征、训练 groups、标准化和正则选择流程。

## 5. 指标

- normalized suffix MSE，attached/slipped 分开与平均；
- common-prefix MSE；
- learned route 与 oracle route 相对 `linear_chunk` 的风险下降；
- two-branch 相对 `per_lead_constant` 的下降；
- predicted/target branch separation ratio；
- set coverage：每个真实 branch 到最近 predicted branch 的 MSE；
- post-feedback guard accuracy、paired ranking；
- pre-feedback guard accuracy 与 shuffled-label train control；
- source-level bootstrap 95% CI。

## 6. 裁决

输出 `BRANCH_REPRESENTATION_FEASIBLE` 需要同时满足：

1. train=102、val=13，val 覆盖至少 8 个独立 source，且所有 pre-feedback parity 检查通过；
2. post-feedback guard accuracy >=85%，pre-feedback accuracy <=60%，train-label shuffle control <=60%；
3. `two_branch_learned_route` 相对 `linear_chunk` 的 source-mean suffix MSE 至少降低 25%，且 paired
   bootstrap 95% CI 的降低下界大于 10%；
4. `two_branch_oracle_route` 相对 `linear_chunk` 至少降低 30%；
5. learned route 在 attached 和 slipped 两支上分别至少降低 15%；
6. oracle-routed two-branch 相对 `per_lead_constant` 至少降低 20%；
7. predicted/target branch separation ratio 的中位数至少 0.5，排除双 head collapse。

否则输出 `STOP_BRANCH_ACTION_CHUNK`。数据或 parity 不完整则输出 `GATE0_INVALID`。

通过只允许进入真实 simulator Gate 1；不允许直接打开 test、微调 Pi0.5 或宣称顶会方法成立。
