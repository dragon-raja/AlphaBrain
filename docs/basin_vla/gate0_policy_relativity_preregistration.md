# BASIN-VLA Gate 0 预注册：Policy Relativity 是否存在

状态：结果前冻结

冻结时间：2026-07-18 UTC

## 1. 唯一问题

本 Gate 不训练 critic，也不主张 BASIN 有效。它只问：

> 对完全相同的 on-policy state 和完全相同的 16 个候选 action chunks，三个冻结 Full-H Pi0.5
> checkpoint 的有限预算 continuation 是否给出实质不同的候选偏好？

若答案为否，停止 policy-conditioned committor，不通过换网络寻找差异。

## 2. 数据边界

- 只使用现有 `/share/longjunyu/fresh-vla/cora-vla/onpolicy-support-v1/cache/seed41-{a,b}`；
- 这些是已使用过的 validation slipped on-policy states，不构成新 holdout；
- 不访问 original test、CORA confirmation 或任何包含 `confirmation`/`sealed` 的路径；
- 候选池固定为 seed41 Full-H 当时生成并缓存的 N=16 chunks；所有 target policies 使用同一候选池；
- 统计单位为 source initial state，不能把 candidate、continuation repeat 或 replanning state 当独立样本。

## 3. 固定状态子集

使用七个既有阶段：

```text
feedback_reveal
failure_continuation
recovery_start
reapproach
preclose
post_regrasp
final_failure
```

每阶段按 `sha256("basin-vla-relativity-v1:" + cache_path)` 排序，取前三个，共目标 21 states。若某阶段
不足三个则全取；不得根据 continuation 结果替换。至少 18 个有效 states 且覆盖全部七阶段，否则
`GATE0_INVALID`。

## 4. Target policies 与 rollout

冻结 checkpoint：Full-H seeds 41、42、43，路径和 SHA256 沿用
`docs/cora_vla/asset_audit.md`。不更新任何参数。

每个 `(state, candidate, target_policy)`：

- 先执行同一 candidate 的 `K=2`；
- target policy 固定继续 8 actions；
- 2 个 matched continuation repeats；
- 不同 target policy 使用相同 candidate、simulator snapshot 和 continuation seed schedule；
- 保存 formal success、regress、transport、lift、stable grasp、progress AUC 的 repeat mean；
- 排序继续使用已冻结的 lexicographic key：

```text
success > no-regress > transport > lift > stable-grasp > progress-AUC
```

不根据结果更换 horizon、repeat、key 或候选数。

## 5. Policy-relativity 指标

每个 state 内对每个 policy 的 16 个 key 转为 tie-aware percentile rank。报告：

1. policy pair 的 comparable candidate-pair fraction；
2. 在两 policy 都非 tie 的 candidate pairs 上，preference flip rate；
3. Kendall-style pair agreement；
4. top-tier candidate set Jaccard；
5. leave-one-policy-out selector：用另外两个 policy 的平均 percentile 选候选，在 target policy 上测 percentile；
6. target-policy Oracle 相对 leave-one-policy-out、candidate0 的 percentile gain；
7. 每阶段、每 policy pair、每 source state；
8. source-state bootstrap 95% CI。

## 6. 裁决

`POLICY_RELATIVITY_EXISTS` 需要同时满足：

1. 全部 policy pairs 的 comparable fraction 中位数至少 25%；
2. 至少两个 policy pairs 的 preference flip rate 至少 15%；
3. target-policy Oracle 相对 leave-one-policy-out selector 的平均 percentile gain 至少 10 pp；
4. 该 gain 的 source-state bootstrap 95% CI 下界大于 2 pp；
5. 结果不只来自一个阶段或一个 source state。

实现澄清（首次 rollout 前冻结）：第 1 项对每个 policy pair 取 state-level comparable fraction 的中位数；
第 2 项先在同一 source 内平均 state-level flip rate，再跨 source 取均值；第 3--4 项先在每个 source 内
平均 target policy 与 states，再进行 source bootstrap；第 5 项要求至少三个阶段、至少三个 source 的
平均 flip rate分别达到 15%。

否则输出 `STOP_POLICY_RELATIVE_COMMITTOR`。数据/rollout 不完整则输出 `GATE0_INVALID`。

通过只允许下一步训练固定小型 fingerprint committor，并做 leave-one-policy-out prediction Gate；不允许
直接微调 Pi0.5、打开 confirmation 或宣称闭环提升。
