# FRESH-VLA Oracle Plan-Commit 最终上界裁决

## 实验完整性

- 冻结 Full-H checkpoints：seeds=[41, 42, 43]，未重训、未修改权重；
- 配对 test snapshot groups：13；对应独立 source initial states：9；
- 该 test split 已参与上一阶段最终裁决，本轮是 locked/post-hoc 上界实验，不是新盲测集；
- `feedback_reveal_time > action_divergence_time` 反向组：0；
- fixed K=1/2/3 复用既有结果；新增方法完成 isolated、end-to-end 和 deterministic reach；
- 统计单位为 snapshot group，先在 group 内跨 seed 平均，再做 paired bootstrap；
- 另以 source initial state 为聚类单位做保守敏感性分析，最终 GO 必须同时通过两套门槛；
- teacher 绝对时钟在偏离轨迹上无效，因此实际 Oracle 是 privileged runtime grasp/lift event interrupt 上界；事件 outcome 从未进入 Pi0.5 输入或动作选择。
- 两个 Oracle 的调度 trace 与行为结果逐行一致；独立 EGL 运行的编码视频配对中，1/117 个 MP4 未达到文件字节一致，其余完全一致。该渲染/编码复现性差异单独披露，不参与方法胜负。

## 主结果

括号为 snapshot-group bootstrap 95% CI。

| 方法 | Overall | Attached | Slip recovery | Isolated recovery | Forward calls | Inference s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_k1` | 3.8% [0.0, 7.7] | 7.7% [0.0, 15.4] | 0.0% [0.0, 0.0] | 2.6% [0.0, 7.7] | 318.3 | 80.34 |
| `fixed_k2` | 5.1% [0.0, 11.5] | 10.3% [0.0, 23.1] | 0.0% [0.0, 0.0] | 15.4% [5.1, 25.6] | 155.2 | 39.17 |
| `fixed_k3` | 14.1% [5.1, 24.4] | 23.1% [7.7, 41.0] | 5.1% [0.0, 12.8] | 15.4% [5.1, 28.2] | 101.9 | 25.73 |
| `oracle_branch_safe_commit` | 14.1% [6.4, 23.1] | 23.1% [10.3, 38.5] | 5.1% [0.0, 12.8] | 15.4% [5.1, 28.2] | 102.4 | 26.66 |
| `oracle_feedback_reveal_commit` | 14.1% [6.4, 23.1] | 23.1% [10.3, 38.5] | 5.1% [0.0, 12.8] | 15.4% [5.1, 28.2] | 102.4 | 26.15 |
| `gripper_commit` | 9.0% [3.8, 15.4] | 17.9% [7.7, 30.8] | 0.0% [0.0, 0.0] | 17.9% [5.1, 33.3] | 111.1 | 28.42 |
| `random_matched_commit` | 15.4% [6.4, 25.6] | 23.1% [7.7, 41.0] | 7.7% [0.0, 15.4] | 15.4% [5.1, 28.2] | 101.6 | 24.19 |
| `self_consistency_commit` | 14.1% [5.1, 23.1] | 28.2% [10.3, 46.2] | 0.0% [0.0, 0.0] | 7.7% [0.0, 15.4] | 1226.9 | 300.01 |

## Oracle 配对差异

| 对照 | Overall | Slip recovery | Isolated recovery | Failure continuation | Premature commitment |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_k3` | +0.0 [-3.8, +3.8] pp | +0.0 [+0.0, +0.0] pp | +0.0 [+0.0, +0.0] pp | -51.9 [-83.3, -16.7] pp | -51.9 [-83.3, -16.7] pp |
| `random_matched_commit` | -1.3 [-5.1, +2.6] pp | -2.6 [-7.7, +0.0] pp | +0.0 [+0.0, +0.0] pp | -40.7 [-74.1, -5.6] pp | -40.7 [-74.1, -5.6] pp |
| `gripper_commit` | +5.1 [-3.8, +15.4] pp | +5.1 [+0.0, +12.8] pp | -2.6 [-12.8, +7.7] pp | -60.0 [-90.0, -30.0] pp | -60.0 [-90.0, -30.0] pp |
| `self_consistency_commit` | +0.0 [-10.3, +11.5] pp | +5.1 [+0.0, +12.8] pp | +7.7 [-7.7, +23.1] pp | -58.3 [-83.3, -33.3] pp | -58.3 [-83.3, -33.3] pp |
| `fixed_k1` | +10.3 [+2.6, +19.2] pp | +5.1 [+0.0, +12.8] pp | +12.8 [+5.1, +23.1] pp | +0.0 [+0.0, +0.0] pp | +0.0 [+0.0, +0.0] pp |

## 边界可识别性限制

本数据的 feedback reveal、action divergence 和物理事件标签完全重合，因此两个 Oracle 调度在本轮没有可识别差异。该事实限制机制归因，但不影响判断‘Oracle 执行承诺上界是否优于固定或近邻控制’。
本轮只能裁决 privileged runtime-event-aligned Plan-Commit 上界，不能独立比较 branch-safe 与 feedback-reveal 两种边界标签；若输出 STOP，其证据含义是连该更强事件上界也未通过。

## 预注册门槛

- `oracle_vs_fixed_k3_primary_effect`: `False`
- `oracle_vs_random_unique_effect`: `True`
- `oracle_vs_gripper_stable_effect`: `False`
- `oracle_vs_self_consistency_stable_effect`: `True`
- `attached_degradation_pp`: `0.0`
- `attached_preserved_within_5pp`: `True`
- `fixed_k1_overall_success_comparable_within_5pp`: `True`
- `policy_forward_call_reduction_vs_fixed_k1`: `0.6782535143190881`
- `fixed_k1_efficiency_gate`: `True`
- `behavior_error_reduced`: `True`

## Source-state 聚类敏感性门槛

- `oracle_vs_fixed_k3_primary_effect`: `False`
- `oracle_vs_random_unique_effect`: `True`
- `oracle_vs_gripper_stable_effect`: `False`
- `oracle_vs_self_consistency_stable_effect`: `True`
- `attached_degradation_pp`: `0.0`
- `attached_preserved_within_5pp`: `True`
- `fixed_k1_overall_success_comparable_within_5pp`: `True`
- `policy_forward_call_reduction_vs_fixed_k1`: `0.6793906445329535`
- `fixed_k1_efficiency_gate`: `True`
- `behavior_error_reduced`: `True`
- `source_state_sensitivity_decision`: `STOP_FRESH_FAMILY`

fixed-K 历史结果缺少逐调用计时，因此其 wall-clock 由相同 seed 新运行的单样本方法每次调用中位数估算；真实 invocation/forward-call count 未估算。最终效率门槛优先使用真实 forward-call count。
历史 fixed-K JSON 记录了精确 seed、split、protocol 和 rows，但远程 policy socket payload 当时没有嵌入 checkpoint hash；本轮保存其 JSON SHA256，并以原 runner 路径约定、当前冻结权重实测 SHA256 和单组逐协议 parity 共同绑定。该限制不会被表述成密码学级历史证明。

机器可读结果：`/share/longjunyu/fresh-vla/runs/libero-oracle-commit-final-v1/summary/results.json`

Pareto 图：`/share/longjunyu/fresh-vla/runs/libero-oracle-commit-final-v1/summary/success_efficiency_pareto.png`

STOP_FRESH_FAMILY
