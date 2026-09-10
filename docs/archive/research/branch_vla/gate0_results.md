# Branch-VLA Gate 0：Contingent Chunk 表示可学性

正式裁决：**STOP_BRANCH_ACTION_CHUNK**

本 Gate 只打开 train/val episode arrays；original test 和 confirmation episode 均未打开。

## 数据审计

- train：102 groups、30 source states、510 lead examples；
- validation：13 groups、9 source states、65 lead examples；
- lead=`1..5`，prediction horizon=`16`；
- pre-feedback agent-view、wrist、robot state、actions parity failures：0；
- test episode files opened：0；confirmation paths opened：0。

## Outcome guard

- post-feedback validation accuracy：100%；source bootstrap 95% CI `[100,100]`；
- pre-feedback accuracy：50%；
- paired ranking：100%；
- 32 次 shuffled-label control mean：50.5%。

这确认结果只在 feedback 后可路由，不是 pre-feedback branch leakage。

## Action branch 风险

| 方法 | normalized suffix MSE |
|---|---:|
| 单条 `linear_chunk` | 0.510 |
| `two_branch_learned_route` | 0.208 |
| `two_branch_oracle_route` | 0.208 |
| `random_precommit` | 0.829 |
| `per_lead_constant` | **0.134** |

- learned route 相对 linear chunk：风险下降 58.0%，source bootstrap 95% CI `[51.4,64.5]`；
- attached：下降 94.1%，95% CI `[93.6,94.6]`；
- slipped：下降 29.6%，95% CI `[14.4,43.0]`；
- predicted/target branch separation ratio：0.933；
- oracle two-branch 相对 per-lead constant：**风险增加 49.6%**，对应 reduction 95% CI
  `[-72.4,-30.3]`。

## 科学解释

线性 chunk 确实因平均两个 outcome suffix 而受损，随机 latent mode 在结果前提交更差，视觉 guard
也能无泄漏地路由。然而 branch-specific actions 在当前数据中高度模板化：知道 `lead` 与 outcome 后，
不看图像和 robot state 的均值动作已经优于状态条件 predictor。

因此当前任务不足以支持“VLA 应生成 state-conditioned contingent chunks”的方法主张。它只能支持
一个廉价的 outcome classifier 加两个脚本，这既不是通用 VLA 问题，也不值得进入 Pi0.5 架构改造。

正式 artifact：

```text
/share/longjunyu/branch-vla/gate0-representation-v1/gate0_results.json
```

最终裁决：**STOP_BRANCH_ACTION_CHUNK**
