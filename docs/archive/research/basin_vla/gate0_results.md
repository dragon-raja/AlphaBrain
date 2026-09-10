# BASIN-VLA Gate 0：Policy Relativity

正式裁决：**STOP_POLICY_RELATIVE_COMMITTOR**

本实验仅使用既有 validation on-policy cache；没有打开 test 或 confirmation。

## 数据

- 有效 states：21；source states：8。
- 七阶段各 3 states：feedback reveal、failure continuation、recovery start、reapproach、preclose、
  post-regrasp、final failure。
- 所有 target policies 使用相同 simulator snapshot、相同 16 candidates 与 matched rollout seed。
- 三个 policy collectors 均为 21/21 valid，same-state direct endpoint hash 审计通过。

## Cross-policy preference

| Policy pair | Comparable median | Flip rate（source mean, 95% CI） | Top-tier Jaccard |
|---|---:|---:|---:|
| `41-42` | 0.0% | 15.7% [1.2, 40.9] | 72.9% |
| `41-43` | 0.0% | 17.7% [3.4, 42.2] | 68.5% |
| `42-43` | 0.0% | 7.2% [2.4, 13.4] | 86.9% |

Target-policy Oracle 相对 leave-one-policy-out selector 为 `+10.6 pp`，source bootstrap 95% CI
`[+2.9,+19.2]`；相对 candidate 0 为 `+24.1 pp`。

## 为什么仍然停止

- `data_valid`：通过；
- 至少两组 policy pair flip >=15%：通过；
- Oracle-minus-LOO 均值与 CI：通过；
- 每组 state-median comparable fraction >=25%：失败，三组均为 0；
- 至少三个阶段和三个 source 广泛成立：失败，只有 2 个阶段、1 个 source。

差异只出现在 preclose/reapproach 的少量状态，多数状态的 16 个候选在有限 continuation key 下完全
并列。当前数据不支持“通用 critic 因混合不同 checkpoint competence basins 而系统失效”的主张。
训练 policy fingerprint committor 会把局部现象升级成未经支持的算法复杂度。

正式 artifact：

```text
/share/longjunyu/basin-vla/policy-relativity-gate0-v1/gate0_results.json
```

最终裁决：**STOP_POLICY_RELATIVE_COMMITTOR**
