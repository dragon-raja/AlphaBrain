# CORA-VLA Sequential Oracle 最终 Gate

正式评测只使用冻结的 13 个 validation snapshot groups；正式 evaluator 未访问 confirmation。所有方法固定 N=16（single 为同一候选流的 candidate 0）、K=2、最长 320 actions，基础 Full-H 参数冻结。

> 流程披露：早期 batch smoke 的通用 NPZ 文件搜索曾打开 confirmation 文件并枚举 archive key 名；没有加载数组、没有查看 observation 内容、没有选择 group，也没有将其用于方法或裁决。正式 evaluator 对 confirmation 路径 fail-closed。故结论是“正式结果未使用 confirmation”，而非“全流程零元数据访问”。

## 闭环主结果

| 方法 | Attached成功 | Slip恢复成功 | Overall成功 | Failure continuation | Premature commitment | 平均actions | 平均wall time |
|---|---:|---:|---:|---:|---:|---:|---:|
| `single_sample` | 94.9% | 71.8% | 83.3% | 28.2% | 35.9% | 123.4 | 27.8s |
| `random_pick_N` | 100.0% | 61.5% | 80.8% | 28.2% | 33.3% | 125.8 | 77.0s |
| `self_consistency_pick` | 82.1% | 30.8% | 56.4% | 15.4% | 15.4% | 188.0 | 63.8s |
| `oracle_teacher_distance` | 23.1% | 0.0% | 11.5% | 5.1% | 5.1% | 296.8 | 106.9s |
| `oracle_short_physical` | 100.0% | 61.5% | 80.8% | 20.5% | 20.5% | 119.1 | 175.6s |
| `oracle_policy_continuation` | 100.0% | 87.2% | 93.6% | 23.1% | 35.9% | 98.5 | 397.1s |

## 相对 Full-H single

- `random_pick_N`：slip -10.3% (95% CI [-28.2%, 10.3%])；overall -2.6%；attached 5.1%。
- `self_consistency_pick`：slip -41.0% (95% CI [-61.5%, -17.9%])；overall -26.9%；attached -12.8%。
- `oracle_teacher_distance`：slip -71.8% (95% CI [-84.6%, -59.0%])；overall -71.8%；attached -71.8%。
- `oracle_short_physical`：slip -10.3% (95% CI [-25.6%, 5.1%])；overall -2.6%；attached 5.1%。
- `oracle_policy_continuation`：slip 15.4% (95% CI [2.6%, 28.2%])；overall 10.3%；attached 5.1%。

## Seed 与行为诊断

- `single_sample`：slip [s41=84.6%, s42=76.9%, s43=53.8%]；stable regrasp=76.9%；被即时启发式判对的选择=81.4%；候选池即时正确率=81.4%。
- `random_pick_N`：slip [s41=84.6%, s42=38.5%, s43=61.5%]；stable regrasp=74.4%；被即时启发式判对的选择=78.2%；候选池即时正确率=78.0%。
- `self_consistency_pick`：slip [s41=38.5%, s42=23.1%, s43=30.8%]；stable regrasp=64.1%；被即时启发式判对的选择=77.6%；候选池即时正确率=78.6%。
- `oracle_teacher_distance`：slip [s41=0.0%, s42=0.0%, s43=0.0%]；stable regrasp=12.8%；被即时启发式判对的选择=99.3%；候选池即时正确率=95.3%。
- `oracle_short_physical`：slip [s41=69.2%, s42=61.5%, s43=53.8%]；stable regrasp=69.2%；被即时启发式判对的选择=96.3%；候选池即时正确率=89.4%。
- `oracle_policy_continuation`：slip [s41=92.3%, s42=92.3%, s43=76.9%]；stable regrasp=94.9%；被即时启发式判对的选择=78.1%；候选池即时正确率=78.5%。

即时正确标签与最终成功并不等价：teacher-distance 的选择有 99.3% 被即时启发式判对，但 slip 成功为 0%；short-physical 的即时正确选择为 96.3%，slip 成功仍只有 61.5%。这说明局部 action/teacher 标签不足以监督长期路由。

## 裁决审计

On-policy correct-mode recall@16=96.2%，候选支持必要条件已满足。
- `oracle_short_physical`：未通过 slip_gain_at_least_15pp、overall_gain_at_least_5pp、significantly_better_than_random、three_seed_direction_or_ci、teacher_distance_same_direction
- `oracle_policy_continuation`：未通过 teacher_distance_same_direction

统计单位为 snapshot group；三个 seed 先在组内聚合，再进行 paired group bootstrap。候选、帧与 replan 没有被当成独立样本。

视频审计：468/468 可解码，成功/失败视频=317/151，全部 H.264/avc1/yuv420p/faststart。

## 科学解释

最强 policy-continuation Oracle 的 slip 提升达到 15.4%，overall 提升 10.3%，证明连续闭环中确实存在可利用的候选路由 headroom。其三个 seed 的 slip 方向均为正。

但该上界平均 wall time=397.1s，是 single 的 14.3 倍；更重要的是，短物理 Oracle 与 random 的 slip 成功同为 61.5%，teacher-distance 的 slip 成功为 0%。因此，headroom 只在昂贵的 frozen-policy future rollout 中出现，当前 CORA 可训练局部 target 没有得到同向验证。

正式停止当前 CORA energy/reranking/flow-guidance 路线。这不等价于“基础策略没有恢复模式”，也不否定一般的 sequential routing 问题；它否定的是用当前 teacher-distance 或 K=2 局部物理标签训练 CORA selector 的证据链。

## 最终结论

STOP_CORA_ROUTING
