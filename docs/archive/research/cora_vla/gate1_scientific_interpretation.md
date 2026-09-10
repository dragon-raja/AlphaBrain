# CORA-VLA Gate 1 科学解释

## 状态区分

原预注册裁决保持不变：**STOP_CORA_CANDIDATE_SUPPORT**。本报告不把 Gate 1 改写成正式通过，也不修改原始指标或门槛。

但这一形式裁决不能解释为“基础策略缺少恢复模式”。更准确的科学状态是：

**CANDIDATE_SUPPORT_EXISTS_BUT_SEQUENTIAL_ROUTING_UNRESOLVED**

## 证据

1. Attached 的 action recall@N 和短时物理 recall@N 均为 100%，但联合 recall@16 仅 74.4%。逐候选审计显示 action/physical 一致率为 98.2%，joint/physical 一致率仅 13.9%。低联合 recall 主要是 EEF-effect 标签拒绝了物理可行但不复现单条 teacher EEF 位移的替代轨迹。
2. Slipped 联合 correct-mode recall@16 和 recall@32 均为 100%，说明在 teacher post-feedback 状态中，冻结 Full-H 能稳定采到 recovery mode。
3. Slipped physical success@1 已达 89.7%，best-of-16 为 100%，单次选择的绝对 headroom 只有 10.3 个百分点，低于预注册 15pp 门槛。
4. Gate 1 的物理标签只执行一个候选 K=2，随后由 teacher 接管。它没有测试 Full-H 连续多次 replan 时的选择错误是否累积，也没有测试正确候选在 Full-H 自己偏离后的 on-policy 状态中是否仍存在。

## 可以与不可以得出的结论

可以得出：teacher post-feedback 状态中的 slipped 候选支持充分；Attached 联合标签存在明显单轨迹偏置；单次 best-of-N 的收益空间有限。

不能得出：CORA 的连续路由一定无效；on-policy failure state 中仍有相同候选支持；一个可学习 scorer 能达到 Sequential Oracle；或者能量路由能提高完整闭环任务成功率。

因此下一步只允许进行 on-policy candidate support 和一次 Sequential Oracle 上界。若 on-policy recall@16 低于 60%，或 Sequential Oracle 不产生规定的闭环收益，立即停止 CORA routing，不训练能量模型。
