# CORA-VLA On-policy Candidate Support 预注册

本 Gate 使用完整 LIBERO v2 的 validation groups 和冻结 Full-H seeds 41/42/43；confirmation 保持封存。状态由 Full-H 从真实 slipped feedback snapshot 以固定 K=2 自主闭环产生，teacher 不参与状态生成。

## 状态阶段

每个 `(seed, group)` 最多保留各阶段首次出现的 replanning boundary：`feedback_reveal`、`failure_continuation`、`recovery_start`、`reapproach`、`preclose`、`post_regrasp`，失败 episode 另保留 `final_failure`。缺失阶段按 intention-to-treat 报告为不可达，不以此筛选 group。

## 候选与主标签

每个状态采样 N=16，候选种子固定由 checkpoint seed、pair id、自然 replan index 和 candidate index 得到；candidate 0 与自然 Full-H 下一次调用同 seed。每个候选从完整 simulator/controller/runtime snapshot 的独立 branch restore 执行 K=2。

主 `immediate_correct_mode`：

- 若状态已抓住物体：K=2 全程保持 grasp，且不错误张开夹爪；
- 若未抓住：K=2 不产生 empty lift 或 premature transport，且至少出现张开/朝物体移动的 recovery action，或在 K=2 内获得 grasp。

这一定义不使用单条 teacher action/EEF 距离，避免 Gate 1 已证实的单轨迹假阴性。Action/effect 距离仅作辅助。每个候选另记录 K=2 后同一 teacher 能否完成任务。

## Gate

按 snapshot group 聚合三个 seed，在全部可达 slipped on-policy states 上报告 recall@1/4/8/16，并分阶段报告。若主 recall@16 <60%，输出 `BASE_POLICY_ONPOLICY_SUPPORT_INSUFFICIENT`，不运行长程 policy continuation 或 Sequential Oracle。

若 recall@16 >=60%，再对缓存候选补充冻结 Full-H continuation completion，并进入 N=16、K=2 Sequential Oracle。该顺序只节省失败 Gate 后的无效长程 rollout，不改变任何候选、标签或阈值。
