# CORA-VLA Sequential Oracle 最终上界预注册

状态：on-policy Gate 已通过；本文在 Sequential Oracle 数值产生前冻结。Confirmation 保持封存，本轮只使用 13 个 validation groups。

## 固定协议

- Full-H checkpoints：seeds 41/42/43，完全冻结；
- 从完整 episode 的 attached/slipped `feedback_reveal_time` 分支状态开始，公共 scripted prefix 不重复计入；
- 每次 replan 固定执行 K=2，直到正式任务成功或 320 actions；
- 主候选数 N=16；候选由一次 batched stochastic flow forward 产生，pool seed 与 candidate index 全量保存；若 batch16 显存不足，只允许固定 4x4 microbatch；
- policy continuation：2 套 matched rollout seeds、固定 8-action lookahead；选择键依次为 formal success、保持/获得 grasp、避免 regress/empty-lift、transport、progress AUC。它是昂贵上界，不是部署方法；
- 所有方法使用相同 group、episode预算、K、图像/state输入、任务成功条件和 replan频率；Oracle 特权只用于候选选择，不输入 Full-H。

## 方法

1. `single_sample`：N=1 标准 Full-H；
2. `random_pick_N`：N=16 pool 中固定随机选择；
3. `self_consistency_pick`：选择 K=2 action prefix 的 medoid；
4. `oracle_teacher_distance`：选择到当前状态 teacher K=2 continuation 动作距离最小的候选；
5. `oracle_short_physical`：clone 当前完整 runtime state，执行每个候选 K=2，按分支特定即时正确行为、teacher 可恢复性和短时 progress 排序；
6. `oracle_policy_continuation`：从每个 K=2 endpoint 运行两套 matched frozen-policy 8-action lookahead并排序。

Attached 的即时正确行为是保持 grasp 且不错误张开；slipped 未抓住时是避免 empty lift/premature transport，并张开或朝物体恢复。单条 teacher EEF effect 不作为 short-physical 主标签。

## 指标与裁决

报告 attached/slip/overall success、stable regrasp、empty-lift、progress AUC、completion actions、replans、选择正确率、wall-clock、policy forwards 和 simulator transitions。统计单位为 snapshot group，三个 seed 先在组内聚合，再做 paired group bootstrap 95% CI。

输出 `GO_CORA_ENERGY_ROUTING` 必须同时满足：physical 或 policy Oracle 相对 single 的 slip recovery >=15pp、overall >=5pp、attached退化 <=5pp；显著优于 random；K固定；on-policy recall@16 >=60%；三 seed同向或 paired CI排除0；teacher-distance 与至少一个 physical/policy Oracle方向一致。

其余结果输出 `STOP_CORA_ROUTING`。不允许通过 N/K/seed/子集 sweep 改写结论。通过时只生成能量训练计划，不在本任务训练；失败时停止 CORA energy、rerank、flow guidance 与当前任务模块扩展。
