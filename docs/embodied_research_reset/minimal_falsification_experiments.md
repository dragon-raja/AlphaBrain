# Research Reset 最小证伪实验记录

## 共同约束

- 使用冻结 Full-H Pi0.5 seeds `[41,42,43]`；不重训动作策略；
- 使用现有 LIBERO test 13 snapshot groups / 9 source initial states；
- branch outcome、未来状态和 oracle 标签不进入 policy；
- 统计以 snapshot group 为单位，并报告 source-state cluster 敏感性；
- 视频直接写 H.264/avc1、yuv420p、faststart；
- 实验输出位于 `/share/longjunyu/fresh-vla/research-reset`。

## 实验 E1：反馈可观测性

**问题**：反馈揭示后，当前双视角与 robot state 是否仍不能区分 attached/slipped？

**强负对照**：feedback 前一帧、单模态、32 次 shuffled labels。

**预注册通过条件**：post-feedback vision+state accuracy >=85%，group CI 下界 >70%，pre <=60%，shuffle mean <=60%。

**结果**：post=100%，pre=50%，shuffle mean=52.5%，全部门槛通过。

**解释方向**：这里“通过”表示可观测性存在，因而否定“不可观测性是主要根因”。不能从线性 probe 推出 VLA 已正确使用信号。

## 实验 E2：局部 mode coverage 与阶段漏斗

**问题**：Full-H 是否拥有正确 recovery mode，但单次采样没有选中？

**强负对照**：sample0、opposite expert continuation、best-of-N、Stage A self-consistency 与 fixed/oracle K。

**预注册 mode-selection 条件**：any-correct >=70%、CI 下界 >50%、sample0-correct <=50%、availability gap >=20 pp，并有 >=20% best-of-N RMSE 收益。

**结果**：any-correct=100%，sample0-correct=97.4%，gap=2.6 pp，best-of-N RMSE 降低=51.3%。`supports_mode_selection_bottleneck=false`。

**行为补充**：isolated K3 从 regrasp 61.5% 降至 transport/success 15.4%；严格逻辑链的 regrasp->transport 条件率 25.0%。

**裁决**：否定纯随机 mode-selection；保留“within-mode 候选质量和物理后果组合”问题。

## 实验 E3：显式 recovery prompt

**问题**：是否只需要把失败后的 subtask 用语言显式路由给冻结 VLA？

**强负对照**：原任务提示与错误的 success-assumption 提示。

**预注册通过条件**：显式提示相对原任务 recovery 至少 +20 pp，相对错误提示至少 +10 pp；CI 排除 0 或三 seed 均同向。

**结果**：原任务 15.4%，显式提示 0%，错误提示 0%；显式相对原任务 -15.4 pp `[-28.2,-5.1]`。

**裁决**：`supports_prompt_grounded_recovery=false`，停止 prompt-only 路线。

## 三实验联合结论

1. 物理结果已进入当前 observation/state；
2. 冻结 Full-H 通常能产生前三步局部正确 recovery mode；
3. 更明确的 recovery 语言不会把局部动作组合成完整恢复；
4. 行为主要断在 regrasp 之后的 transport 阶段；
5. 下一项最小实验必须直接测试 candidate action 的实际短程物理后果，而不是继续改权重、K、prompt 或 detector。

本轮三个实验均已完成，不存在仍在运行的 GPU 任务。
