# CORA-VLA Gate 1 预注册

状态：**在查看 Gate 1 数值结果前冻结**  
数据边界：仅使用 `libero-full-episode-v2-128` 的 `val` split；不得读取新 confirmation groups 的图像、动作或 CORA 指标。

## 问题与固定设置

Gate 1 只回答一个问题：冻结 Full-H Pi0.5 在看到真实抓取结果后，候选分布中是否已经包含与该结果兼容的动作模式。本阶段不训练模型。

- checkpoint：资产审计中冻结的 Full-H seed 41/42/43、step 10353；
- 状态：每个 validation group 的 `feedback_reveal_time`；attached/slipped 分开恢复；
- 候选数：一次固定生成 32 个完整 H=10 action chunks，并报告前缀 N=1/4/8/16/32；
- 执行尺度：固定 K=2；
- 候选随机种子：由 checkpoint seed、pair id 和候选序号确定，成对分支共享；
- 主 action margin：归一化动作 RMSE 差 `0.02`；
- 主 effect margin：两步 EEF 位移 RMSE 差 `0.002 m`；
- simulator、flow steps、语言指令和图像变换均保持现有正式链路不变。

## 标签冻结

### 1. Action 标签

候选前 K=2 步到正确 continuation 的 RMSE 加 `0.02`，严格小于到交换 continuation 的 RMSE，才算 action-compatible。

### 2. EEF-effect 标签

从同一 snapshot 执行候选 K=2 后，将 EEF 位移分别与正确、交换 continuation 的已记录 K=2 位移比较。到正确 effect 的 RMSE 加 `0.002 m`，严格小于到交换 effect 的 RMSE，才算 effect-compatible。

Gate 1 的主 `correct-mode recall@N` 使用 **action-compatible 与 effect-compatible 的交集**。两者各自的 recall 和一致率同时报告，不能事后替换主口径。

### 3. 独立短时物理标签

每个候选从对应 snapshot 单独恢复并执行 K=2，然后由同一个 `FullEpisodeTeacher` 接管，最多执行 320 步。

- attached：K=2 全程保持 grasp、未掉落，并且 teacher 最终完成正式任务；
- slipped：K=2 不出现 empty-lift，至少一步张开夹爪/朝物体移动或 EEF-object 距离净减少至少 1 mm，并且 teacher 最终完成正式任务。

`oracle best-of-N physical success` 表示前 N 个候选中至少一个满足上述分支特定标签。只看 teacher 最终成功不构成候选兼容，因为强 teacher 可能掩盖错误候选。

## 反馈前泄漏审计

在 `feedback_reveal_time - 1` 恢复成对分支，要求：

- 两路 agent-view、wrist-view、robot state 和 simulator state 一致；
- 相同随机种子的冻结策略候选逐元素一致；
- 当前输入不得包含 branch、grasp/contact、teacher phase、success 或未来状态。

任何成体系的输入或候选差异都先按数据泄漏调查；无法排除时输出 `COUNTERFACTUAL_DATA_LEAKAGE` 并停止。

实现时每个 group、每个 checkpoint seed 使用一个成对随机候选作运行时一致性检查；图像、state 与 simulator state 仍逐元素全量检查。这样形成 13 x 3 次独立候选审计，不重复为完全相同输入生成 32 次，不影响 post-feedback 的 N=32 主实验。

## 辅助状态

另报告 attached 的 transport/place 状态、slipped isolated-recovery 状态，以及已有 policy-state correction 的实际失败状态的候选诊断。这些辅助状态不改变主 Gate 阈值，也不允许用于 confirmation 选择。

## Gate 1 决策

通过必须同时满足：

1. attached 主 recall@16 >= 80%；
2. slipped 主 recall@16 >= 50%；
3. slipped oracle best-of-16 physical success 比 sample-1 至少提高 15 个百分点；
4. 反馈前泄漏审计通过。

若 slipped 主 recall@32 < 40%，直接输出 `BASE_POLICY_LACKS_RECOVERY_SUPPORT`。其余未满足通过条件的情况输出 `STOP_CORA_CANDIDATE_SUPPORT`。Gate 1 未通过时不实现或训练 Counterfactual Outcome Energy。

统计单位始终是 snapshot group。报告三个 checkpoint seed 的独立结果、跨 seed 均值，并对组级差值做 paired bootstrap 95% CI；候选或帧不得作为独立样本扩大显著性。
