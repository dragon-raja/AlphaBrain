# 从问题本质出发的课题再审视

## 结论

用户指出的风险成立：热门方向存在近邻是正常的，研究不能把“避开近邻”当优化目标。原候选物理后果 verifier 仍值得做，但只能作为诊断和部署组件，不能单独承担最终方法主张。

修订后的主线是“失败恢复中的 action-level physical process credit assignment”。它正面进入 reward modeling、failure-informed post-training 和 preference optimization 这条热门主线，并尝试解决这些方法仍较粗的部分：

- state-only progress 无法区分同一状态下不同候选动作；
- 成功/失败轨迹对比混合了状态难度与动作贡献；
- expert-action similarity 不等价于真实接触稳定性和阶段推进；
- 只做 inference reranking 不会改善动作生成分布。

进一步从根因复审后，最终方法主张收敛为 **Counterfactual Recovery Advantage Distillation**：同状态物理干预是因果数据获取手段，receding Oracle 是支持集上界，真正需要证明的是无搜索 Pi0.5 能否学会跨重规划持续有效的恢复动作。完整论证与反证条件见 `root_problem_method_review.md`。

## 原方案保留什么

- 同状态 simulator clone；
- N=8 候选的实际物理执行；
- action-conditioned consequence prediction；
- fixed K 闭环公平比较；
- group-preserving split 与 paired bootstrap。

## 原方案删除或降级什么

- “只选哪个候选”不再是最终问题，只是 Gate 0/1；
- “和 A3/SAFE/Reflective VLA 不同”不再作为选择理由；
- K=3 即时 outcome 不再被视作充分标签；
- 任意手调 scalar score 改成结构化、预注册的过程结果；
- 只有 best-of-N 成功不再算策略学习成功。

## 新增的关键识别设计

每个候选执行 K 步后，切换到完全相同的 reference continuation，并共享后续随机数。这样估计的是当前候选对未来阶段的边际影响，而不是不同后续采样造成的偶然差异。

post-regrasp 主状态必须由冻结策略从 feedback 状态闭环执行并在 replanning boundary 捕获。scripted expert 的 post-regrasp 状态过于干净，只能作为控制；否则实验会把“专家给出的容易状态”误当成策略真实恢复分布。

同时保留：

- direct effect：短程接触和位姿结果；
- bridge value：统一 continuation 下的下一阶段/最终结果；
- teacher continuation：仅作可达性上界；
- frozen-policy continuation：作为部署相关主标签。

## 新增的学习目标

同一 restored state 内构造 action preference pairs，用真实执行后的物理阶段结果定义 `DeltaG`。先训练 process critic 验证可预测性，再把同状态 preference 用于 Pi0.5 flow policy 后训练。最终主指标是 posttrained policy 的 sample0 闭环成功率，而不是 reranker 的 best-of-N 上界。

## 与强近邻的态度

不回避 SARM2、AFIL、HAPO/PACT、Reflective VLA、A3 或 LifeLong-RFT。它们是必须击败或解释的强基线。如果 state-only stage reward、失败负指导或普通未配对 preference 已经同样有效，应采用更简单方案并放弃当前主张。

## 当前执行状态

- 已完成同状态 restore parity 初检；
- 已完成 1 个 val group 的开发 smoke，但它来自 dirty worktree 且单 continuation 选择对 held-out continuation 不稳定，只能用于发现测量问题，不能进入裁决；
- 旧 Gate 0 曾在 clean commit `2986a3a` 上完成 seeds `41/42/43`，但后续 full-rollout parity 发现 runtime snapshot 漏掉 observable cache、采样时钟和 OSC/interpolator 状态，原裁决已标记为 `INVALIDATED_REQUIRES_RERUN`；
- training bank 和 simulator 特权 audit bank 将物理分离；
- N=8 positive coverage 为 94.9%，held-out stable regrasp 提升约 10.2 pp，但 transport 仅 +5.6 pp、full success 仅 +4.6 pp 且区间跨 0；
- policy-induced post-regrasp reranking 对 next-stage/transport 无增益；
- 修复后 natural-vs-forced-restore 的动作、图像和 robot state 均为 0 差异，sim state 误差约 `8.3e-15`；
- 下一轮从修复后的完整 runtime state 固定 4 个 replanning boundary，并在每个节点重新做同状态 sibling intervention；先跑 exact receding Oracle，再决定是否做策略后训练；
- 详细结果见 `docs/embodied_research_reset/physical_process_gate0_results.md`。
