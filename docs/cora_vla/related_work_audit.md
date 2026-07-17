# CORA-VLA 最近邻审计

审计日期：2026-07-17 UTC

阶段结论：`CORA_NOVELTY_CLEAR_FOR_PILOT`

## 审计问题

本审计不以“避开热门工作”为目标，而是检查公开工作是否已经完整覆盖以下组合：

1. 同一前史、不同真实物理结果的对称 continuation swap；
2. 反馈前能量中立、反馈后候选偏好翻转；
3. 冻结 VLA 动作生成器；
4. 仅学习 history-action compatibility，并在推理时 sample-and-rerank。

截至审计日，没有发现完整覆盖四项的公开工作。CORA 仍只是可检验假设，不因形式差异自动
构成贡献；Gate 1 必须先证明冻结策略确实采得到恢复模式。

## 强近邻

| 工作 | 是否更新生成器 | 偏好/负样本 | 与 CORA 的关键差异 |
| --- | --- | --- | --- |
| [FlowPRO](https://arxiv.org/abs/2606.05468) | 是 | intervention/rollback 产生正负轨迹，并以 flow preference objective 更新 VLA | 不冻结生成器；不是当前物理结果下的 history-action reranker |
| [Set-Supervised Diffusion Policy](https://arxiv.org/abs/2606.01865) | 是 | 人类纠正形成 desired/undesired action chunks | 学习 diffusion policy 本身；没有反馈前中立与对称 outcome swap |
| [AFIL](https://arxiv.org/abs/2605.08434) | 是 | 成功/失败分布由双 action generator 建模并用于 guidance | 失败动作主要是全局分布负指导，不是同一动作随 outcome 翻转标签 |
| [DEFLECT](https://arxiv.org/abs/2605.19294) | 是 | stale observation chunk 对 future observation chunk 的时间反事实偏好 | 处理异步延迟并更新 VLA；物理 outcome twin 与冻结 rerank 不同 |
| [PC-Flow](https://doi.org/10.1609/aaai.v40i12.37971) | 否，学习 preference classifier | 通用 flow preference | 与轻量 energy/guidance 最接近，但不是机器人 outcome twin 或 pre-feedback neutrality |
| [DreamAvoid](https://arxiv.org/abs/2605.11750) | 冻结 proposer，训练 dream evaluator | 对 N 个候选预测短时未来并排序 | 最接近 sample-and-rerank；依赖 world-model 式 future dreaming 和 critical trigger，CORA 只评分已观察结果与动作兼容性 |
| [Feedback World Model](https://arxiv.org/abs/2605.15705) | 冻结策略，可指导 diffusion | 在线修正预测与真实反馈的偏差 | 显式预测未来；本阶段明确不引入 world model |
| [B2FF](https://arxiv.org/abs/2606.09258) | 不微调低层 generator | 选择预生成的未来视觉 milestone | 路由视觉目标而非基础 action samples，不使用 continuation swap |
| [ReCoVLA](https://arxiv.org/abs/2606.09630) | 冻结 VLA，另训 residual policy | VLM 编译 failure-conditioned reward | 是独立恢复策略与 RL，已被本阶段范围排除 |
| [Dream2Fix](https://arxiv.org/abs/2603.13528) | 是 | world model 合成 failure-correction pairs | 生成并学习恢复轨迹，不是冻结 generator 的状态依赖动作 rerank |

通用 Diffusion Policy 已证明生成策略可以表达多模态动作分布，但没有回答特定物理结果后能否
从基础 VLA 的有限样本中检索正确模式。Flow Motion Policy 的 best-of-N 也说明冻结随机生成器
加外部可行性筛选是合理工程基线，但其筛选目标是碰撞可行性，不是 outcome-conditioned
continuation compatibility。

## 暂时保留的主张

CORA pilot 只主张验证一个窄问题：**真实结果已可观察时，成对反事实监督能否学习一个
状态依赖的 action compatibility function，从冻结 VLA 候选中选回正确模式。**

以下均不作为新颖性主张：best-of-N、action critic、energy guidance、失败动作负样本、冻结
生成器或多帧编码本身。若 generic critic、current-frame scorer、self-consistency 或
DreamAvoid 类 future evaluator 达到同等效果，应采用更简单或更成熟的方法。
