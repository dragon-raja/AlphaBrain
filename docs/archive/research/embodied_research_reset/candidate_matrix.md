# 具身智能 Research Reset：候选问题矩阵

## 评分约定

除“组内重合风险”外，1 分最弱、5 分最强；“组内重合风险”1 分表示低风险，5 分表示高度可能与持续学习、数据或类脑方向重复。这里不求和机械选题：一个高总分但被近期工作直接覆盖的题目仍应淘汰。

| ID | 可证伪问题 | 科学重要性 | 原创空间 | 现有证据关联 | 资源匹配 | 最小实验成本 | 可解释性 | 闭环可评 | 两月扩展 | 组内重合风险 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 当前双视角与 proprioception 是否无法辨别抓取成功/滑落？ | 4 | 2 | 5 | 5 | 5 | 5 | 4 | 2 | 2 |
| C2 | 恢复 mode 是否存在于 Full-H 分布中，却因随机采样未被选中？ | 4 | 3 | 5 | 5 | 5 | 5 | 4 | 3 | 3 |
| C3 | 原任务语言是否掩盖了失败后的 recovery subtask，显式提示能否恢复？ | 3 | 1 | 4 | 5 | 5 | 5 | 4 | 2 | 3 |
| C4 | 局部恢复动作存在时，策略是否缺少对候选 chunk 的物理阶段转移验证？ | 5 | 4 | 5 | 4 | 4 | 4 | 5 | 5 | 3 |
| C5 | 单帧反应策略是否需要 observation-action-consequence 历史才能稳定恢复？ | 5 | 2 | 4 | 3 | 3 | 3 | 5 | 4 | 4 |
| C6 | 遮挡/隐接触是否要求主动视角或安全 probe action？ | 4 | 2 | 3 | 3 | 3 | 4 | 5 | 4 | 4 |
| C7 | 恢复失败是否必须通过 failure-aware RL/post-training 改写策略分布？ | 5 | 2 | 4 | 2 | 1 | 2 | 5 | 4 | 5 |

## C1：反馈后结果可观测性

- **核心矛盾**：标签规定 feedback 已揭示，但 Pi0.5 实际输入中的低维视觉/状态未必能区分 attached 和 slipped。
- **科学假设**：若 held-out source-state 上的简单 probe 不能识别 branch outcome，恢复失败主要是部分可观测性问题，而不是动作选择问题。
- **重要性**：决定后续应改表示/视角，还是改决策与控制。
- **最近邻**：[SAFE](https://arxiv.org/abs/2506.09937)、[ActiveVLA](https://arxiv.org/abs/2601.08325)。
- **差异**：本实验不是训练通用 failure detector，只审计本数据在严格 source-state-disjoint split 上是否含足够信息。
- **最小实验**：对 feedback offset `[-1,0,1,2,3,5]` 的 agent-view、wrist、robot state 训练 ridge probe；正标签不输入 policy；加入 32 次 shuffle control。
- **资源**：现有完整 episode，CPU；无需模型推理或重训。
- **一周信号**：feedback 后准确率显著高于前一帧与 shuffle，或明确保持不可辨。
- **No-Go**：post-feedback accuracy >=85%、group CI 下界 >70%，而 pre-feedback 和 shuffle <=60%；此时“看不见失败”不能作为主因。

## C2：随机 mode 可用性与选择

- **核心矛盾**：Full-H 可能学到了恢复轨迹，但单次 flow sample 总落入错误成功 mode。
- **科学假设**：反馈后 N=8 样本中有正确 mode，而 sample 0 经常错误；一个 mode selector 即可改善恢复。
- **重要性**：若成立，可冻结大模型，仅训练轻量选择器。
- **最近邻**：[A3](https://arxiv.org/abs/2605.11567)、AAC、自一致性选择和 diffusion best-of-N。
- **差异**：按 attached/slip expert continuation 定义恢复 mode，而不是只按样本方差或动作熵。
- **最小实验**：三 seed、13 test groups、反馈前后分别 N=8，比较 any-correct、sample0-correct、best-of-N RMSE 和 opposite mode。
- **资源**：冻结 Full-H，3 张 GPU，约分钟级到小时级。
- **一周信号**：any-correct 明显高于 sample0-correct，且 best-of-N 对 expert action 有稳定收益。
- **No-Go**：sample0 本身已几乎总在正确局部 mode；此时“随机没抽到 recovery”不是主要断点。

## C3：显式 recovery language routing

- **核心矛盾**：原 instruction 始终是 `put the cream cheese in the bowl`，没有显式说明失败后应重新抓取。
- **科学假设**：基础 VLA 的语言知识足以执行恢复，但训练/部署提示没有路由到该 subtask。
- **重要性**：若简单提示就能解决，可避免新增网络和训练。
- **最近邻**：[RoboFAC](https://arxiv.org/abs/2505.12224)、[FailSafe](https://arxiv.org/abs/2510.01642)、TCoT、RACER。
- **差异**：这里只做冻结策略的上界审计，不把 prompt 方法包装成论文贡献。
- **最小实验**：isolated slip、K=3、三 seed；比较原任务、显式失败恢复提示和错误的“已经抓住”提示。
- **资源**：6 张 GPU 并行，现有 test groups，无训练。
- **一周信号**：显式提示相对原任务恢复率至少 +20 pp，且相对错误提示至少 +10 pp；CI 排除 0 或三 seed 同向。
- **No-Go**：提示收益低于门槛或不稳定；不继续堆叠 prompt engineering。

## C4：候选动作的物理阶段转移验证

- **核心矛盾**：模型能开始恢复、甚至重新抓取，却不能稳定完成 `regrasp -> lift -> transport -> place`；动作相似度与采样一致性没有反映实际后果。
- **科学假设**：同一状态下的候选短 chunk 在“下一物理阶段是否完成”上可分，显式的 candidate-conditioned consequence/progress verifier 能比动作熵、自一致性和固定 K 更好地选择候选。
- **重要性**：直接针对 VLA 局部合理但闭环长程失败的问题，主指标落到恢复、进度与任务成功。
- **最近邻**：[Reflective VLA](https://arxiv.org/abs/2606.25215)、[ProgVLA](https://arxiv.org/abs/2605.28231)、[RoboReward](https://arxiv.org/abs/2601.00675)、SAFE、VLA-Corrector。
- **差异**：不预测执行长度；不只检测当前失败；不构建完整像素 world model。输入当前 observation/state、候选 1--3 步 chunk 与短后果历史，输出候选是否完成下一可验证阶段及风险。
- **最小实验**：先用 N=8 mode probe 证明局部正确动作可用，再用行为漏斗证明 regrasp 与最终成功之间存在大断层；下一步在 simulator state clone 上执行候选 1--3 步，测试 oracle consequence ranking 上界。
- **资源**：冻结 Full-H、LIBERO state clone、轻量 verifier；中等 GPU 和 rollout 成本。
- **一周信号**：candidate oracle 在 regrasp/lift/transport transition 上比 sample0/self-consistency 提升 >=15 pp，并能训练轻量 probe 复现至少一半收益。
- **No-Go**：不同候选短 chunk 的物理阶段结果不可分，或 oracle candidate selector 也不能提高阶段转移。

## C5：动作后果历史表征

- **核心矛盾**：当前帧显示结果，但策略不知道“刚才执行了什么以及产生了什么变化”，同一图像可能对应不同控制上下文。
- **科学假设**：短 observation-action-consequence 历史而非更多静态视觉，使策略区分接触稳定性和恢复进度。
- **重要性**：可能解释单帧局部动作正确但连续阶段不稳定。
- **最近邻**：[Reflective VLA](https://arxiv.org/abs/2606.25215) 几乎直接覆盖 action-consequence history。
- **差异**：只有与候选恢复阶段验证结合、并在 failure recovery 而非部署标定偏移上得到独特结果时才有空间。
- **最小实验**：matched current-only、frame-history、action-history、consequence-triplet 四组轻量 probe；不先微调整个 VLA。
- **资源**：现有逐帧 episode；若进入 policy finetune 则成本中等。
- **一周信号**：triplet 在 held-out source-state 的阶段/后果预测显著优于 history-only。
- **No-Go**：普通帧历史已匹配，或 Reflective VLA 复现直接覆盖全部收益。

## C6：主动视角或安全 probe action

- **核心矛盾**：腕部相机可能被夹爪遮挡，静态 observation 对接触状态存在不可消除歧义。
- **科学假设**：一个低风险抬升/轻触/视角调整动作能提高 branch belief，并减少错误恢复。
- **重要性**：在真实接触任务中有明确安全与可观测性价值。
- **最近邻**：[ActiveVLA](https://arxiv.org/abs/2601.08325)、[ProbeAct](https://arxiv.org/abs/2606.09740)。
- **差异**：需要从反事实 value-of-information 和安全约束建立窄边界，不能只是“移动相机看清楚”。
- **最小实验**：只对 C1 中不可辨样本，在 simulator 中比较 passive wait、固定 probe 和 oracle view 的 branch classification/恢复收益。
- **资源**：需要新 intervention 与安全约束；中等。
- **一周信号**：probe 使不可辨样本准确率/恢复率提高 >=20 pp，正常任务退化 <=5 pp。
- **No-Go**：C1 已证明反馈后普遍可辨，或 probe 不改善闭环恢复。

## C7：failure-aware RL/post-training

- **核心矛盾**：行为克隆优化动作匹配，不直接优化失败后的任务恢复与物理进度。
- **科学假设**：带 calibrated progress/recovery reward 的 post-training 能把已有局部 mode 组合成完整恢复。
- **重要性**：最终可能获得最大成功率收益。
- **最近邻**：pi0.6、RL Token、action-chunk PPO、RoboReward、ProgVLA。
- **差异**：当前没有足够窄的独特算法边界；只能作为验证机制后的优化工具。
- **最小实验**：先冻结 policy，训练 candidate value/reward 并做 reranking；只有 reranking 上界成立才进入 LoRA/小规模 RL。
- **资源**：online rollout 与稳定训练成本高；当前不适合作为第一步。
- **一周信号**：冻结 candidate reranking 已提高恢复；否则 RL 信号没有可信归因。
- **No-Go**：reward 不能预测 held-out 物理进度，或收益仅来自大量 online trials。

## 三个最小证伪实验

| 角色 | 对应候选 | 实验 | 强负对照 | 预注册裁决 |
| --- | --- | --- | --- | --- |
| 最稳健 | C1 | feedback observability ridge probe | feedback 前帧、robot-only、shuffle labels、source-state-disjoint split | 高 post、低 pre/shuffle 则否定“不可观测是主因” |
| 最有算法潜力 | C2/C4 | Full-H post-feedback N=8 mode coverage + 行为阶段漏斗 | sample0、opposite expert mode、self-consistency、fixed/Oracle K | 若局部 mode 已存在但 regrasp 后阶段大幅流失，则保留 consequence-composition 问题，否定纯 mode selector |
| 高风险高收益 | C3 | 冻结 Full-H 显式 recovery prompt 闭环上界 | 原 task prompt、错误 success-assumption prompt | +20 pp/+10 pp 且稳定才支持语言路由；否则停止 prompt 路线 |

## 非机械选择原则

- C1、C2 的解释性和成本最好，但若各自假设被证伪，就不能因为“容易做”而保留。
- C3 即使出现小收益，近期语言恢复工作也使论文空间很窄；它更适合作为 C4 的 baseline。
- C5 与 Reflective VLA 重叠最大，只有 candidate-conditioned recovery consequence 的额外机制成立才考虑。
- C6 必须由真实不可观测样本触发，不能在当前反馈已经可见时人为增加主动感知。
- C7 可能提高最终分数，但在机制未定位前最难归因，也最耗 rollout。
- C4 当前与现有行为漏斗最一致，但最终入选仍取决于三个证伪实验及下一步 candidate oracle 上界，不能凭叙事直接选择。

## 完成后的实验裁决

| 实验 | 实际结果 | 候选影响 |
| --- | --- | --- |
| E1 feedback observability | pre 50%，feedback frame 100%，shuffle mean 52.5% | C1 作为根因 No-Go；C6 在当前任务没有触发条件 |
| E2 N=8 mode + funnel | any-correct 100%，sample0-correct 97.4%；isolated regrasp 61.5%，transport/success 15.4% | C2 的纯 mode selector No-Go；C4 的局部动作到物理阶段组合矛盾得到支持 |
| E3 recovery prompt | explicit 0%，original 15.4%，差值 -15.4 pp `[-28.2,-5.1]` | C3 No-Go；语言恢复只保留为 future baseline |

C5 与 Reflective VLA 的近期重叠过强，且当前单帧已经可辨；C6 的必要条件被 E1 否定；C7 在 candidate-level oracle 尚未成立前成本和归因都不合格。三项实验后只有 C4 同时满足“现象存在、强简单 baseline 未解决、当前资源可做、主指标可闭环”的问题选择条件。

入选的是**研究问题**而非已验证方法。C4 的第一道新增硬门槛仍是 state-clone candidate consequence oracle；若它不能把 regrasp->transport 或完整 recovery 提高至少预定幅度，则 C4 也立即 No-Go，不进入 verifier 训练。

SELECT_NEW_RESEARCH_PROBLEM: 候选动作物理后果驱动的恢复阶段验证
