# 具身智能 Research Reset：证据与失败模式

## 1. 审计范围

本报告只整合 AlphaBrain 当前 `exp/fresh-vla-toy-v0` 分支已完成的 FRESH-VLA 证据，不新增 suffix weighting、dynamic K、horizon head、world model、主动视觉或 RL。核心任务为 LIBERO `put_the_cream_cheese_in_the_bowl`。

数据与评测资产：

- 128 个 snapshot groups，每组 attached/slipped 两条完整物理 episode；
- 256 条完整 episode、34,551 个滑动训练窗口；
- attached expert success = 100%，slip expert final recovery = 100%；
- group-preserving train/val/test，test 为 13 groups、9 个 source initial states；source state 跨 split 不重叠；
- Full-H seeds `[41,42,43]`，此前 6 方法 x 3 seeds 统一训练；
- Stage A 冻结 Full-H 的 fixed/oracle/gripper/random/self-consistency 执行评测；
- 所有统计先在 snapshot group 内跨 seed 聚合，再做 group bootstrap；另报告 9 个 source-state clusters 的敏感性分析。

必须保留的限制：test split 已被前序实验重复使用，因此本轮是 locked/post-hoc 机制诊断，不是新盲测；13 groups 的区间仍较宽。任何后续论文结果都需要新的 held-out source states 或第二环境确认。

## 2. 证据链

| 层级 | 已回答的问题 | 结果 | 能支持的结论 | 不能支持的结论 |
| --- | --- | --- | --- | --- |
| Toy 反事实机制 | 分支相关 suffix 是否会污染公共前缀学习？ | 在构造分布中 Oracle FRESH 改善公共动作前段。 | loss allocation 机制在受控 toy 中存在。 | 不能推出真实闭环成功。 |
| 真实 LIBERO 分支 | 是否有相同 conditioning、不同抓取结果与不同 continuation？ | 128 组 attached/slip 物理分支与完整 recovery expert 成立。 | 反事实数据构造和标签可用。 | 不能证明模型能利用标签。 |
| 完整训练数据 | feedback 后是否错误沿用 pre-feedback 短 horizon？ | 34,551 windows；feedback 后恢复 full supervision。 | 数据没有已知的 post-feedback 低权重错误。 | 不保证动作归一化/模型容量足够。 |
| Suffix weighting | Oracle weighting 是否优于 Full/Random/Shuffled/Gripper/Short-H？ | K=3 下 Oracle overall 15.4%、slip 0%、isolated 17.9%；未稳定优于控制。 | `STOP_TRAINING_WEIGHTING_ROUTE`。 | 不否定反馈控制或其他恢复方法。 |
| Deterministic reach | 部署链路是否连基本定向移动都失败？ | 多数方法 K1--K3 达到约 85%--97% reach。 | 仿真部署和基础定向动作不是完全失效。 | reach 不能替代抓取、运输、放置。 |
| Oracle Plan-Commit | privileged event-aligned 执行边界是否提高成功？ | Oracle 与 fixed K3 的 overall/slip/isolated 差值均 0；random 不差，gripper 未被稳定击败。 | `STOP_FRESH_FAMILY`；执行长度不是当前核心作用位置。 | 不等于所有 runtime verifier 都无效。 |
| 反馈可观测性 | feedback 后当前输入能否区分 attached/slip？ | pre-feedback 50%；feedback frame `vision+state` 100%，group CI `[100,100]`；shuffle mean 52.5%。 | 当前任务的主要失败不是“看不见滑落”。 | 线性 probe 可读不等于 Pi0.5 内部一定正确利用。 |
| N=8 mode coverage | Full-H 是否没有生成局部恢复 mode？ | post-feedback slip any-correct 100%，sample0-correct 97.4%，正确 mode 占比 97.1%。 | 随机没有抽到局部 recovery mode 不是主因。 | “正确 mode”只按前三步相对 expert continuation 定义，不等于完整恢复。 |
| 显式 recovery prompt | 是否只缺语言 subtask routing？ | explicit prompt recovery 0%，原任务 15.4%；配对差值 -15.4 pp `[-28.2,-5.1]`。 | 简单 prompt 路线 No-Go。 | 不否定经过专门训练的 recovery language model。 |
| 行为逻辑漏斗 | 失败在哪个阶段集中发生？ | isolated K3：event 100% -> recovery-action proxy 100% -> regrasp 61.5% -> transport 15.4% -> success 15.4%。 | 主要流失位于 regrasp 到 transport 的阶段组合。 | 缺少各子目标首次时间戳，不能声称严格时序因果。 |

## 3. 训练加权失败的具体表现

K=3 主结果：

| 方法 | Overall | Attached | Slip recovery | Isolated recovery | Failure continuation |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_h` | 14.1% | 23.1% | 5.1% | 15.4% | 51.6% |
| `random_soft010` | 17.9% | 30.8% | 5.1% | 17.9% | 71.7% |
| `shuffled_oracle_soft010` | 17.9% | 28.2% | 7.7% | 25.6% | 63.3% |
| `gripper_soft010` | 15.4% | 28.2% | 2.6% | 12.8% | 63.3% |
| `oracle_soft010` | 15.4% | 30.8% | 0.0% | 17.9% | 46.7% |
| `short_h` | 15.4% | 28.2% | 2.6% | 12.8% | 73.9% |

具体失败不是“Oracle 完全没有改变动作误差”：Oracle common-prefix MSE 有改善，行为错误率也有方向性下降；但这些变化没有转化为 slip recovery、overall success 或相对 Random/Shuffled 的独特优势。离线动作指标与闭环结果在这里明确脱钩。

## 4. 执行承诺失败的具体表现

冻结 Full-H 的 Stage A 主结果：

| 方法 | Overall | Attached | Slip recovery | Isolated recovery | Forward calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_k1` | 3.8% | 7.7% | 0.0% | 2.6% | 318.3 |
| `fixed_k2` | 5.1% | 10.3% | 0.0% | 15.4% | 155.2 |
| `fixed_k3` | 14.1% | 23.1% | 5.1% | 15.4% | 101.9 |
| `oracle_branch_safe_commit` | 14.1% | 23.1% | 5.1% | 15.4% | 102.4 |
| `gripper_commit` | 9.0% | 17.9% | 0.0% | 17.9% | 111.1 |
| `random_matched_commit` | 15.4% | 23.1% | 7.7% | 15.4% | 101.6 |
| `self_consistency_commit` | 14.1% | 28.2% | 0.0% | 7.7% | 1226.9 |

Oracle 相对 K3 把 failure-continuation/premature-commitment 降低 51.9 pp，但成功率和恢复率完全不变。这说明“减少明显错误承诺”是必要的行为改进，却不是当前失败的充分解法。N=8 self-consistency 付出约 12 倍 forward calls 也没有提高恢复。

两个 Oracle 的调度 trace 和语义行为逐行等价；117 个配对视频中 116 个文件字节一致，1 个因独立 EGL 渲染出现像素漂移。该差异已单独记录，不参与方法胜负。

## 5. 三个最小证伪实验

### 5.1 反馈可观测性

协议：16x16 RGB block-average 特征，agent/wrist/两视角/state 组合；ridge 正则只在 val 选择；train/val/test source states 完全不重叠；32 次 shuffle labels。

| Offset | Modality | Test accuracy | Group bootstrap 95% | Source-state bootstrap 95% |
| ---: | --- | ---: | --- | --- |
| -1 | vision+state | 50.0% | `[50.0,50.0]` | `[50.0,50.0]` |
| 0 | vision+state | 100.0% | `[100.0,100.0]` | `[100.0,100.0]` |
| +1 | vision+state | 100.0% | `[100.0,100.0]` | `[100.0,100.0]` |

agent-only 和 robot-state-only 在 offset 0 均为 96.2%，两视角 vision 为 100%。这表明信号不是只存在于一个隐藏标签字段；但 probe 使用真实记录 state/image，不代表基础 VLA 已学会利用它。

裁决：**否定“反馈后仍不可观测”作为当前主要失败模式。**

### 5.2 Post-feedback 模式覆盖

协议：每个反馈状态从冻结 Full-H 采样 N=8，只比较前三步与 paired expert continuation；三 model seeds，13 test groups，9 source-state clusters。

| 指标（post-feedback slip） | Snapshot-group 结果 |
| --- | ---: |
| any correct mode | 100.0% `[100.0,100.0]` |
| sample 0 correct mode | 97.4% `[92.3,100.0]` |
| candidates 中 correct mode 比例 | 97.1% `[92.3,100.0]` |
| candidates 中 opposite mode 比例 | 2.9% `[0.0,7.7]` |
| best-of-N 相对 RMSE 降低 | 51.3% `[43.7,59.3]` |

any-correct 与 sample0-correct 的差只有 2.6 pp，远低于预注册 20 pp 门槛。best-of-N 可降低局部动作误差，但 self-consistency 闭环没有收益，因此不能把该结果解释成“只需普通样本一致性选择”。

裁决：**否定纯 mode availability/随机选择瓶颈；保留候选质量/物理后果排序问题。**

### 5.3 显式恢复语言上界

协议：冻结 Full-H、isolated feedback state、固定 K=3、三 seeds、相同 13 groups。比较原任务、显式失败恢复提示和错误 success-assumption 提示。

| Prompt | Recovery success | Regrasp | Final progress |
| --- | ---: | ---: | ---: |
| 原任务 | 15.4% `[5.1,28.2]` | 61.5% `[48.7,74.4]` | 40.4% `[32.1,49.4]` |
| 显式 recovery | 0.0% `[0.0,0.0]` | 17.9% `[5.1,33.3]` | 32.1% `[28.8,35.3]` |
| 错误“已抓住” | 0.0% `[0.0,0.0]` | 38.5% `[25.6,51.3]` | 32.7% `[28.8,37.8]` |

显式 recovery 相对原任务 recovery 差值为 -15.4 pp，三个 seed 为 `[-23.1,0.0,-23.1]` pp；相对错误提示为 0。该实验只裁决现成 Pi0.5 的 prompt-only 上界，不否定专门训练的高层恢复模型。

裁决：**停止 prompt engineering 路线。**

## 6. 行为逻辑漏斗

严格定义：

```text
E = slipped AND intervention_triggered AND event_time != null
A = E AND recovery_switch_observed
R = A AND regrasp_success
T = R AND transport_subgoal
Y = T AND recovery_success
```

`A` 只是“未抓住时首次张开夹爪或朝物体移动”的启发式 proxy。当前 JSON 没有 first-regrasp/first-transport 时间戳，因而 `E->A->R->T->Y` 是逻辑交集漏斗，不是严格时序或因果链。`place_subgoal` 在 evaluator 中与 environment success 等价，不能拆成独立层。

Full-H isolated K3：

| 层级 | 到达率 | Group bootstrap 95% |
| --- | ---: | --- |
| Event（协议固定起点） | 100.0% | `[100.0,100.0]` |
| Recovery-action proxy | 100.0% | `[100.0,100.0]` |
| Regrasp | 61.5% | `[48.7,74.4]` |
| Transport | 15.4% | `[5.1,28.2]` |
| Final success | 15.4% | `[5.1,28.2]` |

条件转化：recovery-action -> regrasp 为 61.5%；regrasp -> transport 为 25.0%，group bootstrap `[8.7,40.9]`；transport -> success 在少量有效分母上为 100%。因此当前最强的定位是：**策略常能开始恢复，也有一定概率重新抓住，但多数重新抓取没有组成有效运输。**

End-to-end K3 的无条件 event/recovery-action/regrasp/transport/success 分别为 38.5%/38.5%/12.8%/7.7%/5.1%。该条件漏斗受“方法能否先到达 intervention event”的选择影响，只作为描述性证据。

## 7. 失败模式裁决表

| 具体失败假设 | 状态 | 依据 |
| --- | --- | --- |
| 当前视觉无法区分抓取成功和滑落 | **否定为主因** | offset 0 vision/state 100%，pre 50% |
| 模型完全没有恢复 mode | **否定局部形式** | N=8 any-correct 100%，sample0-correct 97.4% |
| 单次随机采样经常选错 mode | **否定** | availability gap 仅 2.6 pp |
| 执行过长导致错误承诺 | **不是充分原因** | privileged Oracle Commit 与 fixed K3 成功完全相同 |
| 普通自一致性可选择正确动作 | **否定** | 约 12 倍 calls，isolated recovery 反而 7.7% |
| 原语言没有显式恢复指令 | **否定 prompt-only 解法** | explicit prompt 0%，显著差于原任务 |
| 已切换恢复动作但无法重新抓取 | **部分成立** | recovery-action 100%，regrasp 61.5% |
| 已重新抓取但无法稳定推进至运输 | **强支持** | regrasp -> transport 仅 25.0% |
| 离线动作误差能预测闭环成功 | **否定** | prefix MSE 改善未转化为成功 |
| 当前控制频率/K 完全不匹配 | **部分否定** | K3 最好；K1 更差；Oracle 动态边界无收益 |
| 需要主动视角/probe 才能看到 slip | **当前任务否定** | feedback frame 已完全线性可辨 |
| 候选 chunk 的物理后果可被轻量 verifier 排序 | **尚未验证** | 这是下一阶段第一项 oracle gate，不是现有结论 |

## 8. 视频与产物质量

- 历史 1,905 个 MP4 已从 `mp4v` 转为真实 H.264/AVC、`avc1` tag、`yuv420p`、faststart；原件保留在独立 backup root。
- Stage A 新增 585 个并排/评测视频，artifact verifier 全部解码通过。
- Recovery-prompt 新增 78 个 attached/slipped 并排视频；逐文件完整解码，78/78 已是 H.264/avc1 + yuv420p，无需转码。
- prompt 视频 manifest：`/share/longjunyu/fresh-vla/research-reset/recovery_prompt_video_manifest.json`。

## 9. 当前最窄、可证伪的剩余问题

当前证据不支持继续 FRESH weighting、dynamic commitment、普通 self-consistency、failure observability、主动视角或 prompt-only recovery。仍与证据一致的问题是：

> 当失败已经可见、局部恢复 mode 已在策略分布中、且策略能够重新抓取时，如何验证并选择真正带来下一物理阶段转移的候选短 action chunk，使恢复动作组成 transport/place，而不是停留在局部动作相似性？

这只是问题选择，不是方法已成立。下一步必须先做 simulator state-clone 的 candidate consequence oracle；若 Oracle 候选选择也不能提高 `regrasp -> transport` 或 full recovery，就立即停止该方向。

机器可读证据：

- `/share/longjunyu/fresh-vla/research-reset/feedback_observability.json`
- `/share/longjunyu/fresh-vla/research-reset/post_feedback_modes_summary.json`
- `/share/longjunyu/fresh-vla/research-reset/recovery_prompt_summary.json`
- `/share/longjunyu/fresh-vla/research-reset/recovery_funnel.json`
- `/share/longjunyu/fresh-vla/runs/libero-oracle-commit-final-v1/final_decision.json`
