# 同状态物理过程 Gate 0 结果

## 重要更正（2026-07-15）

本报告的数值保留为历史诊断，但原裁决已撤销，当前状态为：

```text
INVALIDATED_REQUIRES_RERUN
```

后续完整 rollout parity 发现，旧 runtime snapshot 只恢复了 MuJoCo flattened state、body position、friction 和 gripper action，没有恢复 robosuite observable cache、采样时钟以及 OSC/interpolator 内态。恢复当下的 sim state 虽然数值一致，但下一个 replanning boundary 的 policy images、robot state 和动作会分叉。因此旧 Gate 0 的 sibling branches 不是相对于自然 Full-H 过程的严格干预，下面结果不能继续支持 `STOP_SINGLE_CHUNK_RERANKER` 或后续方法裁决。

修复后的 snapshot 已覆盖 controller、interpolator、environment timestep、obs cache 和每个 Observable 的采样状态。真实 LIBERO smoke 中，natural-vs-forced-restore 的动作、图像、robot state 均为 0 差异，sim state 最大差约 `8.3e-15`。新 receding Gate 将从该修复后的状态恢复重新建立证据。

## 裁决

**以下为原历史裁决，已因上述测量缺陷失效。**

正式实验表明，当前 `K=3` action chunk 能稳定影响“是否重新建立抓取”这一级局部物理结果，但这种影响没有稳定传递到 transport 和完整任务成功。按预注册门槛，不应训练单步 action-conditioned critic，也不应把 best-of-8 选择写成完整恢复方法。

```text
STOP_SINGLE_CHUNK_RERANKER
PIVOT_TO_CLOSED_LOOP_RECOVERY_SEGMENT_CREDIT
```

## 实验身份

- Git commit：`2986a3a431d02a7f5e1adfba19d70f689a1ea2d0`
- split：`val`
- Full-H checkpoints：seeds `41, 42, 43`
- snapshot groups：13
- 独立 source initial-state clusters：9
- stages：`feedback`、policy-induced `post_regrasp`
- 每状态候选数：8
- 候选执行长度：`K=3`
- 候选选择 continuation：3
- 独立 held-out continuation：2
- feedback/post-regrasp bridge：120/60 steps
- 阶段判定：带 2-step dwell 的有序状态机
- 训练可见 candidate bank 与 privileged simulator audit bank 物理分离

正式聚合文件：

```text
/share/longjunyu/fresh-vla/research-reset/physical-process-oracle-v2/all_seed_val_summary.json
```

## 三 seed 结果

下表均为 held-out continuation 上、先在 `(seed, source_initial_state_index)` 内聚合后得到的配对百分点差。`next` 在 feedback 状态表示稳定 regrasp。

| seed | N=8 positive coverage | replay semantics | post-regrasp eligible | next | transport | success | progress AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | 92.3% | 92.3% | 38.5% | +5.6 pp | +8.3 pp | +11.1 pp | +0.0154 |
| 42 | 92.3% | 92.3% | 61.5% | +22.2 pp | +0.0 pp | -2.8 pp | +0.0398 |
| 43 | 100.0% | 100.0% | 61.5% | +2.8 pp | +8.3 pp | +5.6 pp | +0.0258 |

三 seed、source-cluster 聚合：

| feedback held-out 指标 | Oracle - sample0 | paired bootstrap 95% CI | 判断 |
| --- | ---: | ---: | --- |
| stable regrasp / next stage | +10.2 pp | [+0.9, +18.5] pp | 局部作用存在 |
| stable grasp at end | +10.2 pp | [+0.0, +20.4] pp | 边界证据 |
| lift | +6.5 pp | [-5.6, +16.7] pp | 不确定 |
| transport | +5.6 pp | [-4.6, +15.7] pp | 未通过门槛 |
| full success | +4.6 pp | [-0.9, +11.1] pp | 未通过门槛 |
| progress AUC | +0.0270 | [+0.0053, +0.0492] | 连续进度改善 |

selection continuation 上 exact Oracle 的 full success 增益为 `+8.6 pp`，95% CI `[+3.1, +14.2] pp`；独立 held-out 后降为 `+4.6 pp` 且区间跨 0。这说明单 continuation 的 best-of-N 会高估作用，因此不能用 selection 指标裁决。

policy-induced post-regrasp 状态只在 21/39 次 seed-state 请求中可达。对可达状态，held-out next-stage 和 transport 差均为 `0`，full success 为 `-3.1 pp`，95% CI `[-9.4, 0] pp`。此处候选结果多样性不足，继续训练 post-regrasp reranker没有依据。

## Gate 判定

### Gate 0A：测量链路

- 原检查只覆盖 restore 当下的数值与像素，没有覆盖下一重规划动作及后续动力学，现已证明不充分；
- 39 个 feedback 状态中，exact candidate + matched continuation 的语义重放率为 94.9%；
- 两次不一致来自后续闭环接触/策略轨迹，而非 candidate action 被修改；
- 多 selection continuation 和独立 held-out continuation 缓解了单轨迹随机性，但没有把系统变成严格确定性。

因此旧测量链路不能用于 decision-bearing SCM intervention。描述性数值仅用于解释为何新增了完整 120-action natural-vs-forced-restore parity，必须由修复后的运行重新裁决。

### Gate 0B：候选支持与持久收益

- N=8 candidate-positive coverage 94.9%，通过 70% 支持门槛；
- held-out stable regrasp 提升约 10 pp，说明同状态动作后果不是伪信号；
- transport 未达到预注册的 `+15 pp`；
- full success 未达到 `+10 pp`，且三 seed 不同向；
- post-regrasp reranking 没有 next-stage 或 transport 收益。

**Gate 0B 未通过。** 不训练单 chunk critic，不扩大 N，不调手工分数，也不把 bridge horizon 再扫一遍寻找显著性。

## 从结果反推真正断点

当前失败链条是：

```text
feedback 可见
  -> action chunk 能提高 stable regrasp
  -> 单个 chunk 的优势被后续 replanning/接触状态覆盖
  -> lift/transport/full success 收益不稳定
```

因此研究单位选错了：一个 `K=3` open-loop chunk 不是恢复技能的充分因果单位。真正需要优化的是跨多个 replanning boundary、能闭环适应新观测并完成有序事件转移的 recovery segment。

## 下一方法：固定预算、逐节点闭环恢复信用

直接从同一初态采样完整 recovery segments 再给 winner 全段正标签仍会产生伪信用：第一步后各 segment 已面对不同观测。下一轮改为固定 `4 replans x K3 = 12 actions`；在每个 replanning 节点都从当前同一 simulator/controller state 重新生成 sibling actions、做 matched physical lookahead、只形成该节点的 action preference，然后执行所选前缀并读取真实新观测。

segment 不由 simulator event 提前停止，总 live control budget 固定为 120 actions；3 套 selection continuation 用于选择，5 套 held-out 只做审计。对照包括 sample0、random-of-4 和只看 K3 直接后果的 myopic stage selector。推理时最终仍要求 posttrained Pi0.5 sample0，不增加 candidate search、critic 或额外观察频率。

详细冻结配置见 `docs/embodied_research_reset/recovery_segment_preregistration.md`。

## 下一轮最小证伪门槛

先做 exact receding recovery-segment Oracle，不训练网络：

- 三 seed 均使用相同 source clusters；
- 每个 replanning 节点 `N=4` sibling actions；
- 固定 4 次 replanning，每次 `K=3`；
- 3 条 matched selection continuations 和 5 条 held-out continuations；
- held-out transport 相对 sample0 至少 `+15 pp`，或 full success 至少 `+10 pp`；
- 三 seed 同向，且 source-cluster paired CI 不跨 0；
- regress/drop 不增加，并优于 random/myopic controls。

未通过时，停止 simulator best-of-N/physical-preference 主线，转向直接收集 expert recovery demonstrations 或采用更简单的 failure-informed post-training。
