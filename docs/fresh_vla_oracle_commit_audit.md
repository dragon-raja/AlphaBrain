# FRESH-VLA Oracle Plan-Commit 资产审计

## 审计结论

本阶段可以直接复用既有 Full-H 策略、13 个 group-preserving test snapshot、完整 attached/slip episode、fixed K=1/2/3 闭环结果和 deterministic reach 结果。无需也不允许重新训练策略权重。

需要新增的只有执行层 commit wrapper、Oracle/随机/夹爪/self-consistency 控制评测、推理效率计数和对应的配对统计。由于录制 teacher 的绝对时钟不能映射到偏离轨迹，本轮实际采用更强的 privileged runtime-event interrupt 上界；事件 outcome、Oracle 标签和未来状态都不进入 Pi0.5 输入、采样噪声或动作选择。

## 冻结策略

三个 Full-H checkpoint 均来自统一 27,607-step 训练预算；仓库模板当前写 2,400，但每个 self-contained final checkpoint 的实际 `framework_config.yaml` 和训练日志都记录 27,607：

| Seed | Checkpoint | 权重大小 |
| ---: | --- | ---: |
| 41 | `/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2/fresh_closed_loop_full_h_seed41/final_model` | 17,591,583,484 bytes; SHA256 `144a3b3d3dcc8421418564a62059a1038c9a7ef3196ac157f5f9ea1997a31f30` |
| 42 | `/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2/fresh_closed_loop_full_h_seed42/final_model` | 17,591,583,484 bytes; SHA256 `98dc52d2ed1983776d218fee7666f3131053d1a55296e93e9f521b1c088ce875` |
| 43 | `/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2/fresh_closed_loop_full_h_seed43/final_model` | 17,591,583,484 bytes; SHA256 `5db16350d9835c1f28d01b660dd6e9234bcab3da79abbce1f092e92b08ac9149` |

运行模板来源为 `configs/experiments/fresh_vla_libero_closed_loop.yaml`，确认性配置以各 final checkpoint 内的 `framework_config.yaml` 为准。Full-H 的 `action_horizon=10`、flow inference steps 为 10，视觉模块冻结，动作维度为 7；Stage A 不修改这些设置。加载仍依赖 `/share/longjunyu/alphabrain/pretrained_models/paligemma-3b-pt-224`。

## 数据与划分

评测数据视图：

`/share/longjunyu/fresh-vla/libero-full-episode-v2-128`

该视图复用 v1 的 episode、视频和 contact sheet，只替换经过验证的 source-initial-state-disjoint manifest 与质量报告。共有 128 个 snapshot group：train/val/test 为 102/13/13，分别覆盖 30/9/9 个互不相交的 source initial state。Stage A 固定使用 13 个 test group，不选择有利 snapshot。同一 split 内允许多个 group 共享 source state，因此 test 的 13 groups 只有 9 个独立 source states。

这 13 个 test groups 已用于上一轮 6 方法 fixed-K final decision，不是新的盲测集。本轮必须标注为 locked/post-hoc execution-layer upper-bound，不能宣称为全新独立 confirmatory holdout；预注册主统计仍以 snapshot group 配对，并额外以 9 个 source initial state 为聚类单位做保守敏感性分析。只有两套门槛都通过时才允许 GO。

每个 group 都有 attached 和 slipped/recovery 两个完整 episode。正式任务为 `put_the_cream_cheese_in_the_bowl`，成功判据继续使用 LIBERO 环境的 `env.check_success()`。

## Oracle 标签来源

`event_time`、`feedback_reveal_time` 和 `action_divergence_time` 来自完整反事实 episode collector：

- `event_time`：真实抓取/滑落物理事件时间；
- `feedback_reveal_time`：从事件开始，任一相机首次出现任意非零像素差的时间；当前实现没有持续性或语义可辨阈值，因此它是敏感但偏弱的视觉差异代理；
- `action_divergence_time`：两条 scripted expert continuation 首次持续动作分歧的时间；
- gripper 控制不读取上述 Oracle 标签，而是从当前已执行夹爪命令到候选 action chunk 的逐步符号变化计算边界；chunk 第一步发生开合切换时只执行该一步后立即重规划；
- random matched 控制从同 seed 的既有 Full-H fixed-K3 runtime `event_time` 构造 outcome-blind boundary map，并做受约束的 group 间置换；每个置换边界必须在目标 group 的 attached/slipped 两条 fixed-K3 branch 结束前都可到达，没有触发事件的 group 保留空边界。最终 verifier 还会比较 Oracle 与 random 的实际调度边界 commit-length 多重集；
- self-consistency 控制只使用同一当前观测下的 8 个 Full-H action sample。

当前数据有一个重要识别限制：128/128 个 group（test 为 13/13）都满足

`event_time == feedback_reveal_time == action_divergence_time`

因此主合法组 `feedback_reveal_time <= action_divergence_time` 为 128/128，反向不等式组为 0。`oracle_branch_safe_commit` 与 `oracle_feedback_reveal_commit` 必须保留为两个独立实现和输出，但在本数据上必须产生相同 commit 调度；本轮无法利用它们之间的差异识别“动作分歧”与“反馈可见”哪个边界更关键。

不能直接把录制 teacher 的绝对 step 当作偏离 expert 轨迹后的 policy 时钟。单组诊断中，teacher boundary 为 56，而 Full-H 的实际 runtime event 到 step 241 才出现，偏差 185 steps。该无效 recorded-clock pilot 已隔离保存为诊断证据。最终 Oracle 使用 runtime event-aligned upper bound：事件发生并产生新观测后，立即截断当前 chunk 的剩余动作并重新调用相同 Full-H。它不把事件 outcome 传给策略；策略仍只能从重新取得的正常图像和 robot state 判断结果。因此本轮只能裁决 privileged event-aligned Plan-Commit 上界，不能独立识别 branch-safe 与 feedback-reveal 标签的差异。若该更强上界仍失败，才按预注册规则输出 `STOP_FRESH_FAMILY`。

## 可直接复用的评测

每个 Full-H seed 已有：

- `closed_loop_isolated.json`：78 rows，即 13 groups × 2 branches × K={1,2,3}；
- `closed_loop_end_to_end.json`：78 rows；
- `deterministic_reach.json`：39 rows，即 13 groups × K={1,2,3}；
- `closed_loop_videos/`：78 个成对视频。

fixed K=1/2/3 的成功率、恢复率、行为诊断、progress、subgoal、completion steps 和 invocation count 均直接复用。机器校验会固定其 JSON SHA256，并核对精确 evaluation seed、split、protocol、row count、pair IDs 和 K 集合。历史 remote-policy JSON 没有嵌入 checkpoint hash，因此 checkpoint 绑定仍依赖原 runner 路径约定、当前冻结权重实测 SHA256 和单组逐协议 parity；报告必须披露这一历史证明限制。历史 rows 也没有逐调用 wall-clock 计时；最终效率分析将保留真实 invocation count，并用同 seed 新增单样本 commit 运行测得的每次推理延迟估算 fixed-K wall-clock，明确标记为估算值，不伪装成历史实测。

## 新增评测

必须新增：

- `oracle_branch_safe_commit`；
- `oracle_feedback_reveal_commit`；
- `gripper_commit`；
- `random_matched_commit`；
- `self_consistency_commit`，固定 N=8、逐动作维 sample variance（`ddof=1`）的 RMS disagreement 阈值 0.15；sample 0 是实际执行 anchor，其他样本只决定 prefix-closed commit 长度；
- 每个方法的 isolated recovery、end-to-end、deterministic reach；
- policy invocation、实际 forward calls 和推理 wall-clock；
- 主合法组与反向不等式组的分离统计；
- paired snapshot-group bootstrap 95% CI 和 success-efficiency Pareto 图。
- source-initial-state 聚类 bootstrap 敏感性门槛。

## 泄漏防火墙

Pi0.5 始终只接收原有 agent view、wrist view、robot state 和 language。policy server 对 RPC example key 做严格白名单检查，拒绝 branch、时间和 Oracle 元数据。wrapper 接收完整 action chunk 后才决定执行前几步。接口不接受 branch outcome，privileged runtime event 只允许中断当前 chunk，不改变动作值、不选择恢复动作、不改变模型权重，也不提供额外图像或未来状态。正式 runner 拒绝脏 Git worktree；matrix preflight 对三份 checkpoint 各现场计算一次 SHA256，单独 evaluator 则自行计算；产物记录 Git SHA、Torch/CUDA/GPU 身份，并用 `flock` 阻止重复矩阵或同一输出并发写入。

## 视频兼容性

旧证据视频原为 MP4 容器内的 MPEG-4 Part 2 `mp4v`。1,905 个目标视频已全部原路径转码为 H.264/AVC `avc1`、`yuv420p`、fast-start MP4；原始文件保存在 `/share/longjunyu/fresh-vla/video_mp4v_backup`。后续 collector/evaluator 也统一直接生成该兼容格式。最终 verifier 会逐协议核对 13 个精确文件名、完整解码、codec/pixel format，并要求视频帧数与对应 episode `completion_steps + 1` 一致。
