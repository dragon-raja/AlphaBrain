# VLA 视角重验证并行执行协议 v4

状态：`ACTIVE_EXECUTION_SCHEDULE`

日期：2026-08-20

范围：第一阶段仅运行 Original LIBERO、LIBERO-Plus Camera Full 和严格机制诊断；
LIBERO-Plus Full 与 RoboCasa 暂时移出关键路径。

本协议只优化执行依赖和资源调度，不改变
`view_revalidation_master_protocol_v3_zh.md` 中冻结的研究问题、数据定义、统计单位或通过门槛。

## 1. 最快有效交付

| 里程碑 | 内容 | 最快时间 | 可得出的结论 |
|---|---|---:|---|
| M-A | seed 41 coverage/pairing、constructed M0/M1、Accel fixed-state/dense | 48-72 小时 | 主线机制 Go/No-Go；使用已有 seed-41 Camera Full 作锚点 |
| M-B | 仅入选方法扩 seeds 42/43，并补 Camera Full 与 finalist retention | 另加 1-2 天 | 三 seed 正式相机泛化结论 |
| M-C | 入选方法 Plus Full、副作用审计、完整统计与报告 | 再加 2-3 天 | 非相机扰动副作用结论；继续后置 |

若 constructed expert、16-worker 评测或训练并发 smoke 失败，允许一次修复波次。M-A/M-B
合计正常为 3-5 天，含一次修复的上限为 7 天。

现有 Official 与 Broad64 seed-41 Camera Full 结果直接复用，不重复运行。当前进行中的 Original Full
Official/Broad64 是基础能力 retention 锚点，但不阻塞 M-A 的数据和模型工程。其余 seed 的 Original Full
只对 M-A 选出的 finalist 运行，并归入 M-B。

## 2. 四条并行流水线

### A. Benchmark 与训练线

1. 不重启当前 Original Full Official -> Broad64 队列，但不等待它再开始 B/C 两线；
2. 运行训练并发容量 smoke；
3. seed 41 完成 Broad64 state-matched、paired FM、paired consistency；
4. 完成 exposure 1x/2x/4x 和 Legacy8/Broad32/Broad64 的 seed 41 诊断；
5. 联合 constructed M0/M1 与 Accel 完成 M-A 主线机制裁决；
6. 只把通过 M-A 的模型扩展到 seeds 42/43；
7. 对入选方法补 Camera Full 与 Original Full retention，形成 M-B 正式结论；
8. Plus Full 后置，获准启动时只运行 Official、Broad64 和最终方法，并复用已完成的 Camera Full 原始记录。

### B. Constructed Blind-Reveal 数据线

与 A 同时执行：

1. 冻结 3-5 个 pilot tasks 和任务实体 registry；
2. 为同一物理状态构造 Canonical、Strong-info、Matched-control、Blind、Look-away 和 blackout；
3. Extreme、Look-away、blackout 仅用于评测，不进入普通动作训练；
4. 仍以任务实体在多相机中的等权可见像素均值计算 `I_task`；
5. 先通过人工 montage、逐相机分解、wrist on/off、物理状态哈希和 expert headroom；
6. 通过 M0 后才生成 Info-pose-support 训练数据并启动完整 M1。

### C. Accel 工程线

与 A/B 同时执行：

1. sampler 返回共享 `x0` 下的逐 flow-step velocity；
2. 实现 `accel_2...accel_10`，主指标固定为 `accel_3`；
3. 完成 fixed-state dense ranking；
4. 分析所选视角与 canonical、最近训练 pose、Strong-info、Reveal 和 oracle@shortlist 的关系；
5. 只有 fixed-state 审计通过后才运行 dynamic shortlist。

Accel 不替代可见性分数，也不用于定义训练信息视角。

### D. 统计与归档线

每个阶段完成即增量运行：

- base-task clustered bootstrap；
- Wilson 95% CI；
- per-seed 与跨 seed 汇总；
- task/suite/difficulty/camera-family 分解；
- W&B 指标记录，失败自动离线；
- 原子 JSONL、AV1/WebM、manifest 和 commit hash 归档。

## 3. 资源调度

### 3.1 LoRA 训练

训练保持 Visual-LoRA、global batch 32、相同更新数和优化器。

优先调度：

```text
4 jobs x 2 GPUs
```

每个作业通过 gradient accumulation 保持 global batch 32。若 2-GPU 吞吐或通信效率不理想，回退到：

```text
2 jobs x 4 GPUs
```

1-GPU 仅用于 20-50 step smoke，不直接用于正式训练，除非与 2/8-GPU 梯度和吞吐审计一致。

### 3.2 闭环评测

当前 8-worker 队列保持运行，不中途重启。下一轮先做 16-worker 容量 smoke，即每张 GPU 两个 policy/eval workers。

16-worker 只有同时满足以下条件才进入正式运行：

- 峰值显存不超过 28 GiB/GPU；
- invalid episode 为 0；
- same-input policy output 与 8-worker 基准一致；
- 物理 snapshot hash 一致；
- 每 episode 中位耗时至少改善 30%；
- CPU 长时间利用率不超过 90%，无持续 I/O stall。

否则保持 8 workers，不通过降低 horizon 或 trials 偷减预算。

### 3.3 数据与 GPU 冲突

- XML、snapshot ledger、候选目录和统计在 CPU 上与训练并行；
- EGL 批量渲染采用独立 shard，避免与正式闭环争抢同一卡；
- 正式训练占满 GPU 时，只运行 CPU 数据准备和小规模渲染 smoke；
- 所有大数据直接写 `/share` 的独立临时目录，通过 manifest 后原子发布。

## 4. 缩短工期但不缩减证据的规则

允许：

- seed 41 跑完整诊断矩阵，seeds 42/43 只扩展通过 gate 的方法；
- Camera Full 已完成的 task 记录直接复用于 Plus Full 汇总；
- 多 checkpoint 的闭环 episode 交错调度，以减少慢任务造成的尾部等待；
- 复用 policy server，避免同一 checkpoint 重复加载；
- 失败 episode 原子重试，不重复有效 episode。

不允许：

- 用 Exact-state 或 Dev40 替代 Camera Full；
- 把三个 seed 的 episode 当作独立样本；
- 为节省时间删掉 Canonical-unique、Canonical-repeat 或 state-matched 控制；
- 将 Extreme/blackout 混入正常动作训练；
- 在测试任务上选择 exposure、Accel prefix 或可见性阈值；
- 仅凭 seed 41 或 MSE 宣称最终方法有效。

## 5. 决策关键路径

```text
seed41 coverage/pairing quick gate
  -> constructed M0/M1 + Accel
  -> M-A mechanism decision
  -> finalists
  -> seeds42/43 + Camera Full + finalist retention
  -> M-B formal view conclusion
  -> deferred Plus Full side-effect audit

Current Official/Broad64 retention
  -> runs in background
  -> supplies the seed41 retention anchor

Constructed scenes
  -> M0 visibility gate
  -> Info-pose-support
  -> full M1

Flow trajectory
  -> Accel fixed-state audit
  -> dense relation
  -> dynamic shortlist
```

前三条路径并行；第一阶段报告只等待三条路径各自通过对应 gate，不等待 Plus Full 或 RoboCasa。
