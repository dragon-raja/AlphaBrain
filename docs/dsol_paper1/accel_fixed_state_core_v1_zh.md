# G6 Accel 固定状态核心 v1

## 范围

本阶段只实现固定物理状态下的候选视角 Accel 排名和关系分析，不实现 dynamic shortlist
或大规模闭环。Accel 是策略兼容性指标，不替代任务实体可见性分数。

公式与 prefix 约定来自 [The Geometry of Flow-Matching Uncertainty](https://arxiv.org/abs/2607.27933)。

## 冻结定义

同一候选组保持物理状态、语言、机器人状态和推理预算一致，只改变相机观察。所有候选显式
共享同一个模型空间高斯噪声 `x0`，在 10 个 denoise step 中记录归一化动作坐标的 velocity：

```text
accel_p = p * sum(||v_t - v_(t-1)||, t=1..p-1)
              / sum(||v_t||, t=0..p-1)
```

记录 `accel_2...accel_10`，主指标固定为 `accel_3`，数值越低表示 flow trajectory 越平滑。
若分母为零，该候选标记为 `degenerate` 并排除选择，不能因分数写成有限的零而被误选。

## 模型接口

`Pi0FlowMatchingHead.sample_actions*()` 新增默认关闭的 `return_velocity_trace`。默认返回和旧版
完全一致；显式开启后返回：

- `velocities`: `[batch, denoise_step, horizon, action_dim]`；
- `times`: `[denoise_step]`；
- `initial_noise`: `[batch, horizon, action_dim]`。

`PaliGemmaPi.predict_action(..., return_flow_trace=True)` 将上述结果转为机器可读字段。trace 始终
位于 `normalized_action` 坐标；动作反归一化不会改变 trace，避免不同 checkpoint 的数据尺度
被混入主指标。

## 工具

固定状态推理库：

```text
scripts/dsol_paper1/accel_inference.py
```

它构造候选间逐元素完全相同的 `x0`，调用模型一次完成候选 batch，并验证模型返回的
`flow_initial_noise` 与输入严格一致。

已有 velocity trace 时，可离线生成排名：

```bash
PYTHONPATH=scripts/dsol_paper1 python scripts/dsol_paper1/rank_accel_candidates.py \
  --trace-npz TRACE.npz \
  --output ranking.json
```

`TRACE.npz` 必须包含 `candidate_ids`、`velocity_trace` 和 `initial_noise`，可选 `flow_times`。

关系分析：

```bash
PYTHONPATH=scripts/dsol_paper1 python scripts/dsol_paper1/analyze_accel_relations.py \
  --ranking ranking.json \
  --candidate-metadata candidates.json \
  --references references.json \
  --output relations.json
```

`references.json` 的键固定为 `canonical/train/strong_info/reveal/oracle`，值为候选 ID 或 ID
列表。输出同时记录 exact match、参考集合内最佳 Accel 排名，以及存在 pose metadata 时的最近
参考视角距离。

## GPU smoke 验收

1. 使用一个 checkpoint、一个恢复状态和 4 个候选视角；
2. `num_denoise_steps=10`，关闭 compile；
3. 候选 `flow_initial_noise` 逐元素完全相同；
4. trace 形状为 `[4,10,H,D]`，全部有限；
5. 默认 `predict_action()` 与显式同 `x0`、trace 开启调用的最终动作逐元素一致；
6. 重复同 seed 的排名完全一致，换 seed 后另存为独立重复；
7. 输出 ranking 与 canonical/train/strong-info/reveal/oracle 关系 JSON。

GPU smoke 通过后才能启动 dense search；dynamic shortlist 仍保持关闭。
