# 视角重验证 M-A 阶段结果

状态：`M_A_COMPLETE_M_B_APPROVED`

## 1. 已完成证据

- Broad64 三个严格机制臂均完成 2,000 step Visual-LoRA 训练；
- Official、Broad64 practical、state-matched、paired FM、paired consistency 共完成 1,050 条 constructed M1 完整闭环；
- 所有 21 个 frame states 均保持相同物理状态哈希，按 6 个来源 demonstration 聚类统计；
- 四个 Broad64 模型均完成 Accel fixed-state 排名与 M1 outcome 联合；
- seed 41 Camera Full 已有 1,599 条 Official 与 1,599 条 Broad64 practical 正式记录。

## 2. M-A 决策

| 方法 | Camera Full / M1 关键信号 | M-B 决策 |
|---|---|---|
| Broad64 practical | Camera Full `+5.23pp`，CI `[+0.12,+10.69]`；信息特异性 `+13.33pp`，CI `[+3.33,+26.67]` | 主 finalist |
| Broad64 paired consistency | 信息特异性 `+11.67pp`，CI `[-6.67,+30.00]` | 机制挑战者，扩 seed 检验 pairing 门槛 |
| Broad64 state-matched | 信息特异性 `+4.17pp`，CI `[-6.67,+15.83]` | 停止扩 seed |
| Broad64 paired FM | 信息特异性 `-1.67pp`，CI `[-21.67,+13.33]` | 停止扩 seed |

Broad64 practical 是唯一同时具有正式被动相机增益和显著 development 信息特异性的模型。paired consistency 尚未通过正式门槛，但它是唯一保留正向机制点估计的 pairing 方法，必须用 seeds 42/43 判断是否满足“相对 unpaired 至少 +3pp 且跨 seed 同向”的预注册门槛。

## 3. Accel 裁决

Accel 在四个模型中主要选择 canonical：每 21 个状态有 15 至 18 个选择 canonical。相对 canonical 的闭环变化介于 `-2.50pp` 到 `+3.33pp`，均不足以证明其为 view-value selector。因此 dynamic shortlist 保持 `HOLD`，当前只把 Accel 解释为策略熟悉度或兼容性指标。

## 4. M-B 执行范围

- Broad64 practical：使用 2 GPU、global batch 32 对 seeds 41/42/43 统一训练；
- Broad64 paired consistency：复用已完成的 seed 41，新增 seeds 42/43；
- 对两个方法的三 seed 运行 Camera Full；
- 完成 Original LIBERO retention，确认基础能力下降不超过 5pp；
- 不扩展 state-matched、paired FM，也不启动 Plus Full 或 dynamic Accel。

本阶段结果仍是开发筛选，不是最终论文结论。正式结论须等待 M-B 三 seed 和 retention gate。
