# LIBERO M1 可见性与位姿支持联合 Gate

## 结论摘要

本轮开发性 quick gate 已完整结束。训练、210 条闭环评测、exact-state 审计、
AV1 视频和 source-demonstration 聚类统计均通过。

核心结果是：当 Strong-info 与 Matched-control 的相机位姿都进入训练 support
后，两者成功率均为 `14/21 = 66.7%`，信息特异性为 `0.0pp`。因此，Info-only
模型中观察到的 `+26.4pp` 不能归因于任务实体可见像素增加，主要解释是训练
位姿支持不对称。

当前 gate 决策为：

> **STOP_CURRENT_NATURAL_VISIBILITY_QUICK_GATE_EXPANSION**

这不是对主动视角研究方向的否定。它只表示不再扩大当前两项自然 LIBERO
任务上的同类像素可见性实验；后续应使用更强、受控且可证伪的 Blind–Reveal
场景检验信息价值。

## 实验问题

本轮专门区分两种解释：

1. 新视角提供了更多任务实体可见像素，因此闭环行为改善；
2. 新视角恰好进入训练相机 support，因此策略更熟悉该位姿。

可见性只用于离线定义候选关系，不输入策略。物理状态、动作预算、评测状态、
随机种子和 replanning 设置在条件间保持一致。

## 数据与训练

| 项目 | 设置 |
|---|---|
| 任务 | `goal_wine_rack`、`libero10_mug_microwave` |
| 测试状态 | 21 个 frame states |
| 独立统计单位 | 6 个 source HDF5 demonstrations |
| 每状态条件 | 10 个 |
| 闭环 episode | 210 |
| 训练 seed | 41 |
| 训练步数 | 2,000 |
| GPU / global batch | 8 / 32 |
| 等待步数 | 0，避免破坏中途恢复的物理状态 |
| 视频 | 210/210，AV1/WebM |

三种关键训练 support：

| 模型 | 相机训练 support |
|---|---|
| Broad32 | Broad-32 常规宽视角 |
| Info-only | Broad32 加 3 个 Strong-info 精确位姿 |
| Info+Control | Broad32 加 3 个 Strong-info 与 4 个 Matched-control 精确位姿 |

Info+Control 数据包含 400 episodes、38,193 records，数据审计通过。训练使用与
前两模型相同的 Pi0.5 初始化、Visual-LoRA/action expert、训练预算和 seed。

## 闭环结果

### 双相机输入

| 模型 | Canonical | Strong-info | Matched-control | Blind |
|---|---:|---:|---:|---:|
| Broad32 | 71.4% | 42.9% | 52.4% | 33.3% |
| Info-only | 66.7% | 71.4% | 47.6% | 61.9% |
| Info+Control | 76.2% | 66.7% | 66.7% | 57.1% |

### 相机通道消融

| 模型 | Canonical external-only | Canonical wrist-only | All-camera blackout |
|---|---:|---:|---:|
| Broad32 | 23.8% | 52.4% | 14.3% |
| Info-only | 28.6% | 71.4% | 19.0% |
| Info+Control | 33.3% | 47.6% | 14.3% |

### Source-demo 聚类 paired 结果

| 模型 | Info - Canonical | Info - Control | Info - Blind |
|---|---:|---:|---:|
| Broad32 | -29.2pp `[-45.8,-12.5]` | -6.9pp `[-29.2,+12.5]` | +8.3pp `[0,+16.7]` |
| Info-only | +5.6pp `[0,+16.7]` | +26.4pp `[+9.7,+43.1]` | +11.1pp `[-11.1,+29.2]` |
| Info+Control | -9.7pp `[-30.6,+8.3]` | **0.0pp `[0,0]`** | +12.5pp `[0,+29.2]` |

区间为按 source HDF5 demonstration 聚类的 paired bootstrap 95% CI；同一 source
中的多个 frame states 不作为独立样本重复计数。

## 解释

1. **Broad32 不足以覆盖 M0 选出的诊断位姿。** Strong-info 在 Broad32 下低于
   Canonical 29.2pp，说明直接把该结果解释为“更多可见像素有害”是不成立的。
2. **Info-only 的正结果存在 support confound。** 只有 Strong-info 精确位姿进入
   训练后，Info 相对 Control 提高 26.4pp，但二者训练 support 不对称。
3. **对称 support 后信息特异性完全消失。** Info+Control 下 Info 与 Control 在
   21 个状态上的成功结果相同，聚类差值为 0.0pp。
4. **训练位姿扩展改善的是一般相机兼容性。** Info+Control 相对 Broad32 同时改善
   Strong-info、Matched-control 和 Blind，而不是只改善任务信息更高的视角。
5. **外部相机不足以单独完成任务。** 三个模型的 external-only 均很低，当前行为
   仍强依赖腕部与外部视觉联合；不能把双相机增益全部归因于外部视角信息。
6. **黑相机负对照有效。** Info+Control 的 blackout 比 canonical 低 61.1pp，
   排除了“策略完全不使用视觉”的解释。

## Gate 决策与下一步

当前自然 LIBERO quick gate 已经回答了 support confound，不继续新增同类模型或
seed。下一步只保留以下路线：

1. 将像素可见性作为候选筛选器，而不是已经验证的 view-value 指标；
2. 在 RoboCasa 或专门构造任务中建立强 Blind–Reveal、Matched-control 和
   wrist-on/off 因果对照；
3. 先要求 Reveal 与 Blind 在任务证据和正确行为上产生明确差异，再启动策略训练；
4. 将训练相机 support 对 Strong-info 与 Control 保持对称；
5. 在更多任务族、source demonstrations 和 seeds 上进行正式确认；
6. Accel 仅用于分析策略兼容性和候选排序，不替代可见性与闭环成功率。

在更强场景完成前，不发布“像素可见性能够选择最优主动视角”的方法结论。

## 可复核产物

- 联合训练：`/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_info-control-pose-support-m1-dev-v1_seed41_g8_gb32_steps2000`
- 联合评测：`/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/m1_visibility/info_control_pose_support_practical_seed41_wait0_v1`
- 三模型统计：`/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/m1_visibility/cross_model_development_v2`
- 模型 SHA-256：`84b3d98650f3e6267dc0d09d822b34a89c19543c9106a67433257b8cc1e3e623`
- 训练源码提交：`9b85ee4caee47c28d73b2e77478ac5e2122f259e`
- 跨模型分析提交：`f1c6a5b`

