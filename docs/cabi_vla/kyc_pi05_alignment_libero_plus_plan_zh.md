# KYC × Pi0.5：视觉对齐与非固定场景验证方案

更新时间：2026-07-30

## 1. 当前判断

现有结果还不能回答“KYC 对 Pi0.5 是否有效”，只能回答：

> 在固定 LIBERO 场景、保留腕部相机、冻结 Pi0.5 全部 VLM，并从随机初始化
> 学习 ray 融合的设置下，真实相机几何没有优于匹配的固定-ray Control。

当前 KYC 为 `43.33%`，Control 为 `44.49%`，差异 `-1.16 pp`，
95% CI 为 `[-7.03,+4.43] pp`。固定 RGB 只替换 ray 时，动作块 RMS
变化约为 `0.00071`，说明当前策略基本没有使用 ray。

后续三种子“场景线索 × 腕部相机”确认实验已经完成：

| 训练/评测场景 | 腕部相机 | Control | KYC | 判断 |
|---|---|---:|---:|---|
| 固定线索 | 开 | 34.34% | 31.03% | KYC 无增益 |
| 随机线索 | 开 | 30.03% | 29.60% | KYC 无增益 |
| 固定线索 | 关 | 0.29% | 0.14% | baseline invalid |
| 随机线索 | 关 | 1.15% | 0.72% | baseline invalid |

因此关闭腕部相机虽然更接近 KYC 的隔离设置，但当前 Pi0.5 数据和控制链路在该条件
下无法完成基本任务。不能用约 1% 成功率继续裁决几何条件化，也不能把低成功率
解释成 KYC 失败。

这不是 KYC 的最终否定，主要有两个尚未拆开的混杂因素：

| 假设 | 当前问题 | 可证伪实验 |
|---|---|---|
| H1：场景捷径 | 固定桌面、背景和机器人几何使 RGB 能直接猜相机位姿；腕部相机又提供稳定参照 | 随机背景/场景线索，并做腕部相机 on/off |
| H2：特征未对齐 | 当前冻结整个 VLM，新建的随机融合层又改变了预训练 RGB token 基底 | 给 SigLIP 加轻量视觉适配，再比较真实 ray 与固定 ray |
| H3：模型仍忽略 ray | 即使闭环分数变化，也未必由真实几何导致 | 固定 RGB，交换为正确、固定、打乱和错误位姿 ray |

## 2. 为什么“冻结 VLM”是实质性差异

当前配置冻结了整个 `vlm_interface`。相机分支先编码 ray，再通过随机初始化的
线性层与 SigLIP 图像 token 融合。视觉主干不能适应新几何通道时，优化器最容易
找到的解是恢复 RGB 通路并把 ray 权重压低。

官方 KYC 的 SmolVLA 配置并没有冻结视觉编码器和整个 VLM。因此，当前实验不是
官方训练条件在 Pi0.5 上的等价迁移。

单卡 32 GB 上不建议全量微调 3B VLM：

- 当前每个训练约占 `22 GB`；
- 全量 VLM 的梯度、Adam 状态和激活会超过剩余显存；
- 多卡参数分片虽可实现，但不适合作为第一轮诊断，成本也没有必要。

应改为只适配视觉塔：

1. **FTM**：全局仿射视觉 token 调制，约 4K 参数，作为极轻基线；
2. **FLA/视觉 LoRA**：只给 SigLIP 27 层的视觉线性层加低秩适配，
   rank 16 约 4.7M 参数；
3. 语言模型与 action expert 保持与现有对照一致。

这不是把视觉适配当作 KYC 的功劳，而是检验“Pi0.5 是否需要先让视觉空间可适配，
才能接纳相机几何”。

## 3. 融合层也需要匹配控制

第一轮保留官方式融合以复核结果；同时做一个小规模稳定性诊断：

```text
当前随机融合：concat(rgb, ray) -> Linear

稳定融合：rgb + alpha * Project(ray), alpha 初始为 0
```

稳定融合在训练开始时严格保持预训练 RGB 表示。若采用它，KYC 和 Control 必须
同时采用相同结构，唯一差别仍然只能是“真实 ray”与“固定默认 ray”。

先用小样本过拟合和 ray-swap 测试选择融合，不直接增加一整轮大规模模型。

## 4. 最小实验顺序

### 阶段 A：完成当前因子实验

保持当前 seed 41 的六项评测继续运行，不重复有效任务。它们用于测量：

- 固定线索、腕部相机关闭；
- 随机场景线索、腕部相机开启；
- 随机场景线索、腕部相机关闭；
- 每个条件下的 Control 与 KYC。

这些结果应标记为 **Frozen-VLM diagnostic**，不能单独作为 KYC 最终结论。

### 阶段 B：视觉对齐筛选

主筛选使用已经确认可学习、同时移除固定场景线索的条件：

> 随机场景线索 + wrist-on + 相同随机多视角训练数据。

`wrist-off` 保留为独立 baseline-repair 任务；只有外部相机策略先达到至少 20%
成功率，才恢复该单元中的 KYC 比较。

| 组别 | RGB/场景数据 | ray | 视觉主干 |
|---|---|---|---|
| B1 | 相同 | 无 | frozen |
| B2 Control | 相同 | 固定默认 ray | frozen |
| B3 KYC | 相同 | 真实匹配 ray | frozen |
| B4 | 相同 | 无 | FLA |
| B5 Control | 相同 | 固定默认 ray | FLA |
| B6 KYC | 相同 | 真实匹配 ray | FLA |

公平比较是：

- `B3 - B2`：冻结视觉空间时，真实几何的增量；
- `B6 - B5`：视觉可适配后，真实几何的增量；
- `(B6-B5) - (B3-B2)`：视觉对齐是否真正解锁了 KYC。

执行顺序：

1. 1–2K updates：显存、梯度、checkpoint 严格加载 smoke；
2. 小样本过拟合：确认视觉 adapter 与 ray 分支都能收到梯度；
3. 约 130 个分层闭环 episode/组做单 seed 筛选；
4. 只有通过停止规则才跑 520 episodes 和 seeds 42/43。

### 阶段 C：ray 因果使用审计

对完全相同的 RGB、state 和语言输入，分别提供：

- 正确实时 ray；
- 固定默认 ray；
- batch 内打乱 ray；
- 明确错误的相机位姿 ray。

一个真正使用相机几何的 KYC 策略应同时满足：

1. 正确 ray 优于固定/打乱 ray；
2. 错误 ray 会造成可重复的性能下降；
3. 中间特征和动作对 ray 交换有明显响应；
4. 这种响应在 Control 中不存在。

只看到“KYC 分数略高”但 wrong-ray 不影响行为，不足以证明模型使用了几何。

### 阶段 D：正式非固定场景评测

通过阶段 B/C 后再做完整矩阵：

| 评测条件 | 目的 |
|---|---|
| Camera only | 标准跨相机位姿泛化 |
| Background only | 检查纯外观背景鲁棒性 |
| Camera × Background | 移除固定背景的位姿捷径，检验组合泛化 |
| wrist-on / wrist-off | 测量腕部相机是否掩盖第三人称视角问题 |
| KYC 式物理场景随机化 | 改变机器人/桌面相对场景关系，最接近原论文假设 |

LIBERO-Plus 的 Background 主要改变纹理和材质，不等于 KYC 的机器人/桌面物理
布局随机化。因此两者都需要，但不能互相替代。

## 5. LIBERO-Plus 下载决策

### 现在需要

| 资源 | 大小 | 用途 | 链接 |
|---|---:|---|---|
| `assets.zip` | 约 6.4 GB | 官方 LIBERO-Plus 环境资产和评测渲染 | https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main |
| `libero_plus_camparam_rlds.zip` | 约 16.6 GB | 2,876 个带外部相机 4×4 外参的 episode，用于训练 KYC | https://huggingface.co/datasets/Sylvest/libero_plus_camparam_rlds/tree/main |

合计约 `23.0 GB`。建议平台下载到：

```text
/share/longjunyu/alphabrain/datasets/libero-plus/archives/
```

代码仓库放在 `/projects/LIBERO-plus`，解压后的数据和资产继续放 `/share`。

相机参数版本包含：

- 外部相机图像与腕部图像；
- state、joint state、action、语言；
- 每个 episode 的 `primary_cam_extrinsics`；
- 2,876 episodes、256 shards。

它没有显式保存内参，但 LIBERO 相机的固定 FOV 和分辨率可用于恢复内参。该数据
主要是 camera-view 子集，并不自带充分的背景随机化，所以仍需与场景随机化训练
或联合评测配合。

### 现在不要下载

| 资源 | 大小 | 暂缓原因 |
|---|---:|---|
| 普通 `libero_plus_lerobot` | 约 16 GB | 没有相机内外参，不能直接训练 KYC |
| 完整 `libero_plus_rlds` | 约 75.5 GB | 当前对齐筛选不需要；通过阶段 B/C 后再考虑 |
| 四套全集组合 | 约 99.9 GB | 目前会造成无必要的存储和迁移成本 |

可选从 4 卡机迁移已有 `/datasets/models/lerobot/pi05-libero`（约 9 GB），作为
标准 LIBERO Pi0.5 起点；迁移前先核对其 checkpoint 格式，不把它当作当前
`libero-bind` 单任务 checkpoint 的直接替代。

## 6. 预注册停止规则

进入三 seed 正式评测至少应满足：

1. `KYC+FLA - Control+FLA >= 5 pp`；
2. 正确 ray 相对错误/打乱 ray 有稳定优势；
3. 不降低默认视角成功率超过 5 pp；
4. 效果在 Camera × Background 或 KYC 式物理随机场景中仍存在。

结果解释：

| 结果 | 结论 |
|---|---|
| KYC+FLA 明显优于 Control+FLA，wrong-ray 会伤害性能 | 视觉对齐解锁了显式几何，继续完整验证 |
| FLA 明显提高所有组，但 KYC≈Control | 视觉适配有效，KYC 没有额外价值 |
| 只在 wrist-off 或非固定背景中 KYC 有效 | 固定场景/腕部相机捷径假设成立，明确适用边界 |
| 所有条件 KYC≈Control，且模型仍不响应 ray | 停止在 Pi0.5 上继续堆 KYC 变体 |

## 7. 近期执行优先级

1. 三种子因子实验已完成并归档，确认 wrist-off baseline invalid；
2. SigLIP 原生低秩适配已实现，两步 GPU smoke、梯度审计和严格重载已通过；
3. 在随机场景线索 + wrist-on 上跑 2K updates 单种子筛选；
4. 同步完成正确/固定/错配 ray 因果诊断；
5. 通过门槛后才扩展 33K updates 与三种子确认；
6. LIBERO-Plus 直连当前超时，保留平台下载/跨机器传输，不用代理拉 23 GB。

这条路线能分别回答“背景让模型猜到了相机”与“冻结视觉主干使 KYC 无法对齐”，
并避免再用一个大而混杂的实验同时解释两个问题。
