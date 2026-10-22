# 匹配训练模型的视角空间诊断：执行计划与边界

后续授权补充：用户已要求补齐 64 状态并加入 Oracle 对照，参见
[完整状态扩展执行说明](full64_view_landscape_execution_20260908_zh.md)。
本页保留首批 8 状态的原始执行记录；首批仍运行，扩展已启动排队接续。

日期：2026-09-08。根据本轮用户授权，执行第 1–3 步；第 4 步实际相机获取、第 5 步 RoboCasa 暂不启动。此处不修改原有实验、原始 PDF 或历史训练 release。

## 本轮要回答的问题

在同样的训练预算与策略输入条件下，外部相机训练覆盖改变以后，候选视角的闭环成功率分布、Accel 排序及其可靠性怎样变化？

这轮是指标与训练覆盖的匹配诊断，不直接宣称提出了可部署主动视角算法，也不把“能查看全部候选图像”当作免费的实际感知。先确定哪些关系确实存在，再决定是否值得设计搜索器及付出相机获取成本。

## 顺序、规模与当前状态

| 阶段 | 内容 | 固定规模 | 状态 |
|---|---|---|---|
| 1 | canonical 最终权重与训练配方验收 | 与 Broad M-B 相同 seed41、2000 步、64000 exposures | 已完成，验收 PASS |
| 2A | 已有 Broad 的空间图、状态级统计、噪声稳定性、Oracle 诊断 | 64 状态 × 97 候选 × O32；Accel8 | 已生成 6 页主图与 64 状态图册，视觉复核通过 |
| 2B | canonical 在相同缓存图像上计算 Accel | 首批 8 状态 × 97 候选 × 8 初噪 | 03:13:18 UTC 全部完成并校验 PASS |
| 3 | canonical 闭环并与 Broad 同状态配对比较 | 首批 8 状态 × 97 候选 × O32 = 24832 条 | 03:16:24 UTC，48 条检查及噪声／物理审计 PASS；03:16:30 UTC 进入 wave00 remainder-a |
| 暂缓 | 实际获取、搜索算法正式验证、RoboCasa | 本轮无预算 | 未启动 |

“单视角训练”在这里准确含义是**单一外部相机位姿训练**。两个模型都保留外部图像、腕部图像与任务语言，不能写成单图输入与双图输入的对比。当前只是一对 matched seed41 模型，不能推断跨训练 seed 或跨架构规律。

## 为什么先做 8 状态，不先跑全量

已有 Broad 的 64 状态结果足够立即出图；因此不必等 canonical 全量闭环才能开始第二步。canonical 第一批按固定 salt 对 development 的 source_group 做哈希排序，每个任务选 1 个来源，共 8 个状态。选择过程不读取成功率、Accel 或测试状态。

所有 97 个候选和 32 次 O-bank 重复均保留，不按先验表现逐轮删候选。这样减少的是第一批状态数，而不是重新引入此前候选裁剪问题。8 状态对照用于检查方向和可测性；每任务仅 1 个来源，不能支撑任务内状态泛化或覆盖全部失败类型的结论。是否补到更大来源集合，需要根据本轮结果及运行代价另行明确预算。

总量为 24832 条**新增**闭环，而不是两种模型各重跑一套。Broad 的对应结果复用已有账本。

实际启动：2026-09-08 03:11:21 UTC，tmux 会话 `dsol-matched-landscape-v1-20260908`。Accel 每状态纯打分 35.6–36.7 秒（另有模型加载和准备时间），每卡观测显存约 11.47 GB；不能把这个速度直接外推为闭环速度。完成全部闭环并通过审计后，总控自动生成 canonical 报告与同 8 状态配对比较图。

该状态是带时间戳的快照；后续以新根的 `controller-status.json` 和各段 `segment-receipt.json` 为准。48 条检查只证明执行协议正常，不根据其成功率选择、删除或扩增正式候选。

## 匹配条件与噪声解释

- 同一物理起点、场景构造、语言、腕部图像和 97 个外部相机候选；对缓存图像和物理状态核验 hash。
- Accel 沿用历史 `asset_source_pair_key` 作为初噪 seed key，保持 8 个初噪、batch16、BF16、动作 horizon10 和相同评分实现。不能改用新实验名字生成 seed。
- 闭环使用已经物化的 O32 初噪库，索引为 `(state, repeat, replan_index)`，同一索引跨候选、跨模型完全相同，不受别的 episode 调用顺序影响。
- 每次执行 5 个动作后重新规划，wait_steps=0；相同的模型去噪配置和求解步数保持固定。不同轨迹可以提前成功或有不同的规划次数，只对共同的 replan 索引配对；不强行让轨迹、结束时刻或实际墙钟时间一致。
- Accel8 与 O32 是两个独立的噪声集合。分析是“平均指标与噪声边缘成功率的关系”，不是“同一初噪上的 Accel 预测本次是否成功”。
- 规范候选保留原始相机朝向，其余候选采用 look-at pivot。规范点在图上单独标记；97 点是非规则三参数采样，不是完整 SE(3) 或规则体素网格。

## 图和统计的预定读法

1. 三维候选散点：同位置分别着色成功率与 Accel；半径分带显示二维投影。不用插值制造不存在的连续精度。
2. 状态 × 候选矩阵：不只画全局均值，防止相反的状态关系被平均抹掉。
3. 状态内 `rho(-Accel, success)`、Accel Top1 相对规范候选收益、不同评分噪声之间的排名与 Top-k 一致性分别报告。稳定不代表有用，相关也不等于选择收益。
4. canonical 与 Broad 只在**同一批 8 状态**上做直接对照，分别看候选整体表现、规范表现、分布差异和指标—成功关系；不能把 Broad64 与 canonical8 的全局均值直接作训练效应。
5. 规范参数距离与最近训练支持距离只作为几何代理，不称为已测得的模型熟悉度，不推断因果机制。
6. Oracle 保留两种读数：O32 上经验最大值明确标注选择偏差；O 前 16 / 后 16 双向交叉评估，在一半噪声上选候选、另一半评估。后者仍是有限搜索的历史诊断，不是严格 Oracle 上界，也不是新学习器的独立确认。

现有 64 状态来自构造遮挡的 HDF5 中间状态续跑，不是自然初始状态的“扰动相机后恢复”实验。因此本轮最多支持静态视角空间中的关系分析；用户提出的扰动恢复贡献仍需后续另设初始条件、可见输入权限、获取预算，以及直接回到规范相机的基线。

## 有界执行与失败规则

8 卡并行：Accel 每卡 1 份模型；闭环沿用已验证的每卡 2 份策略服务、总 32 个模拟器 worker。先测 48 条 smoke，PASS 后执行 wave00 的两个剩余矩阵，再执行 wave01–07：

`48 + 1504 + 1552 + 7 × 3104 = 24832`。

smoke 与 wave00 完整协议逐字段一致，可复用；不另跑 full wave00。每段独立保留协议、run manifest、episode ledger 和噪声审计。任何段失败即停，不无限重试；已有非 PASS 残片必须人工诊断后另定恢复方式。Accel 单 worker 墙钟上限 90 分钟，闭环控制器上限 36 小时；**上限不是预计完成时间**。根据新模型实测吞吐给出预计时间，若超预算则保留部分数据并重新评估，不暗中增加预算。

## 源码、模型和结果的完整路径

已核实的启动日志说明：`No dataset_statistics.json` 是通用 loader 的字典缺省提示，当前模型从 `framework_config.yaml` 的 `framework.normalization` 及保存的 action mean/std buffers 独立完成 MEAN_STD；8 份评分日志均显示启用。旧 runner 用裸 TCP 探测 WebSocket 端口，会产生握手 EOF/InvalidMessage 日志；本次之后正常建立策略连接、48 条 smoke 完成并审计 PASS。不能仅按这些日志关键词判定模型或评测失败。

- 仓库：`/workspace/projects/alphabrain-dsol-paper1`
- 新实验根：`/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908`
- 完成验收：`/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908/checkpoint-audit/completed_run_receipt.json`
- 协议清单：`/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908/protocols/protocol_manifest.json`
- 选择清单：`/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908/protocols/selection.json`
- Broad 图表：`/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908/broad-existing`
- canonical 评分：`/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908/canonical-accel`
- canonical 闭环：`/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908/closed-loop/canonical/O`
- 后续 canonical 报告（完成后生成）：`/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908/canonical-report`
- 后续匹配比较（完成后生成）：`/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908/matched-comparison/matched_view_landscapes_v1.pdf`
- 总控：`/workspace/projects/alphabrain-dsol-paper1/scripts/dsol_paper1/run_matched_view_landscape_pipeline_v1.sh`
- CPU 配对分析：`/workspace/projects/alphabrain-dsol-paper1/scripts/dsol_paper1/compare_matched_view_landscapes_v1.py`
- 总控日志：`/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908/logs/pipeline.log`

canonical 权重：`/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_canonical_unique_mb-matched-v1_seed41_g2_gb32_steps2000/final_model`

canonical SHA256：`7e510b752143bb6fd4987988dfab5e94f7db1de5f7e97579f892a58eed22cc68`。

Broad 权重：`/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_m-b-formal-v1_seed41_g2_gb32_steps2000/final_model`

Broad SHA256：`123786f53c9823b878fb08fe61ef25fe4931d4bcae8134e0a64940b7a8ac3cad`。
