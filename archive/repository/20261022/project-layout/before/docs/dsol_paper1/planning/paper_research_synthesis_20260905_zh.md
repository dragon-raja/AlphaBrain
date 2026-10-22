# 论文研究审计、收敛与后续设计

日期：2026-09-05。执行状态截点：09:37 UTC（北京时间17:37）。

本文依据仓库协议、实际分析代码、已完成产物、v2 development 数据与公开论文整理。当前 v2 test rollout outcome 未被本次审计读取或汇总。本文是研究建议与证据审计，不修改当前冻结协议、选择器、阈值或运行控制器。新增建议需作为补充分析或下一轮独立方案记录。

## 1. 核心判断

目前有一批扎实的实验资产，但尚未形成证据闭合的 CVPR 主会论文。主要缺口是主张之间的因果连接、独立状态覆盖和可部署的正向结论，并非缺少更多 rollout。

建议以一个问题收敛：

> 在经过广视角后训练、同时使用外部与腕部相机的 VLA 中，外部相机选择还会带来多少稳定的闭环收益？这种收益能否通过部署时可获得的观察预测，并在新状态上兑现？

“How to use view in VLA”可继续作为研究计划的总问题。本篇更适合暂用：

> **When Does Viewpoint Choice Still Matter for Multiview-Trained VLAs?**

这个标题不预设最终 Oracle 一定显著，也不要求本文解决图像 token 融合、几何表示和动态相机控制的全部问题。若选择方法最终通过独立验证，可再把副标题改成具体算法结果。

此前“Train, Fuse, Select”的整理有助于理解系统，但三者并不是已经实证闭合的三项贡献。当前多视角训练最成熟，融合是重要条件但因果证据不足，状态选择仍待最终测试。若不补有效的传感器对照，正文应固定 external+wrist 接口，不能声称已经解释融合如何重塑视角价值。

## 2. 研究对象与边界

需要区分四件事：

| 概念 | 当前含义 | 已覆盖情况 |
|---|---|---|
| 多视角训练 | 训练样本覆盖不同 external camera pose | 已有 Broad64 等后训练对照 |
| 多相机输入/融合 | 策略同时接收 external 与随手臂运动的 wrist | 主设置一直存在；有效因果分解不足 |
| 候选视角选择 | 根据起始状态选 external pose，后续 rollout 中保持该位姿 | 当前 v2 的直接问题 |
| 多候选图像融合/动态获取 | 同时融合多个 external 候选，或执行中移动相机 | 当前没有正式验证 |

Broad64不是一次向VLA输入64张候选图像。训练中的跨样本视角覆盖，与推理时的 external+wrist 输入不能混称为同一种“多视角”。

令后训练数据为 D，冻结策略为 π_D，传感器接口为 M，任务为 τ，冻结起始状态为 s，外部位姿为 v。当前量为：

\[
J_{\pi_D,M}(s,v)=\mathbb E_\xi[Y(\operatorname{rollout}(s,v;\pi_D,M,\xi))].
\]

Y为完整 continuation 的成功指示。它依赖当前策略、训练支持、传感器组合、执行时限、replan频率和任务状态；不是相机位姿或单张图像的固有信息量。当前主要状态来自演示中途恢复，不能把其成功率写成从任务初态开始的官方 benchmark score。

同一个 frozen state 的初始 wrist 图像相同；external view 变化后，动作、后续物理状态、external图像与wrist图像序列都会分叉。这是 external pose 干预的总闭环效应。不能把后续轨迹差异视为噪声对齐失败，也不能把该效应全部归因于初始遮挡消除。

实际目标可以是提高视角平均成功率、提高低分位成功率、减少坏视角风险，或者在观察预算内选择收益更高的视角。“压平视角景观”只是描述，降低最好视角成功率也能压平，所以平坦度不能脱离成功率和保留能力单独优化。

## 3. 已完成证据，哪些可以写

| 研究环节 | 核实结果 | 可写结论 | 不能推出 |
|---|---|---|---|
| 原始相机敏感性 | 旧LIBERO-Bind中canonical75%，半径0.925倍30%，1.075倍50%；部分大幅退化仍保留目标可见性 | 相机位姿变化可显著损害当前策略，目标出画不是全部解释 | 某个特定几何/遮挡/背景机制已被确定 |
| Broad64实用后训练 | Camera Full pooled：75.86%→82.16%；Original：96.45%→93.92% | 此后训练recipe改善相机轨道，并付出小幅原能力损失 | 增益完全由“视角数”单变量造成；所有视角问题已解决 |
| coverage机制对照 | seed41小规模Canonical/ImageAug/Broad对照中Broad明显更强 | 支持真实相机覆盖优于当前像素增强方案 | 大规模多seed下单独coverage因果已全部隔离 |
| KYC | 旧matched三seedControl44.49%，KYC43.33%，差−1.16pp，CI[−7.03,+4.43]；ray替换动作弱敏感 | 当前raw-ray适配未显示独特收益 | 显式几何无用；几何信息已被多传感器完全解决 |
| paired consistency | 当前recipe相对practical：Camera task-equal−4.75pp，Original−9.25pp，三个seed均负 | 当前同预算数据组织与loss组合不划算 | consistency loss本身无效；损害来自压制任务证据 |
| 自然M1 | 最初六demo信息特异性+13.3pp；另一自然gate对称pose support后Info/Control均66.7% | 存在方向性信号，但pose support必须控制，结果未普遍成立 | 已证明VLA正确利用了任务相关新证据 |
| 构造可见性 | A2可见像素增约14pp，Strong与canonical均87.5% source宏平均，Control/wrist-only100% | 更多实体像素不足以保证更高闭环价值 | 遮挡无关紧要；视觉信息无价值；wrist获取证据的具体时序已证实 |
| Accel | 跨noise排序Spearman0.287–0.430；单次Top1不稳定；候选选择未稳定超越基线 | 当前Accel跨视角排序用法不能替代闭环价值 | 原论文的失败检测任务无效 |
| 旧v1搜索/选择 | 候选搜索有小幅独立确认信号，冻结全局选择器未泛化 | 值得更密集搜索与独立噪声确认 | 真Oracle不存在；全局最好视角就是canonical |

证据源：[阶段一结果](../results/view_revalidation_stage1_final_20260824_zh.md)、[KYC完整报告](../../cabi_vla/kyc_camera_generalization_completed_report_zh.md)、[自然M1联合support gate](../results/libero_m1_visibility_joint_support_gate_v1_zh.md)、[构造gate](../results/libero_expanded_a_constructed_gate_20260827_zh.md)、[旧view-value发现](../results/view_value_reverse_discovery_pilot_20260827_zh.md)。不同cohort的正负结果不能直接互相相减，也不能把后续不同cohort的负结果称为推翻原cohort结果。

三处必须修正此前表述：

1. **百分比权重。** Camera Full75.86%→82.16%的pooled差为+6.30pp。文档中的+5.15pp是先在40个base task内配对、再task等权后的差值，CI为[+0.99,+9.62]。paired recipe的pooled差是77.78−82.16=−4.38pp，而task等权差为−4.75pp。两者都能报告，但必须标清口径。当前CI先平均三个training seed，再bootstrap任务，并非seed×task的完整层级区间。原程序：[analyze_m_b_multiseed.py](../../../scripts/dsol_paper1/analysis/analyze_m_b_multiseed.py)。
2. **腕部训练对照并非完全未做。** 旧KYC已完成三seed“背景线索×wrist”因子，wrist-off在训练和评测都屏蔽。问题是off条件下两个模型均只达到约0.14%–1.15%，形成地板效应，无法识别腕部对KYC增量的调节作用。Broad64的M1测试置黑则是另一类消融，不能混在一起。旧因子协议：[factorial preregistration](../../cabi_vla/kyc_camera_generalization_factorial_preregistration.md)。
3. **负结果不是机制证明。** paired recipe与practical还改变独立状态数、batch重复和数据组织。有限600步的Info-pose-support继续训练失败，也不足以完全排除pose OOD、优化不足或遮挡外观OOD。NI/ER是原计划中的假设框架，当前尚未证实其为这些负结果的原因。

旧KYC与新Broad64实验在benchmark、执行horizon等条件上也不同。它们可构成研究动机与背景诊断，不能并列成同一张直接方法排行榜。

## 4. 当前 v2 进度与新证据

截至本次核查：

| 阶段 | 数量 | 状态 |
|---|---:|---|
| Development O | 48×97×32=148,992 | 8个wave均PASS_COMPLETE |
| Development P | 48×9×32=13,824 | PASS_COMPLETE |
| Development Q | 48×2×64=6,144 | PASS_COMPLETE |
| 选择规则冻结 | 6种规则，另有canonical基准 | receipt已生成，test_outcomes_read=false |
| Test O | 16×97×32=49,664 | wave00完成，wave01执行中 |
| Test P/Q、seed42/43 transfer、可能的R | 依冻结顺序 | 待完成 |

从development Q原始ledger只读聚合得到：

| Development Q量 | 数值 |
|---|---:|
| Canonical成功率 | 65.85% |
| P冻结非canonical候选成功率 | 68.78% |
| 成对净增益 | +2.93pp |
| 同noise配对rescue | 13.18% |
| 同noise配对harm | 10.25% |
| 状态增益正/平/负 | 26 / 11 / 11 |

上述是48个开发状态、每条件64条独立Q噪声的描述性结果。它说明搜索后的净增益不大，收益与伤害同时存在。它尚未达到测试集结论，也不能替代预注册Q-test的5pp实用门。强制非canonical的搜索方案不等于允许留在canonical的真实最优策略。

逐任务开发集差值为：cream cheese→bowl −0.78pp；top drawer bowl +6.25pp；wine rack +6.77pp；book caddy −4.17pp；bottom drawer +1.56pp；mug microwave +9.38pp；cream cheese basket 0pp；spatial drawer bowl plate +4.43pp。每任务只有6个source，以上仅帮助识别异质性和后续功效设计。

开发CV已冻结两类学习器：geometry-context ridge为65.76%，image-queryable ridge为68.03%。二者分别选择alpha10和1000；它们是选超参数后的O-bank开发CV分数，不能与Q的canonical65.85%直接相减，也不能当作独立测试增益。

产物：[development CV](../../../../../../share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/selector/development-cv-report.json)。实验根目录为`/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2`。

## 5. v2 的正确解释与需要补充的分析

执行底座总体合理。O/P/Q/R独立；每个state/repeat/replan显式共享同一个10×7高斯张量；每次推理使用相同10步Euler网格、每5环境步replan；不同终止时间不造成其他视角随机流错位。当前审计未发现需要停止实验的噪声对齐问题。

| 审计项 | 当前边界 | 补充处理 |
|---|---|---|
| Oracle估计 | O/P选择、Q独立确认得到搜索候选价值，是对最优价值的下界估计；有限Q估计本身仍有误差 | O经验最大值仅诊断；给Q区间；不叫精确Oracle或认证上限 |
| 强制换视角 | P→Q排除canonical，即使canonical在P更好仍选择其他视角 | 保留原主结果；另做P-only决定的canonical fallback，不能用Q择优 |
| 状态依赖 | 自动gate只比较statewise与dev-frozen global | 补statewise−task-fixed及learner−task-fixed配对统计；否则不能区分任务差异与任务内状态差异 |
| 固定基线 | global/task-fixed也由有限development数据估计 | 只能说超过这些冻结基线，不能直接等同理论最优固定策略的regret |
| 测试状态 | 16个source、8个已见任务，每任务2个source | 不把二十多万rollout当成二十多万个独立场景；优先补新source |
| 历史复用 | v2 source与learner dev隔离，但这些物理状态曾在v1评估 | 称v2 source-disjoint与新noise确认；不称整个研究从未接触的新盲测 |
| heldout视角 | 32个pose相对VLA后训练未见；selector标签覆盖全部97 | 不能称selector未见位姿泛化；需要另留新pose验证 |
| 输入语义 | 实际任务编码是one-hot；图像是冻结ImageNet ResNet18+PCA | 当前仅同任务新source泛化；不能声称语言语义组合或新任务能力 |
| seed42/43 | 迁移seed41已经冻结的候选与选择规则 | 是跨checkpoint转移，不是三个模型各自的dense Oracle |
| image-queryable | 使用simulator候选图像银行 | 需明确查询/移动/重建预算；不可当免费RGB部署 |

统计方面，现有区间固定8个task，仅在每任务2个source均值里bootstrap。这种小样本percentile区间较脆弱，应保留主口径并给小样本稳健敏感性分析、逐任务结果及固定状态下的paired Monte Carlo误差。不要简单认为多加Flow噪声就可以解决source覆盖不足，也不要盲目叠加多层bootstrap而重复计算同一噪声来源。

R按区间宽度或前32/全64变化触发，预注册是优点，但不自动保证数据依赖停止后的普通95%区间仍有精确覆盖率。建议保留Q-only主结果、R-only复核和Q+R联合描述，并清楚记录触发原因。当前脚本在R完成后会覆盖同名final-analysis汇总文件；原始Q ledger仍可重建。后续归档应分别保存这些版本。

代码依据：[stage builder](../../../scripts/dsol_paper1/protocols/build_statewise_view_oracle_v2_stage.py)、[analysis](../../../scripts/dsol_paper1/analysis/analyze_statewise_view_oracle_v2.py)、[learner](../../../scripts/dsol_paper1/training/train_statewise_view_selector_v1.py)、[controller](../../../scripts/dsol_paper1/operations/launchers/run_statewise_view_oracle_v2_tail.sh)。这些发现主要是分析和主张边界，不是推倒重跑的理由。

## 6. 什么才是论文的逻辑链

建议正文围绕四个连续问题组织：

1. **强后训练解决了多少？** 用官方Camera与Original结果建立Broad64强基线，再用匹配小矩阵区分普通继续训练和相机覆盖。
2. **训练后还剩什么视角差异？** 在同状态、同传感器接口、同noise条件下估计闭环价值分布；区分平均能力、差视角风险、canonical改进空间、任务固定收益与状态适应收益。
3. **剩余差异由什么预测？** 以可见性、训练pose距离、任务、图像、本体信息等作竞争解释，检查这些变量能解释多少独立确认价值。相关分析只能说可预测，不能自动说因果机制。
4. **这种规律是否值得部署？** 用冻结selector，在新source、独立noise和明确观察预算下检验收益、harm、延时和fallback。

KYC/CVC放在第一问的边界比较中；可见性/Accel放在第三问；当前Oracle是第二问与第四问之间的桥梁。NI/ER、完整融合机制、动态主动感知不再同时充当本文未兑现的主贡献。

单个Broad64 checkpoint上的景观，能回答“强训练后是否仍存在”；要回答“训练怎样改变景观”，必须加跨checkpoint的同状态候选矩阵。若要把fusion写入因果贡献，也必须有基础能力有效、训练评测一致的sensor对照。否则将fusion固定在M条件中即可，不必为凑完整口号额外开新路线。

当前实证结论限定于这些后训练策略和输入接口。它们可以为预训练数据中的视角分配、几何监督和传感器设计提出可检验假设，但还不是预训练阶段的实验证明。论文不必从头预训练VLA才有价值；应明确训练阶段和适用范围，避免将单一后训练体系的结果直接推广到所有VLA。

## 7. 必要的数学抽象

数学的用途是澄清哪些量可以比较，不是给普通现象换一个理论名字。

在固定策略、传感器接口和候选集合下，定义：

\[
V_G^*=\max_v\mathbb E_s J(s,v),\quad
V_T^*=\mathbb E_\tau\max_v\mathbb E[J(s,v)\mid\tau],\quad
V_S^*=\mathbb E_s\max_v J(s,v).
\]

理论上`V_G* ≤ V_T* ≤ V_S*`。三者之差分别刻画任务层和任务内状态层适应的潜在价值。当前有限样本估计的global/task/statewise候选不一定服从这个排序，所以必须报告估计误差，不能把候选ID不同当作状态依赖证明。

真实部署只能访问h（当前external/wrist图像、语言、本体信息及可用历史）。定义：

\[
V_H^*=\mathbb E_h\max_v\mathbb E[J(s,v)\mid h].
\]

`V_H* ≤ V_S*`。因此full-state Oracle空间大，并不保证当前观察能够识别好视角。另一方面，一个ResNet18+ridge失败，也不能证明`V_H*`低；失败还可能来自表示、样本量或学习器不足。只有合理增强表示、控制数据量后仍无法预测，才值得研究增加查询/交互观察。

连续视角问题可以把总体差距分成：有限候选覆盖差距、部署观察的可辨识差距、学习器逼近差距；实验对这些量的估计另有noise和source抽样误差。相机位置与姿态可在受约束SE(3)域或任务centric参数空间表示，但闭环成功率在遮挡边界和接触行为附近不必平滑。可以做邻域扰动实证，不能未经检验就假设全局Lipschitz或直接声称找到了连续好视角流形。

当前静态一次选择主要是contextual decision/bandit问题。只有在每次相机动作影响后续观察、状态和成本后，POMDP或半MDP才进入算法主体。本文无需先解决完整POMDP。

头部相机的可移动性提供了后续扩展入口，但不能直接认为当前结论完美迁移：可达位姿、转动时间、移动期间的状态变化和观测历史都改变决策问题。静态候选没有显著收益，也不能排除执行到某个中间时刻再获取新证据的动态收益。

上述期望最大值、common random numbers、独立确认和bootstrap都是已有数学统计工具。提出记号本身不构成创新。

## 8. 建议的算法：预测“换视角收益”，并允许不换

建议下一版只设计一个主要算法，不并行堆叠多个大模型模块。

输入h包括当前external+wrist、真正的任务语言表示、本体状态；每个候选输入可部署的相机位姿。基础VLA冻结。优先复用其冻结视觉/语言表示，或与较强通用冻结表示比较；现有ResNet18+ridge保留为简单基线。

学习目标为相对参考视角b的闭环优势。默认b=canonical；以冻结task-fixed作为参考的版本单独列为消融，不能在最终测试中择优决定基准：

\[
\Delta(h,v)=\mathbb E[J(s,v)-J(s,b)\mid h].
\]

O银行提供同状态同noise的差值监督；拟合时平均掉Flow noise，不能把单次成功候选当唯一正确类别。可以采用带样本数权重的成功概率/优势回归，或容忍近似等价候选的排序损失。实际容量应受独立source数约束，不能把48×97条候选行误当4656个独立场景。

模型输出预测增益与经开发集校准的不确定性。选择规则将参考视角b纳入候选，并固定其优势、不确定性及新增成本为0：

\[
\hat v=\arg\max_v\{\widehat\Delta(h,v)-\lambda\widehat\sigma(h,v)-\eta c(v)\},
\]

只有候选分数超过在开发校准集冻结的非负阈值才切换，否则保持b。c按相对b新增的切换时间或观察成本计算；当前静态免费相机实验可先设为0。

这提供两个可检验的改进：减少不必要换视角造成的harm；学习一组近似良好候选而非脆弱的精确Top1。模型不确定性惩罚属于经验校准，不能未经独立验证称形式安全保证。rescue/harm是指定共同noise耦合下的对照量，也不是现实部署中已知的逐episode安全标签。

必须与以下基线对比：canonical、dev-frozen global、task-fixed、uniform random、最接近训练pose、visibility、Accel、现有ridge、相同表示的简单task/wrist router。若使用候选图像，所有比较需匹配可用图像和query预算。

主要算法消融保持精简：绝对success预测vs配对优势；强制Top1 vs fallback；简单图像表示vs冻结VLA表示；task-only vs state context；有无candidate image。应在新开发/校准数据上选定最终版本，再评估真正未使用的新测试source。

候选图像全部来自simulator真状态时具有额外观测权限。真实novel-view renderer只能重投影/推断已获取的证据，不能凭空恢复未观测遮挡后的真实状态。若声称实际部署，必须指定相机移动、预装多相机或重建输入是怎样获得的，并把观察预算计入。

该算法骨架本身不是保证新颖的方法。论文价值要由可靠闭环监督相对启发式监督的收益、保守选择相对强制选择的收益、以及独立泛化实证来支撑。

## 9. 最小后续实验包

| 优先级 | 工作 | 要区分的问题 | 最小交付 | 预算判断 |
|---|---|---|---|---|
| P0 | 完成当前冻结v2 | 发现增益能否在新noise/test source兑现 | Q主结果、R复核、checkpoint transfer、完整审计 | 现有运行继续，不变更设计 |
| P1 | 现有数据补充分析 | 任务差异还是任务内状态差异；强制换视角还是可靠收益 | task-fixed配对CI、P-only fallback、noise/source敏感性、Q/R分别归档 | 基本无需新rollout |
| P2 | 匹配跨checkpoint矩阵 | 训练本身怎样改变视角平均、下尾和可选空间 | Official、canonical continuation、Broad practical，同状态同pose同预算 | 冻结几何分层候选子集即可，无须每模型重做全97 |
| P3 | 新source+新selector | 规律能否脱离已反复使用的64状态 | 新demo隔离的dev/calibration/test，配对优势与fallback方法 | 优先扩大独立状态；noise数按功效分配 |
| P4 | 外部验证 | AlphaBrain/单仿真体系依赖有多大 | 官方OpenPI运行核心矩阵；另加第二环境或少量真机 | 精简核心对比，避免重跑全量 |
| 可选 | 有效sensor因子 | wrist互补、冗余还是掩盖external学习 | dual与external-only/dropout匹配训练，先过能力门 | 若要将Fuse写入主贡献才升为必需 |
| 可选 | loss-only一致性对照 | 当前recipe损害来自loss还是状态预算 | 完全相同paired数据的FM-only vs FM+consistency | 不以CVC机制为主贡献则放附录 |

新source规模应由开发集source-level效应方差和目标3–5pp改进做功效估计，不以经验数字宣称足够。可用每任务8–12个全新source作为预算起点，并另设少量未见任务测试；这只是规划量，不是统计保证。单个视角32个Bernoulli rollout在p=0.5时标准误约8.8pp，说明精确Top1区分并不容易。相反，更多相同state的noise也不能代替新的独立场景。

跨checkpoint子集候选应按相机几何和任务覆盖事先冻结，避免仅挑对Broad64有利的候选后外推到所有模型。主对比至少包括Official、canonical continuation和Broad64；KYC/CVC仅在runtime与训练条件可公平匹配时加入。

真机最适合先做一次性选择：当前观察→从3–5个已知外部位姿或预装相机中选择→固定该视角完成任务。这样与当前问题直接对应，并可测延时和成功。完整头部动作策略、腕部操作耦合、背景变化主线和在线重建不进入本篇最低完成条件。

## 10. 近邻覆盖与可区别的问题

截至2026-09-05的有限检索，没有依据声称“首次状态视角选择”或“首次联合鲁棒与主动感知”。

| 工作 | 已经覆盖 | 本文需避开的重复主张/合理比较 |
|---|---|---|
| [LIBERO-Plus](https://arxiv.org/abs/2510.13626) | 七类扰动及VLA脆弱性分析 | 换相机性能下降不是新发现；本文须深化到强后训练后的闭环价值与可兑现性 |
| [KYC](https://arxiv.org/html/2510.02268v1) | 射线条件化、背景位姿捷径、无wrist隔离实验 | raw-ray负结果只能限定适配与设置；完整几何因果需要matched训练与有效sensor对照 |
| [CVC](https://arxiv.org/html/2608.06965v1) | 同state同action的flow一致性，训练测试mask wrist，仿真与真机 | 同paired data的FM-only是必需对照；当前practical对照不能孤立loss效果 |
| [ActiveVLA](https://arxiv.org/html/2601.08325v1) | 观测三维输入的虚拟投影、任务区域、启发式视角排序和zoom | 可移植评分作baseline但应称adaptation；整系统需匹配RGB-D、候选图像、TopK和训练预算 |
| [Selective Perception](https://arxiv.org/html/2602.15543v1) | wrist+语言预测多相机/多模态路由，VLM标签，真实操作 | 已覆盖任务条件传感器选择；应比较相同输入的router及标签来源，不能以输入组合称新颖 |
| [G³VLA](https://arxiv.org/html/2606.24472v1) | ray、PRoPE、跨相机fusion和几何监督，含π0.5验证 | dual-camera不等于几何无需建模；若geometry/fusion为主贡献需现代对照 |
| [SaPaVe](https://arxiv.org/html/2603.12193v1) | 相机与操作动作分解，动态头部感知、数据集、仿真与真机 | 鲁棒执行+主动观察的大框架已有覆盖；本篇应明确冻结策略和静态视角价值研究范围 |

若未来采用virtual view，还需区分[已观测图像还原训练相机的AnyCamVLA](https://arxiv.org/abs/2603.05868)与寻找高闭环价值视角；若改相机坐标动作表示，需考虑[CamVLA](https://arxiv.org/abs/2607.05396)。这两支暂不列为当前必须复现系统。

可以争取的区别是：在训练分布、传感器接口、随机策略和查询预算明确的条件下，测量并学习策略条件化的闭环视角收益，区分部署固定、任务适配、状态适配及事后搜索收益；再通过新source和外部验证证明其实际影响。不能仅靠“residual landscape”这个命名建立创新。

## 11. 五条重要结论与最终贡献的层次

| 五条要回答的结论 | 最简说法 | 现在状态 | 与已有工作的关系 | 最后是否放贡献列表 |
|---|---|---|---|---|
| 训练覆盖的作用与保留代价 | 多看些真实视角有帮助，但还没有消除相机差异 | 正向recipe结果较成熟；单变量机制较弱 | augmentation已知；增量在严格强基线和边界量化 | 作为证据基础，通常不单列独创算法 |
| 附加几何/一致性增量的条件性 | 加ray或一致性，不保证在这个双相机模型上继续涨分 | 当前适配/recipe负结果成熟 | KYC/CVC在其设置有效；不是普遍反证 | 机制支持或附录，不硬凑核心贡献 |
| 视角价值与可见性的差别 | 看见更多，不一定做得更好 | 有多个反例；尚缺大样本完整解释 | 感知质量不等于控制效用是已知原则 | 若系统刻画预测边界，可成经验发现 |
| 强训练之后的剩余选择空间 | 训过多视角后，某些状态仍可能值得换相机 | dev独立Q净+2.93pp；test待完成 | 需要区别既有view routing/active方法 | 若跨基线与外部验证成立，是核心发现 |
| 可学习、允许不换的视角规则 | 学会什么时候值得换，比强制挑一个新视角更实用 | 当前简单learner已冻结，改进方法未做 | advantage/abstention本身已知 | 需有效泛化、成本与消融后才是方法贡献 |

最后的正式贡献列表建议压到三条：

1. 明确训练与传感器条件的闭环视角评估协议及可复用数据资产，并区分不同层级的适应收益。
2. 经受控对照和外部验证确认的剩余视角价值规律；不能提前填写“多视角训练未消除可选择空间”的结论。
3. 若成功，基于闭环优势、具备fallback的选择方法，在新source与明确查询成本下取得实际收益。

前期大量实验可以支持这三条，不必每个阶段硬包装为独立创新。

## 12. 论文结构与停止规则

建议正文顺序：问题与观察契约；强后训练基线；同状态value评估；解释变量与弱selector；最终方法；新source和外部验证。主图只需五类：训练前后视角成功分布；状态×候选value map及不确定性；canonical/global/task/statewise/learner对比；收益—harm—查询成本；第二环境/真机结果。KYC复杂适配历史、失效natural gate、重复PDF版本与工程调度过程放附录或归档。

根据最终证据收敛：

| 最终结果 | 对应论文主张与动作 |
|---|---|
| statewise和learner稳定超过task-fixed，外部验证成立 | 以条件闭环价值学习和实际视角选择为主方法论文 |
| statewise成立，简单learner失败 | 先区分表示/数据不足与可观测性不足；最多补一轮强表示+fallback，不直接宣布必须POMDP |
| 只超过global，未超过task-fixed | 收敛为任务级相机配置，不声称任务内state adaptation |
| 仅个别任务收益且伤害抵消 | 明确机会检测与适用任务，不推广普遍主动视角收益 |
| 独立确认无明显headroom | 保留鲁棒性与评估论文方向，删除selector正向标题；不能说连续/动态视角无价值 |
| learner超过当前搜索候选 | 搜索下界并非最优上界；核查估计误差和搜索漏选，不把超过100%capture当矛盾 |

关于CVPR：当前积累足以写出完整研究报告，尚不足以判断已形成有竞争力的主会论文。正向已完成结果主要是已知数据路线，负结果的因果解释有限，独立测试状态少，方法与外部验证尚缺。改标题本身不能修复这些缺口。

CVPR并不要求每篇必须有新网络或真机；官方评审指导也强调新颖性、影响和实证洞察，而非只看SOTA表格。[CVPR评审指导](https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines)。因此有两条可行路线：一个跨模型/环境、揭示重要且可复用规律的分析与基准论文；或一个有明确独立收益、成本控制和机制消融的方法论文。当前建议优先尝试后者的一次精简闭环，同时保留前者的诚实收敛出口，不为追求完整标题同时扩到fusion、背景、POMDP和连续相机控制。

## 13. 直接执行建议

当前继续原冻结v2，并在完成后先归档原始Q及可能的R。随后优先做不新增rollout的task-fixed、P-only fallback、统计和范围审计；再冻结匹配的跨checkpoint小矩阵与真正新source测试。新selector以优势预测和保持canonical为默认选项。只有这些步骤显示有意义、可预测的收益，再投入小规模物理相机选择验证。

运行期间不整理或重写核心脚本。最终代码清理在冻结产物与复现分析通过后，按[仓库清理交接](../audits/repository_hygiene_handoff_20260901_zh.md)建立独立release工作区进行。
