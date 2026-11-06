# Paper 1 报告层

`prefix_comparison.py --run <持久run> --output <新目录>`只读验收完整32回合G更新与Stay/P3/full端点，绘制带图例的查询、STOP、奖励和在线分母曲线。缺少任一配对端点时默认拒绝汇总；显式`--allow-partial`仅将完整模型出表，其余标记pending，不以插值或部分成功率补齐。核心统计和绘图分别在`analysis/prefix_comparison.py`与`plotting/prefix_comparison.py`。

`model_search_comparison.py --config <输入哈希配置> --output <新目录>` 组合三模型同位置网格对照与π0.5完整PPO结果，复用核心离线选择统计；分别标记任务/初态总体、起点、训练量与查询成本。未知GR00T PPO保留未运行，不把网格端点复用当独立样本；不运行研究任务。

`target_spatial_recovery.py` 在原固定空间报告之外生成p3解释补充：重验全部块和复用来源、检查实际评分噪声/操作噪声配对，输出逐初态选位与好候选存在性、中文双三维图。固定端点复用不冒充新恢复轨迹；不使用PPO选位，不增加模型调用，不把p9追认为主配置。原 `target_spatial_panel.load_verified_panel` 提供共有的只读数据装配。

`coordinate_mechanism.py` 汇总相同视觉输入下的CPU首步坐标干预；`coordinate_recovery.py` 核验冻结G安装偏移检查的144搜索/288操作配对，报告成功、救回、损害和相对零偏移差值。两者不能互相替代：首步坐标敏感性不证明主要依赖捷径，四开发任务的小幅安装偏移也不证明真实头部或未知最佳视角的普遍泛化。

`coordinate_validation.py` 消费完整2268搜索/4536操作，核验全覆盖、各方法同初始数组、同执行噪声、零安装回零对旧参考的逐条一致性与预算结算，然后调用核心统计的预定主机制/PPO成本判定。报告保留所有任务、种子和安装子组，不能把READY解释为真实机器人或所有场景已验证。

`search_tracking.py` 将已有搜索训练曲线、历史标签回放、真实仿真A/B和共同时间上限复核
分别整理为可审查的W&B导出包；默认离线，只上传指标、筛选配置和统计图。
登录、授权及下一轮训练接入边界见[实验看板协议](../../docs/dsol_paper1/protocols/experiment_tracking_zh.md)。

`biweekly_review.py` 根据冻结分析产物生成2026-10-15双周报（PDF、可编辑DOCX、Markdown共用内容块）。
日期、证据截点、输入与输出路径位于`configs/dsol_paper1/analysis/biweekly_20261015.json`；
这是独立汇报快照，不覆盖唯一研究进展PDF。沿用上一份3页A4格式，区分汇报日期与实际数据截止日期。

`initial_results.py` 保留 LIBERO 阶段报告构建；`legacy_style.py` 保留历史训练图与原版样式。
`robocasa_results.py` 构建 LIBERO/RoboCasa 完整矩阵统一报告；
`robocasa_diagnostic_update.py` 在该16页版上保留前14页，更新为18页的官方模型诊断版。
该更新器显式接收原发布目录、诊断与分析目录；先构建、检查预览，再用 `--publish-built` 发布。
完整400初态与40初态诊断分开标注，不运行实验，不改变原始统计。

`confirmation_update.py` 消费已验收的新初态独立确认统计，保留19页历史版的前18页正文，
用五窗口对照与任务差异两页替换末页展望，发布为同一个20页PDF。
它不从闭环成败重新选窗口；构建、逐页像素保留检查、预览与发布分开进行。

科学统计、选择器与候选绘图实现位于 `AlphaBrain/research/dsol/`。报告层读取已核查产物，不训练 VLA，不重新评测，不修改统计总体。

`core_decision.py --config <显式配置>` 合并完整 U/T 筛选、长续训与部署预算证据。
逐阶段核验账本、actor、结果哈希、请求与配对噪声；失败替代必须逐项指定，不自动挑成功重试。
输出三种子原始曲线、同任务/初态配对区间和 CSV；统计位于 `analysis/core_decision.py`，
绘图位于 `plotting/core_decision.py`。只接开发集，Q4/Q6/Q16 必须与同训练检查点比较，
不把512回合筛查与长训主表混为同预算消融，不自动按最高分启动或冻结新实验。

`native_abg_comparison.py` 合并已验收的A/B与G报告，逐搜索检查任务/初态/起点交集与两次执行完整性，再生成同种子主表、未见任务三种子及救回/损害/查询对照。缺失随机对照保留为空，不插补缺失A/B种子；输出必须为新目录，历史报告不覆盖。

`native_online_reward.py` 消费完整从零G训练和恢复评测；严格要求9600次搜索/19200条操作轨迹、三个种子各2048训练回合。历史B只在实际已有的种子/初态/噪声交集配对，拒绝缺失种子插补。只读CPU报告通过独立冻结副本运行，不修改活跃实验源码。

`window_synthesis.py` 消费三环境五窗口统一验收统计，保留前18页图表正文，将原末两页
融合为五页更新（23页同一PDF）。明确候选权限、执行噪声预算、初态拆分及RoboCasa未覆盖
的排除规范位终点；完整数值与不确定性放在同名配套说明中。先构建并预览，再显式发布。

统一构建入口为 `reports/paper1/unified.py`，当前报告薄入口为 `scripts/dsol_paper1/reporting/report_standard_initial_results_v1.py`；该入口维护历史基础报告；当前full版及续接入口以 [报告导航](../../docs/dsol_paper1/reports/README.md) 为准。旧脚本根目录不保留兼容壳。

`historical/` 收纳旧阶段报告构建器；输入口径各不相同，不参与当前报告的默认生成。旧 PDF 监视器已获用户授权结束，报告代码不再因其固定调用路径滞留脚本根目录。

连续训练看板入口：`continuous_training_tracking.py`；独立读取已提交日志，以连续回合数为横轴，不复用历史97点run。

`search_progress_appendix.py`原位更新当前full版的连续搜索续接页，只读取已完成的历史统计与训练日志。
保留前33页正文，区分97点、连续首版和三组修正版，不生成额外历史PDF，不执行模型或W&B上传。

`continuous_stop_audit.py` 读取冻结训练与部署记录，显式 `--cpu-policy` 做CPU策略复算；`native_perturbation_audit.py` 核验已有规范/扰动结果及原生首步轨迹，输出配对恢复与Accel证据，不运行模型或仿真。

`native_view_response.py --run <持久run>` 校验全部72块后输出全27点评分/成功率及3D PDF；只读分析，不运行仿真或模型。

`native_full_recovery.py`消费完整原生基线与恢复评测证据，生成逐任务配对统计和带三维位姿的CSV；尚未完成的评测不能生成完整结论。

`native_baseline_diagnostic.py --run <持久run> --output <新诊断目录>` 只读核验完整40任务基线的评分/结果JSON与配对噪声，输出规范/26扰动视角、分任务/强度及全候选最低Accel诊断。不需要恢复评测完成，不覆盖当前full报告，不把离线遍历候选当作8查询PPO效果；统计实现位于 `analysis/native_baseline_diagnostic.py`。

`recovery_geometry.py --run <run> --baseline-panel <panel.json> --verified-hashes <已验收结果JSON哈希表> --output <新目录>` 复核全部恢复结果JSON，输出连续搜索路径/终点及3D PDF。该报告只描述既有数据，不运行模型，不把缺失的规范起点评测推断为零损害；全二进制产物复核由完整报告另行执行。
# FastWAM 关闭批次诊断

`python -m reports.paper1.target_search_diagnostic --root <closed-run> --output <new-run> --font <CJK-font>` 只读取已完成评测与训练记录，验证使用的科学产物哈希，生成逐状态实测三维图和训练诊断。新增模型调用为零；没有评测过的候选保留未知，p9 重算是离线诊断，不能当作新策略成功率。输出目录不得覆盖冻结报告。


`target_spatial_panel.py`在全量完成后校验新块与旧结果哈希，合并432个固定位置，输出逐初态统计、CSV和16页空间图。重复/缺失结果拒绝汇总；不得从部分完成面板报最终成功率。

`target_task_average.py --root <closed-run> --output <new-output> --font <CJK-font>` 消费已审计行表并核验来源哈希，生成任务等权平均及逐任务的 Accel/成功率三维面。三角面仅连接实测点，相关性只使用实测点；同时报告排除参考位的逐任务和逐初态相关性，避免将总体平均关系冒充场景内关系。

`target_spatial_linear.py --root <closed-run> --native-panel <audited-native-panel.json> --output <new-output> --font <CJK-font>` 补充原始数值Pearson、p3/p9及同任务/初态/角度π0.5对照。正向符号统一为`corr(-Accel,成功率)`，另存原始符号、含/不含参考位、逐初态中心化与全部位置混合结果。仅CPU读取冻结数据，p9分析不授权配置切换。

`selection_screen.py --root <closed-rldx-run> --output <new-output>`校验全部受控结果和原始P3选位后，报告有限开发集参考位/固定扰动/最低分终点成功率。核心统计在 `analysis/selection_screen.py`；选定终点复用于两个起点的配对不膨胀独立操作数，不将稀疏操作标签插值成完整三维成功率面。

`rldx_spatial_completion.py --root <completed-run> --output <new-output> --font <CJK-font>`核验复用及新增操作、原始速度评分、输入与噪声身份，完整864条结算后生成任务平均/逐初态空间图和Pearson对照。P4仅离线诊断；有限候选子集与事后最佳位置不冒充PPO或可部署策略。

## 完整开发分析审查

`python -m reports.paper1.core_review --source AUDITED_REPORT --output NEW_DIR`
消费 `core_decision` 已核验的报告，复核逐条CSV与全部汇总及配对均值，生成含目录的完整PDF、HTML、全表CSV和补充矢量图；不重新评测或改变统计总体。
需要 NumPy、Matplotlib、PyMuPDF；输出目录必须不存在。统计校验归属 `analysis/core_review.py`，绘图归属 `plotting/core_review.py`。
当前审查发布物：持久项目根下 `runs/shared-core-review-20261106/`；原始决策报告继续保留在 `runs/shared-core-decision-20261105/report-r2/`。

