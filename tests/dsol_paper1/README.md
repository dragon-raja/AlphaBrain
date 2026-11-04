# Paper 1 测试导航

前缀与序列接口：`runtime/test_prefix_runtime.py`覆盖动作/trace保持、受控实际入口委派、断联和未知拒绝重放；`runtime/test_rldx_search_entry.py`验证原生历史逐动作更新、序列身份与未知调用保留；`runtime/test_native_pipeline.py`新增有序query布局的完整采集→CPU PPO回归；`execution/test_libero_release.py`检查延迟renderer依赖闭包。Torch测试在缺少依赖时跳过，不能作为GPU证据。

浮点流时间：`acquisition/test_token_time_flow_trace.py`验证原生均匀/非均匀Euler网格、动作/RNG/调用数不变、实际执行窗口与有效维评分，以及RTC前缀、残缺/重复/嵌套记录拒绝。需CPU Torch，未装时跳过，不能算真实RLDX权重或GPU验收通过。

VLM图像token：`data/test_vlm_image_tokens.py`覆盖真实双相机/单ego布局、空间网格与图像占位的一致性，及时间帧、重复身份、padding、文本交错、非有限值拒绝；不加载模型或启动GPU，不能代替新VLA的数值准入。

生产接入候选：`runtime/test_controlled_training.py`覆盖四种小批适配组合的实际采样→CPU学习器→发布入口，以及视觉身份/训练划分/清单/哈希错误；`runtime/test_controlled_campaign.py`验证只连接所选节点并拒绝排除节点，不自动转派。`selectors/test_visual_bridge.py`验证共同视觉池化、缺失视图拒绝和meta非持久缓冲区恢复；`acquisition/test_flow_trace.py`验证非均匀Euler采样记录保持动作/RNG并在异常后恢复原方法。CPU合成测试和权重加载不代表FastWAM真实推理通过。

少量交互适配候选：`data/test_adaptation_reward.py`覆盖新域统计隔离、单位缩放、分歧、窗口淘汰、首批固定对照、混合版本/零尺度拒绝；`selectors/test_adaptation_ppo.py`通过真实CPU PPO入口验证8回合更新、窗口淘汰后的精确恢复、检查点失败拒绝重放及旧完整optimizer热启动拒绝。合成样例在`helpers/adaptation_reward.py`；不加载VLA、GPU或模拟器，不等于迁移成功。

完整安装验证：`data/test_coordinate_validation.py`检查2268搜索的任务/起点/方法平衡及物理边界；`runtime/test_coordinate_validation_runtime.py`覆盖三安装和四类实际搜索入口、同初始输入及返回执行逐值重放；`analysis/test_coordinate_decision.py`检验固定判据、共享基线和完整性拒绝；`reporting/test_coordinate_validation_report.py`以完整合成面板调用实际报告构建器，拒绝缺失和不同初始查询。运行环境依赖跳过必须另报，CPU合成通过不代表真实科学效果。

坐标机制检查：`analysis/test_coordinate_mechanism.py`验证CPU首步干预，`data/test_coordinate_probe.py`验证不按结果选择的配对面板；`runtime/test_coordinate_search.py`覆盖真实search入口及执行位姿逐值重放，`runtime/test_coordinate_controller.py`覆盖历史消耗保留计费、已结算预留与未知结果停止；`reporting/test_coordinate_recovery.py`覆盖144搜索/288操作汇总及缺失、重复结果拒绝。均纳入portable；合成测试不等于实际GPU闭环验收。

在线奖励：`data/test_online_reward.py`验证尺度等变性、局部log势差、计费与零尺度拒绝；`selectors/test_online_ppo.py`验证三组恢复后模型/优化器/RNG/统计严格一致。`runtime/test_controlled_training.py`通过实际采样和学习器入口验证B/G/L完整更新；`runtime/test_controlled_recovery.py`验证独立候选/参考诊断增加四调用且不改变选位和执行。Torch测试需CPU可用Torch环境，center跳过不能冒称通过。

`runtime/test_controlled_campaign.py`防止组合器向authority通道重复注入其禁止覆盖的环境字段；W&B由通道自身强制禁用。`analysis/test_reward_adaptation.py`验证共享基线、缺失/重复拒绝和配对统计；`reporting/test_reward_adaptation.py`使用完整1152条合成面板检验实际报告入口，不启动模型。

测试按被验证的职责归类，而不是全部平铺在论文目录。文件统一采用 `test_*.py` 命名。

| 目录 | 验证对象 |
| --- | --- |
| `data` | 数据 schema、初始化、相机输入与显式噪声 |
| `metrics` | Accel、指标计算、指标规则与评分接口 |
| `selectors` | 选择器与候选筛选 |
| `analysis` | 成功率统计、训练条件对比与数值回归 |
| `protocols` | 候选空间、噪声预算、评测协议与冻结包 |
| `execution` | 调度器、运行控制器及恢复逻辑；通过 mock 验证，不启动正式任务 |
| `diagnostics` | 渲染／轨迹一致性、来源身份和审计 |
| `acquisition` | 尚未作为正式研究结论的相机获取工程原语 |
| `reporting` | 图表与报告生成、研究范围表述 |
| `architecture` | 模块边界、迁移证据、目录约束 |
| `helpers` | 共享合成测试样例，不导入其他测试模块 |

`execution/test_controller_ledger.py` 覆盖接管所有权、预算预留、断联停止与显式恢复，纳入portable配置。
共享存储跨机锁语义和GPU后端准入分别记录，不能用单元测试替代。
`execution/test_controlled_queue.py` 使用持久CPU模拟后端测试真实启动入口和CLI，覆盖进程崩溃、回执丢失、部分失败、断联及显式恢复，不加载研究后端。
worker侧检查还覆盖预留后的身份/代次/许可/任务/槽位/根目录/资产变更、重复交付、worker写入和回执失败，以及独立子进程握手无锁死；同机worker接口拒绝跨主机请求与真实后端。新增authority接口只允许固定CPU假worker，通过实际CLI及独立进程测试实时授权、共享快照无效、旧会话、预留后进程退出、丢ACK、部分结果、检查点故障、过期和显式恢复。共享fixture在 `helpers/controller_cases.py`；默认测试只用本地子进程，真实Fresh SSH故障实验需显式配置，证据在对应持久run目录。

`paths.py` 统一提供仓库根目录，新增测试不再通过自身文件层数推测仓库位置。`conftest.py` 仅配置测试导入环境，不启动模型、GPU 或仿真器。

从仓库根目录运行单模块，例如：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/dsol_paper1/metrics -q
```

完整环境和便携 CI 的区别见 [测试总入口](../README.md)。历史协议测试保留原科学语义；按目录归类并不代表历史结果属于当前论文的有效证据。

`execution/test_dispatch_policy.py` 独立验证28处真实函数边界、旧CLI子进程和未受控队列拒绝，检查在读取执行输入、启动进程或加载GPU包前退出；不使用测试豁免。队列机制及合成驱动回归中仅对明确的CPU假后端使用测试内monkeypatch，不能作为正式准入证据。

`execution/test_model_authority.py`验证实际模型入口的CPU合同模式、异步CHECK/CONTINUE、断联/旧代次/丢完成回执/部分块及显式恢复，验证科学适配器只调用一次冻结循环。`helpers/model_cases.py`提供独立合成绑定。真实模型和GPU不是这些测试的依赖。

双worker候选增加`execution/test_worker_gpu.py`：进程出生身份、同GPU锁、单worker容量不能隐式升级、旧代次/未知占用拒绝及原显存检查的复用。模型入口测试同时覆盖批量CHECK、心跳期间许可撤销、断联/旧代次/丢完成回执及不明预算不重发；仍不加载GPU或模型。远端48-worker CPU对比与源码哈希保存在项目`runs/center-dual-worker-20261023/`，不是实际双worker容量准入。

`analysis/test_continuous_stop_audit.py` 验证初始/返回查询成本、奖励破坏与重复回合拒绝，以及规范成功条件下的挽救/附带损害、共享参考和无损失分母；纳入portable测试。

`data/test_native_view_panel.py` 覆盖完整预算、复用、不受控入口拒绝及实际适配器CPU假后端执行；`analysis/test_native_view_response.py` 防止把低评分失败混成oracle收益、漏掉失败点或重复样本。

驻留候选由`execution/test_resident_authority.py`验证实际队列/账本/远端进程跨块复用、独立预留、丢第二块ACK/完成回执、代次变化、空闲断联及禁止重放；`runtime/test_simulator_session.py`用真实进程/Unix socket和CPU环境替身验证同解释器的新环境会话及失败后禁止重启，替身位于`helpers/simulator_agent.py`。`data/test_native_view_panel.py`比较冷/热块的评分、轨迹和计费。CPU替身不证明EGL/模型数值或72-worker吞吐。

原生全40任务覆盖与预算测试为 `data/test_native_baseline.py`；训练限定校准与泄漏拒绝为 `data/test_reward_calibration.py`；`acquisition/test_continuous_search.py` 验证只改reward、不改观察和查询/返回费用。CPU证据不代替真实冷/热轨迹比较。

固定训练阶段由 `data/test_training_stage.py` 验证原计划/配置身份、完整更新边界和六组覆盖；`selectors/test_continuous_export.py` 验证完整优化器/RNG检查点另存与来源文件不变。`execution/test_fixed_stage_launcher.py` 使用显式 `DSOL_FIXED_STAGE_PREPARATION` 指向run内冻结组合入口，通过CPU替身验证已结算批次不重采、第64次更新停止、未知结果不重发，以及导出失败不进入评测。默认不启动生产入口。

恢复评测优化：`runtime/test_recovery_validation.py` 检验全量预检/当前块基线依赖哈希，包含同尺寸同mtime篡改、其他块错误和全局权重改动拒绝；`data/test_native_recovery.py` 验证合并wave不改顺序/首门槛/预算。`execution/test_prepared_recovery_launcher.py` 使用显式 `DSOL_RECOVERY_OPT_PREPARATION` 验证真实编排的完整23批和首块失败停止；`execution/test_recovery_wave_capacity.py` 在 `DSOL_LARGE_CPU_QUEUE=1` 下以6个CPU假slot跑288块，检查独立结算和驻留复用，不宣称48GPU容量准入。

## 恢复接续组合回归

`runtime/test_recovery_validation.py` 同时调用真实清单准备器与真实恢复校验器，覆盖继承基线清单时遗漏模型资产、变更文件和冲突哈希。
`execution/test_staged_evaluation.py` 使用实际共享阶段入口，覆盖CPU合同、首块门槛、未知启动和部分阶段失败；失败不重发、不生成完整报告。均纳入portable，不能用两边都返回成功的桩替代组合证据。


`data/test_reward_cold_start.py`检查完整G授权/预算、历史采样前缀身份、9600搜索面板及实际编排器64屏障，不启动GPU。

FastWAM接入CPU回归：`runtime/test_fastwam_runtime.py`覆盖官方函数逐值对照（显式提供固定源码）、实际工厂/授权执行器、断联、未知调用拒绝重放、动作差异拒绝和初始化回执。该测试使用合成模型，不计作真实推理或跨模型迁移证据。

`runtime/test_target_operation.py` 覆盖 FastWAM 逐步/分块轨迹、提前成功和未知执行结果不重试。

`runtime/test_controlled_adaptation.py` 校验12组版本屏障与预算；`runtime/test_target_recovery.py` 用模拟后端运行实际配对评测入口，检查查询与操作计费。

`analysis/test_target_adaptation.py` 验证重复结果拒绝、任务级聚类和未完整面板不可晋级。

`runtime/test_rldx_runtime.py`覆盖真实时序索引、相机身份、原生夹爪转换、执行段/有效维度评分、实际受控数值入口、未知调用和断联停止，以及逐动作历史与工程轨迹预算；全部使用CPU合成后端，不作为真实GPU或任务成功率证据。

`analysis/test_selection_screen.py`核对有限选位诊断的任务聚类、终点复用计数及缺失/重复标签拒绝，不把两起点复用变成额外独立操作。

`data/test_spatial_completion.py`检查独立科学身份的覆盖、确定性复用、重复结果冲突拒绝及新增预算；`runtime/test_rldx_spatial_panel.py`通过模拟后端运行实际补齐入口，覆盖模型一次加载/后续重绑、每条操作新环境、断联和不明结果导致lease不可复用。CPU通过不等于跨机GPU准入。

`analysis/test_spatial_completion.py`验证配对重复和原始评分成员保留，拒绝缺失、重复、不明操作及评分/操作位姿不一致；完整网格之前不得插补成功率标签。
