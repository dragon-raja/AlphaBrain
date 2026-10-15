# Paper 1 命令与历史工具

研究问题、结果和唯一当前 PDF 从 [研究入口](../../docs/dsol_paper1/README.md)进入。算法实现与目录规则见 [CONTRIBUTING](../../CONTRIBUTING.md)。

## 日常使用

| 要做什么 | 位置 |
| --- | --- |
| XR-1已完成矩阵的绘图尾项 | [CPU补图入口](analysis/render_xr1_completed_panel.py)；只消费既有数组／统计，独立输出并留存原失败，不重跑仿真 |
| XR-1固定位置校准对照 | [只读诊断入口](analysis/analyze_xr1_seen_rules.py)；仅seen标签选位、heldout评价，探索性对照，不是主动搜索方法 |
| 新实验的标准动态队列与恢复 | [统一执行入口](operations/manage_work_queue.py)；`prepare/status`不导入后端，`run/recover`必须显式`--execute`；接口和实际接入边界见[执行管理协议](../../docs/dsol_paper1/protocols/execution_management_zh.md) |
| 更强RoboCasa模型推理与Accel采集验收 | [XR-1显式工程探针](diagnostics/robocasa_xr1_probe.py)；真实缓存初态，不推进仿真，需完整资源收据，非闭环成绩 |
| 更强模型有界闭环诊断 | [冻结与启动](operations/launch_robocasa_xr1_panel.py) · [worker](runtime/robocasa_xr1_worker.py)；先160条规范基线，扰动阶段另行放行，不修改P0执行器 |
| 已完成三模型全视角面板的3D与任务分层诊断 | [只读分析入口](analysis/analyze_view_landscape.py)；只消费冻结数组，新建独立输出目录，不重跑模型、不替换原报告 |
| 数据资源复制/下载与逐文件校验 | [资源导入入口](operations/import_resources.py)；必须显式 `--execute`，原始资源不覆盖，失败目录保留 |
| 跨数据集实验入口审计 | [只读资源索引](diagnostics/audit_resource_catalog.py)；只验证导航和元数据身份，不启动评测 |
| 完整数据/框架/权重关系检查 | [项目资源图](diagnostics/audit_project_resources.py)；可移植ID清单与本机路径分离，检查后端误配与来源环 |
| RoboTwin 原生示范与相机回放验收 | [显式回放入口](runtime/robotwin_replay_probe.py)；默认不执行，独立环境/资产副本，不加载VLA、不修改原数据 |
| 官方RoboTwin资源完成验收 | [显式CPU验收入口](operations/finalize_robotwin_official.py)；等待两项固定导入完成，核验50任务当前schema，只在全部通过后发布新指针 |
| 当前初态结果核查、统计和选择器 | [薄分析入口](analysis/analyze_standard_initial_results_v1.py)；实现见 [研究包](../../AlphaBrain/research/dsol/README.md) |
| 当前统一汇报 | [统一入口](../../reports/paper1/unified.py)；实现见 [报告模块](../../reports/paper1/README.md) |
| 仿真输入、恢复、闭环与评分适配 | [runtime](runtime/)；不承担通用指标算法 |
| 历史指标、Oracle、闭环统计和画图 | [analysis](analysis/) |
| 历史候选空间、协议、噪声预算构造 | [protocols](protocols/) |
| 数据生成、训练、训练匹配与来源核查 | [training](training/) |
| 输入、渲染、运行身份和资源诊断 | [diagnostics](diagnostics/) |
| RoboCasa训练显存／真实批次 smoke | [显式执行的运行适配器](diagnostics/robocasa_training_smoke.py)；最多3步，可显式提供已发布数据清单检查真实配对；无checkpoint输出，不在默认检查中执行 |
| 历史运行编排 | [Python 控制器](operations/controllers/)、[Shell 启动配方](operations/launchers/) |
| 历史 PDF 构建 | [历史构建器](../../reports/paper1/historical/)；报告收尾脚本在 [reporting](reporting/) |
| 尚未用于正式评测的相机获取原语 | [acquisition](../../AlphaBrain/research/dsol/acquisition/) |

目录分组不是协议统一：历史工具仍有各自的数据 schema、模型与初始化约束。旧快照统计不能直接混入标准初态结果。

当前XR-1双机补齐仍使用已冻结的 `AlphaBrain/research/dsol/data/xr1_rebalance.py`；维护期间不切换源码。该实现不等于通用队列已完成XR-1生产接入，接入须另行完成适配器和实际轨迹准入。

## 根目录不再保留旧入口

原 186 个平铺脚本已分两次完成职责迁移（149 + 37），根目录只保留导航，不留批量转发壳。Accel、可见性和显式噪声实现进入研究包；运行适配、调度、分析、协议和报告各有归属。

用户已授权结束暂停的旧 full64 控制器与旧 PDF 监视器；操作前复核没有评测子进程，保存了身份记录，未启动或重跑任何实验。原源码和路径映射见 [本轮迁移记录](../../archive/repository/20260911/runtime-modules/manifest.json)。

源码布局迁移不意味着旧 release 可以换上新源码继续跑。历史源码身份属于冻结 checkout；新的运行链正式投入实验前仍需首帧和整轨迹等价验收，不能给新文件冒用旧哈希。

## 旧路径在哪里

[迁移清单](../../archive/paper1/maintenance/script-layout-20260911/manifest.json) 记录旧路径、新路径、原始源码备份和受保护文件。旧命令需要按清单换路径，Python 调用方改用完整模块名；不会通过修改 `PYTHONPATH` 搜索所有子目录来掩盖依赖关系。

历史冻结 release 按源码哈希识别原实现。重放已有 release 应使用它自己的冻结 `repo/`；不要把迁移后的源码强行配上旧哈希清单运行。历史文档和实验记录不改写。

## 维护检查

```bash
python tools/paper1/check_architecture.py
python tools/paper1/check_layout.py --list
python tools/paper1/test.py
```

结构检查不执行研究脚本。完整测试走 CPU；少数显式列出的 `--help` 测试只验证参数入口。不要用导入或 `--help` 试探未审查的历史控制器。

新增通用实现进入研究包；新增 CLI 进入相应职责目录。不得恢复已经迁出的平铺路径，也不以 `after_*`、`retry_*` 或复制版本来代替接口设计。
# RoboCasa 第二阶段入口（显式执行）

- `operations/continue_robocasa_xr1_panel.py`：一次性、有锁的160条规范基线→640条扰动对照衔接；检查冻结源码、权重收据、完整轨迹与显式噪声地址，待前阶段worker退出后启动。不以成功率筛选任务，不自动重试或扩展矩阵；异常写入状态并停止衔接。
  显式`--full-config`支持用户另行批准的全97候选扩展：复用前5视角800条，补92视角14,720条后自动汇总；默认不扩展。四分位、排名相关及中文空间图复用研究分析/绘图模块。

- `operations/refresh_robocasa_references.py`：经明确授权冻结本轮参考，全部初态独立进程复验，核验复用候选并补齐缺失；只发布新索引，不改历史参考或失败回执。没有 `--execute` 不启动GPU。
- `analysis/audit_robocasa_render_reference.py`：区分历史RGB一致性与当前渲染栈的跨进程重复性；只核查原始数组与来源，不替换参考图、不放行数据。
- `diagnostics/robocasa_render_artifacts.py`：定位历史图像差异对应的像素、几何体与材质，保存真实224像素模型输入配对；不放行异常数据。
- `diagnostics/robocasa_pixel_action_sensitivity.py`：在显式相同Flow初噪下比较历史／恢复图像的动作块，并加入同输入重复对照；不将动作块差异等同于闭环成功率差异。
- `diagnostics/robocasa_render_repeatability.py`：相机与物理状态不变时的重复渲染定位；材质／阴影等消融只在诊断进程内使用，不构成正式数据。
- `diagnostics/robocasa_policy_replay.py`：使用已校验的逐次规划图像、本体与噪声数组隔离验证推理重复性；不启动仿真，不将重放结果当作成功率。驻留策略适配归属 `benchmarks/Robocasa365/evaluation_policy.py`。
- `diagnostics/calibrate_robocasa_render.py`：在预先选定、布局隔离的seen初态上标定与验证像素数值门槛，独立进程重复，不修改正式验收标准。
- `analysis/summarize_robocasa_render_calibration.py`：检查完整标定来源，输出门槛建议及独立验证结果，禁止根据验证集失败继续增大门槛。

- `operations/prepare_robocasa_evaluation.py`：CPU 审计已冻结的全部初态及97视角目录；显式 `render --execute` 才构造初态候选图、安装参数和本体输入。像素门槛或物理身份失败不发布数据，不训练选择器，不产生成功率。
- `operations/supervise_robocasa_job.py`：显式单任务运行监督，持久化stdout、心跳、子进程PID和退出码；不自动重试、不调整实验协议、不管理无关进程。

- `runtime/robocasa_policy_probe.py`：真实输入、OpenPI Accel 轨迹金标准对照与可选整任务工程测试。默认不执行；不作为正式成功率估计。
- `runtime/robocasa_matrix_worker.py`：冻结协议下的驻留模型／环境矩阵worker；先运行分离目录的有界pilot，正式记录每个初态、候选及4次完整任务执行，并保留8套初态评分轨迹。显式执行、独占worker锁、校验续接，不允许隐式改渲染或任务总体。
- `runtime/robocasa_attribution_worker.py`：P0 后补充权重/冻结选择归因验证的薄入口；各权重阶段独立进程释放显存，复用 P0 闭环实现，不替换旧矩阵。协议见 `docs/dsol_paper1/protocols/robocasa_attribution_v1.md`。
- `operations/prepare_robocasa_matrix.py`：CPU侧冻结400×97×2×4总体、两模型权重哈希、源码与双机配对分工；不启动GPU，不把有界pilot纳入正式统计。
- `diagnostics/robocasa_rig_geometry.py`：左右相机联动几何、原状态及腕部保留检查；渲染位置为校准样本，不是已冻结训练分布。
- `diagnostics/robocasa_training_replay.py`：只读重放原始Human300训练状态；可选15位置rig校准，不更新模型。
- `operations/run_robocasa_training_calibration.py`：有界GPU队列，按固定训练ledger每任务取一帧，保存完整失败清单；不根据评测成绩取舍任务。
- `training/prepare_robocasa_matched.py`：冻结64训练＋32额外评测位姿目录、7498段示范的安装分配及源码哈希；不启动GPU。
- `training/render_robocasa_matched.py`：显式分片重渲染，同一示范安装偏置固定、同一帧腕部RGB共享；不训练模型。
- `operations/run_robocasa_matched_render.py`：8卡构造编排，逐示范留档，不停止原有进程；失败不减总体。
- `training/release_robocasa_matched.py`：全部图像及源文件哈希、源窗口配对验收后生成两组训练输入清单；清单不等于训练已完成。
- `training/recover_robocasa_matched.py`：仅对明确指定的缺失示范做有界摆放初始化重试；仍恢复原XML与状态，保存前后实现哈希，不覆盖失败结果。
- `diagnostics/audit_robocasa_render_equivalence.py`：修复前后训练图像与来源一致性对照，固定像素容差，不启动仿真。
- `training/train_robocasa_matched.py`：正式训练编排；必须先通过同组真实批次FSDP8验收，检查首批输入身份，只写新运行目录。配置与输入审计复用 `benchmarks/Robocasa365/openpi_training.py`，更新和checkpoint编码调用指定版本OpenPI；不实现另一套训练算法。
  中断恢复显式指定`--resume-run`和`--resume-step`，只读完整优化器检查点，重放数据游标，并以原连续运行的3步指标核验续接；原运行目录不改写。
- 训练支持审计属于报告层：`python -m reports.paper1.robocasa_training_audit --help`。旧权重只作有限全相机增强基线，正式外部-only范围见统一计划。
## 几何与可见性审查（2026-09-20）

`diagnostics/active_camera_feasibility.py`：显式`--execute`后用CPU编译原生机器人模型，核对相机关节、TIAGo头部正运动学，以及预选seen初态的候选相机中心几何标记。不创建渲染器，不调用VLA，不把中心点检查视为安装/路径碰撞认证；实现位于`benchmarks/Robocasa365/active_camera_audit.py`。

同一入口加`--kitchen-integration`可运行有界的TIAGo厨房原生控制接口检查，实现位于`benchmarks/Robocasa365/tiago_integration.py`，配置归于`configs/dsol_paper1/analysis/tiago_integration*.json`。不调用渲染或VLA，不评估操作任务成功；失败与超时必须保留，输出要求新目录，不能覆盖原记录。

`analysis/analyze_geometry_visibility.py` 是只读冻结LIBERO矩阵的CPU入口：
真实外参核查、零像素起点分层、独立的可见终点约束诊断。实现位于
`AlphaBrain/research/dsol/analysis/geometry_visibility.py`；静态绘图位于
`AlphaBrain/research/dsol/plotting/geometry_visibility.py`；审查图表编排位于
`reports/paper1/geometry_visibility_review.py`。不启动渲染、训练或闭环，不更新当前PDF。
