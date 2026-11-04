# Paper 1 可复用研究代码

共享搜索核心候选与接口边界见[统一核心合同](../../../../docs/architecture/shared_search_core_zh.md)。`data/search_core.py`枚举结构，`selectors/unified_search_policy.py`统一Actor/Critic计算，`search_policy_factory.py`以新schema显式接入原生布局及来源哈希；不改旧schema含义。四结构筛查与视觉/联合输入是分开的实验因素，RLDX前语言视觉捕获器目前只有CPU验证，不能称为生产GPU准入。

前缀合同与序列表示：`data/score_prefix.py`独立绑定去噪/评分/执行轴，`metrics/score_prefix.py`只计算；`data/query_features.py`将真实有序query池化为明确的四组，保持joint身份。`acquisition/native_search.py`与Gym接收完整layout，禁止根据维数重建图像布局。`analysis/prefix_comparison.py`区分组内奖励趋势与真实配对成功率，不能用奖励幅度跨前缀排名。

`data/change_reward.py`是显式启用的配对视角变化尺度候选，旧G不变；`benchmarks/LIBERO/paired_change_evidence.py`从已绑定查询记录恢复不同位姿差值、排除返回，并核对原始best-score链。首批三模型CPU PPO小验证能恢复FastWAM移动倾向，但纯噪声控制发现缩放后的已访最低分奖励仍有正选择偏差，**候选被拦截，未接入真实GPU采集或大训练**。`analysis/reward_noise_control.py`提供独立终点收益校正的纯数值原语和噪声/有信号对照；实际独立噪声、8查询预算分配和闭环效果尚未验收。证据在持久 `runs/change-scale-minimal-validation-20261102/`。

原生特征合同由`data/native_features.py`与`native_observation.py`负责；`acquisition/native_search.py`复用运动／查询；`selectors/search_policy_factory.py`构造按合同投影的历史策略，PPO共用原更新器。模型／benchmark适配留在外层，见[接口协议](../../../../docs/architecture/distributed_execution_zh.md#原生模型特征接入与搜索训练接口)。`execution/node_health.py`负责借用节点健康及失效关闭，不修改保活和科学协议。

`data/vlm_image_tokens.py`按处理器真实网格、图像占位token和mask拆分深层VLM表示，保留相机身份，拒绝时间帧伪装、文本混入、padding及不匹配的空间布局。只提供可复用CPU原语；THAW等新模型仍须实现真实读取并完成动作、轨迹、RNG的GPU一致性验收，不能凭维度兼容加入生产。

`data/adaptation_reward.py`是独立新域小批适配候选：8回合完整批、32初始查询窗口、分歧启发式与步间限幅，域/协议身份及统计随检查点恢复。`data/domain_adaptation.py`绑定训练域、共同视觉接口、源actor与64回合清单；外层实际采样/学习器入口接受独立8回合schema，旧B/G/L仍保持32回合。真实FastWAM适配、数值验收及新运行许可仍未完成；CPU接入不等于迁移或真机效果成立。

`acquisition/flow_trace.py`的`SchedulerTrace`只观察原生Euler采样器的真实非均匀时间网格、状态和速度，保留输出/RNG，不改Accel定义。独立视觉塔加载与旧池化复用在`AlphaBrain/model/pi05_features.py`，不补缺失相机，不把形状匹配当作真实特征一致性验收。

`data/coordinate_probe.py`冻结开发任务上的安装偏移机制面板与预算/actor身份；`analysis/coordinate_mechanism.py`实现相同视觉输入下首步坐标干预与汇总。不改真实运动学，不将CPU敏感性解释为闭环因果效果。执行适配与报告分别位于`benchmarks/LIBERO/coordinate_probe.py`和`reports/paper1/coordinate_recovery.py`。

`data/coordinate_validation.py`定义完整12任务安装验证的平衡清单、固定门槛与输入/预算身份；`analysis/coordinate_validation.py`执行配对任务bootstrap、种子一致性、损害与PPO成本判定。结果不足时明确HOLD，不自动增加样本或挑选种子。该试验不改变运动学，也不等于去先验新版策略训练或真机准入。

`plotting/coordinate_validation.py`只消费完整判定产物，绘制所有安装组的任务成功率、主总体配对任务区间与全量G终点，输出独立PNG/PDF；终点图是yaw/pitch姿态空间，不是三维平移或捷径归因。绘图不计算新成功标签或修改判据。

分布式训练、推理与评测的当前复用合同、维护步骤和未完成扩展统一见
[分布式执行维护手册](../../../../docs/architecture/distributed_execution_zh.md)。

- `data/`：产物 IO、身份核查、标准初态矩阵读取与显式 Flow 噪声库。
- `metrics/`：Accel 定义与推理适配、任务实体可见性、候选视角指标规则。
- `selectors/`：共享 attention actor/critic、PPO，以及历史视觉排序器。
- `analysis/`：Oracle、配对统计、结果汇总、配置驱动分析流水线。
- `plotting/`：实测相机空间、成功率矩阵等可复用绘图。
- `acquisition/`：当前 RL 搜索环境、动态局部候选及连续位姿接口；真实头部适配尚未完成。
- `execution/`：冻结任务身份、跨主机资源槽动态分配、持久状态、失败收尾和显式恢复；不导入模型或仿真。运行细节由显式后端适配器提供。
- `execution/staged_evaluation.py`：共享阶段接续，持久化启动意图，先CPU合同再首块门槛，失败停止且禁止原地自动重放；科学后端与报告通过显式回调组合。
- `execution/dispatch_policy.py`：通用队列限定注册后端及root/operation/进程/线程；真实模型另需实时授权作用域，旧直接入口仍关闭。历史冻结副本不受本文件约束，不是操作系统级隔离。
- `execution/controller_ledger.py`：持久所有权、恢复代次、科学地址去重与预算；由受控适配器用于模拟和已准入生产阶段。保管记录不授予执行许可，历史冻结后端不自动获得新所有权。
- `execution/controlled_queue.py`：标准队列的账本适配器，在实际 `backend.start` 前校验独立许可、owner/epoch及全部peer，并持久化预算预留。`controller_cli.py` 保持CPU假后端入口，真实模型由注册的model适配器组合；旧入口不自动接管。
- `execution/model_authority.py`：复用center预留/GO边界的持久在线传输；同轮worker CHECK共享一次新鲜账本/许可核验及全节点心跳，逐worker发对应序号CONTINUE，不发未来额度或缓存健康。科学执行位于LIBERO适配器，CPU合同模式禁止模型导入。
- `execution/resident_session.py`、`resident_runtime.py`：可选跨块驻留的槽位/所有者/代次/许可钉住、独立任务记账和不可变运行时租约；失败使会话失效，不提供模型导入或科学执行。每块仍需center重新预留和GO。
- `execution/worker_gpu.py`：新center适配器的物理GPU启动锁及同绑定/同代次worker进程出生身份验证；复用`gpu_guard`的可信保活、未知占用拒绝及加载峰值预留。`resources.worker_slots`只生成资源槽，不改变科学任务身份。双worker候选的CPU通过不代替真实模型容量测量。
- `execution/authority_backend.py`、`authority_wire.py`、`authority_worker.py`：center本地账本的单点授权、共享盘不可变检查点和SSH会话内PREPARE/GO/ACK；固定CPU假worker不读取NFS许可。断联/超时/不明消耗保留预留，只能显式核对恢复；无自动接管或生产执行通道。
- `execution/worker_fence.py`：worker独立重读许可与预留，持久化一次性启动意图和回执，拒绝重复/未知启动。当前只允许同主机CPU假后端；实测NFS跨机原子替换可能暂时读到旧内容，不能据共享路径建立生产启动授权。
- `protocol.py`：此前已存在的协议校验，保留接口。

推荐入口：

```bash
python -m AlphaBrain.research.dsol.analysis.pipeline \
  --config configs/dsol_paper1/analysis/standard_initial_v1.json
```

该命令只做离线核查与分析，不启动闭环评测。路径在配置中，核心实现不依赖作者机器。

报告由 `reports/paper1/` 消费结果；薄分析入口在 `scripts/dsol_paper1/analysis/`，原脚本根目录不保留转发文件。

独立恢复确认：`analysis/confirmation_results.py` 只消费验收后的冻结选位及执行结果，
输出窗口1–5、规范和32视角扰动基线的配对统计；主窗口不随结果改变。
RoboCasa原生复位、噪声地址及完整任务验收属于 `benchmarks/Robocasa365/confirmation_audit.py`，
使用CPU读取已有证据，不启动仿真或GPU。报告发布位于 `reports/paper1/confirmation_update.py`。

历史分析、协议构造、训练、诊断与控制器已按职责迁出脚本平铺区，详见 `scripts/dsol_paper1/README.md`。这属于可维护性整理，不意味着所有历史工具都已抽象成通用算法接口。

实际仿真适配在 `benchmarks/`，薄运行入口在 `scripts/dsol_paper1/runtime/`。新通用调度实现位于 `execution/`，旧导航入口是 `scripts/dsol_paper1/operations/manage_work_queue.py`（run/recover/audit-results动态后端已关闭），CPU模拟使用 `manage_controlled_queue.py`；`operations/controllers/` 保留历史兼容控制器。新布局不沿用旧源码身份；正式运行前仍需协议等价与轨迹验收，旧冻结实验使用自身的源码快照。接口、恢复和当前接入边界见 [执行管理协议](../../../../docs/dsol_paper1/protocols/execution_management_zh.md)。

## 当前 RL 搜索代码

- `data/online_reward.py`：保存原始改善/查询费用，正式完整批次后估计累计初始尺度与噪声启发式下限，先结算reward再GAE；统计与PPO检查点一起恢复。`data/reward_pilot.py`限定已批准B/G/L短适配协议，不更改历史A/B协议。
- `analysis/reward_adaptation.py`：完整三臂/三种子面板的任务等权净救回、损害、查询与独立评分诊断；共享基线去重，任务/种子配对重采样仅作小规模探索性区间。报告组合在`reports/paper1/reward_adaptation.py`，不参与训练或选位。
- `acquisition/search_env.py` / `local_view_search.py`：固定表和动态局部搜索，查询计费与停止。
- `selectors/search_attention.py` / `search_attention_ppo.py`：历史97点的编码、离散actor/critic与PPO；连续actor只复用兼容的历史编码。
- `selectors/continuous_search_policy.py` / `continuous_ppo.py`：当前连续STOP/角度策略及完整训练驱动；`data/continuous_training_store.py`：日志、事务与恢复。
- `benchmarks/LIBERO/continuous_training.py` / `continuous_training_queue.py`：当前在线连续查询训练适配与标准队列，默认只读验证；不是旧缓存评价入口。
- `analysis/search_attention_training.py`：已有训练流程；`data/local_search_plan.py`：下一轮划分/预算核查。
- `analysis/search_task_alignment.py` / `execution_trace_audit.py`：配对任务统计、回放边界、已有执行记录核查。
- `benchmarks/LIBERO/search_runtime.py` / `search_simulator.py` / `continuous_camera.py`：仓库外层的模型/仿真适配。

旧 97 点训练和动态候选设计不可混称为同一个已完成方法。当前状态见[项目首页](../../../../docs/dsol_paper1/README.md)。

`analysis/continuous_stop_audit.py`：已有训练奖励一致性及规范/扰动配对的条件恢复、附带损害统计；不启动模型。

`data/native_view_panel.py` 固定原生27视角响应面及768条/43,072调用预算；`analysis/native_view_response.py` 验证完整覆盖，分离评分选位与oracle，统计附带损害及任务敏感性；`plotting/native_view_response.py` 只画实测点及明确标注的插值。

`data/native_comparison_plan.py` 核算固定视角面板、局部Accel与冻结PPO在线搜索的配对执行/评分上限；只生成预算草案，不签发运行许可。

原生全40任务基线的分块与身份位于 `data/native_baseline.py`；训练集固定奖励尺度估计位于 `data/reward_calibration.py`。连续搜索支持独立 reward_scale，默认保留旧行为，观察归一化不随奖励校准改变。

原生A/B扩展：`data/native_recovery.py`定义与结果无关的恢复面板，`execution/learner_publication.py`通过CPU子进程stdout校验版本交接，并在启动前持久保存每版本唯一尝试；异常保留、不自动重做更新。`analysis/recovery_summary.py`计算任务等权配对净救回和损害；实际VLA/仿真仍属于benchmark适配层。

`data/training_stage.py` 验证用户固定的训练前缀修订、原配置哈希和完整更新边界；`selectors/continuous_export.py:export_stage` 保留原绑定并另存优化器/RNG完整检查点。修订是显式科学协议变化，不是将旧8192回合标记为完成。

`data/native_recovery.py` 的wave组合保持完整块顺序/首块验收和预算；基线依赖去重拒绝同文件冲突哈希。实际文件逐字节核验仍由benchmark适配层完成，不使用NFS mtime充当内容证据。

`analysis/native_baseline_diagnostic.py` 汇总完整27视角/两执行种子面板，复用任务聚类配对统计；分离规范、扰动保持、评分与成功率的初态内关系，以及全候选最低Accel的离线诊断。`plotting/native_view_response.py:baseline_figure` 仅绘制实测点，不把插值当证据。

`analysis/recovery_geometry.py` 统计连续终点是否位于离散面板、光轴角距离与配对损害，明确区分扰动起点搜索和规范起点搜索。`plotting/recovery_geometry.py` 绘制相对光轴三维分布、角度/独立Accel/结果散点及逐任务路径；相机光心固定，不能把方向点标成实际相机位置。

`data/reward_cold_start.py`定义原生G从零完整训练/评测协议，复用在线奖励与标准调度，独立于已冻结暖启动pilot。

`analysis/recovery_summary.py:paired_method_contrast` 按策略种子、初态、视角及执行重复严格配对新旧方法；缺失/重复记录拒绝汇总，不自动补齐历史不存在的种子，任务与种子区间标记为探索性统计。

`data/operation_contract.py` 管理显式模型操作协议；FastWAM 32/10、float64 和不额外裁剪独立于旧 10/5 协议。

`analysis/target_adaptation.py` 按任务聚类汇总少量适配的配对净恢复、损害与三种子对照；开发面板不冒充留出确认。

`data/native_observation.py` 提供原生观测/特征的纯 CPU 元数据合同：真实相机身份与角色、显式缺失、统一时钟下的新鲜度/同步、per-view 或 joint 映射及查询/历史分支身份。它不验证图像或特征张量内容，不加载模型，也尚未接入生产网络；时序/生成特征、几何有效性、原生读取与数值回归仍需独立实现和验收。

`AlphaBrain/model/fastwam_features.py` 提供独立、显式开启的 eager `prepare` 特征捕获候选，只读取同次调用的动作去噪前联合 tokens。CPU stand-in 测试覆盖输出/RNG不变与异常恢复；不支持并发/编译路径，尚未接生产适配器或完成真实模型数值验收。旧 `pi05_features.py` 不变。
# 2026-10-29 FastWAM 诊断与原生特征候选

`analysis/target_search_diagnostic.py` 对已完成搜索做逐状态/视角去重、未知候选标签保留、奖励结算和离线 p3/p9 诊断；`plotting/target_search_diagnostic.py` 仅绘制实测点；报告入口为 `reports/paper1/target_search_diagnostic.py`。

`selectors/native_search_policy.py` 是显式启用的候选：复用 `model/fastwam_features.py` 的一次 eager 前向读取，验证观测身份后按已声明空间网格做 2×2 池化，独立投影/融合真实视图或联合表示，再复用连续 STOP/运动分布及 PPO 目标。支持单移动相机、独立多相机及联合表示；不复制头图冒充腕图，不重训操作模型。新历史编码器需要新的搜索权重与配置，禁止隐式加载旧投影；旧 π0.5 路径未改。CPU 小张量测试不替代真实原生特征轨迹验收；真实 grid/tap 身份、生产采集器/账本绑定及实际特征存储仍需下一阶段接入。


固定空间评分有效性：`analysis/spatial_score_validity.py`按初态内排序，统计预定低分组相对均匀候选的成功率增益，并按任务聚类；包含/排除参考位分报。`plotting/spatial_score_validity.py`仅展示实测27点，不插值成功率。模型执行由LIBERO的controlled_target_panel账本入口负责，CPU报告不启动模型。
## 固定候选空间的解释边界（2026-10-29补充）

`analysis/spatial_recovery.py` 区分好候选存在性、仅按p3选出的最低分端点、以及复用固定端点结果的救回/损害配对。事后查到的好候选不作为可部署选择器；端点结果跨起点重复使用不增加独立操作样本量。`plotting/spatial_score_validity.recovery_figures` 对全部初态绘制评分/完整操作双三维图，不插值、不混入PPO训练结果；字体由调用方传入。报告组合入口为 `reports/paper1/target_spatial_recovery.py`，沿用原始结果哈希和固定点验证，旧报告不覆盖。
在线变化尺度确认小试（2026-11-02）：`data/confirmed_reward.py`采用实际起点到最终位置的独立有符号终点分差；`data/change_reward.py`仅提供训练交互变化尺度估计。每个完整on-policy批次冻结尺度，确认噪声不参与探索、选位或尺度。LIBERO `terminal_confirmation.py`在原8查询内预留确认并对失败锁定，`confirmed_pilot.py`以单独来源合同组合标准采集/CPU PPO/完整操作入口。旧G路径不变；此前“只缩分母＋best-so-far”仍被纯噪声控制拦截。三模型128回合/模型的真实小试尚不能代替完整主实验或无损害证明。持久运行证据：`runs/confirmed-change-small-pilot-20261102/`。

