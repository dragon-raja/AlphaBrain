# 下一阶段资产、证据与 checkpoint 匹配清单

日期：2026-09-07。状态：`READ_ONLY_AUDIT_COMPLETE / NEW_EXECUTION_RELEASE_HOLD`。

同日后续组件实现与独立工程 smoke 记录见[脚手架进展](next_stage_scaffold_status_20260907_zh.md)。下文“本轮”指此次只读资产审计本身，不表示后续尚未开始实现；科学执行仍未放行。

本清单对应 [中心计划 v2](paper_master_plan_v2_active_bridge_20260907_zh.md) 的 P0/T/S/A1 资产准备。本轮仅检查本仓库及本项目已归档的数据、训练和评测产物，并新增本文；没有修改代码、checkpoint、旧协议或结果，没有启动训练/rollout，没有读取其他 workspace 研究线的实验结果。同仓库旧 snapshot/probe 工具只审查实现可复用性，不引入其科学结论。仓库及上级未发现适用的 AGENTS.md。

“已核实”指本轮读到实际文件、配置、日志或代码；历史 audit 的 PASS 不等于本轮重新执行了其全部检查。“待核实”不自动表示错误。本文不是新的执行授权回执。

## 1. 结论与当前阻碍

1. 七臂 seed41 quick-gate 的真实 checkpoint、保存配置、训练 manifest、最终 step2000 日志均存在。Canonical-unique 与 Broad32-practical 是可优先复用的匹配候选：相同源数据文件、同 seed/拓扑/预算/学习率计划，保存配置仅训练臂和输出身份不同。
2. 旧 Canonical-unique 与 **v2 实际使用的 Broad64 M-B** 不是完全匹配训练：两者都实际训练2000步、看到64000个模型样本，但总学习率 schedule 分别3000/2000；末步日志 LR 分别 `1.6284082688567607e-5` / `5e-6`。此外8GPU/2GPU拓扑不同，不能直接称“只差coverage”。
3. 存在与旧七臂同3000-step schedule的 Broad64 quick checkpoint，但它不是 v2 O bank 对应的 checkpoint。改用它就不能直接复用 v2 Broad行为记录。
4. 当前DSOL基策略实际消费 external+wrist RGB+language，不消费接口携带的8维 robot state；两个 ridge selector 则使用 robot state。不得只在评测打开state或将当前配置默认为原生OpenPI等价。
5. v2全部完成；其64来源、O/P/Q及已查看的其他历史test只能作为历史开发/分析材料，不能重新命名为新来源盲测。
6. 主动相机原语与状态恢复可以复用，但时间化路径、持位控制、动态标定、观察权限/历史、代价账本及恢复等价性尚需新协议和验证。当前保持新训练/rollout release HOLD。

## 2. 七臂精确 checkpoint 入口

以下每行均核实包含 `model.safetensors` 和 `framework_config.yaml`；每个权重文件为17,610,456,556字节。训练manifest位于该 `final_model` 目录的父目录下，文件名 `run_manifest.json`；训练曲线为同父目录的 `metrics.jsonl`。本轮没有重新计算七个17.6GB权重的全文hash，不把文件存在性称为权重内容重新验真。

| 训练臂 | checkpoint绝对目录 | 模型样本 / source item 每update |
|---|---|---:|
| Canonical-unique | `/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_canonical_unique_quick-gate-v1_seed41_g8_gb32_steps2000/final_model` | 32 / 32 |
| Canonical-repeat | `/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_canonical_repeat_quick-gate-v1_seed41_g8_gb32_steps2000/final_model` | 32 / 16 |
| ImageAug-unique | `/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_image_augmentation_unique_quick-gate-v1_seed41_g8_gb32_steps2000/final_model` | 32 / 32 |
| Broad32 practical | `/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_quick-gate-v1_seed41_g8_gb32_steps2000/final_model` | 32 / 32 |
| Broad32 state-matched | `/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_state_matched_quick-gate-v1_seed41_g8_gb32_steps2000/final_model` | 32 / 16 |
| Broad32 paired FM | `/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_paired_fm_quick-gate-v1_seed41_g8_gb32_steps2000/final_model` | 32 / 16 |
| Broad32 paired consistency | `/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_paired_consistency_quick-gate-v1_seed41_g8_gb32_steps2000/final_model` | 32 / 16 |

完整文件名示例，不应误开另一个同名框架目录：

- Canonical配置：`/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_canonical_unique_quick-gate-v1_seed41_g8_gb32_steps2000/final_model/framework_config.yaml`
- Canonical训练manifest：`/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_canonical_unique_quick-gate-v1_seed41_g8_gb32_steps2000/run_manifest.json`
- Broad32配置：`/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_quick-gate-v1_seed41_g8_gb32_steps2000/final_model/framework_config.yaml`
- Broad32训练manifest：`/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_quick-gate-v1_seed41_g8_gb32_steps2000/run_manifest.json`

七臂共同事实：seed41，8GPU，global model examples32，实际2000 optimizer updates，scheduler_total_steps3000；最终日志均为step2000、examples_seen64000。共同训练数据manifest hash为 `3fe408f2cfb034ae703de32223f80367337a37890480a1662d3f116715d81c7b`；共同初始化manifest hash为 `f4afc8f39bb16fb8c40ae57bbfe3417ae3d91068c0b87029ba8cb0910cf22172`；训练记录commit为 `ab2e62af04840c2934795fd162ffacf788530dad`，记录为clean。

这里source item不是“互不重复的独立状态”。单项臂累计64000次source-item抽取，双项臂累计32000次；它们均来自31440条train记录。不能把相同model-example预算写成相同unique-state实际曝光。保存的逐样本训练抽样/flow-noise完整trace未在本轮核实。

训练定义入口：

- `/workspace/projects/alphabrain-dsol-paper1/configs/experiments/dsol_libero_broad_pairing.yaml`
- `/workspace/projects/alphabrain-dsol-paper1/configs/dsol_paper1/libero_training_budget_quick_gate_v1.json`
- `/workspace/projects/alphabrain-dsol-paper1/scripts/dsol_paper1/run_libero_pair_train.sh`
- `/workspace/projects/alphabrain-dsol-paper1/AlphaBrain/dataloader/paligemma_datasets.py:679`

数据loader确证：canonical_unique取canonical图；Broad practical取broad_a；state-matched为同state的broad_a重复两次；paired FM/consistency取broad_a、broad_b。因此这不是所有臂同时融合两张外部图，正常有效输入仍为一张external加一张wrist。

## 3. Broad64候选及匹配裁决

| 候选 | checkpoint绝对目录 | 已核实身份 |
|---|---|---|
| Broad64 quick | `/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_broad64-quick-gate-v1_seed41_g8_gb32_steps2000/final_model` | seed41、8GPU、2000 updates、schedule3000、GB32，模型/loader关键代码hash与旧七臂相同 |
| Broad64 M-B seed41，即v2主模型 | `/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_m-b-formal-v1_seed41_g2_gb32_steps2000/final_model` | seed41、2GPU、2000 updates、schedule2000、GB32；v2 manifest记权重SHA256为 `123786f53c9823b878fb08fe61ef25fe4931d4bcae8134e0a64940b7a8ac3cad` |
| Broad64 M-B seed42 | `/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_m-b-formal-v1_seed42_g2_gb32_steps2000/final_model` | 已有正式训练/评测资产；本轮未重新全文hash |
| Broad64 M-B seed43 | `/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_m-b-formal-v1_seed43_g2_gb32_steps2000/final_model` | 已有正式训练/评测资产；本轮未重新全文hash |

同父目录的 `run_manifest.json` 是训练身份依据。三个Broad64机制臂seed41也不是尚未训练：

- `/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_state_matched_broad64-parallel-formal-v1_seed41_g2_gb32_steps2000/final_model`
- `/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_paired_fm_broad64-parallel-formal-v1_seed41_g2_gb32_steps2000/final_model`
- `/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_paired_consistency_broad64-parallel-formal-v1_seed41_g2_gb32_steps2000/final_model`

三臂manifest均为2000 updates、schedule2000、2GPU、GB32、16source items/update；与practical的source曝光组织仍不同。paired FM对consistency比practical对consistency更接近loss-only问题，但还须核对全部保存配置和训练数据/随机流，不靠臂名直接发布因果结论。

### 3.1 代码hash变化的实际含义

旧七臂PaliGemmaPi代码hash为 `81600149a86303c04133379f7185075b2ff1cde5aaec5d86411866f7d4bf23d1`；M-B记录为 `76d2b7c0a031bd986d17f06ced6e7595c8dd47ab0245e6fac621f05037fc447f`。本轮已读本地git diff：`ab2e62a...` 到 `8dc6029...` 在该文件仅新增predict_action的可选flow-trace返回，未改训练forward。因此**此hash变化不能单独作为训练数学变化的证据**。整个训练依赖树的全面等价性没有据此得到认证。

schedule不匹配则是实质事实，不是命名差：两者实际都停在2000步，但LR轨迹不同，末步记录不同。8GPU/2GPU也改变执行拓扑；是否影响数据抽样序列、训练flow RNG和最终优化路径未做逐样本等价审计，不能因为同seed就假定bit-exact。

### 3.2 当前允许的资产决策

- **匹配优先**：Canonical-unique quick与Broad32 quick；保存config diff只有dsol_arm、run_id、output_dir。同源、同预算候选已成立，可用于新开发验证。Canonical quick与Broad64 quick也有同source身份、预算和关键代码的支持，但跨数据容器完整图像/标签bitwise比较尚未完成。
- **复用v2 O优先**：固定Broad64 M-B及其行为记录；旧canonical quick只标recipe比较，不宣称纯coverage效应。是否新增一个匹配canonical训练需单独决策和新release，本轮未执行。
- “Narrow”需冻结具体含义。上述canonical是单规范位姿，不是已训练的Narrow-8。Legacy8 catalog存在不等于当前同预算Narrow-8 checkpoint已经存在；本轮在该训练根目录未找到已完成的匹配Narrow-8资产。

## 4. 数据与源状态身份

| 资产 | 绝对路径 | 已核实内容 |
|---|---|---|
| Broad32 collection | `/share/longjunyu/alphabrain/datasets/dsol-libero-broad-pairs-v1/quick_gate_seed41_broad32_stride2/manifest.json` | 38193 records，train/val/test=31440/2701/4052；本轮manifest全文hash与训练记录一致 |
| Broad64 collection | `/share/longjunyu/alphabrain/datasets/dsol-libero-broad-pairs-v1/quick_gate_seed41_broad64_stride2/manifest.json` | 同样38193 records及split数；hash `e8fd2b606c03eb169db6b726b1895d7c8f0b0a0e6c54026d65caecbf6bf0bf94`与训练记录一致 |
| Broad32数据audit | `/share/longjunyu/alphabrain/datasets/dsol-libero-broad-pairs-v1/quick_gate_seed41_broad32_stride2/audit.json` | 已归档audit；本轮不重新渲染 |
| Broad64数据audit | `/share/longjunyu/alphabrain/datasets/dsol-libero-broad-pairs-v1/quick_gate_seed41_broad64_stride2/audit.json` | PASS；记录all_shard_hashes_match、无duplicate、无episode split leakage，抽样state/action/robot state正确 |
| 官方HDF5源根 | `/share/longjunyu/alphabrain/datasets/libero-original-hdf5-v1` | 八任务来源由下列collection plan及每shard manifest指定；不是新任务采集 |
| Broad64来源计划 | `/workspace/projects/alphabrain-dsol-paper1/configs/dsol_paper1/libero_pair_broad64_quick_gate_v1.json` | 八任务、stride2、horizon10、224、JPEG95、episode SHA split |
| 相机目录 | `/workspace/projects/alphabrain-dsol-paper1/configs/dsol_paper1/libero_view_catalog_v2.json` | 训练及诊断pose分组；新实验需冻结自己的允许动作域 |

本轮对八个shard的全部records.jsonl逐行比较 `(sample_id, episode_id, frame, split)`：Broad32/Broad64全部38193条身份和次序一致；八个shard manifest的source_hdf5、expected source SHA、stride、action_horizon、resolution均一致。这证实同源同frame/split，不等于本轮逐条解码pairs.bin验证了canonical/wrist JPEG、动作张量和物理state完全相同；已有各自audit中的抽样检查也不能替代全量跨collection比较。

具体源例：`/share/longjunyu/alphabrain/datasets/libero-original-hdf5-v1/libero_goal/open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5`。其source expected SHA记录为 `1b50d99a053a7e4547c9ae9ccd2e3cd762543d27be78f240f13440d7ae487ae8`，本轮未重hash全部HDF5。

## 5. 输入、动作及初始化契约

当前实际主checkpoint配置：

`/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_m-b-formal-v1_seed41_g2_gb32_steps2000/final_model/framework_config.yaml`

- `pi05=true`、`discrete_state_input=false`、dataset `include_state=false`。
- 三image slots，mask为true/true/false：有效external+wrist，第三槽不提供真实额外观察；224 RGB。
- 基策略有效输入为两路RGB和language。8维state被websocket接收，但没有进入当前base-policy的文本或连续state通道。
- action_dim7、prediction horizon10、flow inference10 steps。v2每次执行K5环境步，恢复后wait0。
- 冻结语言主干，视觉rank16低秩适配同时训练action expert；不是仅训练几百万参数的纯LoRA。

代码确证入口：

- `/workspace/projects/alphabrain-dsol-paper1/AlphaBrain/model/framework/PaliGemmaPi.py:1019`：仅discrete_state_input为true时将state编码进文本。
- `/workspace/projects/alphabrain-dsol-paper1/AlphaBrain/model/framework/PaliGemmaPi.py:1330`、`:1566`：训练/推理连续state只在not pi05时构造；当前state=None。
- `/workspace/projects/alphabrain-dsol-paper1/scripts/cabi_vla/serve_alphabrain_pi05_websocket.py:32`：接收与传递state，不等于基策略消费。
- `/workspace/projects/alphabrain-dsol-paper1/scripts/dsol_paper1/train_statewise_view_selector_v1.py:287`：两个ridge拼接canonical/wrist embedding与robot_states后PCA；selector另有本体权限。

初始化记录：`/share/longjunyu/alphabrain/pretrained_models/openpi/pi05_libero_pytorch/source_manifest.json`。当前新实验不得静默开启state、改token化/动作/控制器后继续沿用旧行为值；任何修订形成新的模型/协议身份，需要新哨兵和兼容性裁决。旧原生参考和AlphaBrain结果也不能因都叫Pi0.5就视为完全相同接口。

## 6. v2结果、O bank复用与数据边界

本轮再次汇总development/test/transfer实际阶段audit：22份均PASS_COMPLETE，episode_count总和242816。控制器最后完成时间2026-09-06T04:11:42Z；最终分析PASS_Q_ANALYZED，R=DO_NOT_ACTIVATE_R，reserve_complete=false意为未使用备用，不是未完成主实验。

| Q方法 | seed41 | seed42 transfer | seed43 transfer |
|---|---:|---:|---:|
| canonical | 59.08% | 55.86% | 58.30% |
| development global固定 | 59.18% | 58.01% | 58.30% |
| development task固定 | 60.45% | 57.52% | 57.91% |
| statewise O/P非canonical选择 | 62.99% | 60.84% | 61.72% |
| geometry/context ridge | 58.11% | 55.08% | 55.66% |
| image-queryable ridge | 59.57% | 57.42% | 59.47% |

seed41 state-search−canonical为+3.90625pp，原95%区间[0,7.8125]，未过原5pp及区间下界严格正的门。两个ridge门均未通过。原state-dependence自动门比较state与global，不足独立证明任务内状态价值；state−task的+2.54pp、区间[-0.68,5.76]属于后续事后补充。Q所确认为强制非canonical替换的有限搜索程序，不是含keep的真实Oracle，也不是主动动作价值上界。

| 复用对象 | 绝对入口 | 限制 |
|---|---|---|
| 最终Q分析 | `/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/final-analysis/analysis.json` | 已开封；不能给新方法充当未看过的确认 |
| 完成日志 | `/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/logs/v2-tail-controller.log` | 正式完成时间依据，不以旧动态状态文档替代 |
| population | `/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/population/population-v2.json` | 48dev+16test、source-disjoint，但64来源来自旧研究资产 |
| O原始ledger根 | `/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/development/O` | wave-00到07，每wave48×97×4=18624；选择/评价必须使用不同repeat子集 |
| O前4重复 | `/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/development/O/wave-00` | 每个episodes-shard-00至31.jsonl中记录repeat0–3；以protocol身份筛，不靠行号 |
| O后4重复 | `/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/development/O/wave-01` | repeat4–7；与wave00组成计划中前8条O |
| wave00运行身份 | `/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/development/O/wave-00/run_manifest.json` | 明确绑定Broad64 M-B seed41权重、K5、wait0、显式noise；不能换成quick checkpoint |
| wave00冻结协议 | `/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/protocols/dense-O-development-wave-00.json` | repeat_ids、候选、state、population hash的依据 |
| O噪声manifest | `/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/noise-banks/bank_O.manifest.json` | shape[64,32,104,10,7]，root seed2026090311；key按bank/state/repeat/replan |
| O噪声本体 | `/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/noise-banks/bank_O.npy` | 本轮重hash为 `2dcdbeb3f544fe4ee08b4d08ec2ca61dcdb05e0734a18c1f28000971ffbab9e6`，与manifest一致 |
| v2冻结科学配置 | `/workspace/projects/alphabrain-dsol-paper1/configs/dsol_paper1/statewise_view_oracle_v2.json` | 明确固定外部相机，排除路径成本/rollout内移动/本体耦合 |

S开发若采用16个F拟合来源+16个搜索来源，应从历史development中事先冻结不重叠名单并保留旧pair_key→新用途映射。Broad O复用要求同checkpoint、state/XML、pose、输入、动作/时限和noise；若P0改变接口或restore语义须重新裁决。新选出的candidate若没有旧P/Q记录，不得用O最大值填作确认。

以下不能再称新holdout：v2全部64来源及其O/P/Q；期望v1来源；已查看的M1/A97/constructed/42来源selector测试；已用于阈值、任务/pose规则选择的validation。新noise只能提供新的采样确认，不会把旧source变成新source。相同demo另一邻近帧也不是新的独立来源。seed42/43 transfer复用seed41候选，不是各自独立密搜。

## 7. 主动获取底座：可复用原语与尚无的执行能力

| 原语入口 | 可复用内容 | 必须补的边界 |
|---|---|---|
| `/workspace/projects/alphabrain-dsol-paper1/scripts/dsol_paper1/evaluate_dsol_libero_hdf5_views.py` | HDF5/XML恢复、hash、固定pose、closed-loop、noise和视频 | 当前只在开始前安装pose；calibration也只获取一次。新主动runner不能默默改旧runner行为 |
| `/workspace/projects/alphabrain-dsol-paper1/scripts/cabi_vla/libero_camera_pose.py` | install/set_camera_pose可即时写pos/quat并刷新observables | 是瞬移原语，不含路径、速度、耗时/碰撞；动态必须刷新标定 |
| `/workspace/projects/alphabrain-dsol-paper1/scripts/dsol_paper1/libero_constructed_view.py` | task-centric候选、镜像控制、无碰撞视觉遮挡 | world-body相机假设，不适用于直接操纵腕部/头部关节；像素不等于正确决策依据 |
| `/workspace/projects/alphabrain-dsol-paper1/scripts/fresh_vla/libero_snapshot_collector.py` | 控制器、插值器、runtime/observable缓存、robot buffer捕获恢复思路 | 有cream-cheese对象/摩擦等特定字段；只复用通用实现，重新验证新任务、action queue和历史 |
| `/workspace/projects/alphabrain-dsol-paper1/scripts/verify_vla/collect_probe_gate.py` | 4步机械臂probe执行、external/wrist序列、对象/EEF漂移和teacher viability记录骨架 | 旧抓附研究；lift/release改变物理状态，不是纯相机peek；未引入旧结果 |
| `/workspace/projects/alphabrain-dsol-paper1/scripts/cabi_vla/libero_wrist_camera.py` | EEF×hand-eye生成wrist外参 | 不提供IK、可达peek、路径或成本模型 |

本轮未找到已验证的时间化独立相机路径执行器、相机安全域/碰撞/速度控制、基于新图反馈的两次查询runner，或能合法利用peek-and-return证据的基策略历史接口。组件脚手架可独立建设；真实release前需持位漂移、模拟时间推进、总预算扣减、路径失败分母、未请求图像不泄漏、noise流隔离及同状态恢复哨兵。静态S无显著收益不作为A1工程或科学门的先验否决。

## 8. 运行时环境与当前进程

| 入口 | 已核实 / 未核实 |
|---|---|
| `/workspace/projects/alphabrain-dsol-paper1` | 当前研究代码根；其他同名AlphaBrain目录不得自动替换 |
| `/alphabrain/.venv/bin/python` | policy/training launcher默认解释器，路径存在；导入应由PYTHONPATH指向当前repo；本轮不import大模型、不执行推理 |
| `/workspace/envs/fresh-libero/bin/python` | 仿真解释器，路径存在 |
| `/share/longjunyu/alphabrain/datasets/libero-plus/runtime/LIBERO-plus` | 冻结仿真runtime，路径存在 |
| `/share/longjunyu/alphabrain/envs/libero-plus-runtime-config-v1` | LIBERO配置根，路径存在 |
| `/usr/bin/ffmpeg` | 视频工具路径存在 |
| `/workspace/projects/alphabrain-dsol-paper1/scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh` | 环境/解释器/日志契约入口；执行会写产物并调度GPU，不能把启动当只读检查 |

2026-09-07本轮ps与GPU进程核查：未见本项目train_alphabrain、policy server、rollout或v2 controller；8卡占用来自既有gpu_compute_keepalive，不是实验。本轮只检查进程，没有停止/重启keepalive。旧execution_status_20260903仍写wave00是历史快照，不能用来判断现在仍在跑。

## 9. 下一执行release前的最少待办

1. 选择T身份：匹配quick pair，还是以v2 Broad M-B为锚补必要匹配训练；在选择前维持HOLD，不因64000样本数相同忽略LR schedule。
2. 固定输入契约，不在评测独自开启state；明确selector权限与A2/W是否需要共同记忆。
3. 对选定旧checkpoint做只读内容/配置hash复核和输入/action/restore哨兵；若使用跨Broad32/64容器的canonical对照，补必要数据一致性检查，不重跑全部旧研究。
4. S开发冻结历史来源与O repeat切片；新增确认另建source/noise，所有旧test显式标历史。
5. A1先完成可审计路径/持位/时限/观察账本和两类正控制，再发布有预算上限的新实验协议；不能把当前静态set-pose接口当主动感知已完成。

本文可作资产入口和阻碍清单，不代替新的机器可读manifest、参数冻结或执行receipt。
