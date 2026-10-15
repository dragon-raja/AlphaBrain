# LIBERO 零步起点验证：执行完成不等于原生基线等价

## 当前有效结论：正确参考640次已完成

正式修正版目录为`/share/longjunyu/alphabrain/experiments/dsol-libero-native-baseline-panel-20260923-independent-noise`，640条全部完成，配对来源及噪声审计通过。原生参考规范/四扰动为92.97%/21.88%；规范后训练为92.97%/15.43%；多视角后训练为89.06%/70.90%。这是真实完整任务重新评测，不是重新训练；旧0%及错误补噪98条均未合并。结果与研究解释统一见[差异归因审查第8节](../results/libero_robocasa_attribution_20260923_zh.md#8-libero-正确原生参考640次重新评测已完成)。

下文保留定位过程；其中明确作废的原生8/8不能替代这份640次结果。后续优先事项已从“补跑参考”变为“依据正确参考解释训练与跨环境差异”，没有自动启动新训练。

## 定位更新：反归一化干预已改变完整任务结果

后续噪声独立性审计修订：首次原生32维诊断补出的25维与原7维复用了同一PCG64随机流（两者使用不同浮点生成精度，仍共享随机比特）。6000个预定随机种子的跨坐标检查发现最高相关约0.579。因此，下文原生**8/8只保留为该错误补噪协议下的观察，不能作为合格原生基线成绩或验收依据**。AlphaBrain 7维的0/8→7/8反归一化干预不涉及补噪，仍成立。已停止随后启动的相关批次，保留其98条完成记录，不拼入修正版。新协议保留原7维、为额外25维使用SHA256域分离的独立PCG64子流，并增加回归检查；正式640次改按该协议重新评测。原生32维成绩以这次修正版完整面板为准。

在原8个任务中各取第一个官方初态、规范视角、原噪声索引0，保持零步权重、图像、任务指令、时限和闭环执行规则不变，仅把动作解码从本实验mean/std恢复为原检查点quantile：**旧接口0/8，修正解码后7/8**。这是完整任务闭环，不是首帧指标或事后Oracle。已逐条核查相同物理初态、首帧来源和共同执行前缀上的噪声地址/哈希。

这项小消融已证明动作解码失配确实影响成败；不再仅称其为“可能原因”。模型按预训练quantile坐标生成动作，直接换用mean/std解码会改变幅度及偏置。例如第三平移维在归一化输出为0时，quantile解码约−0.0001875，当前mean/std解码为−0.09225027。桥接权重并不会自动重新标定输出坐标。

同8个缓存观测上，修正解码后的AlphaBrain七维策略与原生OpenPI三十二维参考，最终7维动作块平均绝对差为0.001759、最大绝对差为0.009332。两者共享原7维噪声；原生另外25维由固定规则生成，**不得宣称完整噪声逐值一致或两种实现完全等价**。原生参考的8次闭环现已完成，结果为**8/8**；AlphaBrain修正后仍失败的spatial_drawer_bowl_plate在原生参考中成功。动作维数、精度、实现细节等残余差异不能据此忽略，也不能仅凭这个样本归因到其中某一项。

| 工程诊断条件 | 成功任务数 | 是否更新神经网络权重 |
| --- | ---: | --- |
| AlphaBrain零步、旧mean/std解码 | 0/8 | 否；复用原匹配结果 |
| 同一AlphaBrain零步权重、quantile解码 | 7/8 | 否；新增8次闭环 |
| 原始转换权重、原生OpenPI组件、内部32维及quantile解码 | 8/8 | 否；新增8次闭环 |

原生参考使用本地同源PyTorch转换权重及冻结OpenPI模型/数据变换组件；按OpenPI要求在独立目录安装Transformers 4.53.2并应用其官方补丁。不是JAX与PyTorch逐值等价认证。原生配置构造仅关闭编译，不改变模型、去噪步数、任务指令或任务时限；为避开不相关训练数据加载依赖，直接组合原生Policy与相同输入/输出变换。原有训练和评测环境未修改。

这些8次诊断**不能替代原32初态×5视角×4噪声的640次基线**，也不能把7/8写成正式基准成绩。旧640条0%结果保留为接口失配证据，不继续作为原生能力或后训练收益的基线。

已排除一个代码阅读误判：普通Transformers的PaliGemma带图像特征缩放，但OpenPI要求的`transformers_replace`取消该缩放，与当前AlphaBrain一致。因此本轮没有修改图像缩放，也不将它列为根因。

**已训练的规范/多视角模型不可直接切换成quantile解码。** 它们的训练目标使用mean/std，必须继续按自身训练坐标解码。后续应以恢复原生能力的参考评测判断：现有训练收益中有多少属于接口/动作坐标再适配，有多少属于视角覆盖，而非直接启动重训。

诊断目录：`/workspace/libero-initialization-diagnostic-20260923`，完整结果及24条新旧匹配记录核查见`analysis/summary.json`和`analysis/audit.json`。两版实际执行的策略源码及调度源码已留存，字节哈希与各自计划记录匹配。新代码位于`benchmarks/LIBERO/initialization_diagnostic.py`和`initialization_dispatch.py`，纯统计核验位于`AlphaBrain/research/dsol/analysis/libero_initialization_diagnostic.py`。历史检查点、冻结源码、原评测和PDF均未覆盖。

## 已完成结果与解释限制

2026-09-23，LIBERO 小面板640次完整任务完成，四个工作进程正常退出；自动核查完成，新旧合计1920条来源记录归档。RoboCasa 800次起点小面板也已完成。

LIBERO：8任务、原全部32初态、规范＋4个几何预选留出候选、原执行噪声索引0–3。

| 模型 | 规范成功率 | 四扰动平均成功率 |
| --- | ---: | ---: |
| AlphaBrain桥接重建的零步起点 | 0.00% | 0.00% |
| 规范后训练 | 92.97% | 15.43% |
| 多视角后训练 | 89.06% | 70.90% |

**0%是当前适配接口下的观测，不是官方原生π0.5的成功率，不能用于断言原生模型没有能力。** 该零步基线用于重建当前训练起点；没有证明桥接具有策略行为等价性。初态、首帧与噪声检查通过，不能替代这个语义验收。

## 已核实的接口差异

1. 原生OpenPI π0.5配置使用分位数动作归一化；原权重自带q01/q99。当前AlphaBrain训练和推理使用该实验配置的均值/标准差。零步桥接复制神经网络张量，不会自动把预训练动作坐标变换成新坐标。第一平移维的归一化输出为1时，原生反变换约0.937126，而本实验mean/std反变换为0.402663。这是已证实的行为差异，不是仅名称不同。
2. 原始转换权重内部动作维数32；AlphaBrain本实验为7。桥接切片了action_in_proj权重及action_out_proj权重/偏置。它不是原生策略先在32维内部去噪、最后仅截取7维输出的同一操作。
3. 本地原始转换权重config.json的action_horizon已经是10，与本实验相同；不能把“官方一定50步而这里10步”作为本轮已确认的根因。
4. 本轮加载与原训练日志一致：814/935匹配、121缺失、3投影适配、0形状错误、0未映射源张量。121缺失包括108视觉适配器张量、2个动作归一化buffer、11个SigLIP pooling-head张量。不能称为“13%预训练能力丢失”；当前图像token使用last_hidden_state而非pooling输出。

动作反归一化的闭环干预见本页更新；其余差异尚未通过逐项反事实消融确定贡献。也未证明不存在其它预处理、token、模型实现差异，不宣称所有桥接细节已经完全等价。

## 对研究结论的影响

- 当前LIBERO规范/多视角模型仍是在同一适配框架、数据预算和输入接口内的匹配对照。其观察到的扰动差异仍可报告，但结论限定该训练与实现。
- 不再将LIBERO与RoboCasa都描述成“在保留原生策略能力的官方模型上做少量视角微调”。RoboCasa使用原生JAX策略，本轮LIBERO零步接口未显示任务能力。
- 不能用LIBERO零步0%证明视角后训练从无到有获得了通用能力，也不能用它排除“原生模型本有鲁棒性，适配过程改变了它”的可能。
- 下一步优先做同观测下的原生/适配推理诊断与少量规范视角验收，建立保留原生动作坐标与维度的参考链。新的训练配置需在此基础上明确冻结/更新模块、任务范围和有效预算；不直接重启全量训练。

## 证据入口

- `/share/longjunyu/alphabrain/experiments/dsol-libero-baseline-panel-20260923/analysis/summary.json`
- `/share/longjunyu/alphabrain/experiments/dsol-libero-baseline-panel-20260923/analysis/audit.json`
- `/share/longjunyu/alphabrain/experiments/dsol-libero-baseline-panel-20260923/step-zero-receipt.json`
- `/share/longjunyu/alphabrain/pretrained_models/openpi/pi05_libero_pytorch/config.json`
- `/share/longjunyu/alphabrain/pretrained_models/openpi/pi05_libero_pytorch/assets/physical-intelligence/libero/norm_stats.json`
- 原冻结runtime的`openpi/training/config.py:create_base_config`与`openpi/transforms.py:Unnormalize`。
- `AlphaBrain/model/framework/PaliGemmaPi.py:predict_action`及`AlphaBrain/model/modules/action_model/pi0_flow_matching_head/weight_bridge.py:_adapt_action_projection`；正式执行使用原冻结快照，不修改历史源码。

本次新增隔离的小规模归因验证，不改变已完成评测、旧权重和PDF，不启动重训。
