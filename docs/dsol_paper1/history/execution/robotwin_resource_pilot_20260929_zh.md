# RoboTwin 官方资源工程试点（2026-09-29）

## 文件导入已完成

7项资源、10,932个文件、20,852,709,823字节（约20.85GB）已完成导入与逐文件校验。
包括两个官方原始任务包、Aloha本体、物体和支持资产、仿真源码、π0.5的29999步权重。
统一入口：`/share/personal/longjunyu/robotwin/registry/current.json`。
发布清单：`/share/personal/longjunyu/robotwin/registry/resource_release_20260929.json`。
文件级导入完整；本轮又完成两个任务各一条原生示范的回放与相机配对工程验证（见下）。
不代表50任务全部验收，也没有认证全部RGB、模型训练域或VLA闭环能力。

## 官方组成与我们实际具备的范围

官方RoboTwin 2.0不是单一训练压缩包：包含50个双臂任务代码、机器人/相机/随机化配置、
物体与本体资产、专家规划与数据采集流程、发布示范及策略接口。
参见[官方安装与资产](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)、
[官方采集流程](https://robotwin-platform.github.io/doc/usage/collect-data.html)及
[官方数据仓库](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main)。

| 组成 | 本项目已取得 | 未覆盖/不能混称的部分 |
| --- | --- | --- |
| 任务与仿真代码 | 50任务代码、配置、专家路径回放链路的实际源码快照 | 不是所有任务均已运行；带已记录的降噪器开关补丁 |
| 示范数据 | 官方旧包装 `aloha-agilex_clean_50`，2任务×50条 | 未下载全部50任务，也不是新版 `demo_clean` 全包 |
| 物体/本体/支持资产 | Aloha及所用物体、支持文件，文件校验通过 | 不含全部本体与随机背景库；当前clean试点未启用随机背景 |
| 策略权重 | 来源已核验的π0.5 joint/29999权重及归一化等文件 | 真实训练视角覆盖仍UNKNOWN；未加载验证推理接口 |
| 策略工具链 | 原生仿真与数据适配 | 未将整套XPolicyLab及其他策略子仓库复制进来 |

选择旧包装是因为这两个包保留 `_traj_data`，便于回放构造同状态视角对；新版采样包
在文件命名、图像编码与部分轨迹上存在差别，不能按相同episode编号混用。
当前是“工程试点所需资源齐备”，不是“完整官方benchmark已准备并验证完毕”。

## 已冻结的范围

此项不属于正式性能实验，不启动训练或全矩阵评测。只取得两个官方旧格式 clean
任务包，复用核验过的共享资产、仿真源码和权重，建立可追踪的后续回放入口。
两个任务按自由物体/关节物体类型选取，不根据方法成功率筛选。

资源根：`/share/personal/longjunyu/robotwin`。
原执行：`imports/official-pilot-20260929`；续接：`imports/official-pilot-20260929-r1`。
失败原因为无关XPolicyLab策略子目录包含软链接。保留失败状态，续接只排除无关策略
子仓库和旧评测输出，不放宽软链接安全检查，也不重复复制已验收资源。

## 已完成的 CPU 数据检查

| 任务 | 示范 | 保存帧数 | seed/场景/轨迹/动作图像长度 | 六关节机械臂命令数值范围 |
| --- | ---: | ---: | --- | --- |
| adjust_bottle | 50 | 7,238 | 通过 | 两臂均未超出对应URDF范围 |
| open_microwave | 50 | 24,383 | 通过 | 两臂均未超出对应URDF范围 |

两个压缩包均通过固定官方revision的完整SHA256核对；解包内容记录逐文件SHA256。
动作数组为14维，检查全部数组有限值；图像检查每路首末帧压缩格式头及整段记录长度，
没有宣称逐帧像素语义已通过。额外 front camera 保留原样，不自动作为策略第四路输入。
这一阶段未反序列化执行pickle、未启动SAPIEN。后续原生回放另用仅允许NumPy数组的
受限反序列化器，拒绝任意类/函数，不把两个episode的结果扩展到全部100条。

数据结构回执位于原执行目录 `structure-audit/`；关节命令筛查位于
`command-limit-screen/`，各自保存所用源码及输入哈希，不覆盖前一次回执。

## 跨数据集管理检查

`/share/personal/longjunyu/research_registry/snapshots/20260929/resource_catalog_audit.json`
是初始5入口快照。本轮最终索引包含7个主线资源/实验入口，索引44个历史实验目录。
33个历史目录没有根级标准清单，标记为
`LEGACY_METADATA_NOT_AT_ROOT`；这不等于目录内部没有记录，也不等于这些实验被重新验收。
不移动它们，不修改原始 release/plan，不复用已经迁移删除的旧共享根路径。

统一规则见 [资源管理](../../protocols/resource_management_zh.md)，来源判定见
[前置审计](../../audits/robotwin_resource_admission_20260928_zh.md)。

## 本轮原生回放及相机探针结果

验收入口：`/share/personal/longjunyu/robotwin/experiments/replay-admission-20260929/summary.json`。
依赖环境：`/workspace/envs/robotwin-replay-20260929`；可变运行副本：
`/workspace/projects/robotwin-replay-runtime`。运行副本拥有独立资产，不通过链接回写导入资产；
仅在副本中重建左右CuRobo路径配置。源码/配置哈希、环境完整freeze、命令和日志已归档。
使用物理GPU1，未停止其他保活进程，未修改RoboCasa依赖与运行协议。

| 任务/episode | 降噪器 | 完整任务成功 | 已保存关节状态帧数 | 对原HDF5关节状态最大差 | 渲染日志验收 |
| --- | --- | --- | ---: | ---: | --- |
| adjust_bottle / 0 | OIDN | 是 | 140×14 | 0 | **不通过**：unsupported CUDA / invalid handle |
| adjust_bottle / 0 | OptiX | 是 | 140×14 | 0 | 通过 |
| open_microwave / 0 | OptiX | 是 | 427×14 | 0 | 通过 |

OptiX两条分别完整消费5段和12段已记录路径。这里是专家示范重放，不是策略产生动作，
不能将两条成功写成VLA成功率。关节序列相同也不声称所有原始物体隐藏状态逐位恢复。

同一初态内依次执行“原位→原位重复→头部相机world-x平移3cm→恢复”：

- 可观测刚体/关节物理状态保持不变，不执行 `scene.step`。
- 左右腕部外参及RGB逐位不变。
- 头部图像确实改变；恢复后RGB与本轮原位逐位一致。
- 原包首帧与OptiX重渲染图并非逐位相同：三路RGB MAE约0.50–0.89/255，
  仅作诊断记录，**没有据此放宽像素验收阈值**。
- 原始压缩RGB经OpenCV直接解码的通道顺序与仿真RGB更接近；额外翻转R/B反而扩大差异。
  正式策略输入仍须沿官方转换链核对，不凭字段名额外交换通道。

OIDN失败记录没有覆盖。原始运行字段 `runtime_replay_verified` 实际仅检查物理路径与
关节状态，不包含C++渲染日志；独立postflight将该尝试明确标为 `NOT_ADMITTED`。
后续代码改名为 `physics_replay_verified`，并要求额外日志/图像验收。
资源总指针仍保持“文件级通过、非全量运行认证”；两个局部试点不能提升整库认证状态。

**相机探针是仿真相机位姿编辑，不是Aloha原生头部关节动作，不能当作可执行主动相机闭环。**
这次仅证明初态配对渲染的局部工程可行性，跨进程重复渲染、更多示范及策略输入仍需单独验证。

## 后续准入门槛

1. 已完成文件导入与资源汇总；后续仍不得把 `.partial` 当成完整资源。
2. 已建立独立环境及工作副本，保存依赖与源码身份；OptiX仅是本轮工程配置，正式协议另冻结。
3. 已适配旧包episode命名和记录路径回放，原始数据/源码快照不改。
4. 已完成两个episode的完整回放与初态同状态相机探针，不宣称全库通过。
5. 下一步先核对π0.5的官方加载、归一化、图像通道、状态/动作表示，做极小端到端策略验证；
   再按研究设计冻结正式任务/初态/视角，决定是否补齐官方其余任务，避免先下载后改研究定义。

不因资源齐备和两个示范成功就宣称模型可用或正式评测已开始。未启动新训练、全矩阵评测，
未改已有论文PDF。维护检查和资源/回放测试在本轮代码更新后重新执行。

最终检查：三项仓库静态检查通过，66项portable测试通过，28项资源/数据/回放相关测试通过；
独立环境147个包依赖兼容检查通过。GPU回放进程已退出并释放显存，原RoboCasa进程仍在。
最终索引验收：`/share/personal/longjunyu/research_registry/snapshots/20260929/after_robotwin_replay.json`，
7个入口与资源哈希链通过；旧共享索引按内容哈希保存，不覆盖历史证据。
