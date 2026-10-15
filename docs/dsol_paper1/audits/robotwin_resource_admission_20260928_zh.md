# RoboTwin 2.0 资源准入审计（2026-09-28）

## 结论与执行边界

当前共享盘资源不足以直接启动“官方 clean 示范的配对重渲染”主线。官方来源的权重和多数资产可作为复用候选；官方原始 clean 示范在本轮检查的数据目录中未找到；75 GiB 的 Fast-WAM 加工版不进入正式数据链路。不得因官方托管权重就宣称它是 clean-only 或未经相机增强的模型。

本轮仅做文件、代码、公开元数据和压缩包目录审计，没有启动 GPU、训练、仿真或完整数据包下载，也未恢复复制。上轮导入已取消：唯一复制的权重目录已删除，资产和示范正文未复制；原共享文件不变。撤销记录：`/share/personal/longjunyu/robotwin/imports/share-20260928/CANCELLED.json`。

## 证据版本与范围

- 官方 HF 仓库：`TianxingChen/RoboTwin2.0`。
- 本轮固定 revision：`3dc3b798668feb99ac61cc9086d84cbcc3d79186`。
- 共享资源下载收据 revision：`785feb15aa4a4f532395ad2b1d2be5f28cb561ad`。
- 共享代码快照：`/share/bizhihai/robot-eval-code/20260913T081701Z-b8f2e902/project`；RoboTwin 锁定提交 `96c1feab536306b50c26af200044fcdf126e8904`。
- 原始审计元数据：`/workspace/research-reviews/robotwin-resource-audit-20260928/`。包括完整官方文件清单、四个示范 ZIP 的索引与 seed、小型本体配置及版本比较。
- 已查主要数据目录包括 bizhihai、zhaoguoyang、jiangchenyang、cjl、sunguoying、psa/innerflow、jiawei、zhulingbo 等的可见数据路径；部分广域目录扫描有深度/时间上限。因此“未找到”不是对整个共享盘不存在该文件的证明。

## 共享资源与官方对应

| 资源 | 共享位置 | 审计结果 | 准入意见 |
| --- | --- | --- | --- |
| π0.5 checkpoint | `/share/bizhihai/robot-eval-data/checkpoints/robotwin-pi05/RoboTwin-lerobot_v30-aloha_agilex-joint-0/29999` | 原下载收据的大小与 SHA256 对上当前官方权重包；约 12.44 GB。提取目录有参数、归一化和 checkpoint 元数据 | 不必仅为取得同包再次下载；复制前完整性验收，训练域身份仍待确认 |
| 物体资产 | `/share/bizhihai/robot-eval-data/robotwin-assets/objects` | 原包 SHA256 与当前官方相同 | 优先本地复用；解包文件完整性不能只用原包收据代替 |
| 背景纹理 | `/share/bizhihai/robot-eval-data/robotwin-assets/background_texture` | 原包 SHA256 与当前官方相同 | 不需因版本重新下载；clean 小验证只取必需资源 |
| Aloha 本体 | `/share/bizhihai/robot-eval-data/robotwin-assets/embodiments/aloha-agilex` | 82 个官方条目的文件大小和 CRC32 全部与当前官方 ZIP 相同；约 780 MB 解包字节 | 可优先复用，运行路径和控制器仍需验收 |
| 加工示范 | `/share/zhaoguoyang/data/robotwin2.0/robotwin2.0` | Fast-WAM LeRobot v2.1 加工版；27,500 episodes、三路视频；不是官方原始 clean 包 | 不复制到正式数据链路，不作为重渲染源 |
| 官方原始 clean | 本轮已查位置未发现 | 待取得 | 需要补官方小样本包或找到准确本地副本 |

新旧 `embodiments.zip` 的整包哈希不同，但 ZIP 条目的大小/CRC 差异仅出现在 ARX-X5、Franka、Piper、UR5 的 `curobo_tmp.yml`，未涉及 Aloha 条目。Aloha 的 `config.yml` 已读取官方新旧两份小文件，SHA256 相同且与本地字节一致。CRC 核验用于完整性检查，不作为抗恶意篡改的密码学证明。

物体/纹理/权重仅完成“官方包身份对上历史收据”，没有本轮重新扫描全部解包数据，也没有完成当前机器的加载/闭环验收。

## 官方 clean 不是单一包装

官方固定 revision 的文件元数据汇总（十进制 GB，压缩包大小）：

| 包类别 | 任务包数 | 总大小 |
| --- | ---: | ---: |
| `dataset/<task>/demo_clean.zip` | 50 | 23.376 GB |
| `dataset/<task>/aloha-agilex_clean_50.zip` | 50 | 23.781 GB |
| `dataset/<task>/aloha-agilex_randomized_500.zip` | 50 | 391.196 GB |
| `dataset/<task>/arx-x5_clean_50.zip` | 50 | 15.230 GB |

这不能用于否认“约 4 GB 的 clean 数据”存在；较小文件可能是其他转换、分辨率、字段和编码版本，具体需要对应链接。不能把文件名含 clean 的不同发布物视为同一实验身份。

通过 HTTP Range 读取目录和 seed，未下载 HDF5/视频正文，抽查 `adjust_bottle` 与 `open_microwave`：

| 内容 | 新版 `demo_clean.zip` | 旧格式 `aloha-agilex_clean_50.zip` |
| --- | --- | --- |
| HDF5 示范 | 每任务 50 个 | 每任务 50 个 |
| `seed.txt` | 有，50 个 seed | 有，50 个 seed |
| `scene_info.json` | 有 | 有 |
| `_traj_data/episode*.pkl` | 两个抽查包都没有 | 两个抽查包都有，每任务 50 个 |

`adjust_bottle` 新旧 seed 在 episode 索引 16 不同；`open_microwave` 的 50 个 seed 一致。不可仅按 episode 序号跨包配对。seed 一致也不能保证轨迹、物理状态、语言及图像完全一致。

新版官方发布说明记录了 RGB 编码修正和部分越界关节示范的替换。旧格式虽然更接近重放所需资源，也不能未经关节范围与重放验收就直接用于主线。

## 权重是否有相机位姿多视角训练

目前未查明实际 checkpoint 的训练 manifest：

- 官方当前 `demo_clean.yml` 和 `demo_randomized.yml` 都是 `random_head_camera_dis: 0`。
- 相机实现中该参数是随机平移半径，朝向来自本体配置；它不是完整 SE(3) 采样，也不是可执行的主动头部控制。
- clean 同时输入外部与左右腕相机，不等于对外部相机安装位姿做增强；腕相机随手臂运动也应单独说明。
- randomized 中桌高、背景、光照和杂物变化仍可能改变观察几何/分布，不能等同于“视角完全没有变化”。
- 共享适配器引用旧官方 `pi05_aloha_full_base` 模板。该模板在提交 `ac5859dbc1e0059c4899ffce6627123351c81a3f` 中为 `repo_id=your_repo_id`、默认 20,000 步；当前共享 XPolicyLab 的默认模板又以 RoboDojo 数据为默认项。二者都不能证明发布的 `29999` checkpoint 的实际数据范围和启动覆盖参数。
- checkpoint 自带 `RoboTwin_lerobot_v30` 归一化，但 `_CHECKPOINT_METADATA` 的 `custom_metadata` 为空，未找到实际相机增强配置或训练数据清单。

因此可将其暂定为“官方发布的 benchmark-adapted checkpoint，训练域细节未核实”，不能作为已证实的 clean-only 对照。通用 π0.5 base 的更早预训练视角覆盖也不能由这里的配置排除。

## 重渲染的实际缺口

官方采集代码使用 seed 初始化、`_traj_data` 内左右关节规划路径、任务脚本回放，再保存图像/状态/动作。因此保留 `_traj_data` 的原始包优于视频型转换数据，但它仍不等于完整仿真状态快照：机器人关节路径不保存全部物体速度、接触求解状态或关节物体状态。

建议先验收两个不同物理类型的任务（工程试点，不是正式研究任务筛选）：自由物体 `adjust_bottle` 与关节物体 `open_microwave`。旧格式官方原包合计 **1,281,322,944 字节（约 1.28 GB）**。新版对应包合计约 1.267 GB，可在必须比对图像/动作时另取，不先把两套全量都下载。

验收顺序：

1. 锁定代码、资产、本体、数据 revision 和 seed；不得混用新旧 episode 序号。
2. 原相机下重放，检查机器人/物体状态、动作时间对齐、关节合法性、任务成功与图像解码。
3. 同一物理时刻，仅调整相机进行多次渲染；场景和控制器不步进，腕部安装保持不变，记录相机内外参及状态摘要。避免相机随机数调用污染任务初始化 RNG。
4. 若旧示范不能可靠重放，不将其直接宣称为原数据增强；改为经确认后使用官方任务生成器重新采集，明确标记“按官方任务重生成”，同步存储足够的状态/轨迹及多视角配对。
5. 按原始 episode/初态先划分训练与评测，再扩充视角。后训练与否另行决策；本轮不启动。

## 下一步资源建议（尚未执行）

- 不需要重新下载全部约 12 GB 权重及约 22 GiB 资产；现有共享副本有明确官方来源，择需复制并验收即可。
- 需要补的是官方 clean 原始示范。优先上述两个原包验证重放，不先下载 50 任务、randomized 大包或 Fast-WAM 转换包。
- 权重训练域的缺口需要真实训练配置/清单，而不是重新下载相同权重。若公开资料仍不足，可经用户同意向发布者询问；本轮未发送外部消息。
- 通过小样本重放与权重身份核验后，向用户确认正式任务数、视角空间和是否增加训练对照，再恢复资源拷贝及后续实验。

## 官方参考

- [官方数据目录](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset)
- [clean 更新说明](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/discussions/94)
- [官方采集与回放流程](https://robotwin-platform.github.io/doc/usage/collect-data.html)
- [官方相机实现](https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/main/envs/camera/camera.py)
- [clean 配置](https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/main/env_cfg/task_config/demo_clean.yml)
- [randomized 配置](https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/main/env_cfg/task_config/demo_randomized.yml)
- [权重发布记录](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/discussions/91)
- [旧 π0.5 训练模板](https://github.com/RoboTwin-Platform/RoboTwin/blob/ac5859dbc1e0059c4899ffce6627123351c81a3f/policy/pi05/src/openpi/training/config.py)
