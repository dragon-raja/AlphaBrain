# 开发与仓库维护规范

本分支以 Paper 1 的 VLA 视角利用研究为主线。规范适用于人和自动化开发代理；历史冻结协议不因本规范改变。

全仓库层次、依赖方向和兼容边界统一见 [架构说明](docs/architecture/README.md)，测试环境与覆盖范围见 [测试说明](tests/README.md)。模块化不以文件数量为目标，而以职责、依赖、单一实现和可验证接口为目标。

## 1. 代码放在哪里

| 内容 | 目录 | 不应承担的职责 |
| --- | --- | --- |
| 框架模型、训练与数据加载 | `AlphaBrain/model`、`training`、`dataloader` | 某次论文结果的硬编码 |
| 核心共享工具 | `AlphaBrain/common` | 反向依赖训练、研究或部署 |
| 论文数据结构与核查 | `AlphaBrain/research/dsol/data` | 训练、画图、发布 PDF |
| 指标定义 | `AlphaBrain/research/dsol/metrics` | 查测试标签后选择规则 |
| 特征与选择器 | `AlphaBrain/research/dsol/selectors` | 读写某台机器的固定实验目录 |
| 统计与研究流程 | `AlphaBrain/research/dsol/analysis` | 依赖报告层、启动仿真 |
| 可复用科学绘图 | `AlphaBrain/research/dsol/plotting` | 修改数据、训练模型、发布报告 |
| 命令行入口 | `scripts/dsol_paper1` 的职责子目录 | 新增一整套算法实现或恢复平铺区 |
| 获取运动与输入工程原语 | `AlphaBrain/research/dsol/acquisition` | 冒充已验证的主动感知方法 |
| 报告布局、说明与发布 | `reports/paper1` | 重写统计估计或偷偷重新评测 |
| 实验参数与路径 | `configs/dsol_paper1` | 替代已冻结的 release |
| 测试 | `tests` | 依赖作者个人 GPU 或隐式启动正式任务 |
| 退出主线的研究 | `archive/research` | 被新主线 import |

使用现有 `AlphaBrain` 包，不为了追求目录形式整体迁移成新的 `src/` 包。Paper 1 脚本全部按职责进入子目录，根目录禁止新增 Python／Shell 和批量兼容壳。纯指标和噪声实现进入研究包，仿真环境适配在 `scripts/dsol_paper1/runtime/`。测试同样按职责分层，共享样例在 helpers，不能从另一测试模块 import。历史源码哈希对应冻结 checkout，不能给迁移后的实现冒用旧哈希。

## 2. 新功能的最小流程

1. 先确认研究问题与变更范围。新增任务、初始化、遮挡、噪声预算、训练条件、指标定义或成功判据，需要用户明确确认。
2. 检查现有模块可否复用。不要复制 `*_v2.py`、`after_*`、`retry_*` 来替代接口设计。
3. 算法和统计写入可导入模块；CLI 只处理参数、加载配置和组合模块。
4. 为变化增加对应测试。科学行为变更与纯重构分开，不混在同一次验收里。
5. 运行维护检查和适用测试，记录结果、输入范围、版本与来源。
6. 更新唯一研究入口和当前报告指针，不新增相互冲突的“最终版”。

探索代码允许短期存在，但必须标记目的、输入、退出条件和归档位置。一次性运维操作记录放在相应实验目录或 `archive/paper1/maintenance`，不要新增默认主线入口。

## 3. 依赖方向与参数管理

- `data / metrics / selectors / analysis` 不 import `scripts`、`reports` 或 `archive`。
- `plotting` 使用明确传入的数组，不从当前目录猜测实验数据。
- `reports` 消费经过核查的统计产物；排版变化不得改变统计总体。
- 可复用实现不硬编码 `/share`、`/workspace`、个人缓存或机器地址。实际路径位于配置，CLI 允许显式指定。
- IO、数据验证、数值算法分开。导入核心模块不得启动服务、训练、下载或评测。
- 评测执行、GPU 调度与纯 CPU 分析分别配置环境，不能因画图依赖升级改变仿真环境。

## 4. 实验与重构的验收

科学协议包括任务、初态、相机、遮挡、模型、噪声、去噪网格、时限、数据划分和统计权重。这些不是仓库整理时可自由改变的默认值。

重构至少验证：

- 候选顺序、选择结果、标签配对、数据划分与输入身份不变。
- 数值输出对照迁移前基准；浮点容差显式给出，不把大偏差当作“只是重构”。
- 测试标签不参与 PCA、归一化、超参数或规则选择。
- 已发布 PDF 与冻结结果保持字节不变；需要重建时单独输出并核对，不覆盖原证据。
- 修改实际运行链时，CPU 测试不能替代首帧和完整轨迹一致性验收。

## 5. 归档而不是按名字删除

归档前检查 import、Shell、配置、报告来源和运行中进程。记录原路径、新路径、内容 SHA256、保留原因与恢复办法。

`invalidated`、失败日志、旧协议不等于垃圾。原始结果、权重、冻结源码和 release 留在原位置。旁支代码按依赖组归档；若仍有公共消费者，先抽出共享实现，再留必要兼容入口。

历史代码保留原字节，可能依赖原布局；重放使用独立恢复 checkout，而不是直接在归档目录启动旧脚本。新代码禁止反向依赖归档。

## 6. 日常检查

```bash
python tools/repository/check.py
python tools/paper1/check_architecture.py
python tools/paper1/check_layout.py
python tools/repository/test.py --profile portable
python tools/repository/test.py --profile full
```

前两项为只读静态检查。完整 CPU 测试部分依赖本地历史资产；公共 CI 只运行不依赖私有数据的核心与架构测试。正式任务不属于默认测试流程。

新文件不得无说明地加入旧脚本平铺区；维护检查以登记的兼容入口为基线，并禁止核心实现依赖归档或报告。维护规范本身也需要代码检查约束，而不是只靠口头记忆。

上传前只 stage 已审查的改动，运行 `python tools/repository/check.py --staged`。该检查不会打印匹配到的凭据值，也不能替代人工确认仓库可见性、私有研究规则和大文件来源。不使用 `git add -f .`，不擅自更改上游仓库或默认分支。

## 7. 参考与许可证

本仓库参考 [PyPA 包布局讨论](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)、[pytest 集成实践](https://docs.pytest.org/en/stable/explanation/goodpractices.html) 与 [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) 关于可导入实现、测试以及数据／模型／报告分离的建议，按现有框架渐进应用，不原样套用模板。

这些是工程组织参考，不是更换项目开源许可证。本轮不修改 `LICENSE`，保留上游版权和已有依赖许可。
