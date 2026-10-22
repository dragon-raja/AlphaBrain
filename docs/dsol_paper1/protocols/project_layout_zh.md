# Paper 1 项目管理

## 一个当前入口

当前任务、完成状态和结论限制只维护在 [项目首页](../README.md)。
带日期的旧计划和执行记录进入 `docs/dsol_paper1/history/`，正文保留，不作为运行授权。
实验参数以冻结清单为准；维护操作不修改任务、噪声、时限或成功判据。

## 源码职责

| 所有者 | 内容 |
| --- | --- |
| `AlphaBrain/research/dsol/acquisition` | 搜索环境、动态候选、连续相机接口 |
| `AlphaBrain/research/dsol/selectors` | PPO、共享 attention actor/critic |
| `AlphaBrain/research/dsol/data` / `analysis` | 数据身份、预算核查、训练流程与离线分析 |
| `AlphaBrain/research/dsol/execution` | 与具体模型无关的队列、进程身份、资源准入 |
| `benchmarks/LIBERO` | 真实图像查询、冻结 π0.5、仿真操作适配 |
| `configs/dsol_paper1` | 版本化设计和协议；本机资产路径用 `*.local.json` |
| `reports/paper1` / `tests/dsol_paper1` | 报告生成和相应测试 |

RL 属于 dsol，不另建一套外部 RL 源码项目。仿真适配保留在 benchmarks，是依赖分层，不是第二个项目。
共享 AlphaBrain 框架和其他模型入口属于上游框架依赖；本次没有删库、改包名或将它们搬进论文研究模块。

## 运行产物

源码根目录的本机 `artifacts` 链接指向持久化 Paper 1 产物根；真实路径写在被 Git 忽略的 `WORKSPACE.md`。

```text
artifacts/
  current/                 # 当前结果、下一轮设计及报告的少量明确入口
  runs/<run-id>/           # 今后的新实验；每次独立目录
    protocol/ inputs/ checkpoints/ logs/ results/ analysis/
    run.json              # 源码 commit、协议 SHA256、状态与职责
  evidence/<experiment>/  # 历史冻结实验的索引链接，原数据不复制
  reports/                # 已交付报告入口，不把旧 PDF 标成最新
  assets/                 # 私有机器人等外部资产，禁止上传
  archive/                # 结束队列、旧计划、制图中间物
  maintenance/            # 移动前后哈希、位置、删除依据、恢复说明
```

原冻结实验目录包含绝对路径、源码身份和结果哈希，保留原址；`evidence` 统一索引。
这类保留属于可追溯兼容边界，不是新实验继续散落的理由。
`/workspace` 根部过去的 `formal-*`、`rl-search-*` 是队列控制状态/句柄/日志；它们不是训练算法源码。
只有确认终态、无在用进程、锁可获得，且持久副本逐文件校验通过后才移入 archive；归档队列不得直接恢复运行。
新队列临时控制文件放源码根目录 `.runtime/queues/<run-id>/<host>`，持久结果放 `runs/<run-id>`。

创建目录及身份记录（不会启动任何服务或实验）：

```bash
python tools/paper1/project.py init-run --artifact-root /path/to/persistent/paper1 \
  --run-id example-reviewed-run --protocol /path/to/reviewed-protocol.json \
  --purpose "Reviewed experiment purpose"
```

创建结果为 `PREPARED_NOT_RUNNING`。科学授权、来源哈希、资源检查和独立准入仍由执行协议控制。
失败记录、负结果和缺失标签保留；不移动权重、Python 环境、共享动态库、SSH/代理或监控配置。

## 历史与清理

- 文档移动保留 before/after 哈希；只修复正文链接，旧科学文字不重新改写。
- 当前首页替换前的完整版本放 history/navigation；旧计划不与当前计划并列展示。
- 唯一研究数据不删除。只清理已核验重复的临时文件和可再生缓存；每次记录删除依据。
- 先备份并核对未提交源码；仍有用途的分析和测试应正式纳入版本控制，不能只存在于某台开发机。
- 私有 URDF、模型、原始数组和本机身份配置与可开源源码分离。
- 检查通过只说明本次维护边界成立，不宣称整个历史仓库已达到最终公开发布标准。

维护验证：`python tools/paper1/project.py check`、仓库/架构/布局检查和 portable 测试。
本机附加执行 `python tools/paper1/project.py check --artifact-root /path/to/persistent/paper1` 核对索引目标。
