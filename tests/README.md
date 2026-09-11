# 测试入口

默认是 CPU-only、无仿真、无模型下载。不要通过遍历导入全部脚本来发现测试。

Paper 1 测试按数据、指标、选择器、统计、协议、执行、诊断、报告和架构分层，见 [模块导航](dsol_paper1/README.md)。根目录只保留共享配置与路径工具；新测试不能恢复平铺，也不能导入其他测试文件获取样例。

```bash
python tools/repository/check.py
python tools/paper1/check_architecture.py
python tools/paper1/check_layout.py
python tools/repository/test.py --profile portable
```

`portable` 使用 `pyproject.toml` 的 `test-portable` 依赖，覆盖边界、归档、纯数值方法、兼容性和安全 CLI。CI 使用此配置，不需要私有实验结果。

具备本项目完整依赖时运行：

```bash
python tools/repository/test.py --profile full
```

`full` 额外覆盖模型组件、数据加载、全部 Paper 1 与共享运行工具。只读私有数据的 golden 测试在缺少其固定数据时跳过。它仍不启动 VLA 训练或正式 rollout，也不能替代实际仿真轨迹、不同硬件和真机验收。

归档目录不参与默认测试。活动兼容工具对应的测试留在 `tests/cabi_vla`；其他旁支的原测试随源码归档，恢复原 checkout 后使用。
