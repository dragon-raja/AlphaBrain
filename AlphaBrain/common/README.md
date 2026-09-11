# 核心共享原语

- `images.py`：图像 uint8 转换、等比例填充缩放、保持结构的 PIL 转换。
- `pair_records.py`：现有 DSOL 配对记录格式的读写、JPEG 编解码和内容哈希。

这些实现可被模型、数据加载、研究和部署共同调用，不导入任何上层模块。图像与记录格式行为由迁移前源码和字节级回归测试校验。

`deployment/model_server/tools/image_tools.py` 与旧配对记录入口只是兼容别名，不维护第二份实现。
