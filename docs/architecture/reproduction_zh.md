# 安装、验证与复现

本页区分可安装代码、CPU 回归与已授权科学运行。安装不自动下载模型、不打开派发。

## 交付内容

wheel 包含 `AlphaBrain` 及 `benchmarks.LIBERO` / `benchmarks.Robocasa_tabletop`。
不包含运行目录、模型/数据、认证、历史归档、脚本树或其他 benchmark。
历史结果用原冻结源码和 manifest 复现，不依赖当前 wheel 的文件名相同。

源码 checkout：Python 3.10/3.12 基础 CPU 检查；搜索网络的实际本轮测试环境为 Python 3.12、Torch 2.6.0+cpu、NumPy 1.26.4、Gymnasium 0.29.1、SciPy 1.14.1、pytest 8.3.4。
这些是 CPU 工程证据，不是 GPU 科学环境锁文件。直接依赖版本保存在 `configs/dsol_paper1/environments/cpu-test-constraints.txt`；它不冒充完整传递依赖锁。

```bash
python -m venv .venv
.venv/bin/python -m pip install '.[test-portable]'
.venv/bin/python tools/repository/check.py
.venv/bin/python tools/paper1/check_architecture.py
.venv/bin/python tools/paper1/check_layout.py
.venv/bin/python tools/paper1/project.py check
.venv/bin/python tools/repository/test.py --profile portable
```

搜索核心测试另需 Torch/Gym：在独立 CPU 环境按平台安装 CPU Torch，再安装 `.[search-test]`。不要为了跑检查改动研究 worker 环境。精简 portable 环境的跳过项不算通过。

```bash
CUDA_VISIBLE_DEVICES='' WANDB_MODE=disabled python -m pytest -q \
  tests/dsol_paper1/selectors/test_unified_search_policy.py \
  tests/dsol_paper1/selectors/test_task_conditioned_policy.py \
  tests/dsol_paper1/runtime/test_gr1_trial_runtime.py \
  tests/dsol_paper1/runtime/test_gr1_block_runtime.py
```

## 干净安装检查

```bash
python -m pip install build
python -m build --wheel
python -m venv /tmp/alphabrain-wheel-check
/tmp/alphabrain-wheel-check/bin/python -m pip install dist/*.whl numpy
cd /tmp
/tmp/alphabrain-wheel-check/bin/python -I /absolute/checkout/tests/repository/installed_smoke.py
```

`-I` 和独立解释器禁止源码树/PYTHONPATH 掩盖缺包；该检查不导入 Torch/JAX/MuJoCo，不创建模型、仿真或 worker。完整 GPU 后端仍需按其冻结依赖恢复。

## 从配置到科学结果

1. 模型合同：权重、processor、normalization、特征读取位置/相机/时序、去噪/评分/动作前缀，全部显式且有哈希。
2. 搜索合同：同一核心结构配置、T、可选视图 mask、初始化来源、PPO 和奖励、查询预算。checkpoint 保存优化器、RNG 与在线尺度状态。
3. 科学清单：任务/初态/起点、训练/开发/测试划分、噪声身份、成功判据和时限。不能通过新机器绑定改变样本。
4. 发布：用 `execution.source_release.source_manifest` 收集代码，`isolated_identity` 验证新发布物单独可导入；另保存环境/数据/权重身份。
5. 资源绑定：节点实例、GPU UUID、解释器、可信保活身份及新 owner/epoch/期限。不能沿用旧 PID 或许可。
6. 默认关闭；新后端按授权预算完成数值、轨迹与容量验收，再由唯一 controller 派发。
7. 保存原始结果、调用账本、完整失败、进度、配置、代码和恢复回执；统计严格配对并披露覆盖。

当前生产 LIBERO 的入口及 GR1 CPU 组合见各 benchmark README。GR1 的真实生产注册/绑定与 GPU 准入仍待完成；不会通过这份安装说明自动开启。

## 发布前清单

- `git diff --cached` 逐项确认，`tools/repository/check.py --staged` 不含凭据和意外大文件。
- wheel 不携带资产；模型和 benchmark 的上游许可证单独遵守，代码仓库许可证不授予权重再分发权。
- 本次整理只更新开发树；在途发布物必须哈希不变。
- 公开结果只对应明确冻结的配置；旧结构、旧输入和旧前缀不能改标为新版。
