# AA0 双机完整评测执行记录

**本矩阵已停止继续执行，保留为旧恢复快照的辅助实验。当前主线已改为官方标准初态整任务评测，见 [当前执行入口](standard_initial_aa0_execution_20260908_zh.md)。下述 397,312 是旧冻结计划，不是当前运行预算。** 两机中止审计为 `hosts/{fresh,gnu}/superseded-by-standard-initial.json`，不删除或改写原 release 和已完成数据。

用户授权：2026-09-08，停止论文/PDF讨论，接入 gnu 并重新启动修复协议评测。gnu 的保活不得杀死或暂停；本次不修改其脚本或进程参数，与评测共存。

## 唯一实验目录

`/share/longjunyu/alphabrain/experiments/dsol-aa0-full64-dualhost-v1-20260908`

共享目录包含本轮专用代码快照、依赖副本、冻结 release、按主机分离的状态与日志。gnu 原有 `/alphabrain`、`/projects/openpi`、`/workspace/ai2r` 内容未修改。新进程使用主机原生 Python 3.12 和本轮隔离的依赖路径；仿真使用共享的同一 Python 3.8 和依赖副本。原生 Python 存在小版本差异，因此跨机完整轨迹门槛不可跳过。

## 主机与资源

- fresh：本机；gnu：`root@192.168.242.120:183`，8 张 RTX 5090。
- gnu 原 NumPy/Transformers 与 fresh 不同；专用依赖覆盖后二者均为 NumPy 1.26.4、Transformers 4.57.0、Torch 2.11.0+cu128。仿真二者使用 MuJoCo 3.2.3、NumPy 1.22.4。
- 已核验两机 PyTorch 核心二进制相同：`libtorch_cuda.so` SHA256 `2aa4ad61474d40f1e385ba01de26695833b4de220124436aec104157051e860f`；`libtorch_cpu.so` SHA256 `0c560ca9edf091adda46c1c9dbc48ea5a869125e6ab51c97d4f41f898656e04b`。这不替代跨机轨迹实测。
- 当前采用每卡 2 个策略副本、每主机 32 个模拟 worker，保留保活显存空间。没有自动采用桥接时不含保活竞争的 48 并发速度。
- 新执行器只清理自己创建的子进程组，不调用旧 runner 的保活管理逻辑，不对保活发送信号。
- gnu 缺少 EGL 加载器：首次验收在仿真导入阶段停止，未产生有效闭环。首次运行记录保留为 `hosts/gnu-attempt1-missing-egl` 与 `gnu-controller-attempt1.log`。仅下载 Ubuntu Jammy 的 `libegl1_1.4.0-1_amd64.deb` 并解包到本任务 `runtime/egl-jammy`，未安装系统包；gnu 本次进程通过独立 `LD_LIBRARY_PATH` 加载，EGL 导入检查通过后显式重启验收。
- 后续补齐同样缺少的 MagickWand 及其运行库，均只解包到同一独立目录；第二次未产生闭环的启动记录保留为 `gnu-attempt2-missing-magick`。已在 gnu 实际完成状态索引 1 的全部 97 候选 RGB/可见性渲染，物理恢复哈希通过，再恢复闭环验收。
- 原契约中的 catalog 使用了本机非共享路径；在产生任何有效闭环前，将其改为共享代码快照内的相同字节文件，校验 SHA 不变。原 release 保留为 `release-initial-before-catalog-relocation.json`，新 release 重新冻结；失败的路径检查目录也保留。

## 执行顺序

1. 原安排为暂停旧 AA4 两个调度器并等待 wave-05 自然结束。随后确认该批次不属于 AA0 验收的科学前置条件，且阻塞本机验收，已取消等待并结束自有旧 AA4 进程组；已完成的旧记录和未完成批次全部保留，明确标记为中止，不能当作完整批次。退场审计为 `old-aa4-cancelled-for-aa0.json`。未对任何保活发送信号；旧 runner 自己的退出清理恢复了它此前管理的本机保活。新 AA0 控制器直接启动。
2. 两主机各做 192 次 AA0 验收：8 个事前选定状态 × 3 视角 × 2 噪声 × 2 模型 × 2 次重复。
3. 对同键四份运行（两机、两次）比较完整输入/动作/逐步物理签名链与成败。任何不一致均停止在门槛，不自动进入大矩阵。
4. 通过后各主机负责 32 个状态，两主机均评测两个模型。状态按固定顺序奇偶分片，每个任务在两机均有覆盖；不把一个模型绑定一台机器。
5. 相应 97 候选图、可见性和 8 初噪 Accel 重新生成。完整闭环为 2 × 64 × 97 × 32 = 397,312 次；不混入旧 AA4 数据。

## 状态与日志

- 冻结契约：`release.json`
- 旧评测退场：`old-aa4-retirement.json`、`old-aa4-cancelled-for-aa0.json`、`fresh-queue.log`
- gnu：`gnu-controller.log`、`hosts/gnu/status.json`
- fresh：`fresh-controller.log`、`hosts/fresh/status.json`
- 跨机验收：`hosts/{fresh,gnu}/gate-pass.json`
- 正式完成：`hosts/{fresh,gnu}/dense-complete.json`
- tmux：本机 `dsol-aa0-dualhost-fresh-now-v1`；gnu `dsol-aa0-dualhost-gnu-v1`

每条记录带 release 哈希、模型哈希、主机、状态/视角/噪声键及完整轨迹签名。验收与正式记录分目录；存在的不完整结果需要明确审查，不覆盖原记录。模型调用或模拟失败会使本机控制器退出并清理自有进程，不触碰另一主机或其他用户任务。

时间预算需以保活共存条件下的实际吞吐重估；先前 3.5–5 天是两机完整资源可用时的外推，不视作当前保证。本文仅记录执行，不修改论文/PDF。
