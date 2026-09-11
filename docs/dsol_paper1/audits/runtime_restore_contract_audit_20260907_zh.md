# P0：LIBERO runtime restore 契约审计（2026-09-07）

状态：只读代码审计 + 无场景的运行库符号探针；没有运行模型、创建仿真场景或产生研究结果。本文是下一步实现建议，不是 restore 已通过的证明。没有修改历史 evaluator、HDF5 或已有实验记录。追加纠正：首次参考了 `/projects/openpi` 的 LIBERO 源码，并非任务实际 LIBERO-plus runtime；现已对相关函数做实际 runtime 的只读 diff 核对，见第 2.1 节。

## 1. 结论与两种状态语义

现有 `scripts/fresh_vla/libero_snapshot_collector.py` 已有不少值得复用的字段清单，但不能直接作为通用分叉 API：它绑定单 Panda/OSC/cream-cheese，恢复时调用有副作用的 LIBERO observation regeneration，且没有应用层动作队列、随机流和完整模型/任务状态。

必须保留两个名字和两套入口：

| 模式 | 状态来源与承诺 | 不承诺什么 |
| --- | --- | --- |
| `archived_hdf5_policy_restart` | 原 DSOL 流程：reset、载入原 XML、恢复 HDF5 物理状态、按旧规则处理相机/wait、清空策略队列并从 call 0 开始评测 | 不恢复采集该 HDF5 时的控制器、原策略记忆或动作队列；不是原 rollout 的无缝续跑 |
| `live_runtime_fork_v1` | 从新 rollout 的明确控制边界捕获 simulator + controller + task + observations + runner/RNG，恢复后不干预地续跑应与未恢复分支一致 | 未通过绑定环境的哨兵前，不声称任意任务、版本、控制器、任意内部时刻均可精确恢复 |

两者都可以是合法实验，但 estimand 不同。不能为“修复”新主动实验而暗改旧 evaluator，也不能给旧 HDF5 补造未知队列/控制器字段并称其原始 runtime。旧物理状态也可作为新 rollout 的初始化条件；只有其后真实执行并捕获的 runtime 才属于后者，来源链必须写明。

## 2. 直接代码证据

- `libero_snapshot_collector.py:144–384`：捕获 `ctrl`、warmstart/applied forces、mocap、若干求解器缓存；`gripper.current_action`；OSC goals/reference/interpolator；env `cur_time/timestep/done`；observable 采样状态和 cache；robot recent buffers。
- 同文件 `:395`：friction 通过几何名包含 `cream_cheese_1` 定位，且所有 robot/controller 访问都固定 `[0]`。`model_body_pos` 有捕获，`body_quat`、camera pose 等没有。
- 同文件 `:322–384`：恢复经 `env.regenerate_obs_from_state()`，再 `controller.update(force=True)`，最后部分回填；dynamic constraint buffer 长度不符会跳过，返回的 skipped 列表没有被调用者记录。
- 实际 LIBERO-plus `env_wrapper.py:383–393`：`set_init_state`/`regenerate_obs_from_state` 内部会 `set_state → forward → check_success → _post_process → force update observables`。这些操作不能当成纯赋值；此函数块与首次参考的 `/projects/openpi` 版本相同。
- 实际 robosuite `utils/binding_utils.py:213–241,1140–1166`：`MjSimState.flatten()` 只含 `time/qpos/qvel`，其反序列化还要求 `na == 0`。并不包含 `ctrl`、激活状态、warmstart、env 的时间计数或策略状态。
- `evaluate_dsol_libero_hdf5_views.py:162–164,175–179,263–310`：HDF5 初始化；`action_plan=deque()`、`inference_calls=0`；仅空队列触发推理，入队 `chunk[:replan_steps]`，每次环境步弹出一个动作。显式 bank 索引是 `(pair_key, repeat_id, inference_calls)`。
- `view_acquisition_executor.py`：`PersistentOSCGoalHold.__init__` 自身会 `controller.update(force=True)`；hold 现在显式固定 gripper actuator `ctrl` 并保留 command，而不是将零 gripper action 误认为任意 runtime 的目标保持。相机 backend 会每步修改 `model.cam_pos/cam_quat`；endpoint `observe()` 会 force-update 全部 observables。

上面行号对应审计时工作树。本轮检查的 robosuite 源码位置：`/workspace/envs/fresh-libero/lib/python3.8/site-packages/robosuite`；实际任务指定的 LIBERO-plus runtime root 是 `/share/longjunyu/alphabrain/datasets/libero-plus/runtime/LIBERO-plus`。正式 receipt 仍需在真实执行进程中绑定实际导入的 `module.__file__` 及内容 hash，不能把指定路径、默认 import 探针或代码文件存在当成运行时 provenance 已认证。

### 2.1 实际 LIBERO-plus 的差异核对与补充限制

只读 `diff`/SHA-256 核对结果：

| 文件/结论 | 核对结果 |
| --- | --- |
| 首次参考 `/projects/openpi/third_party/libero/libero/libero/envs/env_wrapper.py` | SHA-256 `a782fb76c9792268d28979474fe72849e1e98ada49c8e997a65359a8d6b6acd0`；不是本任务实际 wrapper |
| 实际 `/share/longjunyu/alphabrain/datasets/libero-plus/runtime/LIBERO-plus/libero/libero/envs/env_wrapper.py` | SHA-256 `e91d7b7b35cc3ad2b073606c99860def6b3ac43b66eba00ef0ddb6bfd8f39c3c`；新增 view/initstate/noise 文件名解析、机器人变体参数与 reset/step 图像后处理 |
| 物理状态与恢复副作用 | 实际 wrapper 的 `get_sim_state/check_success/set_state/set_init_state/regenerate_obs_from_state` 与参考版相同；原结论成立 |
| base domain 的 step/success/counters/post_process/seed | 两个 `bddl_base_domain.py` 只差 `register_problem` 末尾新增 `return target_class`；实际 `:163,801–825` 的相关运行逻辑相同，原结论成立 |
| Python task object 状态 | 两个 runtime 的 `object_states/base_object_states.py` SHA 均为 `ca6f53123336b6487bdb4feb02534ddb3a0627a4b847fe89da4460520ef01366`；`objects/articulated_objects.py` 均为 `f5187b0381d69a2a8736c1dcb646e5fa7c138a42950f92e30c21c2efaed52faa`；FlatStove/`vis_site_names` 结论成立 |

必须补充的接口限制：LIBERO-plus wrapper 的 `self.noise != 0` 时，`step()`/`reset()` 会对 external RGB 做后处理，其中 motion/fog/glass blur 等会消耗 NumPy global RNG。可是 `regenerate_obs_from_state()` 和当前 acquisition backend 的 `observe()` 直接读底层 observables，不走这段 wrapper 后处理。因此它们在有 wrapper noise 时不是相同的策略观测通道；仅保存底层 `_obs_cache` 也不等于保存策略最后收到的图像。

首版 P0/A1 应将 **实际 `env.noise == 0`** 作为显式支持条件和运行时断言，并绑定 constructor 的 view/initstate/noise 解析后配置及机器人实际类型；不能因为 config 里没写噪声就默认为零。若未来允许 `noise != 0`，需要独立的 wrapper-observation adapter，统一起始/step/endpoint 的后处理语义、随机流和 query 计账，另加哨兵后才能放行。这是接口支持边界，不要求本轮扩展噪声研究。

`_configure_runtime()` 会在导入 LIBERO 前修改 `sys.path`，但不会自动卸载已加载的同名 Python modules；真实 provenance 检查必须发生在正确配置后的实际执行进程。第 4 节的默认 fresh-Python `import mujoco` 只证明该解释器当时可见的 MuJoCo 符号，不证明 LIBERO-plus/robosuite 在正式进程中的完整 import 来源。

## 3. 最小字段与边界

| 层 | `live_runtime_fork_v1` 必须保存/核对 | 现有实现可复用与缺口 |
| --- | --- | --- |
| 身份及配置 | schema、source/cluster/branch ID、源 checkpoint、XML/BDDL/assets/代码 hash、MuJoCo/robosuite/LIBERO-plus 版本及实际 module 路径；机器人/控制器/相机索引与尺寸；physics/control timestep、horizon、ignore_done、观测采样配置；Plus wrapper 的 noise/view/initstate 解析后配置 | 原 collector 没有通用 binding 验证；不能仅检查 qpos 长度；首版要求实际 `env.noise == 0` |
| MuJoCo 输入/积分状态 | `time,qpos,qvel,act,plugin_state,qacc_warmstart,ctrl,qfrc_applied,xfrc_applied,eq_active,mocap_pos,mocap_quat,userdata`，具体依版本 state spec | 原 flattened state 远不足；优先 native `mjSTATE_INTEGRATION`，不要把 packed `efc_*` 长度一致当图结构/索引语义相同 |
| 运行时 model | 至少 `body_pos/body_quat`、完整允许变化的 `geom_friction`、`cam_pos/cam_quat`、任务视觉状态对应的 `site_rgba`；其他字段冻结并验证无未登记变化 | 旧 collector 遗漏 body quaternion 和本实验会改的 camera pose。不必无限复制所有 model arrays，但未登记的可变字段不能静默忽略 |
| Robot/controller | `gripper.current_action` 与相关 actuator `ctrl` **分别**保存；OSC initial/goal/reference/interpolator；`new_update`；当前 controller/robot torque 和运行所依赖的 state/dynamics caches；recent Delta/RingBuffer 内容、指针、size | 已有目标/缓冲字段可提取；完整 cache 尚缺。若拟通过“下一步会重算”省略，必须限定控制边界并由哨兵证明，不能只恢复 flag 而不解释 cache |
| Env/task | `cur_time,timestep,done`；外层 success/termination/horizon；受任务函数更新的 Python leaves（如 `object_properties['vis_site_names']`）；任何 adapter 注册的 episode counters | LIBERO 的 `step()` 返回 done 可被 success 重写，但 `env.done` 仍有 horizon 语义，两者须分别记录。`FlatStove.turn_on/off()` 会改 Python object properties，`_post_process()` 不是无副作用查询 |
| Observations | `_obs_cache`、每 observable 的 timer/delay/current value/sampled；enabled/active/sampling rate 和 sensor/filter/corrupter/delayer 身份；最后一次已合法返回的 policy/selector 输入的副本 | collector 已覆盖前一部分。若 filter/delayer 自带记忆，必须有 adapter，否则拒绝该配置；`_get_observations(force_update=False)` 可读已有 cache，恢复不能强制新采样 |
| Runner | 未执行 action deque（顺序、dtype、shape）、其 chunk/hash、当前 chunk offset、下一 base-policy call index、已执行环境步数、剩余总 budget、phase、终止状态、最近动作及合法历史 | 旧 DSOL 清空队列是其 restart 语义；新 fork 不能再清空。不能只存 inference_calls 而漏掉队列 |
| Noise/RNG | bank ID/manifest+file hash/source key/repeat/next call index；Python random、NumPy global RandomState、所有显式 Generator 的 bit-generator state；本进程 torch CPU/CUDA RNG 或声明不加载 torch；selector/camera 各自命名流 | collector 不保存 RNG。`env.seed()` 实际是 `np.random.seed()`，只改 global stream；不恢复独立 `default_rng` 对象，也不恢复已经消耗到的位置 |
| Model/selector 记忆 | 当前模型明确 `history=current_frame`，不存在跨请求 KV continuation；selector history 单独保存。若未来引入 RTC、temporal ensemble、图像历史或递归状态，新增 adapter | `PaliGemmaPi.predict_action` 每次重新计算 prefix；flow head 的 KV cache 是单次去噪调用内部 cache，不能误当跨 rollout 记忆。当前 pi05 分支不使用 state 张量，不代表 selector 可悄悄继承 privileged state |

特别说明：固定阻抗 OSC 的 kp/kd、action scaling、插值器类型/总步数等可以视为绑定配置而不做每步可变数据；如果运行期间允许修改，必须升级为 runtime 字段。`control_timestep/model_timestep` 通常是配置，但需要验证比值对应实际每控制步的 simulator substeps；只恢复 `data.time` 而让 `timestep` 归零会重置终止预算。

## 4. 运行库探针及推荐恢复路径

本轮唯一执行性探针是用 `/workspace/envs/fresh-libero/bin/python` 按其默认 import 环境导入 `mujoco` 并检查符号；没有创建 `MjModel`/`MjData` 或场景，没有加载 VLA/GPU。该探针没有配置/导入实际 LIBERO-plus，不是实际实验进程 provenance 的证据：

```json
{"mujoco_version":"3.2.3","mjSTATE_INTEGRATION":8191,
 "python_api":{"mj_copyData":false,"mj_stateSize":true,"mj_getState":true,"mj_setState":true}}
```

`MjData` 类另有 `__copy__/__deepcopy__/__getstate__/__setstate__` 槽，但本轮没有验证其场景副本语义。C header 声明 `mj_copyData` 不等于 Python 可调用。不要直接 `deepcopy(env)`，不要先使用未经审计的 ctypes/ABI 绕过；替换 `sim.data._data` 可能使 robosuite 封装、已缓存 numpy views、renderer/controller 的引用仍指向旧对象。

推荐第一版为 **同进程、同一个已绑定环境、完整控制步边界上的 integration-state + 显式应用状态恢复**：

1. 新模块独立实现 `capture_runtime(env, runner_state, rng_registry)`、`restore_runtime(env, snapshot, ...)`、`audit_runtime(snapshot, env, ...)`；不 import 整个任务专用 collector 作为通用后端。用小型 adapter 复用其字段思路。
2. `capture` 不调用 reset/forward/controller.update/check_success/force observation，直接深拷贝已存在的数值/JSON leaves；不序列化函数、sim/env 指针、未审计 Python 对象。capture 前后验证 zero env/query/policy/RNG 消耗。
3. 最初只支持一台 Panda、fixed OSC、已审计无历史 filter 的相机/观测、实际 LIBERO-plus wrapper `noise == 0`、无其他任意 wrapper 的环境；没有登记的可变 task/controller 类型 fail-closed。拒绝在 `env.step` 的 physics substep、推理中途或 active hold override 内捕获。
4. 恢复前先核对 binding，原位恢复允许变化的 model fields；以 `mj_setState(..., mjSTATE_INTEGRATION)` 载入积分状态，显式同步派生量，再恢复会被同步改变的积分输入/必要 caches。具体调用顺序必须经下节实际接触哨兵验证，不能把 `mj_forward` 自动等同于完美还原。
5. 恢复 controller/robot/task leaves、env counters 和 observable caches，返回保存的合法输入副本；最后恢复 RNG，使恢复路径自身可能消耗的随机数不污染下一个真实事件。不要在最后追加“保险”性的 `regenerate_obs_from_state`、`check_success` 或强制渲染。
6. 恢复 runner queue/counters/ledger/history，与 snapshot 一并校验 hash。不得把 buffer/device/env 指针整体替换成别的环境中的引用。
7. receipt 明确标注 `restoration_mode=integration_state_plus_application_adapters`、适用配置、覆盖字段、未支持字段、瞬时数据差异和后缀差异。**不是全 MjData 字节同一承诺**。若接触后缀仍不一致，停止 formal 放行，再单独研究 native full-data copy 与 wrapper 绑定，而不是忽略动态 buffer 差异继续跑。

## 5. VLA 队列、噪声和相机插入点

首版主实验建议只允许 `phase=before_base_policy_call` 且 `len(action_queue)==0` 的采集/选择机会：这保留自然 replan cadence，避免额外引入“中断旧 chunk”机制。通用 snapshot 仍保存队列，非空队列哨兵用于检查恢复正确，但非空不代表 A1 自动获得中途清空权限。

- continue：在构造 `PersistentOSCGoalHold` 之前直接返回；不调用 controller update、不刷新图像、不前进任何 RNG/budget，然后按原 runner 继续。
- hold/move：从相同 snapshot 出发；固定原 OSC goal 和原 gripper actuator controls；真实前进相同 τ 个控制步；不执行/消耗 base-policy action queue，不申请 base flow noise；到达后仅做协议允许的 endpoint query。
- acquisition 后下一次 base-policy call 使用原 snapshot 的 `next_call_index`。noise bank 无隐式游标，恢复的是 key/index，而不是重播一次 global seed。取景/selector 调用不应改变 base-policy index。
- continue 与 acquire 的相同 call index 对应相同 flow 初始噪声；**不意味着相同物理时刻**。τ 的物理时间差正是 acquisition 成本，必须计入共同总 horizon；不得在 acquisition 后重新给满 horizon。
- 若未来允许非空 queue 插入，需事先定义保留队列还是丢弃重规划，且所有相关分支匹配 interruption/replan 规则；否则收益混入更频繁控制修正。
- 同一个 runtime 中尚未执行的源策略动作不能换个 checkpoint 后称为新策略自己的历史。跨训练臂的共同状态评估宜在空队列边界，并记录 source-generating policy 与各被评估策略，明确条件状态分布，不混称两者自然 on-policy continuation。

服务端 `serve_alphabrain_pi05_websocket.py:108–153` 每请求重置 torch CPU/CUDA seed，并接受显式 `_eval_noise`、返回 noise/action hash。请求不能在共享 global torch RNG 下并发交错；正式 harness 需绑定服务器实例/串行化策略、checkpoint、preprocess、action norm/remap、horizon、去噪 num_steps/时间网格及数值后端。当前 metadata 没有把全部 solver 配置回传，需后续补可审计身份，不必为此复制整个网络进 env snapshot。

当前 pi05 无跨请求模型记忆时，远程服务无需由 env 进程保存 torch RNG 全状态；但这一点必须通过相同请求跨其他请求后的重复输出哨兵验证，且所有随机预处理也必须确定。若将来模型有跨请求 cache/RNG，则另需服务端 capture/restore/session contract，不能只在客户端补 `torch.get_rng_state()`。

## 6. 实际 LIBERO 哨兵：最小放行集合

以下是建议执行设计，**本轮均未执行**。先不调用模型：用明确固定的动作脚本和可审计 fake-policy/noise 请求层覆盖控制与 runner。只在开发环境产生独立工程 receipt，不计算研究成功率、不进入 confirmation 来源。

| 哨兵 | 实际构造与对照 | 必须检查 |
| --- | --- | --- |
| 捕获自身无操作 | 一次完整 env.step 后，capture 前后比较；另连续 capture 两次 | physics integration、model/task/controller/cache/RNG/queue/counter 相同；无 env/query/policy 消耗 |
| 原轨迹 vs 恢复后缀 | fresh runtime 执行固定 prefix，在未接触、gripper closing/contact、grasp 或放置接触三个事先规定时点捕获；未恢复支路执行固定 suffix，然后恢复重放同 suffix，至少重复两次 | 每步动作/ctrl/goal/current_action、时间、qpos/qvel、接触概要、task/obs/cache/counters 一致；非只比较恢复瞬间 qpos hash。覆盖至少两个不同任务名，其中一个不含 cream-cheese；不预筛只留下容易相等的点 |
| 分支污染与逆序恢复 | 捕获 A 后执行会改变 camera/ctrl/goal/task 的分支 B，再恢复 A；与直接从 A 续跑对照 | body quaternion/camera pose/calibration、friction/visual task leaves、gripper command 与 ctrl 都回到 A；第一次重放与第二次重放同样正确 |
| 队列与噪声 | fake policy 每次产生有唯一标识的 chunk；在队列非空和空队列两种边界 snapshot；恢复后连续执行超过两次 replan | action 次序/offset、下一 bank key/hash、call 数、query 数、ledger 不丢不重；插入额外 camera/selector 随机 draw 不改变 base bank 序列 |
| 真正 no-acquisition | 从同一 live snapshot 比较完全直通 runner 与 continue 分支，不构造 hold；再执行相同完整 suffix | 不只 physics no-op：controller.new_update/cache、合法初始图像、全 RNG、队列及后缀也相同 |
| matched acquisition | 同一 snapshot 的两次 hold 和同 τ move；仅 endpoint 可见；覆盖原 ctrl 与 gripper command 不一定能互推的开发状态 | hold vs move 的机械轨迹一致、时钟/horizon/next call 匹配；camera/image 不同；分别报告 EEF/物体/手指漂移，机械漂移小是额外门，不由匹配等价自动证明 |
| 终止及未知绑定 | horizon 前一控制步、already-success、terminated ledger、错误XML/controller类型/observable配置/漏字段、restore 中抛错 | 不把 horizon 当成功、不越预算、不重置计数；拒绝未知类型/漏字段，不 silently fallback 到旧 HDF5 restore |

任务范围若含 stove 等会变化的 Python task state，需加对应 task-leaf 哨兵；否则第一版在支持声明中明确排除，不能一边宣称全 LIBERO 通用一边未测。底层 obs 噪声/延迟和 Plus wrapper 图像噪声是不同层：首版逐层断言关闭；若要允许，增加真实 corrupter/delayer 或 wrapper 后处理的重复采样、起始/step/endpoint 同通道哨兵。

数值门：离散字段、动作队列、hash、计数、bank 索引、RNG states 要精确相同；绑定同软件/硬件的 CPU MuJoCo 后缀先要求逐步 exact 并报告 max error。若非 bitwise，相应数值容差必须在独立开发探针上冻结，并证明误差不改接触/终止/任务谓词；不能看见评测结果后放宽。图像 exact 在相同 renderer 上检验，跨 renderer 不外推；相机变化导致的新图像差异只在指定干预分支出现。

待无模型哨兵通过后，再用真实 VLA 做小规模独立的请求重复/restore-noacq 后缀等价：至少跨一次 replan，验证 flow noise/hash、动作及 closed-loop 后缀。无模型工程通过不等于真实 VLA P0 已放行。

## 7. 当前停止线

现在仅能说已有 acquisition scaffolding 与 archived-development engineering smoke；后者故意使用 `current_pose` hold，不能认证完整 runtime fork。下一步是新 snapshot adapter + 上述最小哨兵，不是扩大候选或正式评测。

只有 binding、零操作、动作/噪声续接、接触后缀、相机/任务恢复、预算终止分别通过，才把相应 restore gate 从 unresolved 改为有 receipt 支持的 PASS。任一失败保留原始差异和失败分母，formal 继续拒绝；不要改旧结果的含义或将新旧数据无标记合并。
