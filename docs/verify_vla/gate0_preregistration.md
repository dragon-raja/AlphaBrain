# VERIFY-VLA Gate 0 预注册：安全物理验证动作是否存在

状态：结果前冻结（实现澄清已于首次正式运行前写入）

冻结时间：2026-07-18 UTC

## 1. 目标

Gate 0 不训练 VLA，也不主张方法有效。它只回答：

> 对视觉近似相同但接触结果不同的 matched states，是否存在一个短 probe，使 held-out outcome 更可辨，
> 同时保留两个 outcome 的任务可恢复性？

若答案是否定，停止 VERIFY-VLA，不增加网络、loss 或数据规模。

## 2. 数据边界

- 基础数据：`/share/longjunyu/fresh-vla/libero-full-episode-v2-128`；
- 只使用 `train` 和 `val` groups；不访问 `test` 或任何 confirmation 路径；
- `libero-grasp-slip-0000` 已用于 0.25--15 mm offset 的开发校准，只能留在 train，不进入 held-out 裁决；
- 统计独立单位为 snapshot group，不能把两个 outcome、probe frames 或 simulator steps 当独立样本。

## 3. Latent-contact twin 构造

从每组 attached episode 的 `prefix_steps` observation 恢复真实 runtime state。该时刻夹爪已经闭合，原始
状态必须满足 `object_grasped=True`。

- `attached`：不改变 object pose；
- `detached`：沿该组预先记录的 slip XY direction，以 0.25 mm 网格寻找最小位移，使 simulator grasp
  predicate 变为 false；最大 3 mm；
- 两分支 robot state 必须一致；初始双视角平均 pixel MAE 必须不超过 raw RGB 2.0（即归一化
  `2.0/255`）；超过则该组标记无效，
  不能放宽阈值重试。

实现澄清：若有效 train 少于 80 组、有效 val 少于 10 组，或任一 split 有效率低于 75%，则判为
`GATE0_INVALID`，不能用少量易构造 twin 裁决方法。

这是一项 simulator intervention。privileged grasp predicate 只用于构造 outcome label，不作为 classifier
或未来 policy 输入。

## 4. 固定 probes

每个 probe 固定 4 个 control steps，夹爪保持闭合，除 `release` 负对照：

| Probe | 每步 normalized action | 角色 |
|---|---|---|
| `hold_closed` | `[0,0,0,0,0,0,+1]` | 额外帧/自然动力学基线 |
| `micro_lift` | `[0,0,+0.25,0,0,0,+1]` | 主验证候选 |
| `full_lift` | `[0,0,+1,0,0,0,+1]` | 高进展/较高风险候选 |
| `micro_lateral` | `[+0.25,0,0,0,0,0,+1]` 与反向交替 | 非特异小运动对照 |
| `release` | `[0,0,0,0,0,0,-1]` | 高信息但破坏任务的负对照 |

不根据 val 结果调幅度、方向或步数。

## 5. 可辨识性评测

- 特征：agent + wrist 的 16x16 block-average RGB，加 8D robot state；
- classifier：固定 ridge regression，regularization=1.0；
- fit：全部有效 train groups；
- test：全部有效 val groups；
- 报告 sample accuracy、attached/detached accuracy、paired ranking accuracy、group bootstrap 95% CI；
- 同时报告 pre-probe 和每个 probe endpoint；不在 val 上选正则或特征。

## 6. 物理可恢复性

从每个 val probe endpoint 启动同一 `FullEpisodeTeacher`，最大 320 actions：

- attached completion success；
- detached final recovery success；
- completion steps；
- probe 后是否 grasped、object displacement、drop/不可逆失败；
- teacher 是 privileged viability upper bound，不是部署方法。

另运行一个不执行 action 的 `no_probe` viability-only control，用于判断 twin 本身是否可恢复，以及计算
probe 新增的不可逆失败。它不参与 probe 候选选择，也不改变固定 classifier 条件。若 `no_probe` 的 attached
成功率低于 90% 或 detached recovery 低于 80%，则 teacher/twin 基线无效，裁决为 `GATE0_INVALID`。

## 7. Gate 通过条件

至少一个非 `release` probe 同时满足：

1. pre-probe val accuracy <= 65%；
2. post-probe val accuracy >= 85%；
3. 相对 `hold_closed` 提升 >= 20 pp；
4. paired ranking accuracy >= 85%；
5. attached teacher completion >= 90%；
6. detached teacher recovery >= 80%；
7. 相对无 probe 的 object displacement 中位数 <= 3 cm，且没有新增不可逆失败；
8. 结果不是只由 robot state 区分；vision-only 必须同方向通过。

控制性要求：原始大 slip 的 feedback frame 已知可见。额外 probe 在该 visible-control 上不应被解释为必要；
报告中必须明确区分“latent-contact Gate 0”与既有 visible-feedback task。

## 8. 裁决

- 通过：`PROCEED_TO_LEARNED_DVOV`
- 不通过：`STOP_VERIFY_VLA`
- twin 构造或 teacher viability 本身无效：`GATE0_INVALID`

通过只允许下一步训练轻量 DVoV predictor，并做 vision-gated scripted continuation。还不允许直接微调 Pi0.5、
宣称闭环提升或写论文结论。
