# 2026-11-03：完整前缀主线的零派发边界恢复

原主线已完成六组v64。RLDX seed0在v42之后、GR00T seed1/2在v50之后停止；另六组评测继续。RLDX下一批仅完成远端CPU store准备，center阶段目录/许可未建立，store没有task、resident或模型调用，旧进程与四张目标卡已只读审计。原512调用预留继续占用，按零调用审计结算尝试，不冒充已完成32回合。

首次CPU PREPARE的HELLO在共享输入校验之后返回，但原Channel使用固定10秒接收时限。新的可选hello_timeout只在明确的PREPARE入口采用既有launch_timeout_seconds（600秒上限）；普通Channel保持已有超时，模型执行的心跳/逐调用授权不延长。已用actual CPU transport故障测试检验默认时序与失败保留，拒绝非法时限。

通用组合模块 `benchmarks/LIBERO/prefix_resumption.py`复用PrefixCampaign、frozen_stage、PhaseBudget、FleetSlots和原CPU learner。启动前核验已停止owner、固定v42/v50边界、无不明optimizer意图、所有原非委派阶段已结算，以及哈希绑定的零派发审计。剩余1600回合使用同清单/种子/评分/G/PPO，原初始化、优化器、RNG、历史和规范发布目录保持；不重新初始化或重放完成回合。

实际采集/评测采用独立冻结source-r3；与evaluation-r8的科学实现相同，仅初次握手与CPU接续组合改变。原CPU learner仍按原配置和原源码执行，采集批次保持同一演员哈希与32完整回合屏障。新的Fresh模型/主机/suite组合须重新通过数值、完整轨迹与原生基准逐值比较，CPU49通过不替代GPU准入。

资源先仅接管旧owner停止后已审计的Fresh4–7；旧评测的交接表不包含这块训练隔离资源的最终交接键，旧owner状态保持不可变，因此不会在新v64发布后抢占新owner。其余机器继续当前评测。只有旧评测owner结束、其六组最终操作和基线完成、无新增失败和未结算叶子记录且GPU实际空闲后，才允许新owner接收释放的卡；新source仍逐模型/主机/suite验收。每卡一个worker、单评测阶段最多八张物理卡，cy只用1–6。

原520万上限不变。原根剩余训练上界25712、额外工程核验预留上界10000，与现有全部委派预留合计最大5172603；最终三组操作使用原evaluation p1额度，不新开科学预算。旧未知调用不退款，未验证的失败不自动重发。

持久证据在 `runs/full-prefix-mainline-20261103/boundary-recovery-20261103/`；当前组合监测为父目录 `progress-combined.json`。1087/1088的第一版CPU回归暴露默认HELLO时序回归，已在任何GPU派发前拒绝并保留；修正后1096通过、82跳过，实际Torch CPU49通过，GPU准入单独留证。所有旧release、失败日志、结果和Git索引保留。
