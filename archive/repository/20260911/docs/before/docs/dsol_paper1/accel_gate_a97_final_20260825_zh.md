# Accel Gate A97 最终结果

## 实验范围

- 正式选择评测：8 个任务、96 个冻结状态、4 个 Broad64 模型。
- 每个状态先在 97 个 operational views 上运行 6 个共享 flow-noise seeds，再完整闭环执行 6 种预注册选择条件。
- 共 2,304 条正式选择 episode。
- 额外对 Broad64 practical 的 15 个 canonical 失败态穷举全部 97 个视角，共 1,455 条 episode。
- 主统计单位为 source HDF5 demonstration；同一演示的多个中间状态不作为独立样本扩大显著性。

## 正式 Gate A97

| 模型 | Canonical | Accel 单噪声 | Accel 六噪声 | 可见性最高 | Accel Top10 + 可见性 | 随机视角 | Oracle@6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Broad64 practical | 84.4% | 84.4% | 83.3% | 82.3% | 86.5% | 80.2% | 91.7% |
| Broad64 state-matched | 85.4% | 80.2% | 87.5% | 81.2% | 83.3% | 77.1% | 94.8% |
| Broad64 paired FM | 85.4% | 79.2% | 81.2% | 79.2% | 79.2% | 77.1% | 93.8% |
| Broad64 paired consistency | 86.5% | 80.2% | 80.2% | 80.2% | 81.2% | 82.3% | 89.6% |

六噪声 Accel 相对 canonical 的差值依次为 `-1.0pp`、`+2.1pp`、`-4.2pp`、`-6.2pp`。只有 paired-consistency 的下降区间排除 0；没有模型显示稳定正收益。

## Canonical 失败态 Oracle@97

| 选择规则 | 状态级救援率 | 演示分组均值 | 95% CI |
|---|---:|---:|---:|
| Canonical 重放 | 6.7% | 3.0% | [0.0%, 9.1%] |
| Accel 单噪声 | 40.0% | 50.0% | [22.7%, 77.3%] |
| Accel 六噪声 | 33.3% | 40.9% | [13.6%, 68.2%] |
| 可见性最高 | 33.3% | 39.4% | [12.1%, 66.7%] |
| Accel Top10 + 可见性 | 40.0% | 50.0% | [22.7%, 77.3%] |
| 随机视角 | 33.3% | 45.5% | [18.2%, 72.7%] |
| Oracle@97 | 80.0% | 83.3% | [63.6%, 100.0%] |

- Oracle@97 相对本轮 canonical 的演示分组差值为 `+80.3pp`，95% CI `[+57.6,+100.0]`。
- 六噪声 Accel 相对随机为 `-4.5pp`，95% CI `[-27.3,+13.6]`。
- 六噪声 Accel 相对可见性为 `+1.5pp`，95% CI `[-25.8,+28.8]`。
- 平均每个失败态有 31.3% 的视角可成功，但单状态成功候选数从 `0/97` 到 `86/97`，搜索难度高度异质。
- Accel 与候选成功的平均 point-biserial correlation 为 `+0.023`，接近 0；低 Accel 没有稳定对应成功行为。

## 结果 4 模型范围

M1 底层已覆盖 Official、Broad64 practical、Broad64 state-matched、Broad64 paired FM 和 Broad64 paired consistency。旧报告只展开 practical 的任务级案例；整合报告 v4 已增加五模型总表。

## 判定

97 视角候选池确有显著可救援空间，但当前 Accel 不能稳定优于随机视角或纯可见性，也不能在四个训练组织上稳定改善完整闭环。当前 `argmin accel_3` 作为主动视角选择器应保持 `HOLD`；后续若继续，应学习任务和状态条件化的 view value，而不是继续调整 Accel 前缀或噪声平均方式。
