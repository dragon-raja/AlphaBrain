"""Measured local-view diagnostics. No dense SE(3) interpolation or task claims."""
from __future__ import annotations

import numpy as np

from AlphaBrain.research.dsol.acquisition.image_geometry import image_motion_jacobian

AXES = ("Tx", "Ty", "Tz", "Rx", "Ry", "Rz")
BLUE, ORANGE, GREEN = "#2665A8", "#D67832", "#148579"


def setting_name(state, candidate):
    task = "奶酪入碗" if state == 0 else "开抽屉放碗"
    return task + " · " + ("规范起点" if candidate == 0 else "扰动起点")


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dce2e8", linewidth=.6)
    ax.set_axisbelow(True)


def process_figure(bank):
    import matplotlib.pyplot as plt

    rgb = bank["probe_images"].astype(float)
    depth = bank["depth_optical"][0]
    jac, _ = image_motion_jacobian(rgb[0, :, ::-1], depth, bank["geometry_intrinsics"])
    # Fixed illustrative setting: drawer, perturbed view, Ry; not chosen by model score.
    predicted = (jac[..., 4] * bank["twists"][10, 4])[:, ::-1].mean(-1)
    actual = ((rgb[10]-rgb[9])/2).mean(-1)
    limit = max(abs(predicted).max(), abs(actual).max())
    fig = plt.figure(figsize=(14, 7.7), layout="constrained")
    grid = fig.add_gridspec(3, 4, height_ratios=[.7, 2, .65])
    labels = ["当前 RGB-D + 语言/本体", "冻结 VLA → Accel 图像梯度", "几何链式组合 → 相机方向", "真实微动取图 → 独立噪声复核"]
    for j, label in enumerate(labels):
        ax = fig.add_subplot(grid[0, j]); ax.axis("off")
        ax.text(.5, .5, label, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=.6", fc="#edf3f8", ec="#bad0e0"))
    axes = [fig.add_subplot(grid[1,j]) for j in range(4)]
    axes[0].imshow(rgb[0].astype(np.uint8)); axes[0].set_title("当前外部图像（策略输入）")
    im = axes[1].imshow(depth[:, ::-1], cmap="viridis", vmin=depth.min(), vmax=depth.max())
    axes[1].set_title("对齐深度（米，仿真真值）")
    fig.colorbar(im, ax=axes[1], orientation="horizontal", shrink=.9, label="optical z / m")
    for ax, value, title in zip(axes[2:], [predicted, actual], ["几何预测的图像变化", "实际 +0.2° / −0.2° 中心差分"]):
        im = ax.imshow(value, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(title)
    fig.colorbar(im, ax=axes[2:], orientation="horizontal", shrink=.8,
                 label="有符号 RGB 通道均值变化 / 0–255 像素单位（共用色条）")
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(grid[2,:]); ax.axis("off")
    ax.text(.5,.55,"示例：开抽屉放碗 · broad_train_000 · Ry\n"
        "四个场景设置共 52 张真实渲染图；两模型 × 8 初噪 = 832 次评分。机器人不执行任务。\n"
        "这里只画空间微动诊断，没有多步恢复轨迹；深度只用于几何，不输入 VLA。", ha="center", va="center", fontsize=11)
    fig.suptitle("01｜验证过程：图像变化能否预测，不等于 Accel 变化能否预测", fontsize=16)
    return fig


def direction_figure(records):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2,4,figsize=(15,7.8),layout="constrained")
    for ax, r in zip(axs.flat, records):
        predicted = np.asarray(r["geometric_predictions"]["all_valid"]["predicted_half_difference"])*1000
        actual = np.asarray(r["measured_half_difference"])*1000
        x = np.arange(6)
        ax.bar(x-.18,predicted,.36,color=BLUE,label="链式预测")
        ax.bar(x+.18,actual,.36,color=ORANGE,label="真实微动")
        bound = max(abs(predicted).max(), abs(actual).max())*1.18
        ax.set(xticks=x,xticklabels=AXES,ylim=(-bound,bound),title=setting_name(r["state_index"],r["candidate_index"]))
        ax.axhline(0,color="#475569",lw=.8)
        ax.text(.03,.96,f"方向余弦 {r['geometric_predictions']['all_valid']['scaled_direction_cosine']:.2f}",
                transform=ax.transAxes,ha="left",va="top",fontsize=9)
        _clean(ax)
    axs[0,0].set_ylabel("规范视角训练\nΔAccel × 1000")
    axs[1,0].set_ylabel("多视角训练\nΔAccel × 1000")
    axs[0,3].legend(loc="lower right",fontsize=9)
    fig.suptitle("02｜六轴方向：蓝橙异号，意味着预估与实测指向相反\n"
        "每柱 = [F(+步长)−F(−步长)] / 2；正值表示正方向比负方向更高。8 初噪均值。\n"
        "Tx/Ty/Tz：2 mm；Rx/Ry/Rz：0.2°。每格独立纵轴，不能仅比柱高。",fontsize=12)
    return fig


def noise_figure(cases, seeds):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2,4,figsize=(15,9),layout="constrained")
    for ax, case in zip(axs.flat,cases):
        scores = case["scores"]
        values = (scores[:,2::2]-scores[:,1::2])*500
        limit = max(float(abs(values).max()),1e-12)
        im = ax.imshow(values,cmap="RdBu_r",vmin=-limit,vmax=limit,aspect="auto")
        ax.set(xticks=range(6),xticklabels=AXES,yticks=range(len(seeds)),yticklabels=seeds,
               title=setting_name(case["state_index"],case["candidate_index"]))
        for i in range(len(seeds)):
            for j in range(6):
                ax.text(j,i,f"{values[i,j]:.1f}",ha="center",va="center",fontsize=8,
                    color="white" if abs(values[i,j]) > .55*limit else "#111827")
        ax.axhline(3.5,color="black",lw=1.5)
        fig.colorbar(im,ax=ax,shrink=.8,pad=.02,label="ΔAccel × 1000")
    axs[0,0].set_ylabel("规范视角训练 · Flow seed")
    axs[1,0].set_ylabel("多视角训练 · Flow seed")
    fig.suptitle("03｜只换 Flow 初噪，同一轴的方向也可能反转\n"
        "红：正向比负向高；蓝：正向比负向低。每格自适应色条，数字为原始差分 ×1000。\n"
        "黑线以上 41–44 用于定方向，以下 45–48 用于独立噪声检查；不是任务 train/test 划分。",fontsize=12)
    return fig


def transfer_figure(cases):
    import matplotlib.pyplot as plt
    methods = ["geometry_chain","actual_image_chain","finite_difference","local_best_with_stay"]
    names = ["几何链式\n当前图可算", "真实图差分链式\n诊断：需邻近图", "有限差分方向\n诊断：需邻近图", "局部最低分含原位\n诊断：需邻近图"]
    fig,axs = plt.subplots(2,4,figsize=(16,8.5),layout="constrained")
    for ax,case in zip(axs.flat,cases):
        ctr = case["controls"]
        values = np.array([ctr[m]["mean_test_change"] for m in methods])*1000
        ax.bar(np.arange(4),values,color=[BLUE,"#999fb0",ORANGE,GREEN],width=.65)
        for i,m in enumerate(methods):
            delta = np.asarray(ctr[m]["paired_test_deltas"])*1000
            ax.scatter(i+np.linspace(-.16,.16,len(delta)),delta,color="#303947",s=12,zorder=3)
        ax.axhline(0,color="#263849",lw=1)
        ax.set(xticks=range(4),xticklabels=names,title=setting_name(case["state_index"],case["candidate_index"]))
        ax.tick_params(axis="x",labelsize=8)
        _clean(ax)
    axs[0,0].set_ylabel("规范视角训练\n到位 Accel − 原位 Accel（×1000）")
    axs[1,0].set_ylabel("多视角训练\n到位 Accel − 原位 Accel（×1000）")
    fig.suptitle("04｜前 4 噪声确定视角，后 4 噪声检查：低于零才是改善\n"
        "柱：独立噪声均值；点：每个配对噪声结果。零线 = 不移动。所有值仍是 Accel，不是成功率。\n"
        "后三项使用真实邻近图，仅用于拆分误差；不是无需移动的新算法，也不是全空间 Oracle。",fontsize=12)
    return fig


def axis_curves_figure(cases, *, state=4, candidate=1):
    import matplotlib.pyplot as plt
    fig, axs=plt.subplots(2,3,figsize=(13,8),layout="constrained")
    x=np.array([-1.,0.,1.])
    for case,color,model in zip([c for c in cases if c["state_index"]==state and c["candidate_index"]==candidate],
                               [BLUE,ORANGE],["规范视角训练","多视角训练"]):
        scores=case["scores"].mean(0)
        pred=np.asarray(case["record"]["geometric_predictions"]["all_valid"]["predicted_half_difference"])
        for j,ax in enumerate(axs.flat):
            y=(scores[[1+2*j,0,2+2*j]]-scores[0])*1000
            ax.plot(x,y,"o-",color=color,label=model+"：实际三点")
            ax.plot(x,x*pred[j]*1000,"--",color=color,alpha=.7,label=model+"：一阶预测")
            ax.axhline(0,color="#7e8a98",lw=.6)
            ax.set(xticks=x,xticklabels=(["−2 mm","原位","+2 mm"] if j<3 else ["−0.2°","原位","+0.2°"]),
                   title=AXES[j],ylabel="相对原位 ΔAccel ×1000")
            _clean(ax)
    axs[0,0].legend(fontsize=8)
    fig.suptitle("05｜局部剖面：真实变化并不总像一条直线\n"
        "开抽屉放碗 · broad_train_000 · 8 初噪均值；实线只连接三个实测点，不代表中间位姿已评测。",fontsize=13)
    return fig
