"""Standalone figures from supplied cached-search results; no model calls."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


METHODS = {
    "stay": ("不换位", "#7b8794"), "scan": ("固定扫描", "#3578a9"),
    "random": ("随机搜索", "#d59832"), "local_accel": ("局部 Accel", "#2c8b68"),
    "ppo": ("PPO 搜索", "#8a4ba5"),
}


def style(font=None):
    if font:
        from matplotlib import font_manager
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})


def training_figure(updates):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), constrained_layout=True)
    panels = [
        ("episode_mean_total_reward", "训练初态：搜索回报"),
        ("episode_mean_queries", "每回合实际查询次数"),
        ("train/value_loss", "critic 的价值拟合误差"),
        ("train/explained_variance", "critic 的回报解释率"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        for seed, rows in updates.items():
            valid = [r for r in rows if r.get(key) is not None]
            ax.plot([r["steps"] for r in valid], [r[key] for r in valid], label=f"seed {seed}", lw=1.8)
        ax.set(title=title, xlabel="搜索决策步")
        ax.grid(alpha=.2)
    axes[0, 1].set_ylim(.5, 8.5)
    axes[1, 1].axhline(0, color="gray", lw=.8)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("PPO 训练过程：三个种子，固定参数，全部保留\n训练回报不等于机器人任务成功率", fontsize=16)
    return fig


def evaluation_figure(summary):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [
        ("success", "完整任务成功率（旧标签重放）", 100),
        ("queries", "实际查询次数（含起点）", 1),
        ("accel_gap", "距全候选最低 Accel 的差距（越低越好）", 1),
        ("total_reward", "搜索目标：改善量减查询与移动成本", 1),
    ]
    for ax, (metric, title, scale) in zip(axes.flat, panels):
        for method, (label, color) in METHODS.items():
            rows = sorted([r for r in summary["rows"] if r["method"] == method], key=lambda r: r["budget"])
            x = [r["budget"] for r in rows]
            y = [r["metrics"][metric]["mean"] * scale for r in rows]
            ax.plot(x, y, "o-", color=color, label=label, lw=2 if method == "ppo" else 1.5)
            if metric == "success":
                lo = [r["metrics"][metric]["ci95"][0] * scale for r in rows]
                hi = [r["metrics"][metric]["ci95"][1] * scale for r in rows]
                ax.fill_between(x, lo, hi, color=color, alpha=.07)
        ax.set(title=title, xlabel="查询上限", xticks=[4, 8, 16])
        ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("%")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=9)
    axes[0, 1].set_ylim(.5, 17)
    fig.suptitle("同一批 16 个评价初态 × 32 个扰动起点\n主预算为 8；4 和 16 为预算变化诊断", fontsize=16)
    return fig


def task_figure(rows, budget=8):
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(tasks))
    names = {
        "goal_cream_cheese_bowl": "奶酪放碗",
        "goal_top_drawer_bowl": "上抽屉\n放碗",
        "goal_wine_rack": "酒瓶入架",
        "libero10_book_caddy": "书放收纳架",
        "libero10_bowl_bottom_drawer": "碗入下抽屉",
        "libero10_mug_microwave": "杯入微波炉",
        "object_cream_cheese_basket": "奶酪入篮",
        "spatial_drawer_bowl_plate": "空间任务\n抽屉/碗/盘",
    }
    labels = [names.get(task, task) for task in tasks]
    task_values = {}
    for offset, method in enumerate(("stay", "local_accel", "ppo")):
        values = [100 * np.mean([r["success"] for r in rows if r["task"] == task
                                 and r["method"] == method and r["budget"] == budget]) for task in tasks]
        task_values[method] = np.array(values)
        label, color = METHODS[method]
        ax.bar(x + (offset - 1) * .24, values, width=.23, label=label, color=color)
    ax.set(ylabel="成功率（%）", title="主预算 8：逐任务成功率；每任务仅 2 个评价初态")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=.2)
    delta = task_values["ppo"] - task_values["local_accel"]
    axes[1].bar(x, delta, color=["#2c8b68" if d >= 0 else "#b84c49" for d in delta], width=.6)
    for i, value in enumerate(delta):
        axes[1].text(i, value + (.4 if value >= 0 else -.4), f"{value:+.2f}",
                     ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set(xticks=x, xticklabels=labels, ylabel="PPO − 局部 Accel（百分点）",
                title="平均接近，任务间仍有明显得失", ylim=(min(delta)-2.5, max(delta)+2.5))
    axes[1].grid(axis="y", alpha=.2)
    return fig


def architecture_figure():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set(xlim=(0, 14), ylim=(0, 8))
    ax.axis("off")
    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.12",
                                   fc=color, ec="#4b5563", lw=1))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=12, linespacing=1.55)
    def arrow(a, b, label=None):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#374151", lw=1.7))
        if label:
            ax.text((a[0]+b[0])/2, (a[1]+b[1])/2 + .16, label, ha="center", fontsize=10)
    ax.text(7, 7.6, "当前训练：缓存中的主动搜索，不运行机器人操作模型", ha="center", fontsize=18, weight="bold")
    box(.3, 4.7, 3.5, 1.6, "可见输入\n已访问图像特征与 Accel\n相对候选位置、访问历史、预算", "#e5eff8")
    box(5.1, 4.9, 3.2, 1.2, "搜索 actor：MLP 128 × 2\n选下一位置 / STOP", "#eadff1")
    box(10, 4.7, 3.5, 1.6, "可查询的旧数据环境\n访问后才解锁图像和分数\n每候选 8 套评分噪声", "#e4f1e9")
    arrow((3.95, 5.5), (4.95, 5.5))
    arrow((8.45, 5.5), (9.85, 5.5), "合法动作屏蔽")
    ax.annotate("", xy=(2, 4.55), xytext=(11.8, 4.55),
                arrowprops=dict(arrowstyle="->", connectionstyle="bar,fraction=-.08", color="#374151"))
    ax.text(7, 4.02, "新观测；奖励 = 最佳 Accel 改善 − 查询成本 − 移动成本",
            ha="center", fontsize=11, bbox=dict(facecolor="white", edgecolor="none", pad=3))
    box(.3, 1.65, 3.5, 1.5, "critic：MLP 128 × 2\n预测剩余搜索回报\n辅助 PPO 更新 actor\n仅参与训练", "#fff0d4")
    arrow((2, 4.55), (2, 3.3))
    box(5.1, 1.95, 3.2, 1.2, "停止后\n选择已访问候选中\nAccel 最低的位置", "#eadff1")
    arrow((6.7, 4.75), (6.7, 3.3))
    box(10, 1.95, 3.5, 1.2, "独立评价侧\n查询历史完整任务成绩\n成功率不输入 actor / critic", "#f7e4e1")
    arrow((8.45, 2.55), (9.85, 2.55))
    ax.text(7, .85, "后续在线接法：缓存查询 → 移动外部相机 → 冻结 PaliGemmaPi05 推理并提取 Accel\n"
            "搜索结束后才执行操作；头部迁移另加运动学与输入适配。", ha="center", va="center", fontsize=12)
    fig.tight_layout()
    return fig
