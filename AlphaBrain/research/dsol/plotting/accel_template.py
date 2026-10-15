"""Static scientific figures for measured-site template diagnostics."""

import numpy as np


def coverage_figure(curves):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")
    colors = ["#1763aa", "#de7c28"]
    for m, ax in enumerate(axes):
        for color, kind, label in zip(colors, ["same_task", "other_tasks"], ["Other 3 initials of same task", "Other 7 tasks (target task excluded)"]):
            ys = curves[f"m{m}_{kind}_low24"] * 100
            x = np.arange(1, ys.shape[1]+1) / ys.shape[1] * 100
            med = np.median(ys, axis=0)
            q1, q3 = np.quantile(ys, [.25, .75], axis=0)
            ax.plot(x, med, color=color, lw=2, label=label)
            ax.fill_between(x, q1, q3, color=color, alpha=.12)
        ax.plot(x, x, "--", color="gray", label="Random expected coverage")
        ax.plot(x, np.minimum(x*4, 100), ":", color="black", label="Perfect ordering reference")
        ax.axhline(80, color="gray", lw=.6)
        ax.set(title=["Canonical-trained", "Multiview-trained"][m],
               xlabel="Retained measured locations (%)", ylabel="Recall of actual lowest 24 / 96 views (%)",
               xlim=(0, 100), ylim=(0, 102))
        ax.grid(alpha=.15)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("How much of the measured view catalog can a template exclude?\nMedian and IQR over 32 initials; NOT closed-loop success or continuous space volume", fontsize=11)
    return fig


def decomposition_figure(rows):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7), layout="constrained")
    methods = ["affine", "affine_linear", "affine_quadratic"]
    split_names = [("full_map_descriptive", "Full-map fit (descriptive)", "#aab2bb"),
                   ("random", "Random camera holdout", "#1763aa"),
                   ("spatial", "Spatial-block holdout", "#de7c28")]
    for m in range(2):
        for j, kind in enumerate(["same_task", "other_tasks"]):
            ax = axes[j, m]
            for si, (split, label, color) in enumerate(split_names):
                vals = [[r["r2"] for r in rows if r["model"] == m and r["template"] == kind and r["split"] == split and r["method"] == method] for method in methods]
                med = np.median(vals, axis=1)
                q1, q3 = np.quantile(vals, [.25, .75], axis=1)
                ax.bar(np.arange(3)+(si-1)*.25, med, width=.24, color=color, label=label,
                       yerr=np.stack([med-q1, q3-med]), capsize=2)
            ax.axhline(0, color="black", lw=.6)
            ax.set(xticks=range(3), xticklabels=["Offset + scale", "+ spatial linear", "+ spatial quadratic"],
                   ylabel="R-squared (median, IQR)",
                   title=["Canonical-trained", "Multiview-trained"][m] + " | " + ["Same-task template", "Other-task template"][j])
            ax.grid(axis="y", alpha=.15)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Is amplitude calibration enough, or is a spatial residual needed?\nCalibration uses target-initial scores on training views; diagnostics, NOT a deployed search algorithm", fontsize=11)
    return fig
