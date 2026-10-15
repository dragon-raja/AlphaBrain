"""Closed-loop endpoint references, explicitly distinct from acquisition trials."""

import numpy as np


def outcome_figure(result):
    import matplotlib.pyplot as plt

    methods = ["stay_heldout32", "dev_global", "dev_same_task", "dev_other_tasks", "min_accel96", "min_accel97", "known_canonical"]
    labels = ["Keep perturbed start", "One model-level template view", "Task template (2 dev initials)",
              "Other-task template (dev only)", "Per-initial min Accel (same 96 views)", "Per-initial min Accel (97, + canonical)", "Known canonical view"]
    colors = ["#a6afb8", "#287c8e", "#3070aa", "#657baa", "#b66b28", "#d48b3b", "#6e747b"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), layout="constrained")
    for m, ax in enumerate(axes):
        rows = result["models"][m]["populations"]["evaluation16"]
        values = np.array([rows[k]["success"]["mean"]*100 for k in methods])
        bars = ax.barh(np.arange(len(methods)), values, color=colors)
        for bar, v in zip(bars, values):
            ax.text(v+.7, bar.get_y()+bar.get_height()/2, f"{v:.2f}", va="center", fontsize=10)
        ax.set(xlim=(0, 105), yticks=range(len(methods)), yticklabels=labels if m == 0 else [],
               xlabel="Archived full-task success (%)", title=["Canonical-trained", "Multiview-trained"][m])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("Does a low-Accel template view recover task performance?\nSame 16 evaluation initials; template fitting uses dev initials 0/1 only | no new rollout or camera motion test", fontsize=11)
    return fig
