"""Current-image field diagnostic, with proxy and task metrics separated."""

import numpy as np


def summary_figure(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ['#94a3b8', '#2563eb', '#0d9488']
    methods = ['template', 'pose_only', 'current_rgb_pose']
    labels = ['Template', 'Pose only', 'Current RGB + pose']
    for col, model in enumerate(['canonical', 'broad']):
        for row, (metric, title) in enumerate([('low_quartile_hit', 'Low-Accel quartile hit (%)'),
                                              ('archived_endpoint_success', 'Archived full-task endpoint success (%)')]):
            ax = axes[row, col]
            for j, (method, label, color) in enumerate(zip(methods, labels, colors)):
                y = [next(r[metric] for r in summary['models'] if r['model'] == model and r['method'] == method and r['split'] == split) * 100
                     for split in ['initial_holdout', 'task_holdout']]
                bars = ax.bar(np.arange(2) + (j - 1) * .24, y, .23, label=label, color=color)
                ax.bar_label(bars, fmt='%.1f', fontsize=9)
            ax.set_xticks([0, 1], ['Held-out initials', 'Held-out task'])
            ax.set_ylim(0, 110)
            ax.set_title(model + ' trained | ' + title, fontsize=11)
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
            if row == 0:
                ax.axhline(25, color='black', linestyle=':', alpha=.5)
            else:
                stay = next(r['stay_success'] for r in summary['models'] if r['model'] == model) * 100
                ax.axhline(stay, color='black', linestyle=':', label='Stay perturbed')
                ax.legend(fontsize=8)
    fig.suptitle('Current-image field baseline: no target images; known reference coordinates\n4 scoring noises -> opposite 4; starts averaged within initial; exploratory, no new rollout', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .91))
    return fig
