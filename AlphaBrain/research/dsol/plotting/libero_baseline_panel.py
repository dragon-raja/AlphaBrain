"""Static figure for a matched LIBERO baseline panel."""
import numpy as np


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = ['canonical', 'mild', 'strong', 'four_perturbed']
    fig, ax = plt.subplots(figsize=(10, 5))
    for m, (name, label, color) in enumerate([
        ('baseline', summary.get('baseline_plot_label', 'Reconstructed training start'), '#7b8794'),
        ('canonical', 'Canonical adaptation', '#2761ad'),
        ('broad', 'Multi-view adaptation', '#178678'),
    ]):
        rows = summary['groups'][name]
        means = np.array([rows[g]['mean'] for g in groups])*100
        ci = np.array([rows[g]['ci95'] for g in groups])*100
        x = np.arange(4)+(m-1)*.25
        ax.bar(x, means, .24, label=label, color=color)
        ax.errorbar(x, ci.mean(1), yerr=(ci[:, 1]-ci[:, 0])/2, fmt='none', ecolor=color, capsize=3)
    ax.set_xticks(np.arange(4), ['Reference', 'Mild (2 views)', 'Strong (2 views)', 'Perturbed (4 views)'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Full-task success (%)')
    ax.set_title('LIBERO: 8 tasks, all 32 original initials, 4 paired execution repeats')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    fig.text(.5, .01, summary.get('plot_note', 'AlphaBrain step-zero bridge, not native official policy. Five geometry-selected poses, not full-space average.'),
             ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .04, 1, 1])
    fig.savefig(output, dpi=180)
    plt.close(fig)
