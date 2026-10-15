"""Cross-benchmark diagnostic figure; deliberately separate from main PDF."""
import numpy as np


def plot(result, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = result['benchmarks']
    rows = [data[b]['evaluation'][m] for b in ('libero', 'robocasa')
            for m in ('canonical', 'broad')]
    labels = ['LIBERO\nCanonical', 'LIBERO\nMultiview',
              'RoboCasa\nCanonical', 'RoboCasa\nMultiview']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    for j, (key, name, color) in enumerate((('reference', 'Reference', '#8c99a5'),
                                          ('stay', 'Stay perturbed', '#d39c40'),
                                          ('min_accel', 'Min Accel', '#26877e'))):
        bars = axes[0].bar(np.arange(4) + (j - 1) * .26,
                          [r[key] * 100 for r in rows], .25, color=color, label=name)
        axes[0].bar_label(bars, fmt='%.1f', fontsize=7)
    axes[0].set_xticks(range(4), labels, fontsize=8)
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel('Closed-loop success (%)')
    axes[0].set_title('Different populations / model recipes')
    axes[0].legend(fontsize=8, loc='upper center', ncol=3)
    for i, row in enumerate(rows[:2]):
        full = (row['min_accel'] - row['stay']) * 100
        sampled = row['downsample4']['min_accel_gain']['mean'] * 100
        axes[1].plot([32, 4], [full, sampled], 'o-', label=('Canonical', 'Multiview')[i])
        axes[1].annotate(f'{sampled:.1f} pp', (4, sampled), xytext=(6, 5), textcoords='offset points')
    axes[1].set_xlim(35, 0)
    axes[1].set_xticks([32, 4])
    axes[1].set_ylim(0, 50)
    axes[1].set_xlabel('Execution repeats per initial/view')
    axes[1].set_ylabel('Min Accel minus stay perturbed (pp)')
    axes[1].set_title('LIBERO: fewer repeats do not erase gain')
    axes[1].legend(fontsize=8)
    for i, (key, name, color) in enumerate((('global_view_rank_correlation', 'Aggregate over initials', '#245faa'),
                                          ('median_view_rank_correlation', 'Median per initial', '#d39c40'))):
        values = [np.mean([p[key] for p in row['cross_repeat_splits']]) for row in rows[2:]]
        bars = axes[2].bar(np.arange(2) + (i - .5) * .34, values, .32, label=name, color=color)
        axes[2].bar_label(bars, fmt='%.2f', fontsize=9)
    axes[2].set_xticks(range(2), ['Canonical', 'Multiview'])
    axes[2].set_ylim(0, 1.2)
    axes[2].set_ylabel('Spearman rank agreement')
    axes[2].set_title('RoboCasa: select 2 / check other 2 noises')
    axes[2].legend(fontsize=8)
    fig.suptitle('Existing data only: LIBERO 16 evaluation initials; RoboCasa 200 heldout-scene initials', fontsize=12)
    fig.savefig(target, dpi=180)
    plt.close(fig)
