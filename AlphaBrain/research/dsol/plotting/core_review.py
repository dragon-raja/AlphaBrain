"""Additional ablation-review plots from explicit, verified summary inputs."""
import numpy as np


def figures(summaries, curves, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def save(fig, name):
        for ext in ('png', 'pdf'):
            fig.savefig(output / f'{name}.{ext}', dpi=180)
        plt.close(fig)

    variants = ['U0', 'U1', 'U2', 'U3', 'U12', 'U13', 'U23', 'U123', 'U0T', 'U123T', 'U12T']
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    x = np.arange(len(variants))
    data = [summaries[f'{v}@512'] for v in variants]
    axes[0].bar(x, [100*d['success_rate'] for d in data], color='#b6cce0', label='3-seed mean')
    for seed, marker in enumerate(['o', 's', '^']):
        axes[0].scatter(x+(seed-1)*.16, [100*d['seed_rates'][seed] for d in data],
                        marker=marker, s=28, label=f'Seed {seed}', zorder=3)
    axes[0].axhline(100*data[0]['stay_rate'], color='gray', ls='--', label='Stay')
    axes[0].axhline(100*data[0]['reference_rate'], color='black', ls=':', label='Reference')
    axes[0].set(ylabel='Operation success (%)', ylim=(0, 100))
    axes[0].legend(ncol=6, fontsize=8)
    axes[1].bar(x, [d['mean_queries'] for d in data], color='#5588aa')
    axes[1].set(ylabel='Mean deployment queries', ylim=(0, 8.5))
    for ax in axes:
        ax.set(xticks=x, xticklabels=variants)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('All 11 structural / task-input screens: 512 episodes; shared development panel')
    save(fig, 'structure-all-seeds')

    names = ['U12', 'U12-cost005', 'U12-cost020', 'U12-no_score', 'U12-fixed_query']
    labels = ['Full: cost .01', 'Cost .005', 'Cost .02', 'No score INPUT', 'No learned STOP']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    data = [summaries[f'{v}@512'] for v in names]
    for ax, key, scale, title in zip(axes, ['success_rate', 'mean_queries'], [100, 1],
                                    ['Operation success (%)', 'Mean deployment queries']):
        ax.barh(labels, [scale*d[key] for d in data], color='#6395b6')
        ax.invert_yaxis()
        ax.set(xlabel=title)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Matched 512-episode mechanism screen; no-score-input still uses Accel reward / selection')
    save(fig, 'mechanisms')

    metrics = ['policy_loss', 'loss', 'stop_entropy', 'motion_unit_entropy',
               'normalized_gain', 'query_cost', 'initial_stop_rate', 'at_cap_fraction',
               'mean_initial_accel', 'mean_best_accel', 'exploration_regularizer', 'queries']
    fig, axes = plt.subplots(4, 3, figsize=(13, 13), constrained_layout=True)
    for variant, color in [('U12', '#2166ac'), ('U3', '#d6604d')]:
        group = [v['points'] for k, v in curves.items() if f'-{variant}-' in k]
        x = [p['episodes'] for p in group[0]]
        for ax, metric in zip(axes.flat, metrics):
            values = np.asarray([[p[metric] for p in ps] for ps in group])
            for seed in values:
                ax.plot(x, seed, color=color, alpha=.22, lw=.6)
            ax.plot(x, values.mean(0), color=color, lw=1.2, label=variant)
            ax.set(title=metric, xlabel='Training episodes')
            ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle('Additional raw PPO diagnostics: thin = seed, thick = mean; no smoothing')
    save(fig, 'training-extra')
