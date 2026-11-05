"""Export transparent development curves; lines connect measured endpoints only."""
import numpy as np


def figures(summaries, curves, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'U12': '#2166ac', 'U3': '#d6604d'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for variant in colors:
        xs = sorted(int(k.split('@')[1]) for k in summaries if k.split('@')[0] == variant)
        data = [summaries[f'{variant}@{n}'] for n in xs]
        for seed in range(3):
            axes[0, 0].plot(xs, [100*d['seed_rates'][seed] for d in data],
                            color=colors[variant], alpha=.25, lw=.9)
        axes[0, 0].plot(xs, [100*d['success_rate'] for d in data], 'o-',
                        color=colors[variant], label=variant)
        axes[0, 1].plot(xs, [d['mean_queries'] for d in data], 'o-',
                        color=colors[variant], label=variant)
        axes[1, 0].plot(xs, [d['mean_best_accel'] for d in data], 'o-',
                        color=colors[variant], label=variant)
        axes[1, 1].plot(xs, [100*d['initial_only_fraction'] for d in data], 'o-',
                        color=colors[variant], label=variant)
    baseline = summaries['U12@512']
    axes[0, 0].axhline(100*baseline['stay_rate'], ls='--', color='gray', label='Paired Stay')
    axes[0, 0].axhline(100*baseline['reference_rate'], ls=':', color='black', label='Reference')
    titles = ['Operation success (%) — thin lines: individual seeds',
              'Deployment queries (includes return verification)',
              'Best visited Accel on development panel', 'Initial-only search (%)']
    for ax, title in zip(axes.flat, titles):
        ax.set(title=title, xlabel='Training episodes')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Shared-core development evidence: 8 tasks, 3 seeds; not the final test')
    for ext in ('pdf', 'png'):
        fig.savefig(output / f'development-curves.{ext}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
    metrics = ['mean_reward', 'mean_raw_improvement', 'queries', 'scale',
               'mean_stop_probability', 'approximate_kl', 'clip_fraction', 'value_loss', 'gradient_norm']
    for variant in colors:
        group = [v['points'] for k, v in curves.items() if f'-{variant}-' in k]
        x = [p['episodes'] for p in group[0]]
        for ax, metric in zip(axes.flat, metrics):
            a = np.asarray([[p[metric] for p in points] for points in group])
            for seed in a:
                ax.plot(x, seed, lw=.6, alpha=.2, color=colors[variant])
            ax.plot(x, a.mean(0), lw=1.1, color=colors[variant], label=variant)
            ax.set(title=metric, xlabel='Training episodes')
            ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle('Raw 32-episode PPO updates: settled G reward; no smoothing / convergence claim')
    for ext in ('pdf', 'png'):
        fig.savefig(output / f'training-diagnostics.{ext}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for episodes, color in [(512, '#777777'), (3072, '#2166ac')]:
        caps = [4, 6, 8] + ([16] if episodes == 3072 else [])
        ds = [summaries[f'U12{("-Q"+str(q)) if q != 8 else ""}@{episodes}'] for q in caps]
        axes[0].plot(caps, [d['success_rate']*100 for d in ds], 'o-', color=color, label=f'{episodes} episodes')
        axes[1].plot(caps, [d['mean_queries'] for d in ds], 'o-', color=color, label=f'{episodes} episodes')
    for ax, title in zip(axes, ['Operation success (%)', 'Actual mean queries']):
        ax.set(title=title, xlabel='Deployment query cap (training cap remains 8)', xticks=[4, 6, 8, 16])
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle('Matched checkpoint budget diagnostic — Q16 is not Q16-trained')
    for ext in ('pdf', 'png'):
        fig.savefig(output / f'query-budget.{ext}', dpi=180)
    plt.close(fig)
