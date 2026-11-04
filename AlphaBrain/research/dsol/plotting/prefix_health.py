"""Static, labeled training diagnostics; seeds are never pooled as episodes."""
from pathlib import Path


def learning_figures(curves, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    groups=[('pi05',10),('groot_n15',4),('rldx',4)]
    panels={
        'learning-behavior': [('mean_reward','Settled G reward'),('queries','Queries / episode'),
            ('initial_stop_rate','Initial-only fraction'),('scale','Online G denominator')],
        'optimizer-losses': [('policy_loss','PPO policy loss'),('value_loss','Value loss'),
            ('loss','Total optimizer loss'),('gradient_norm','Gradient norm')],
        'optimizer-stability': [('approximate_kl','Approximate KL'),('clip_fraction','Clipped sample fraction'),
            ('mean_stop_probability','STOP probability (optimization states)'),('stop_entropy','STOP entropy')],
        'gain-and-cost': [('mean_raw_improvement','Raw initial minus best Accel'),('normalized_gain','Normalized score gain'),
            ('query_cost','Extra-query cost'),('at_cap_fraction','Eight-query fraction')],
        'score-levels': [('mean_initial_accel','Mean initial Accel'),('mean_best_accel','Mean best queried Accel'),
            ('mean_raw_improvement','Mean paired initial minus best'),('scale','Online G denominator')]}
    for name,metrics in panels.items():
        fig,axes=plt.subplots(3,4,figsize=(15,8.5),squeeze=False)
        for i,(family,prefix) in enumerate(groups):
            for seed,color in enumerate(['#2374AB','#D55E00','#009E73']):
                points=curves[f'{family}-p{prefix}-seed-{seed}']['points']
                x=[p['episodes'] for p in points]
                for j,(metric,title) in enumerate(metrics):
                    raw=[p[metric] for p in points]
                    ax=axes[i,j];ax.plot(x,raw,color=color,alpha=.18,lw=.7)
                    # Show all 64 raw points faintly; a trailing four-update mean clarifies trends.
                    smooth=[sum(raw[max(0,k-3):k+1])/len(raw[max(0,k-3):k+1]) for k in range(len(raw))]
                    ax.plot(x,smooth,color=color,lw=1.4,label=f'Seed {seed}')
                    ax.set_title(f'{family} P{prefix}: {title}',fontsize=9)
                    ax.set_xlabel('Completed training episodes');ax.set_xlim(0,2048);ax.grid(alpha=.2)
                    if metric in ('initial_stop_rate','at_cap_fraction','clip_fraction','mean_stop_probability'):ax.set_ylim(-.02,1.02)
            for ax in axes[i]:ax.legend(fontsize=7)
        subtitle = ('Training batches have changing initial states; best = minimum visited query score, not independent confirmation'
                    if name == 'score-levels' else 'Reward is not operation success; reward/loss scales differ across models')
        fig.suptitle('Full-prefix G/PPO: 3 seeds, 64 updates each; thin = raw, thick = trailing 4 updates\n'+subtitle,fontsize=11)
        fig.tight_layout(rect=(0,0,1,.94))
        for ext in ['png','pdf']:fig.savefig(output/(name+'.'+ext),dpi=160)
        plt.close(fig)


def development_figure(progress,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,family in zip(axes,['pi05','groot_n15','rldx']):
        for seed,color in enumerate(['#2374AB','#D55E00','#009E73']):
            points=progress[family][str(seed)]['points']
            ax.plot([p['episodes'] for p in points],[100*p['success_rate'] for p in points],'-o',color=color,label=f'Seed {seed}')
        ax.set_title(family);ax.set_xlabel('Completed training episodes');ax.set_ylabel('Operation success (%)')
        ax.set_ylim(0,100);ax.set_xticks([512,1024]);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Same fixed development panel: 64 paired operations per seed/checkpoint\nTwo noisy checkpoints are not a convergence guarantee; final full panel still running',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.9))
    for ext in ['png','pdf']:fig.savefig(output/('development-success.'+ext),dpi=160)
    plt.close(fig)
