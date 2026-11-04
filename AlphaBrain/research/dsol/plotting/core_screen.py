"""Completed small-screen figures with all declared seeds and unsmoothed optimizer points."""
from pathlib import Path


def figures(summary,curves,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    out=Path(output);names=list(summary);x=np.arange(len(names))
    seeds=sorted(next(iter(summary.values()))['per_seed'],key=int)
    colors=['#b1440e','#16446c','#6f4099','#168475','#c58b00']
    if any(set(d['per_seed'])!=set(seeds) for d in summary.values()):raise ValueError('Mixed seed matrix')
    fig,ax=plt.subplots(figsize=(10,4.5))
    ax.bar(x,[100*summary[v]['policy_success'] for v in names],color='#6096ba',label=f'{len(seeds)}-policy-seed mean')
    for i,seed in enumerate(seeds):
        color=colors[i%len(colors)];offset=(i-(len(seeds)-1)/2)*.12
        ax.scatter(x+offset,[100*summary[v]['per_seed'][seed]['policy_success'] for v in names],color=color,s=42,label='Policy seed '+seed,zorder=3)
    base=next(iter(summary.values()));ax.axhline(100*base['stay_success'],color='#444444',linestyle='--',label='Matched perturbed Stay')
    ax.axhline(100*base['reference_success'],color='#407d35',linestyle=':',label='Matched reference')
    ax.set(xticks=x,xticklabels=names,ylim=(0,105),ylabel='Complete operation success (%)',title='pi0.5 P10/G: 512 training episodes per seed; fixed development panel')
    ax.legend(fontsize=8,ncol=2);ax.grid(axis='y',alpha=.2)
    fig.text(.5,.01,f'8 tasks; 64 physical cases reused by {len(seeds)} policy seeds. Screening, not final test or convergence.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1))
    for ext in ['png','pdf']:fig.savefig(out/('success-by-structure.'+ext),dpi=170)
    plt.close(fig)
    for stem,metrics in [('behavior',[('mean_reward','Settled G reward'),('queries','Queries per episode')]),('optimizer',[('value_loss','Value loss'),('approximate_kl','Approximate KL')])]:
        fig,axes=plt.subplots(len(names),2,figsize=(11,2.1*len(names)),squeeze=False)
        for i,name in enumerate(names):
            for si,seed in enumerate(seeds):
                color=colors[si%len(colors)]
                points=curves[f'pi05-{name}-seed-{seed}']['points']
                for j,(metric,label) in enumerate(metrics):
                    ax=axes[i,j];ax.plot([p['episodes'] for p in points],[p[metric] for p in points],'-o',color=color,lw=1,ms=2,label=f'Seed {seed}')
                    ax.set_title(f'{name}: {label}',fontsize=10);ax.set_xlabel('Completed training episodes');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('All 16 updates shown; changing training batches. Reward is not operation success.',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.98))
        for ext in ['png','pdf']:fig.savefig(out/('training-'+stem+'.'+ext),dpi=130)
        plt.close(fig)
