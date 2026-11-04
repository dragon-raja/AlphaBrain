"""Scientific prefix-pilot learning curves with explicit axes and legends."""
from pathlib import Path

def learning_curves(cells,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,4,figsize=(15,9),squeeze=False)
    metrics=[('mean_reward','G reward (within-arm)'),('queries','Queries / episode'),('initial_stop_rate','Stop at initial view'),('scale','G denominator')]
    families=[('pi05',10),('groot_n15',4),('rldx',4)]
    for row,(family,full) in enumerate(families):
        for prefix,color in [(3,'#2468b4'),(full,'#d87719')]:
            points=cells.get(family+'-p'+str(prefix),[])
            for col,(key,title) in enumerate(metrics):
                axes[row,col].plot([r['episodes'] for r in points],[r[key] for r in points],'-o',ms=3,color=color,label='P'+str(prefix))
                axes[row,col].set_title(family+' | '+title,fontsize=10);axes[row,col].grid(alpha=.25);axes[row,col].set_xlabel('Completed training episodes')
        for ax in axes[row]:ax.legend(fontsize=8);ax.set_xlim(0,256)
        axes[row,2].set_ylim(-.03,1.03)
    fig.suptitle('Matched G/PPO prefix pilot: one seed, 8 updates; reward is not operation success',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96));output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'));fig.savefig(output.with_suffix('.png'),dpi=170);plt.close(fig)
