"""Static figures for a small matched checkpoint diagnostic, not full P0."""
from pathlib import Path
import numpy as np


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    groups=summary.get('groups',['canonical','mild','strong','four_perturbed'])
    labels=(["Reference", "Train poses (64)", "Extra poses (32)", "Perturbed (96)"]
            if 'all_perturbed' in groups else ['Reference','Mild (2 views)','Strong (2 views)','Perturbed (4 views)'])
    names=summary.get('models',['official','canonical','broad'])
    titles=names
    colors=['#7b8794','#80a8de','#84c8b7','#2761ad','#178678'] if len(names)==5 else ['#7b8794','#2761ad','#178678']
    width=.8/len(names)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,split,title in zip(axes,['pretrain','target'],['Seen layouts/styles','Heldout layouts/styles']):
        for m,(name,color) in enumerate(zip(names,colors)):
            rows=summary['splits'][split][name]
            means=np.array([rows[g]['mean'] for g in groups])*100
            ci=np.array([rows[g]['ci95'] for g in groups])*100
            ax.bar(np.arange(4)+(m-(len(names)-1)/2)*width,means,width*.95,color=color,label=titles[m])
            ax.errorbar(np.arange(4)+(m-(len(names)-1)/2)*width,(ci[:,0]+ci[:,1])/2,
                        yerr=(ci[:,1]-ci[:,0])/2,fmt='none',ecolor=color,capsize=3)
        ax.set_xticks(np.arange(4),labels,rotation=15,ha='right')
        ax.set_title(title);ax.set_ylim(0,100);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].set_ylabel('Full-task success (%)');axes[1].legend(fontsize=8)
    fig.suptitle('Matched starting-checkpoint diagnostic: 10 tasks, 2 initials/task/split, 4 repeats')
    note=('97-view catalog, 40 initials only; paired task/initial bootstrap; no confirmed oracle.'
          if 'all_perturbed' in groups else 'Geometry-selected heldout views; NOT the 97-view mean or an oracle. Paired task/initial bootstrap.')
    fig.text(.5,.01,note,ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(output/'checkpoint_comparison.png',dpi=180);plt.close(fig)
