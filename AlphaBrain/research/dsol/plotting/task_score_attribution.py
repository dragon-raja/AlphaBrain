"""Visualize frozen scorer/executor crossovers without new method claims."""
from pathlib import Path
import argparse
import numpy as np
from ..data.artifacts import read


def plot(summary_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=read(summary_path);fig,axes=plt.subplots(2,len(r['tasks']),figsize=(12,8),squeeze=False)
    for col,(task,item) in enumerate(r['tasks'].items()):
        broad=item['models']['broad'];cross=np.array(item['crossover_rows_scorer_columns_executor'])
        values=100*np.array([broad['canonical'],broad['stay_extra32'],cross[0,1],cross[1,1]])
        ax=axes[0,col];bars=ax.bar(range(4),values,color=['#7d8995','#a3aab1','#2865ac','#da8133'])
        ax.bar_label(bars,fmt='%.2f',padding=4);ax.set_ylim(0,100)
        ax.set_xticks(range(4),['Canonical\nview','Perturbed\nno adjustment','Canonical-model\nAccel selects','Multiview-model\nAccel selects'],fontsize=9)
        ax.set_ylabel('Existing complete-rollout success (%)');ax.set_title(task+'\nSame multiview executor in all four bars')
        ax=axes[1,col]
        for name,label,color in [('canonical','Canonical-trained','#2865ac'),('broad','Multiview-trained','#da8133')]:
            values=100*np.array(item['models'][name]['per_initial_accel_quartile_success'])
            ax.plot(range(4),values,'o-',label=label,color=color)
        ax.set_xticks(range(4),['Lowest Accel','Q2','Q3','Highest Accel'])
        ax.set_ylim(40,85);ax.set_ylabel('Complete-rollout success (%)');ax.legend(frameon=False)
        ax.set_title('Rank 96 views within each initial, then average');ax.grid(alpha=.15)
    fig.suptitle('Task-specific diagnostic | 20 heldout initials/task | 8 scoring + 4 execution noises',fontsize=14)
    fig.text(.5,.018,'Selection: lowest mean Accel among 97 views. Perturbed baseline: registered extra32. Existing endpoint rollouts, not physical search or a new evaluation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94));output=Path(summary_path).parent/'task-score-attribution.png';fig.savefig(output,dpi=160);plt.close(fig)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);args=p.parse_args();print(plot(args.summary))
