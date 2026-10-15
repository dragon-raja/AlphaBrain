"""Separate pixel sensitivity from model-dependent view selection."""
from pathlib import Path
import argparse
import numpy as np
from ..data.artifacts import read


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);summary=read(root/'summary.json');plan=read(root/'plan.json')
    with np.load(root/'scores-crossed.npz') as z:scores=z['scores'].mean(-1)
    fig,axes=plt.subplots(2,len(summary['tasks']),figsize=(12,8),squeeze=False)
    for col,(task,item) in enumerate(summary['tasks'].items()):
        ax=axes[0,col]
        for source,(key,label,color) in enumerate([('canonical_archive','Identical image set A','#326bb0'),('broad_archive','Identical image set B','#cf8439')]):
            values=100*np.array(item['crossover_by_common_image_source'][key])[:,1]
            bars=ax.bar(np.arange(2)+(source-.5)*.32,values,.32,label=label,color=color)
            ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
        ax.set_xticks([0,1],['Canonical-model Accel selects','Multiview-model Accel selects'],fontsize=9)
        ax.set_ylim(0,100);ax.set_ylabel('Existing complete-rollout success (%)')
        ax.set_title(task+'\nSame multiview executor; same image set for both scorers')
        ax.legend(frameon=False,fontsize=9)
        ti=[i for i,j in enumerate(plan['jobs']) if j['task']==task]
        ax=axes[1,col];allvalues=[]
        for m,(name,color) in enumerate([('canonical','#326bb0'),('broad','#cf8439')]):
            a=scores[m,0,ti,1:].mean(0);b=scores[m,1,ti,1:].mean(0)
            allvalues.extend(a);allvalues.extend(b)
            changes=item['models'][name]['winner_changes']
            ax.scatter(a,b,s=16,alpha=.6,color=color,label=f'{name}: {changes}/20 winner changes')
        low,high=min(allvalues),max(allvalues);margin=(high-low)*.05
        ax.plot([low-margin,high+margin],[low-margin,high+margin],'k--',lw=.8)
        ax.set_xlabel('Mean Accel, archived images A');ax.set_ylabel('Mean Accel, archived images B')
        ax.set_title('96-position average field; individual winner changes in legend')
        ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.15)
    fig.suptitle('Native micro-pixel sensitivity | two image versions x two models | paired 8 scoring noises',fontsize=13)
    fig.text(.5,.018,'A/B differ by at most one uint8 level; task/state/noise fixed. Scoring re-evaluated on changed cells. No new execution or training.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94));path=root/'render-score-replay.png';fig.savefig(path,dpi=170);plt.close(fig);return path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();print(plot(a.root))
