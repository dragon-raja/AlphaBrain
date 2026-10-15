"""Small factorial matrices from completed same-state diagnostics."""
import numpy as np


def plot_matrices(contrasts,cases,destination):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(cases),3,figsize=(11,6),squeeze=False,constrained_layout=True)
    for row,case in enumerate(cases):
        records=[next(r for r in contrasts if r['initial']==case['initial'] and r['step']==step
                      and r['metric']=='baseline__score') for step in case['steps']]
        means=[np.mean(r['cells'],axis=0) for r in records]
        low=min(x.min() for x in means);high=max(x.max() for x in means)
        for col,(record,mean) in enumerate(zip(records,means)):
            ax=axes[row,col];im=ax.imshow(mean,vmin=low,vmax=high,cmap='Blues')
            ax.set_xticks([0,1],case['views']);ax.set_yticks([0,1],case['views'])
            ax.set_xlabel('Installed view');ax.set_ylabel('State/history branch')
            ax.set_title(case.get('label',case['initial'][:8])+f" | step {record['step']}")
            for i in range(2):
                for j in range(2):ax.text(j,i,f'{mean[i,j]:.5f}',ha='center',va='center',color='red')
        fig.colorbar(im,ax=axes[row,:].tolist(),label='Mean Accel (8 paired noise draws)')
    fig.savefig(destination,dpi=180,bbox_inches="tight");plt.close(fig)
