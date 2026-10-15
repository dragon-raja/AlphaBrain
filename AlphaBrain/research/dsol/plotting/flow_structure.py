"""Static publication-exportable views of existing-data flow decomposition."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def phase_figures(alignment, output):
    out=Path(output);out.mkdir(exist_ok=False)
    titles={'libero-05':'LIBERO drawer initial 5','libero-06':'LIBERO drawer initial 6',
            'xr1-39dd9788':'XR-1 counter to cabinet','xr1-a90105e3':'XR-1 sink to counter'}
    names={'original':'Original','near':'Executed portion','near_radial':'Radial only','near_angular':'Angular only'}
    paths=[]
    for key,case in alignment['cases'].items():
        fig,ax=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
        steps=np.asarray(case['steps'])
        for r in case['comparisons']['original']['repeats']:
            ax[0].plot(steps,r['delta_good_minus_bad'],color='0.65',alpha=.35,lw=.7)
        for rule,data in case['comparisons'].items():
            ax[0].plot(steps,data['cumulative_delta_mean'],marker='o',label=names[rule],ls='--' if rule=='near' else '-')
        ax[0].axhline(0,color='black',lw=.8);ax[0].legend(fontsize=8)
        ax[0].set(title='Cumulative cost: better minus worse',ylabel='Negative favors historical better candidate',xlabel='Executed steps at scoring')
        for channel,delta in case['cumulative_channel_delta'].items():
            label=channel.removeprefix('baseline_').removesuffix('__accel_contribution')
            ax[1].plot(steps,delta,marker='o',label=label)
        ax[1].axhline(0,color='black',lw=.8);ax[1].legend(fontsize=8)
        ax[1].set(title='Additive channel attribution of original cost',ylabel='Contribution to better-minus-worse difference',xlabel='Executed steps at scoring')
        for c,label,color in zip(case['candidates'],['Historical better','Historical worse'],['tab:blue','tab:orange']):
            distances=np.array([r['target_distance_m'] for r in c['rows']])*100
            ax[2].plot(steps,distances.mean(0),marker='o',color=color,label=label)
            ax[2].fill_between(steps,distances.min(0),distances.max(0),color=color,alpha=.12)
            for row in c['rows']:
                step=row['first_contacts'].get(row['target_body'])
                if step is not None:ax[2].axvline(step,color=color,alpha=.18,lw=.8)
        ax[2].set(title='EEF to target body center; shaded repeat range',ylabel='Distance (cm)',xlabel='Executed steps; vertical lines = first target contact')
        ax[2].legend(fontsize=8)
        fig.suptitle(titles[key]+' | 8 paired scoring repeats; endpoint labels from separate execution repeats',fontsize=12)
        for a in ax:a.grid(alpha=.15)
        for ext in ('png','pdf'):
            path=out/(key+'.'+ext);fig.savefig(path,dpi=180);paths.append(str(path))
        plt.close(fig)
    return paths
