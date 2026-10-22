"""Plots of saved telemetry; bands are observed ranges, not confidence limits."""
import numpy as np


def plot_physical(branches,destination):
    import matplotlib.pyplot as plt
    cases=sorted({b['case'] for b in branches if b['model']=='xr1'})
    fig,axes=plt.subplots(3,len(cases),figsize=(11,9),sharex='col',squeeze=False,constrained_layout=True)
    for col,case in enumerate(cases):
        views=list(dict.fromkeys(b['view'] for b in branches if b['case']==case))
        for index,view in enumerate(views):
            bs=[b for b in branches if b['case']==case and b['view']==view];steps=[f['step'] for f in bs[0]['frames']]
            color=('tab:blue','tab:orange')[index]
            for row,key,factor in ((0,'target_distance_m',100),(1,'raw_joint_separation_m',1000)):
                data=np.array([[f[key]['obj_main'] if row==0 else f[key] for f in b['frames']] for b in bs])*factor
                axes[row,col].plot(steps,data.mean(0),color=color,label=view)
                axes[row,col].fill_between(steps,data.min(0),data.max(0),alpha=.12,color=color)
            contact=np.array([['obj_main' in f['contact_bodies'] for f in b['frames']] for b in bs])
            closed=np.array([[f['closing_command'] is True for f in b['frames']] for b in bs])
            axes[2,col].step(steps,contact.mean(0),where='post',color=color,label=view+' contact')
            axes[2,col].step(steps,closed.mean(0),where='post',color=color,linestyle=':',label=view+' close cmd')
        axes[0,col].set_title(case);axes[0,col].set_ylabel('EEF to object center (cm)')
        axes[1,col].set_ylabel('Raw finger joint separation (mm)');axes[2,col].set_ylabel('Fraction of 8 branches')
        axes[2,col].set_xlabel('Recorded action step');axes[2,col].set_ylim(-.05,1.05)
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(destination,dpi=170,bbox_inches='tight');plt.close(fig)


def plot_view_channels(effects,destination):
    import matplotlib.pyplot as plt
    initials=sorted({r['initial'] for r in effects});fig,axes=plt.subplots(len(initials),2,figsize=(12,7),squeeze=False,constrained_layout=True)
    for row,initial in enumerate(initials):
        origins=sorted({r['origin'] for r in effects if r['initial']==initial})
        for col,origin in enumerate(origins):
            rr=sorted([r for r in effects if (r['initial'],r['origin'])==(initial,origin)],key=lambda r:r['step']);steps=[r['step'] for r in rr];ax=axes[row,col]
            ax.plot(steps,[r['channels']['baseline__score']['mean'] for r in rr],'ko--',label='Original Accel',linewidth=2)
            for channel in ('translation','rotation','gripper','base','control_mode'):
                key='baseline_'+channel+'__accel_contribution'
                ax.plot(steps,[r['channels'][key]['mean'] for r in rr],'o--',label=channel,alpha=.8)
            ax.axhline(0,color='grey',linewidth=.7);ax.set_title(initial[:8]+' | state/history '+origin)
            ax.set_xlabel('Scored step (no intermediate scores)');ax.set_ylabel('Historical-good view minus historical-bad view')
            ax.set_xticks(steps);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(destination,dpi=170,bbox_inches='tight');plt.close(fig)


def plot_recovery(groups,destination):
    import matplotlib.pyplot as plt
    panels=['libero_canonical','libero_broad','plus_canonical','plus_broad'];fig,axes=plt.subplots(1,4,figsize=(13,4),sharey=True,constrained_layout=True)
    for ax,panel in zip(axes,panels):
        for j,mode in enumerate(('include','exclude')):
            rr=sorted([r for r in groups if r['panel']==panel and r['baseline']=='disturbance_uniform' and r['mode']==mode],key=lambda r:r['window'])
            ax.bar(np.arange(3)+(j-.5)*.35,[100*r['gain'] for r in rr],width=.35,label=mode+' canonical')
        ax.set_title(panel.replace('_',' '));ax.set_xticks(range(3),['W1','W3','W5']);ax.axhline(0,color='black',lw=.7);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Original recovery gain vs frozen disturbance (pp)');axes[-1].legend(fontsize=8)
    fig.savefig(destination,dpi=170,bbox_inches='tight');plt.close(fig)
