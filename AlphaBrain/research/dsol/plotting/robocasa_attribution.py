"""Small diagnostic figure; same y scale across training arms."""
import numpy as np


def plot_diagnostics(result, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1,3,figsize=(15,4.7), constrained_layout=True)
    colors=['#245faa','#149181']
    for m,arm in enumerate(('canonical','broad')):
        r=result['models'][arm]['target']
        values=[r['canonical'],r['extra32'],r['empirical_oracle'],r['crossfit_catalog_ties']['mean']]
        x=np.arange(4)+(m-.5)*.36
        bars=axes[0].bar(x,np.array(values)*100,width=.34,color=colors[m],label=arm+' training')
        axes[0].bar_label(bars,fmt='%.1f',fontsize=8)
        axes[1].plot(np.arange(4)+1,np.array(r['success_by_accel_quartile'])*100,'o-',color=colors[m])
    axes[0].set_xticks(range(4),['Reference','Stay\nextra32','Empirical\nmax of 4','Select 2 /\ntest 2'])
    axes[0].set_ylim(0,100);axes[0].set_ylabel('Closed-loop success (%)')
    axes[0].set_title('Noise selection diagnostic (not true Oracle)');axes[0].legend(fontsize=8)
    axes[1].set_xticks([1,2,3,4],['Lowest','Q2','Q3','Highest'])
    axes[1].set_xlabel('Per-initial Accel quartile');axes[1].set_ylim(0,60)
    axes[1].set_title('Relative score vs success');axes[1].set_ylabel('Closed-loop success (%)')
    groups=['reference','inner_noncanonical','outer_noncanonical','extra32']
    stats=[result['training_geometry']['target'][g]['paired_gain'] for g in groups]
    means=np.array([s['mean'] for s in stats])*100
    ci=np.array([s['ci95'] for s in stats])*100
    axes[2].bar(range(4),means,color=['#af4358' if v<0 else '#149181' for v in means])
    axes[2].errorbar(range(4),means,yerr=np.vstack([means-ci[:,0],ci[:,1]-means]),fmt='none',color='black',capsize=4)
    axes[2].axhline(0,color='gray',linewidth=.8);axes[2].set_xticks(range(4),['Reference','|az|<20°','|az|≥20°','Extra32'])
    axes[2].set_ylabel('Broad minus canonical training (pp)');axes[2].set_title('Geometric redistribution (exploratory)')
    fig.suptitle('RoboCasa P0 — heldout scenes, same 200 initials; no new rollouts in this figure',fontsize=12)
    fig.savefig(target,dpi=180);plt.close(fig)
