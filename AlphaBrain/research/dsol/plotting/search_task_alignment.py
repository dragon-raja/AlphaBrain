"""Static figures from audited tables, never recomputes scientific labels."""
import numpy as np

LABELS={
    'goal_cream_cheese_bowl':'Cream cheese → bowl',
    'goal_top_drawer_bowl':'Bowl → top drawer',
    'goal_wine_rack':'Wine → rack',
    'libero10_book_caddy':'Book → caddy',
    'libero10_bowl_bottom_drawer':'Bowl → bottom drawer',
    'libero10_mug_microwave':'Mug → microwave',
    'object_cream_cheese_basket':'Cream cheese → basket',
    'spatial_drawer_bowl_plate':'Drawer bowl → plate',
    'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it':'Stove + moka pot',
    'KITCHEN_SCENE8_put_both_moka_pots_on_the_stove':'Both moka pots → stove',
    'open_the_middle_drawer_of_the_cabinet':'Open middle drawer',
    'push_the_plate_to_the_front_of_the_stove':'Push plate forward',
    'pick_up_the_alphabet_soup_and_place_it_in_the_basket':'Alphabet soup → basket',
    'pick_up_the_bbq_sauce_and_place_it_in_the_basket':'BBQ sauce → basket',
    'pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate':'Bowl between objects → plate',
    'pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate':'Centre bowl → plate',
}


def figures(summary,common,output):
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.facecolor':'white'})
    colors={'stay':'#8693a1','local_accel':'#d28b26','ppo':'#1677b8'}
    names={'stay':'Stay','local_accel':'Local Accel','ppo':'PPO'}
    for cohort in ('A','B'):
        table=[r for r in summary['table'] if r['cohort']==cohort]
        tasks=sorted(set(r['task'] for r in table));ys=np.arange(len(tasks))
        lookup={(r['task'],r['method']):r for r in table}
        fig,axes=plt.subplots(1,3,figsize=(15,6.5),sharey=True,gridspec_kw={'width_ratios':[1.25,.8,1.]})
        for m,offset in zip(('stay','local_accel','ppo'),(-.22,0,.22)):
            axes[0].barh(ys+offset,[100*lookup[t,m]['success_rate'] for t in tasks],height=.21,color=colors[m],label=names[m])
        for m,offset in (('local_accel',-.12),('ppo',.12)):
            axes[1].scatter([lookup[t,m]['queries'] for t in tasks],ys+offset,color=colors[m],label=names[m])
            axes[2].barh(ys+offset,[100*lookup[t,m]['rescue_rate'] for t in tasks],height=.23,color=colors[m])
            axes[2].barh(ys+offset,[-100*lookup[t,m]['harm_rate'] for t in tasks],height=.23,color=colors[m],alpha=.45,hatch='//')
        axes[0].set_yticks(ys,[LABELS.get(t,t) for t in tasks]);axes[0].invert_yaxis()
        axes[0].set_xlim(0,105);axes[1].set_xlim(0,8.5)
        axes[0].set_title('Task success (%)');axes[1].set_title('Mean queried views');axes[2].set_title('Harm ← 0 → Rescue (% of pairs)')
        axes[2].axvline(0,color='#555',linewidth=.8)
        for a in axes:a.grid(axis='x',alpha=.15);a.set_axisbelow(True)
        axes[0].legend(loc='lower left',bbox_to_anchor=(0,1.06),ncol=3,frameon=False)
        fig.suptitle(f'{cohort}: complete fresh simulator execution | 300-step horizon',y=1.01,fontsize=16)
        fig.text(.02,.01,'A: 512 executions/method/task. B: 128. Paired seeds; tasks, not episodes, are the generalization units.\nQueries include the initial view. Hatched left bars: stay succeeded but search failed; right bars: stay failed but search succeeded.',fontsize=10)
        fig.tight_layout(rect=(0,.07,1,.96));fig.savefig(output/f'{cohort}-tasks.png',dpi=160,bbox_inches='tight');plt.close(fig)
    table=common['table'];tasks=sorted(set(r['task'] for r in table));lookup={(r['task'],r['method']):r for r in table}
    fig,axes=plt.subplots(1,3,figsize=(15,5.8),sharey=True)
    ys=np.arange(len(tasks))
    for ax,m in zip(axes,('stay','local_accel','ppo')):
        old=[100*lookup[t,m]['replay_success_rate'] for t in tasks]
        live=[100*lookup[t,m]['success_rate'] for t in tasks]
        for i,(a,b) in enumerate(zip(old,live)):ax.plot([a,b],[i,i],color='#aaa',linewidth=2)
        ax.scatter(old,ys,facecolors='white',edgecolors=colors[m],s=65,label='Replay: same 8 seeds')
        ax.scatter(live,ys,color=colors[m],s=25,label='Fresh execution')
        ax.set_title(names[m]);ax.set_xlim(-2,103);ax.set_xlabel('Success at common horizon (%)');ax.grid(axis='x',alpha=.15)
    axes[0].set_yticks(ys,[LABELS.get(t,t) for t in tasks]);axes[0].invert_yaxis()
    axes[0].legend(loc='lower left',bbox_to_anchor=(0,1.05),frameon=False)
    fig.suptitle('A: replay vs fresh execution, matched seeds AND time limit',y=1.03,fontsize=15)
    fig.text(.02,.01,'Common horizon = min(original suite limit, 300): spatial 220, object 280, goal/long tasks 300.\nAll 1,536 search paths, selections and query counts match. Residual execution differences are retained; B has no archived replay labels.',fontsize=10)
    fig.tight_layout(rect=(0,.08,1,.96));fig.savefig(output/'A-replay-common-horizon.png',dpi=160,bbox_inches='tight');plt.close(fig)
