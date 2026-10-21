"""Figures for the single native-feature mixed-budget search version."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .search_ppo import style


METHODS={'stay':('扰动不调整','#7b8794'),'local_accel':('局部 Accel','#2c8b68'),
         'random':('随机搜索','#d59832'),'ppo':('Attention PPO','#8a4ba5')}


def training_curves(updates):
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    panels=[('episode_mean_total_reward','搜索回报（不是任务成功率）'),
            ('episode_mean_queries','实际查询数：混合 4/8/16 上限'),
            ('train/value_loss','Critic 拟合误差'),('train/explained_variance','Critic 回报解释率')]
    for ax,(key,title) in zip(axes.flat,panels):
        for seed,rows in updates.items():
            valid=[r for r in rows if r.get(key) is not None]
            ax.plot([r['steps'] for r in valid],[r[key] for r in valid],label=f'seed {seed}')
        ax.set(title=title,xlabel='搜索决策步');ax.grid(alpha=.2)
    axes[0,1].set_ylim(.5,16.5);axes[1,1].axhline(0,color='gray',lw=.8)
    axes[0,0].legend(frameon=False)
    fig.suptitle('原生 SigLIP + 混合预算 + 特权 Critic：固定三种子，全部保留',fontsize=15)
    return fig


def replay_curves(summary):
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,(metric,title,multiplier) in zip(axes.flat,[('success','历史完整任务成绩上的选位回放',100),
                                               ('queries','实际查询数（含起点）',1),
                                               ('harmed_fraction','换位损失：占全部配对任务的比例',100)]):
        for method,(label,color) in METHODS.items():
            rows=sorted([r for r in summary['rows'] if r['method']==method],key=lambda r:r['budget'])
            x=[r['budget'] for r in rows];y=[r['metrics'][metric]['mean']*multiplier for r in rows]
            ax.plot(x,y,'o-',label=label,color=color)
            if metric=='success':
                ax.fill_between(x,[r['metrics'][metric]['ci95'][0]*100 for r in rows],
                                [r['metrics'][metric]['ci95'][1]*100 for r in rows],color=color,alpha=.08)
        ax.set(title=title,xlabel='查询预算',xticks=[4,8,16],ylabel='%' if multiplier==100 else '次数')
        ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False,fontsize=9)
    for method,(label,color) in METHODS.items():
        rows=sorted([r for r in summary['rows'] if r['method']==method],key=lambda r:r['budget'])
        x=[r['metrics']['queries']['mean'] for r in rows]
        y=[r['metrics']['success']['mean']*100 for r in rows]
        axes[1,1].plot(x,y,'o-',label=label,color=color)
        if method=='ppo':
            for a,b,row in zip(x,y,rows):axes[1,1].annotate(f"B={row['budget']}",(a,b),xytext=(4,4),textcoords='offset points')
    axes[1,1].set(title='成功率与实际查询成本',xlabel='实际查询数',ylabel='成功率（%）')
    axes[1,1].grid(alpha=.2)
    fig.suptitle('16 个评价初态 × 32 个扰动起点；新评分与旧任务标签配对，非新闭环成功率',fontsize=14)
    return fig


def live_task_figure(receipt):
    tasks=receipt['tasks'];fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    methods=('stay','local_accel','ppo')
    for j,method in enumerate(methods):
        rows=[r for r in tasks if r['method']==method]
        wins=sum(r['success'] for r in rows)
        axes[0].bar(j,100*wins/len(rows),color=METHODS[method][1])
        axes[0].text(j,100*wins/len(rows)+2,f'{wins}/{len(rows)}',ha='center')
    axes[0].set(xticks=range(3),xticklabels=[METHODS[m][0] for m in methods],ylim=(0,112),
                ylabel='成功率（%）',title='真实仿真完整任务：每方法 8 条')
    matrix=np.asarray([[next(r['success'] for r in tasks if r['initial']==i and r['method']==m and r['repeat']==s)
                        for s in range(4)] for i in (0,4) for m in methods],int)
    axes[1].imshow(matrix,cmap='RdYlGn',vmin=0,vmax=1,aspect='auto')
    for (i,j),v in np.ndenumerate(matrix):axes[1].text(j,i,'成功' if v else '失败',ha='center',va='center')
    axes[1].set(xticks=range(4),xticklabels=range(4),xlabel='配对执行种子',yticks=range(6),
                yticklabels=[task+' / '+METHODS[m][0] for task in ('奶酪放碗','上抽屉放碗') for m in methods],
                title='相同任务、初态和执行种子的逐条结果')
    fig.suptitle('两例工程验证：保留失败；不足以推断整体泛化收益',fontsize=15)
    return fig
