"""Question-led figures that relabel existing evidence without changing scores."""
import numpy as np


def choice_history(cases,destination):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(cases),figsize=(11,3.8),sharey=True,constrained_layout=True)
    for ax,case in zip(axes,cases):
        colors=['#287eaa' if view=='A' else '#d98235' for view in case['choices']]
        ax.bar(range(3),case['successes'],color=colors,width=.55)
        for i,(n,view) in enumerate(zip(case['successes'],case['choices'])):
            ax.text(i,n+.17,f'选视角{view}\n历史成功{n}/8',ha='center',fontsize=11)
        ax.set_xticks(range(3),['只看第1次评分\nW1','综合前3次评分\nW3','综合前5次评分\nW5'])
        ax.set_ylim(0,10.1);ax.set_yticks([0,2,4,6,8]);ax.set_title(case['title'],fontsize=13)
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('被选视角的历史完整任务成功次数（共8次）')
    fig.savefig(destination,dpi=160,bbox_inches='tight');plt.close(fig)


def cabinet_progress(rows,destination):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for view,label,color in [('extra_008','视角A：历史成功8/8','#287eaa'),('train_003','视角B：历史成功2/8','#d98235')]:
        rr=sorted([r for r in rows if r['case']=='xr1-39dd9788' and r['view']==view],key=lambda r:r['step'])
        x=[r['step'] for r in rr]
        values=[[100*r['distance_m']['obj_main']['mean'] for r in rr],
                [r['metrics']['baseline__score']['mean'] for r in rr],
                [r['metrics']['baseline_gripper__accel_contribution']['mean'] for r in rr]]
        for ax,y in zip(axes,values):ax.plot(x,y,'o-',label=label,color=color,lw=2)
    titles=['① A走得更接近物体','② 但A后期的评分反而升高','③ 升高主要出现在夹爪通道']
    ylabs=['末端到物体中心距离（厘米）','每次预测的Accel（越低越受原规则偏好）','夹爪通道对Accel的贡献']
    for ax,title,ylabel in zip(axes,titles,ylabs):
        ax.set_title(title,fontsize=12);ax.set_xlabel('原短轨迹已经执行的动作步');ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.set_xticks([0,16,32,48,64])
    axes[0].legend(fontsize=8);fig.savefig(destination,dpi=160,bbox_inches='tight');plt.close(fig)


def same_state_explained(cells,destination):
    import matplotlib.pyplot as plt
    y=np.asarray(cells).mean(0)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True,constrained_layout=True)
    titles=['固定：A原来走到的状态和历史','固定：B原来走到的状态和历史']
    for i,ax in enumerate(axes):
        ax.bar([0,1],y[i],color=['#287eaa','#d98235'],width=.55)
        for j,value in enumerate(y[i]):ax.text(j,value+.003,f'{value:.4f}',ha='center',fontsize=12)
        winner=int(np.argmin(y[i]));ax.text(.5,.91,'此时原规则更偏好视角'+('A' if winner==0 else 'B'),ha='center',fontsize=11,transform=ax.transAxes,color='#244b62')
        ax.set_xticks([0,1],['换成视角A观察','换成视角B观察']);ax.set_title(titles[i],fontsize=12)
        ax.set_ylim(0,.145);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Accel（越低越受原规则偏好）')
    fig.suptitle('入柜第48步：面对同一个状态时，换个视角会怎样？',fontsize=14)
    fig.savefig(destination,dpi=160,bbox_inches='tight');plt.close(fig)


def outcome_overlap(pairs,destination):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,panel,title in zip(axes,['libero_broad','plus_broad'],['LIBERO多视角模型：7个改善、7个退化','Plus多视角模型：只有1个改善、6个退化']):
        rr=[r for r in pairs if r['panel']==panel and r['mode']=='include' and r['start']==3 and r['changed']]
        for i,label in enumerate(('improved','degraded')):
            values=sorted(r['added_delta']['near_rotation_variation'] for r in rr if r['outcome']==label)
            offsets=np.linspace(-.12,.12,len(values)) if len(values)>1 else [0]
            ax.scatter(values,i+np.asarray(offsets),s=48,color=['#14836e','#c04b45'][i],edgecolors='white')
        ax.axvline(0,color='#555',ls='--',lw=1);ax.set_yticks([0,1],['换位后成功率提高','换位后成功率降低']);ax.set_ylim(1.4,-.4)
        ax.set_title(title,fontsize=11);ax.set_xlabel('新增阶段：新候选−旧候选的旋转命令变化量\n小于0：新候选更平滑；大于0：新候选变化更大',fontsize=9);ax.grid(axis='x',alpha=.2)
    fig.suptitle('如果“更平滑就是更好”，绿色和红色能分开吗？每个点代表一个初态',fontsize=13)
    fig.savefig(destination,dpi=160,bbox_inches='tight');plt.close(fig)
