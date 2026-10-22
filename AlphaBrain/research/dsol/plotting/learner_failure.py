"""Figures from frozen learner attribution tables; no model execution."""
import numpy as np


def label_stability(rows, destination):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),sharex=True,sharey=True,constrained_layout=True)
    for ax,method,title in zip(axes,('geometry_context_ridge','image_queryable_ridge','statewise_P_top1'),
                              ('几何＋上下文学习器','增加候选图的学习器','用 O/P 任务结果搜索')):
        rr=[r for r in rows if r['method']==method]
        x=np.array([r['O']['gain'] for r in rr])*100;y=np.array([r['Q']['gain'] for r in rr])*100
        ax.scatter(x,y,c=np.where(y<0,'#c75249',np.where(y>0,'#258978','#888')),s=48,alpha=.65)
        ax.plot([-17,52],[-17,52],':',color='#bbb');ax.axhline(0,color='#777',lw=.8);ax.axvline(0,color='#777',lw=.8)
        ax.set_title(title);ax.set_xlabel('O 批次相对参考视角增益（百分点）');ax.grid(alpha=.15)
        ax.set_xlim(-17,52);ax.set_ylim(-17,52)
        ax.text(.03,.96,f'16 个状态（重叠点未抖动）\nO 均值 {x.mean():+.2f} → Q {y.mean():+.2f}',transform=ax.transAxes,va='top',fontsize=9)
    axes[0].set_ylabel('独立 Q 批次增益（百分点）')
    fig.suptitle('标签稳定性：好坏是否跨执行噪声保持？\n右图候选由 O/P 选出，O 上的高值包含选择偏差',fontsize=13)
    fig.savefig(destination,dpi=160);plt.close(fig)


def decomposition(panels,destination):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True,sharex=True,sharey=True)
    for ax,(title,rows,model) in zip(axes.flat,panels):
        rr=sorted([r for r in rows if r['model']==model and r['method']=='local_rgb_accel'],key=lambda r:r['budget'])
        b=[r['budget'] for r in rr]
        for key,label,color in [('actual','冻结学习器实际选位','#287ba5'),('cross_best_visited','已访问中选点：分半交叉评估','#c58a35'),('cross_best_all','全部候选选点：分半交叉评估','#258978')]:
            ax.plot(b,[r['metrics'][key]['mean']*100 for r in rr],'o-',color=color,label=label)
        ax.set_title(title);ax.set_xticks(b);ax.set_ylim(0,100);ax.grid(alpha=.15);ax.set_xlabel('查询视角数')
    axes[0,0].legend(fontsize=8,loc='lower right');axes[0,0].set_ylabel('成功率（%）');axes[1,0].set_ylabel('成功率（%）')
    fig.suptitle('搜索没找到，还是找到后排错？\n橙、绿线用一半旧标签选点、另一半评价后交换；仅作诊断，非部署方法',fontsize=13)
    fig.savefig(destination,dpi=160);plt.close(fig)


def cross_scores(panels,destination):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(panels),figsize=(4*len(panels),4.3),constrained_layout=True)
    for ax,(title,rows,model) in zip(axes,panels):
        matrix=np.array([[next(r['success']['mean'] for r in rows if r['model']==model and r['budget']==16 and r['search']=='local_'+search and r['scorer']==scorer)*100 for scorer in ('accel','rgb','rgb_accel')] for search in ('accel','rgb','rgb_accel')])
        ax.imshow(matrix,cmap='YlGnBu',vmin=matrix.min()-1,vmax=matrix.max()+2)
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white' if matrix[i,j]>(matrix.min()+matrix.max())/2 else 'black')
        ax.set_xticks(range(3),['Accel','RGB','RGB＋Accel'],rotation=25);ax.set_yticks(range(3),['Accel','RGB','RGB＋Accel'])
        ax.set_title(title);ax.set_xlabel('同一访问集合，换评分器选终点')
    axes[0].set_ylabel('固定由哪种评分器引导的原搜索路径')
    fig.suptitle('预算 16：固定已经看过的图，再比较评分器（成功率 %）',fontsize=13)
    fig.savefig(destination,dpi=160);plt.close(fig)


def example_images(examples,destination):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(examples),3,figsize=(10,3.3*len(examples)),constrained_layout=True,squeeze=False)
    for row,example in enumerate(examples):
        e=example['event'];images=example['images']
        for col,(idx,label) in enumerate(zip((e['start'],e['selected'],e['best_visited']),('扰动起点','学习器实际选中','访问过的历史最高成绩候选'))):
            ax=axes[row,col];ax.imshow(images[idx]);ax.set_xticks([]);ax.set_yticks([])
            ax.set_title(f'{label}\n成功 {e["success"][col]*100:.1f}% · 最少实体像素 {e["external_min_pixels"][col]}',fontsize=10)
        axes[row,0].set_ylabel(f'{e["model"]} / 初态 {e["initial"]}\n'+('恢复正例' if e['stable_gain'] else '换位反例')+f'\n分半增益 {e["half_gain"][0]*100:+.1f}/{e["half_gain"][1]*100:+.1f} pp',fontsize=9)
    fig.suptitle('可见性正反例：每个模型展示分半同方向的最大改善和最大退化\n第三列按全部旧标签挑选，仅作解释；并非无偏最优值。完整 1024 个起点记录均保留。',fontsize=12)
    fig.savefig(destination,dpi=150);plt.close(fig)
