"""Figure primitives for audited P0 data. No IO, training or data selection."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

BLUE='#285FAD';TEAL='#198879';GRAY='#99A4AF';INK='#17304A';MUTED='#627184'


def space_axes(fig, rect, parameters, values, title, *, percentage=False, canonical=None):
    ax=fig.add_axes(rect,projection='3d')
    p=np.asarray(parameters);v=np.asarray(values)* (100 if percentage else 1)
    low,high=float(v.min()),float(v.max())
    if canonical is not None:
        reference=canonical*(100 if percentage else 1)
        low,high=min(low,reference),max(high,reference)
    if low==high:low,high=low-.01,high+.01
    norm=Normalize(low,high)
    cmap='viridis' if percentage else 'viridis_r'
    artist=ax.scatter(p[:,0],p[:,1],p[:,2],c=v,cmap=cmap,norm=norm,s=31,depthshade=False,alpha=.95)
    if canonical is not None:
        ax.scatter([0],[0],[1],marker='*',s=115,c=[canonical*(100 if percentage else 1)],cmap=cmap,norm=norm,edgecolor='black',linewidth=.6)
    ax.view_init(elev=24,azim=-57)
    ax.set_xlabel('方位角（°）',fontsize=8,labelpad=-1);ax.set_ylabel('仰角（°）',fontsize=8,labelpad=-1)
    ax.set_zlabel('半径倍率',fontsize=8,labelpad=-3);ax.tick_params(labelsize=7,pad=0)
    ax.set_title(title,fontsize=12,pad=0,color=INK)
    bar=fig.colorbar(artist,ax=ax,fraction=.035,pad=.05,shrink=.65)
    bar.ax.tick_params(labelsize=8)
    bar.set_label('成功率（%）' if percentage else '初态 Accel',fontsize=8)
    return ax


def grouped_success(ax, values, categories, labels=('规范训练','多视角训练')):
    x=np.arange(len(categories));colors=[BLUE,TEAL]
    for m in range(2):
        bars=ax.bar(x+(m-.5)*.33,100*np.asarray(values[m]),.33,color=colors[m],label=labels[m])
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{bar.get_height():.1f}',ha='center',fontsize=10,color=INK)
    ax.set_xticks(x,categories,fontsize=10);ax.set_ylim(0,100);ax.set_ylabel('完整任务成功率（%）',fontsize=11)
    ax.legend(frameon=False,fontsize=10)


def success_heatmap(ax, values, rows, columns, *, difference=False):
    v=100*np.asarray(values)
    lim=max(float(np.max(np.abs(v))),1.)
    artist=ax.imshow(v,aspect='auto',cmap='RdBu_r' if difference else 'viridis',
                     vmin=-lim if difference else 0,vmax=lim if difference else 100)
    ax.set_yticks(np.arange(len(rows)),rows,fontsize=9)
    ax.set_xticks(np.arange(len(columns)),columns,fontsize=9)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            color='white' if (abs(v[i,j])>lim*.6 if difference else v[i,j]<55) else INK
            ax.text(j,i,f'{v[i,j]:+.1f}' if difference else f'{v[i,j]:.1f}',ha='center',va='center',fontsize=9,color=color)
    return artist


def budget_curves(ax, summary, model):
    curves=[('local_accel','Accel 搜索',BLUE,'-'),('random_accel','随机＋Accel',BLUE,'--'),
            ('local_rgb_accel','RGB＋Accel 搜索',TEAL,'-'),('random_rgb_accel','随机＋RGB/Accel',TEAL,'--')]
    for method,label,color,style in curves:
        rows=summary['models'][model][method];budgets=sorted(map(int,rows))
        ax.plot(budgets,[100*rows[str(b)]['success']['mean'] for b in budgets],style,marker='o',markersize=3,color=color,label=label)
    ax.set_xlabel('已查询候选数（含起点）',fontsize=10);ax.set_ylabel('成功率（%）',fontsize=10)
    ax.set_xticks([1,4,8,16,32]);ax.legend(frameon=False,fontsize=8)
