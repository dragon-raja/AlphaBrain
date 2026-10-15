"""Plot CPU feasibility evidence; neither success-rate nor clearance claims."""
import numpy as np


def audit_figure(result):
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(14,6),layout='constrained')
    ax=fig.add_subplot(1,2,1,projection='3d')
    samples=result['tiago']['samples'];p=np.array([x['position'] for x in samples]);f=np.array([x['forward'] for x in samples]);q=np.array([[x['pan'],x['tilt']] for x in samples])
    col=ax.scatter(*p.T,c=np.rad2deg(q[:,0]),cmap='coolwarm',s=30)
    chosen=np.array([0,6,24,42,48]);ax.quiver(*p[chosen].T,*f[chosen].T,length=.11,arrow_length_ratio=.2,color='#244f74')
    points=np.vstack([p,p[chosen]+.11*f[chosen]]);c=(points.min(0)+points.max(0))/2;h=np.ptp(points,axis=0).max()*.58
    ax.set(xlim=(c[0]-h,c[0]+h),ylim=(c[1]-h,c[1]+h),zlim=(c[2]-h,c[2]+h),xlabel='机器人模型 X (m)',ylabel='Y (m)',zlabel='Z (m)',title='TIAGo原生头部：49组关节位姿\n相机安装不改，非头部正运动学位置不变')
    ax.set_box_aspect((1,1,1));ax.view_init(20,-60);fig.colorbar(col,ax=ax,shrink=.55,label='头部第一关节 (°)')
    bx=fig.add_subplot(1,2,2)
    rows=result['initial_point_audits'];valid=[r for r in rows if 'error' not in r]
    bx.barh(np.arange(len(valid)),[r['flagged_views'] for r in valid],color='#bd4e45')
    bx.set_yticks(np.arange(len(valid)),[r['task'] for r in valid],fontsize=9);bx.invert_yaxis()
    for i,r in enumerate(valid):bx.text(r['flagged_views']+.3,i,str(r['flagged_views']),va='center')
    bx.set(xlim=(0,max(r['flagged_views'] for r in valid)+5),xlabel='相机中心进入碰撞几何的候选组数 / 97',title='RoboCasa安装初筛：每任务一个预选seen初态\n零标记 ≠ 实体相机、支架及路径均可行')
    bx.grid(axis='x',alpha=.15)
    fig.suptitle('主动相机可行性：原生可动头部存在；原候选安装中也确有几何冲突',fontsize=15)
    return fig
