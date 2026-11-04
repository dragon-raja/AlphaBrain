"""Observed continuous search endpoints and paths; never interpolate success."""
from pathlib import Path
import numpy as np
from ..analysis.recovery_geometry import rays
from ..data.native_view_panel import views


def explanatory_figures(records, output, font_path):
    """Chinese reading guide with discrete outcome legends and explicit units."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.backends.backend_pdf import PdfPages
    font_manager.fontManager.addfont(str(font_path))
    family=font_manager.FontProperties(fname=str(font_path)).get_name()
    output=Path(output);panel=np.array(views());colors=['#d55e00','#0072b2','#009e73']
    groups=[[r for r in records if r['split']=='unseen_test' and r['method']==f'B-seed-{s}'] for s in range(3)]
    handles=[Line2D([],[],marker='o',linestyle='',color=c,label=f'{i}/2 次操作成功') for i,c in enumerate(colors)]
    handles += [Line2D([],[],marker='*',linestyle='',color='black',markersize=13,label='规范方向：仅事后对照'),
                Line2D([],[],marker='x',linestyle='',color='#777777',label='基础评测的 27 个固定方向')]
    with plt.rc_context({'font.family':family,'axes.unicode_minus':False}), PdfPages(output/'view-guide-zh.pdf') as pdf:
        fig,axes=plt.subplots(1,3,figsize=(15,6))
        fig.subplots_adjust(left=.055,right=.98,top=.78,bottom=.26,wspace=.23)
        for seed,(ax,rows) in enumerate(zip(axes,groups)):
            pos=np.array([r['selected_angles'] for r in rows]);success=np.array([sum(r['success']) for r in rows])
            ax.scatter(panel[:,0],panel[:,1],color='#777777',marker='x',s=35,alpha=.65)
            for count,color in enumerate(colors):
                points=pos[success==count];ax.scatter(points[:,0],points[:,1],color=color,s=14,alpha=.48,edgecolors='none')
            ax.scatter([0],[0],color='black',marker='*',s=180,zorder=5)
            ax.axhline(0,color='gray',linewidth=.6,alpha=.35);ax.axvline(0,color='gray',linewidth=.6,alpha=.35)
            ax.set(xlim=(-32,32),ylim=(-22,22),xlabel='左右转角 yaw（度；正=右，负=左）',
                   ylabel='上下转角 pitch（度；正=抬头，负=低头）',title=f'B 种子 {seed} · 全部 960 次搜索')
            ax.grid(alpha=.2)
        fig.suptitle('先看这个二维投影：策略最终把相机朝向了哪里？',fontsize=17,y=.96)
        fig.text(.5,.875,'每个彩色圆点 = 一次搜索的最终方向；三列分别是三个独立训练种子。',ha='center',fontsize=12)
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.105),ncol=5,frameon=False,fontsize=10)
        fig.text(.5,.035,'颜色不是 Accel，也不是可靠成功概率。黑星是事后标注；当前策略的机械坐标输入仍可能形成回中心先验。',ha='center',fontsize=11)
        fig.savefig(output/'view-guide-2d-zh.png',dpi=170);pdf.savefig(fig);plt.close(fig)
        for seed,rows in enumerate(groups):
            fig=plt.figure(figsize=(10,9));ax=fig.add_axes([.06,.24,.83,.58],projection='3d')
            ax.computed_zorder=False
            v=rays([r['selected_angles'] for r in rows]);v0=rays(panel);success=np.array([sum(r['success']) for r in rows])
            ax.scatter(*v0.T,color='#777777',marker='x',s=45,alpha=.65)
            for count,color in enumerate(colors):
                p=v[success==count];ax.scatter(*p.T,color=color,s=17,alpha=.55,depthshade=False)
            ax.scatter([0],[1],[0],color='black',marker='*',s=230,depthshade=False,zorder=10)
            ax.set(xlabel='向右的方向分量',ylabel='向前的方向分量',zlabel='向上的方向分量',
                   xlim=(-.55,.55),ylim=(.78,1.02),zlim=(-.37,.37))
            ax.set_yticks([.8,.9,1.]);ax.set_box_aspect((1.1,.24,.74));ax.view_init(elev=22,azim=-65)
            fig.suptitle(f'三维读图：B 种子 {seed} 的全部 960 个最终朝向',fontsize=17,y=.96)
            fig.text(.5,.89,'想象从相机中心沿每个朝向伸出一根等长的箭头，图中画的是这些箭头的尖端。',ha='center',fontsize=11)
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.10),ncol=2,frameon=False,fontsize=11)
            fig.text(.5,.035,'这些点不是相机的位置：相机光心固定，变化的只有左右、上下两个转角。\n黑星是规范朝向（0, 1, 0）；所有点均为已测搜索终点，没有插值。',ha='center',fontsize=11)
            fig.savefig(output/f'view-guide-3d-seed-{seed}-zh.png',dpi=160);pdf.savefig(fig);plt.close(fig)


def figures(records, baseline, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    output=Path(output); panel=np.asarray(views()); nominal=rays([[0,0]])[0]
    groups=[[r for r in records if r['split']=='unseen_test' and r['method']==f'B-seed-{s}'] for s in range(3)]
    cmap=plt.get_cmap('viridis'); norm=plt.Normalize(0,1)
    def decorate(ax):
        ax.set(xlabel='Yaw (deg)',ylabel='Pitch (deg)',xlim=(-32,32),ylim=(-22,22));ax.grid(alpha=.25)
    with PdfPages(output/'geometry.pdf') as pdf:
        fig=plt.figure(figsize=(18,6));fig.subplots_adjust(left=.02,right=.9,wspace=.32,top=.82,bottom=.1);axes=[]
        for seed,rows in enumerate(groups):
            ax=fig.add_subplot(1,3,seed+1,projection='3d');axes.append(ax)
            v=rays([r['selected_angles'] for r in rows]);v0=rays(panel)
            ax.scatter(*v0.T,c='gray',s=28,marker='x',alpha=.6)
            ax.scatter(*v.T,c=[np.mean(r['success']) for r in rows],cmap=cmap,norm=norm,s=9,alpha=.55,depthshade=False)
            ax.scatter(*nominal,c='red',marker='*',s=180,depthshade=False)
            ax.set(xlabel='Right',ylabel='Forward',zlabel='Up',title=f'B seed {seed}: {len(rows)} starts',xlim=(-.55,.55),ylim=(.78,1.02),zlim=(-.37,.37))
            ax.set_yticks([.8,.9,1.]);ax.tick_params(labelsize=8)
            ax.set_box_aspect((1.1,.24,.74));ax.view_init(elev=22,azim=-65)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.94,.2,.012,.52]),label='Success fraction (2 executions)')
        fig.suptitle('Unseen PPO tasks: relative optical-axis endpoints on unit sphere\nFixed camera centre; these are NOT camera positions. Gray crosses: 27 panel directions; red star: nominal.',fontsize=11)
        fig.savefig(output/'orientation-3d.png',dpi=170);pdf.savefig(fig);plt.close(fig)
        fig=plt.figure(figsize=(18,6));fig.subplots_adjust(left=.02,right=.9,wspace=.32,top=.82,bottom=.1);axes=[]
        zmax=max(r['independent_accel'] for rows in groups for r in rows)*1.03
        for seed,rows in enumerate(groups):
            ax=fig.add_subplot(1,3,seed+1,projection='3d');axes.append(ax);v=np.array([r['selected_angles'] for r in rows])
            ax.scatter(v[:,0],v[:,1],[r['independent_accel'] for r in rows],c=[np.mean(r['success']) for r in rows],cmap=cmap,norm=norm,s=8,alpha=.6,depthshade=False)
            decorate(ax);ax.set(zlabel='Independent Accel',title=f'B seed {seed}',zlim=(0,zmax));ax.view_init(elev=25,azim=-60)
            ax.set_zticks([0,.005,.01,.015,.02]);ax.tick_params(labelsize=8)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.94,.2,.012,.52]),label='Success fraction (2 executions)')
        fig.suptitle('Measured endpoints: yaw / pitch / independent Accel\nDifferent tasks and states pooled; not a fitted response surface or a causal effect estimate.',fontsize=11)
        fig.savefig(output/'score-3d.png',dpi=170);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
        bins=[np.linspace(-30,30,31),np.linspace(-20,20,21)]
        maximum=max(np.histogram2d(*np.array([r['selected_angles'] for r in rows]).T,bins=bins)[0].max() for rows in groups)
        for seed,(ax,rows) in enumerate(zip(axes,groups)):
            a=np.array([r['selected_angles'] for r in rows]);hist=ax.hist2d(a[:,0],a[:,1],bins=bins,cmap='Blues',vmin=0,vmax=maximum)
            ax.scatter(panel[:,0],panel[:,1],c='gray',s=15,marker='x');ax.scatter([0],[0],c='red',s=90,marker='*')
            decorate(ax);ax.set_title(f'B seed {seed}: endpoint counts')
        fig.colorbar(hist[3],ax=axes,label='Search endpoints / bin',shrink=.8)
        fig.suptitle('All 960 held-out-task starts per seed; bins show location density, NOT success probability',fontsize=10)
        fig.savefig(output/'angle-density.png',dpi=170);pdf.savefig(fig);plt.close(fig)
        # Outcome-independent coverage: first state in sorted identity per unseen task.
        for task in sorted({r['task'] for r in groups[0]}):
            state=min(r['state_key'] for r in groups[0] if r['task']==task)
            b=[r for r in baseline if r['state_key']==state]
            fig=plt.figure(figsize=(15,5.2),layout='constrained')
            for seed,group in enumerate(groups):
                rows=[r for r in group if r['state_key']==state];ax=fig.add_subplot(1,3,seed+1,projection='3d')
                ax.scatter([r['yaw_deg'] for r in b],[r['pitch_deg'] for r in b],[np.mean(r['success']) for r in b],c='gray',s=28,marker='x',alpha=.65)
                ax.scatter([r['selected_angles'][0] for r in rows],[r['selected_angles'][1] for r in rows],[np.mean(r['success']) for r in rows],c=[np.mean(r['success']) for r in rows],cmap=cmap,norm=norm,s=50,marker='^',depthshade=False)
                for r in rows:
                    a=np.array(r['query_path']);ax.plot(a[:,0],a[:,1],np.full(len(a),-.1),alpha=.35,linewidth=1)
                canon=next(r for r in b if r['view_id']==0);ax.scatter([0],[0],[np.mean(canon['success'])],c='red',s=100,marker='*')
                decorate(ax);ax.set(title=f'B seed {seed}',zlabel='Measured success fraction',zlim=(-.15,1.05))
            fig.suptitle(f'{task}\n{state.rsplit("::",1)[-1]}: gray = baseline; triangles = search endpoints; paths on floor show queried angles only',fontsize=9)
            pdf.savefig(fig);plt.close(fig)
