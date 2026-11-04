"""Observed angular response plots; no fitted surface or invented success labels."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from ..analysis.target_search_diagnostic import endpoint_outcomes, pose_key


def figures(cases, training, output, font):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.backends.backend_pdf import PdfPages
    font_manager.fontManager.addfont(str(font))
    family=font_manager.FontProperties(fname=str(font)).get_name()
    out=Path(output);outcomes=endpoint_outcomes(cases)
    states=sorted({c['state_key'] for c in cases})
    scores=defaultdict(list)
    for c in cases:
        for q in c['queries_data']:
            scores[pose_key(c['state_key'],q['angles'])].append(q)
    colors=['#d95f02','#7570b3','#1b9e77']
    handles=[Line2D([],[],marker='o',linestyle='',color=c,label=f'{n}/2 次操作成功') for n,c in enumerate(colors)]
    handles+=[Line2D([],[],marker='.',linestyle='',color='#aaaaaa',label='只有评分，未测操作成败'),
              Line2D([],[],marker='*',linestyle='',color='black',label='原始参考位（仅右图；Accel未测）')]
    def decorate(ax):
        ax.set(xlim=(-31,31),ylim=(-21,21),xlabel='Yaw（度）',ylabel='Pitch（度）')
        ax.view_init(elev=27,azim=-60);ax.tick_params(labelsize=8)
    with plt.rc_context({'font.family':family,'axes.unicode_minus':False}), PdfPages(out/'fastwam-spatial-and-training.pdf') as pdf:
        for index,state in enumerate(states):
            fig=plt.figure(figsize=(14,7.8));a=fig.add_subplot(121,projection='3d');b=fig.add_subplot(122,projection='3d')
            rows=[(k,v) for k,v in scores.items() if k[0]==state]
            for k,v in rows:
                z=float(np.mean([np.mean(q['p3']) for q in v]));label=outcomes.get(k)
                success=None if label is None else label['success']
                color='#aaaaaa' if success is None else colors[int(round(2*success))]
                a.scatter(k[1],k[2],z,color=color,s=9 if success is None else 35,alpha=.3 if success is None else .9,depthshade=False)
            for k,v in outcomes.items():
                if k[0]!=state or v['success'] is None:continue
                reference=k[1:]==(0.,0.)
                b.scatter(k[1],k[2],v['success'],color='black' if reference else colors[int(round(2*v['success']))],
                          marker='*' if reference else 'o',s=160 if reference else 32,depthshade=False)
            decorate(a);decorate(b);a.set(zlabel='实测 Accel p3',title='评分空间：灰点的操作结果未知')
            b.set(zlabel='操作成功比例（仅2次）',zlim=(-.05,1.05),zticks=[0,.5,1],title='实际执行过的视角：没有插值曲面')
            fig.suptitle(f'FastWAM · 初态 {index+1}/{len(states)}\n{state.split("::")[-2]} / initial {state.split("::")[-1]}',fontsize=13)
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.08),ncol=3,frameon=False,fontsize=10)
            fig.text(.5,.025,'每页固定任务与初态；去重同角度/执行噪声。颜色不是可信成功概率。第三轴是评分或成功比例，不是相机高度。',ha='center',fontsize=10)
            fig.subplots_adjust(left=.02,right=.97,top=.81,bottom=.23,wspace=.18)
            fig.savefig(out/f'state-{index:02d}-3d.png',dpi=150);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
        methods=['source-first_batch_fixed','source-window','random-first_batch_fixed','random-window']
        for method,color in zip(methods,['#0072b2','#d55e00','#777777','#009e73']):
            rows=[r for r in training if r['run'].rsplit('-seed-',1)[0]==method]
            for ax,key in zip(axes.flat,['mean_reward','stop_probability','kl','mean_cost']):
                ys=np.array([[r[key] for r in rows if r['update']==u] for u in range(1,9)])
                ax.plot(range(1,9),ys.mean(1),label=method,color=color)
                ax.fill_between(range(1,9),ys.min(1),ys.max(1),color=color,alpha=.10)
        for ax,title in zip(axes.flat,['结算奖励（不是任务成功率）','训练状态的 STOP 概率','近似 KL（优化步骤均值）','查询成本（不含必需初始查询）']):
            ax.set(title=title,xlabel='更新次数（每次8回合）');ax.grid(alpha=.2)
        axes[0,0].axhline(0,color='black',lw=.6);axes[1,0].legend(fontsize=8)
        fig.suptitle('FastWAM 少量适配 · 全部四组/三种子；阴影为种子范围，不是置信区间')
        fig.savefig(out/'training-diagnostic.png',dpi=160);pdf.savefig(fig);plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,5.8),layout='constrained')
        for group,color in [('source-first_batch_fixed','#0072b2'),('source-window','#d55e00')]:
            xs=[0,1,4,8] if group.endswith('fixed') else [4,8];means=[]
            for u in xs:
                vals=[np.mean([e['success'] for c in cases if c['method']==f'actor:{group}-seed-{s}:u{u}' for e in c['executions']])*100 for s in range(3)]
                means.append(np.mean(vals));ax.scatter([u*8]*3,vals,s=28,color=color,alpha=.5)
            ax.plot(np.array(xs)*8,means,marker='o',color=color,label=group)
        ax.axhline(25,color='gray',ls='--',label='扰动保持 25%')
        ax.axhline(90.625,color='black',ls=':',label='已知参考位 90.63%（32条）')
        ax.set(xlabel='目标域适配回合数 / 每种子',ylabel='任务成功率（%）',ylim=(0,100),title='适配增加并未带来单调恢复；圆点为三个种子，折线为均值')
        ax.legend(fontsize=9);ax.grid(alpha=.2);fig.savefig(out/'success-learning-curve.png',dpi=160);pdf.savefig(fig);plt.close(fig)
    (out/'figures-index.json').write_text(__import__('json').dumps({f'state-{i:02d}-3d.png':s for i,s in enumerate(states)},indent=2)+'\n')


def model_comparison_figures(cases,comparison,output,font):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.backends.backend_pdf import PdfPages
    font_manager.fontManager.addfont(str(font));family=font_manager.FontProperties(fname=str(font)).get_name()
    outcomes=endpoint_outcomes(cases);groups=defaultdict(list)
    for c in cases:
        for q in c['queries_data']:groups[pose_key(c['state_key'],q['angles'])].append(np.mean(q['p3']))
    colors=['#d95f02','#7570b3','#1b9e77'];out=Path(output)
    handles=[Line2D([],[],marker='o',color=v,linestyle='',label=f'{i}/2 次成功') for i,v in enumerate(colors)]
    handles.append(Line2D([],[],marker='.',color='#aaaaaa',linestyle='',label='FastWAM仅有评分、操作未知'))
    with plt.rc_context({'font.family':family,'axes.unicode_minus':False}),PdfPages(out/'pi05-fastwam-spatial-comparison.pdf') as pdf:
        for i,state in enumerate(sorted({c['state_key'] for c in cases})):
            fig=plt.figure(figsize=(14,8));left=fig.add_subplot(121,projection='3d');right=fig.add_subplot(122,projection='3d')
            for r in comparison['pi05_panel']:
                if r['state_key']!=state:continue
                left.scatter(float(r['yaw_deg']),float(r['pitch_deg']),float(r['accel_selection']),
                             color=colors[int(r['success_repeats'])],s=45,depthshade=False)
            for k,v in groups.items():
                if k[0]!=state:continue
                success=outcomes.get(k,{}).get('success');color='#aaaaaa' if success is None else colors[int(round(success*2))]
                right.scatter(k[1],k[2],np.mean(v),color=color,s=8 if success is None else 35,alpha=.3 if success is None else .9,depthshade=False)
            for ax,title in [(left,'π0.5：27个固定视角，均有执行标签'),(right,'FastWAM：自适应搜索访问点，部分有标签')]:
                ax.set(xlabel='Yaw（度）',ylabel='Pitch（度）',zlabel='各自模型的 Accel p3',xlim=(-31,31),ylim=(-21,21),title=title)
                ax.view_init(elev=27,azim=-60);ax.tick_params(labelsize=8)
            fig.suptitle(f'相同任务与初态 {i+1}/16\n{state.split("::")[-2]} / initial {state.split("::")[-1]}',fontsize=13)
            fig.subplots_adjust(top=.81,bottom=.23,left=.02,right=.97,wspace=.18)
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.08),ncol=4,frameon=False)
            fig.text(.5,.025,'初态编号与视角匹配；初始化协议、评分尺度和覆盖未统一。竖轴不共用，不作模型架构因果对比或拟合响应面。',ha='center',fontsize=10)
            if i==0:fig.savefig(out/'pi05-fastwam-same-state.png',dpi=150)
            pdf.savefig(fig);plt.close(fig)
