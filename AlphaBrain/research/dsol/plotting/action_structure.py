"""Scientific figures from the frozen offline action audit."""
import numpy as np

PANELS=('libero_canonical','libero_broad','plus_canonical','plus_broad')
NAMES={'libero_canonical':'LIBERO 规范模型','libero_broad':'LIBERO 多视角模型','plus_canonical':'Plus 规范模型','plus_broad':'Plus 多视角模型'}
LABELS={'close_fraction':'预测闭合比例','gripper_switch':'预测开合切换比例','gripper_variation':'夹爪命令变化','rotation_magnitude':'旋转命令大小','rotation_variation':'旋转命令相邻变化','rotation_reversal':'旋转命令反向比例'}
COLORS={'improved':'#14836e','degraded':'#c04b45','equal':'#939da8','unknown':'#8356a8'}


def view_label(view):
    return view if len(view)<27 else view[:12]+'…'


def style(font_file):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(font_file))
    font=font_manager.FontProperties(fname=str(font_file))
    plt.rcParams.update({'font.family':font.get_name(),'font.size':9,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})


def finish(fig,path):
    import matplotlib.pyplot as plt
    fig.savefig(path,dpi=155,bbox_inches='tight');plt.close(fig)


def design(path):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,4));ax.axis('off')
    blocks=[('已有输入','128份原动作/去噪归档\n4面板 × 32初态记录\n独立历史闭环标签'),('身份与复现','原始字节哈希\n原分数 / 通道逐值复现\nW1 / W3 / W5赢家不变'),('固定结构分解','前5步与完整10步\n开合、旋转6项描述量\n不改Accel、不拟合权重'),('正反例审计','原窗口切换逐初态配对\n各面板 / 去规范候选\n任务剔除与缺失项'),('证据交付','总体曲线 + 逐例过程\n全部记录与可下载表格\n保留 / 放弃假设')]
    for i,(title,body) in enumerate(blocks):
        x=.01+i*.20
        ax.text(x+.085,.72,title,ha='center',va='center',fontsize=12,weight='bold',bbox=dict(boxstyle='round,pad=.7',facecolor='#dfebf2',edgecolor='#a5bbcc'))
        ax.text(x+.085,.40,body,ha='center',va='center',fontsize=9,linespacing=1.8)
        if i<4:ax.annotate('',xy=(x+.20,.72),xytext=(x+.17,.72),arrowprops=dict(arrowstyle='->',color='#536a7b'))
    ax.text(.5,.08,'本轮模型调用 = 0　｜　动作 / 仿真步进 / 渲染 = 0　｜　输出写入新目录；历史文件只读',ha='center',fontsize=11,color='#245c53')
    finish(fig,path)


def transitions(rows,path):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(13,6),sharey=True,constrained_layout=True)
    for col,panel in enumerate(PANELS):
        for row,mode in enumerate(('include','exclude')):
            ax=axes[row,col]
            for x,(start,end) in enumerate(((1,3),(3,5))):
                rr=[r for r in rows if (r['panel'],r['mode'],r['start'],r['end'])==(panel,mode,start,end)]
                bottom=0
                for label in COLORS:
                    n=sum(r['outcome']==label for r in rr)
                    ax.bar(x,n,bottom=bottom,color=COLORS[label],label=label if x==0 else None)
                    if n:ax.text(x,bottom+n/2,str(n),ha='center',va='center',color='white',fontsize=10)
                    bottom+=n
            ax.set_title(NAMES[panel]+('\n保留规范' if row==0 else '\n排除规范'))
            ax.set_xticks([0,1],['W1→W3','W3→W5']);ax.set_ylim(0,35);ax.grid(axis='y',alpha=.2)
    axes[0,0].set_ylabel('初态记录数（每格32）');axes[1,0].set_ylabel('包含不换候选的相等项')
    axes[0,-1].legend(fontsize=7,loc='upper right');finish(fig,path)


def concordance(groups,path):
    import matplotlib.pyplot as plt
    keys=[s+'_'+f for s in ('near','full') for f in LABELS]
    selected=[next(g for g in groups if g['panel']==p and g['mode']==m and g['start']==3) for m in ('include','exclude') for p in PANELS]
    data=np.array([[g['features'][k]['concordance'] if g['features'][k]['concordance'] is not None else np.nan for g in selected] for k in keys])
    fig,ax=plt.subplots(figsize=(12,7),constrained_layout=True);im=ax.imshow(data,vmin=0,vmax=1,cmap='RdBu_r',aspect='auto')
    for i in range(len(keys)):
        for j,g in enumerate(selected):
            a=data[i,j];overlap=g['features'][keys[i]]['ranges_overlap']
            ax.text(j,i,'缺失' if not np.isfinite(a) else f'{a:.2f}'+(' *' if overlap is False else ''),ha='center',va='center',fontsize=8,color='white' if abs(a-.5)>.32 else 'black')
    ax.set_yticks(range(len(keys)),[('前5步 ' if k.startswith('near') else '完整10步 ')+LABELS[k.split('_',1)[1]] for k in keys])
    ax.set_xticks(range(8),[NAMES[g['panel']]+('\n保留规范' if g['mode']=='include' else '\n排除规范')+f"\n正{g['changed_outcomes'].get('improved',0)} / 负{g['changed_outcomes'].get('degraded',0)}" for g in selected],rotation=25,ha='right')
    ax.set_title('W3→W5：新增阶段的候选差能否区分改善和退化？\nP(改善样本特征差 > 退化样本特征差)；0.5无排序方向；*仅表示当前样本值域不重叠',pad=14)
    fig.colorbar(im,ax=ax,label='描述性排序概率；不是新规则准确率',shrink=.7);finish(fig,path)


def distributions(rows,path):
    import matplotlib.pyplot as plt
    features=('near_gripper_variation','near_rotation_magnitude','near_rotation_variation')
    fig,axes=plt.subplots(4,3,figsize=(12,10),constrained_layout=True)
    for row,panel in enumerate(PANELS):
        rr=[r for r in rows if r['panel']==panel and r['mode']=='include' and r['start']==3 and r['changed']]
        for col,key in enumerate(features):
            ax=axes[row,col]
            for label in ('improved','degraded','equal'):
                values=sorted(r['added_delta'][key] for r in rr if r['outcome']==label and r['added_delta'][key] is not None)
                if values:
                    ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',color=COLORS[label],label=label+f' n={len(values)}')
                    ax.scatter(values,np.arange(1,len(values)+1)/len(values),s=12,color=COLORS[label])
            ax.axvline(0,color='grey',lw=.6);ax.grid(alpha=.2);ax.legend(fontsize=7)
            ax.set_title(NAMES[panel]+' / '+LABELS[key[5:]])
            ax.set_xlabel('新候选−旧候选；新增评分阶段平均');ax.set_ylabel('经验累积分布')
    finish(fig,path)


def case_plot(case,pair,path):
    import matplotlib.pyplot as plt
    candidates={c['view']:c for c in case['candidates']};a=candidates[pair['old_view']];b=candidates[pair['new_view']]
    fig,axes=plt.subplots(3,3,figsize=(13,9),constrained_layout=True)
    steps=np.array([0,5,10,15,20]);names=[view_label(pair['old_view']),view_label(pair['new_view'])];colors=['#2775a1','#dc8737']
    def band(ax,x,y,color,label):
        y=np.asarray(y,float);ax.plot(x,y.mean(0),color=color,label=label);ax.fill_between(x,y.min(0),y.max(0),alpha=.12,color=color)
    for candidate,color,name in zip((a,b),colors,names):
        s=np.asarray(candidate['scores']);band(axes[0,0],steps,s,color,name)
        band(axes[0,1],steps,np.cumsum(s,axis=1)/np.arange(1,6),color,name)
        for ax,key in zip(axes[1],('near_rotation_variation','near_gripper_variation','near_close_fraction')):
            band(ax,steps,candidate['features'][key],color,name)
        for ax,w,key in ((axes[2,0],2,'rotation'),(axes[2,1],4,'rotation'),(axes[2,2],4,'gripper')):
            band(ax,np.arange(1,11),np.asarray(candidate[key])[:,w],color,name);ax.axvspan(.5,5.5,color='#7f929e',alpha=.05)
    for channel in ('translation','rotation','gripper'):
        axes[0,2].plot(steps,pair['channel_gap_by_stage'][channel],'o-',label=channel)
    axes[0,2].plot(steps,pair['score_gap_by_stage'],'k--',label='总分差');axes[0,2].axhline(0,color='grey',lw=.7)
    titles=['各阶段原Accel','累计Accel（原选择依据）','新−旧候选通道贡献','前5步旋转命令相邻变化','前5步夹爪命令变化','前5步预测闭合比例','第3次预测：旋转命令范数','第5次预测：旋转命令范数','第5次预测：夹爪命令']
    for i,ax in enumerate(axes.flat):
        ax.set_title(titles[i],fontsize=10);ax.grid(alpha=.2)
        ax.set_xlabel('已执行动作步（各自分支）' if i<6 else '预测块内位置（阴影为前5步）')
        if i<6:ax.set_xticks(steps)
    axes[0,0].legend(fontsize=7);axes[0,2].legend(fontsize=7)
    axes[1,2].set_ylim(-.03,1.03)
    label=lambda x:'缺失' if x is None else f'{x:.3f}'
    fig.suptitle(f"{NAMES[case['panel']]} · 初态{case['initial']:02d} · {case['task']}\nW{pair['start']}→W{pair['end']}：{names[0]} → {names[1]}；历史成功率 {label(pair['old_success'])} → {label(pair['new_success'])}",fontsize=12)
    finish(fig,path)


def denoising_plot(case,pair,path):
    import matplotlib.pyplot as plt
    candidates={c['view']:c for c in case['candidates']};fig,axes=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
    for row,w in enumerate((2,4)):
        for name,color in zip((pair['old_view'],pair['new_view']),('#2775a1','#dc8737')):
            c=candidates[name]
            for col,channel in enumerate(('translation','rotation','gripper')):
                data=np.sqrt(np.asarray(c['denoise_channel_energy'][channel])[:,w]);ax=axes[row,col]
                ax.plot(np.arange(1,10),data.mean(0),'o-',color=color,label=view_label(name))
                ax.fill_between(np.arange(1,10),data.min(0),data.max(0),alpha=.12,color=color)
                ax.axvspan(.5,2.5,color='#83a4b6',alpha=.12)
                ax.set_title(f'第{w+1}次预测 / {channel}');ax.set_xlabel('相邻去噪输出对（1–2,…,9–10）');ax.set_ylabel('该通道去噪增量范数');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.suptitle('去噪中间过程：原Accel只使用阴影中的前两次差分；全程展示只作诊断',fontsize=12)
    finish(fig,path)


def associations(rows,path):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True,constrained_layout=True)
    for ax,key in zip(axes,('near_rotation_magnitude','near_rotation_variation','near_gripper_variation')):
        for panel in PANELS:
            data=[[r['rho'] for r in rows if r['panel']==panel and r['feature']==key and r['stage']==stage and r['rho'] is not None] for stage in range(1,6)]
            ax.plot(range(1,6),[np.median(x) if x else np.nan for x in data],'o-',label=NAMES[panel])
        ax.axhline(0,color='grey',lw=.7);ax.set_ylim(-1,1);ax.set_xticks(range(1,6));ax.set_xlabel('评分阶段');ax.set_title(LABELS[key[5:]]+' ↔ 对应通道贡献');ax.grid(alpha=.2)
    axes[0].set_ylabel('每初态候选间Spearman相关的中位数');axes[-1].legend(fontsize=7)
    finish(fig,path)
