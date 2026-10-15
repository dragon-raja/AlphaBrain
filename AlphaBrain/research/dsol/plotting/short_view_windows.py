"""Chinese matched-window success curves; no cross-noise protocol mixing."""
def plot(result,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for font in font_manager.findSystemFonts():
        if 'NotoSansCJK' in font:font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False})
    models=list(result['models']);fig,axes=plt.subplots(len(models),2,figsize=(12,5*len(models)),squeeze=False,sharey=True)
    for m,model in enumerate(models):
        for c,(split,label) in enumerate([('seen','Seen'),('heldout','Heldout')]):
            row=result['models'][model]['populations'][split];ax=axes[m,c]
            values=[w['success']*100 for w in row['windows']]
            ax.plot(range(1,6),values,'o-',color='#178776',label='短探测 Accel 选位')
            ax.axhline(row['stay']*100,color='#888888',ls='--',label='扰动不调整')
            ax.axhline(row['canonical']*100,color='#b78025',ls=':',label='已知规范安装')
            for k,v in enumerate(values,1):ax.annotate(f'{v:.2f}',(k,v),xytext=(0,8),textcoords='offset points',ha='center')
            title='官方 π0.5' if model=='official' else 'XR-1'
            ax.set_title(f'{title} · {label} · {row["initials"]}个初态');ax.set_xticks(range(1,6));ax.set_ylim(0,100)
            ax.set_xlabel('评分窗口（重规划次数）');ax.set_ylabel('完整任务成功率（%）');ax.grid(alpha=.16);ax.legend(fontsize=9)
    fig.suptitle('匹配对照：窗口1严格复现原初态选位，再增加早期执行信息',fontsize=15)
    fig.text(.5,.02,'每候选8条独立评分短探测；选位后复用原4次完整任务结果。不是执行途中换视角。\n五个窗口全部报告；未按Heldout挑选赢家。π0.5每次25动作步，XR-1每次16动作步。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.08,1,.95));fig.savefig(output/'matched_windows_zh.png',dpi=180);plt.close(fig)
