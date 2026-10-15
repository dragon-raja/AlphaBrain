"""Chinese presentation of retrospective, disjoint-noise temporal checks."""
def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for font in font_manager.findSystemFonts():
        if 'NotoSansCJK' in font:
            font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    for ax, split, title in zip(axes, ('seen', 'heldout'), ('Seen：20个初态', 'Heldout：20个初态')):
        rows = [result['results'][str(k)]['populations'][split] for k in (1, 3, 5)]
        baseline = rows[0]['baseline']
        vals = [baseline['stay'], baseline['canonical'], baseline['initial_accel8'], *[r['success'] for r in rows]]
        ax.bar(range(6), [v*100 for v in vals], color=['#999999', '#b98630', '#4268a8', '#188675', '#188675', '#188675'])
        ax.set_xticks(range(6), ['扰动\n不调整', '已知规范\n安装', '初态Accel\n8次评分', '执行前1次\n重规划', '执行前3次\n重规划', '执行前5次\n重规划'])
        for i,v in enumerate(vals): ax.text(i, v*100+1, f'{v*100:.2f}', ha='center', fontsize=10)
        ax.set_ylim(0, 100); ax.set_title(title); ax.set_ylabel('完整任务成功率（%）')
        ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    fig.suptitle('XR-1：更多早期执行信息，是否让最低Accel选位更可靠？', fontsize=15)
    fig.text(.5, .035, '绿色柱：2组执行噪声算分选位，另2组检验成败；全部6种划分平均。\n使用候选执行分支的事后记录，不是初态可部署方法，也不是实际移动相机实验。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .12, 1, .93))
    fig.savefig(output/'temporal_diagnostic_zh.png', dpi=180); plt.close(fig)
