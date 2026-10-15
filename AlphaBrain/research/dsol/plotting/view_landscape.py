"""Measured rig-parameter fields with explicit separate color ranges."""
import numpy as np


def plot(panel, result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from .view_space import adaptive_limits, spatial_field
    # Discover system CJK fonts even when matplotlib's font cache predates them.
    for font in font_manager.findSystemFonts():
        if 'NotoSansCJK' in font:
            font_manager.fontManager.addfont(font)
    if 'Noto Sans CJK JP' not in {f.name for f in font_manager.fontManager.ttflist}:
        raise RuntimeError('Noto Sans CJK font required for Chinese research figures')
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
    names = {'official': '官方原始权重（未做我们的后训练）', 'canonical': '规范视角后训练（8250步）',
             'broad': '多视角后训练（8250步）', 'xr1': 'XR-1 官方权重（未做我们的后训练）'}
    split_names = {'all': '诊断子集：seen与heldout合并', 'pretrain': '诊断子集：seen初态',
                   'target': '诊断子集：heldout初态'}
    for split in ('all', 'pretrain', 'target'):
        keep = np.ones(len(panel['initials']), bool) if split == 'all' else panel['splits'] == split
        fig = plt.figure(figsize=(13, 4.3*len(panel['models'])))
        for m, name in enumerate(panel['models']):
            a = panel['accel'][m, keep, 1:].mean(0)
            q = panel['y'][m, keep, 1:].mean((0, 2))*100
            for col, (values, label, cmap) in enumerate(((a, 'Accel（8次评分噪声均值）', 'viridis_r'),
                                                        (q, '完整任务闭环成功率（%）', 'viridis'))):
                ax = fig.add_subplot(len(panel['models']), 2, m*2+col+1, projection='3d')
                limits = adaptive_limits(values, minimum_span=max(float(np.ptp(values))*.05, 1e-8),
                                         bounds=(0, float('inf') if col == 0 else 100))
                im = spatial_field(ax, panel['coordinates'], values, limits=limits, cmap=cmap)
                ax.view_init(elev=23, azim=-57)
                ax.set_xlabel('方位角偏移（度）');ax.set_ylabel('俯仰角偏移（度）')
                ax.set_zlabel('半径比例')
                r = result['splits'][split][name]['association']['spatial_mean_spearman']
                detail = f'平均 Accel = {values.mean():.4f}' if col == 0 else (
                    f'平均空间图排名相关 = {r:+.3f}' if r is not None else '空间相关未定义')
                ax.set_title(f'{names.get(name,name)}\n{label}；{detail}', fontsize=10)
                cb = fig.colorbar(im, ax=ax, shrink=.60, pad=.10)
                cb.ax.tick_params(labelsize=8)
        fig.suptitle(f'RoboCasa {len(panel["models"])}模型同初态对照｜{split_names[split]}\n{keep.sum()}个初态，每个视角4次完整闭环；不是原200个heldout初态总体', fontsize=13)
        fig.text(.5,.012,'96个实测扰动候选，未插值；坐标为相机组候选参数，不是世界XYZ；规范视角不画入空间。\n'
                 '各图独立色标：左图黄色=低Accel，右图黄色=高成功率。跨模型绝对大小请读色标，不直接比颜色。',
                 ha='center', fontsize=10)
        fig.subplots_adjust(top=.90,bottom=.07,hspace=.22,wspace=.16)
        fig.savefig(output/f'landscape_{split}.png',dpi=170);plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(14,5.2))
    for name in panel['models']:
        row = result['splits']['all'][name]
        axes[0].plot(range(1,5),100*np.asarray(row['quartiles']),'o-',label=names[name])
    axes[0].set_xticks(range(1,5),['最低组','次低组','次高组','最高组'])
    axes[0].set_xlabel('每初态内部按Accel排序后分四组（96个扰动候选）')
    axes[0].set_ylabel('完整任务闭环成功率（%）');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    rows = [result['splits']['all'][name]['per_task'] for name in panel['models']]
    values = np.array([[r['gain']*100 for r in model] for model in rows])
    bound = max(1,abs(values).max())
    im=axes[1].imshow(values,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto')
    axes[1].set_yticks(range(len(rows)),[names.get(n,n) for n in panel['models']])
    task_names = {'CloseBlenderLid':'关闭搅拌机盖', 'CloseFridge':'关闭冰箱', 'CoffeeSetupMug':'放置咖啡杯',
                  'NavigateKitchen':'厨房导航', 'OpenCabinet':'打开橱柜', 'OpenDrawer':'打开抽屉',
                  'PickPlaceCounterToCabinet':'台面物体放入橱柜', 'PickPlaceSinkToCounter':'水槽物体移至台面',
                  'TurnOffStove':'关闭炉灶', 'TurnOnMicrowave':'开启微波炉'}
    axes[1].set_xticks(range(len(rows[0])),[task_names.get(r['task'],r['task']) for r in rows[0]],rotation=65,ha='right',fontsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            axes[1].text(j,i,f'{values[i,j]:+.1f}',ha='center',va='center',fontsize=7)
    axes[1].set_title('最低Accel选位 − 32额外扰动视角平均（百分点）')
    fig.colorbar(im,ax=axes[1],shrink=.65)
    fig.suptitle('低值区域的关系 ≠ 最低点选位收益｜40初态诊断，seen与heldout合并')
    fig.tight_layout();fig.savefig(output/'regional_and_task_diagnostic.png',dpi=180);plt.close(fig)
