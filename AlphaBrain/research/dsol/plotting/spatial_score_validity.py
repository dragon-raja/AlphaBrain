"""Measured spatial points only; no interpolated success surfaces."""
from pathlib import Path
import textwrap
import numpy as np


def figures(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    output = Path(output)
    cmap = plt.get_cmap('viridis'); norm = Normalize(0, 1)
    with PdfPages(output/'fixed-spatial-maps.pdf') as pdf:
        for index, state in enumerate(sorted({r['state_key'] for r in rows})):
            values = sorted((r for r in rows if r['state_key'] == state), key=lambda r: r['view_id'])
            angles = np.asarray([r['angles_deg'] for r in values])
            success = np.asarray([np.mean(r['success']) for r in values])
            fig = plt.figure(figsize=(16, 5.5))
            for panel, score in enumerate(('p3', 'p9', None), 1):
                ax = fig.add_subplot(1, 3, panel, projection='3d')
                z = success if score is None else np.asarray([np.mean(r[score]) for r in values])
                ax.scatter(angles[:, 0], angles[:, 1], z, c=success, cmap=cmap, norm=norm,
                           s=55, edgecolors='black', linewidths=.3, depthshade=False)
                ref = next(i for i, r in enumerate(values) if r['view_id'] == 0)
                ax.scatter(*angles[ref], z[ref], marker='*', color='red', s=160, label='Reference view')
                ax.set(xlabel='Yaw (deg)', ylabel='Pitch (deg)',
                       zlabel='Observed success (2 repeats)' if score is None else f'Accel {score}',
                       title='Operation outcomes' if score is None else f'{score}: mean of 2 score members')
                if score is None: ax.set_zlim(0, 1); ax.set_zticks([0, .5, 1])
                ax.legend(loc='upper left', fontsize=8)
            fig.suptitle(textwrap.fill(state, 110), fontsize=10)
            fig.subplots_adjust(left=.02, right=.90, bottom=.15, top=.80, wspace=.20)
            cax = fig.add_axes([.93, .25, .012, .45])
            fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[0, .5, 1],
                         label='Success fraction: 0/2, 1/2, 2/2')
            fig.text(.03, .03, 'Fixed optical center; yaw/pitch rotations. Z is score or outcome, not camera height. No interpolation.', fontsize=9)
            pdf.savefig(fig)
            if index == 0: fig.savefig(output/'first-state.png', dpi=160)
            plt.close(fig)


def recovery_figures(rows, analysis, output, *, font_path, task_labels=None):
    """P3 and operation outcomes at identical measured poses; all states shown."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.cm import ScalarMappable
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    output = Path(output)
    cmap = ListedColormap(['#65479a', '#279b9c', '#f1cc37'])
    norm = BoundaryNorm([-.25, .25, .75, 1.25], 3)
    task_labels = task_labels or {}
    index = []
    with plt.rc_context({'font.family': family, 'axes.unicode_minus': False}), \
            PdfPages(output/'p3-spatial-recovery-all-states.pdf') as pdf:
        for number, item in enumerate(analysis['per_state'], 1):
            state = item['state_key']
            values = sorted((x for x in rows if x['state_key'] == state), key=lambda x: x['view_id'])
            angles = np.asarray([x['angles_deg'] for x in values])
            scores = np.asarray([np.mean(x['p3']) for x in values])
            sr = np.asarray([np.mean(x['success']) for x in values])
            ref = next(i for i, x in enumerate(values) if x['view_id'] == 0)
            argmin = next(i for i, x in enumerate(values) if x['view_id'] == item['argmin']['view_id'])
            low_ids = {x['view_id'] for x in item['ranked_nonreference'][:7]}
            low_indices = [i for i, x in enumerate(values) if x['view_id'] in low_ids]
            fig = plt.figure(figsize=(14, 7.4))
            for column, z in enumerate((scores, sr), 1):
                ax = fig.add_subplot(1, 2, column, projection='3d', computed_zorder=False)
                ax.scatter(angles[:, 0], angles[:, 1], z, c=sr, cmap=cmap, norm=norm,
                           s=65, edgecolors='#333333', linewidths=.5, depthshade=False, zorder=1)
                ax.scatter(angles[low_indices, 0], angles[low_indices, 1], z[low_indices],
                           s=145, facecolors='none', edgecolors='#315bc4', linewidths=1.2,
                           depthshade=False, zorder=2, label='蓝圈：非参考位中最低分7点')
                ax.scatter(*angles[ref], z[ref], marker='*', c='#e23f37', s=190,
                           depthshade=False, zorder=3, label='红星：参考位（不参与最低分选位）')
                ax.scatter(*angles[argmin], z[argmin], marker='X', c='black', s=145,
                           edgecolors='white', linewidths=.6, depthshade=False, zorder=4,
                           label='黑X：仅按 Accel 选中的最低分点')
                ax.text(*angles[argmin], z[argmin], f"  v{item['argmin']['view_id']}", fontsize=9)
                ax.set(xlabel='Yaw / 左右偏转（度）', ylabel='Pitch / 上下俯仰（度）',
                       zlabel='Accel p3（2成员均值）' if column == 1 else '完整操作成功次数 / 2',
                       title='同一批视角：评分地形（算法偏好低值）' if column == 1 else '同一批视角：实际操作结果')
                ax.zaxis.labelpad = 15
                ax.view_init(elev=26, azim=-58)
                if column == 2:
                    ax.set_zlim(-.03, 1.03); ax.set_zticks([0, .5, 1]); ax.set_zticklabels(['0/2', '1/2', '2/2'])
                else:
                    ax.ticklabel_format(axis='z', style='plain', useOffset=False)
            label = task_labels.get(item['task'], item['task'])
            fig.suptitle(f"S{number:02d}  {label} / 初态 {state.split('::')[-1]}\n"
                         f"扰动点平均 {100*item['perturbed_success']:.1f}%  |  "
                         f"最低分点 {item['argmin']['successes']}/2  |  "
                         f"参考位 {int(item['reference_success']*2)}/2", fontsize=13)
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.48, .11), ncol=1, fontsize=10)
            fig.subplots_adjust(left=.015, right=.89, bottom=.24, top=.84, wspace=.10)
            cax = fig.add_axes([.93, .34, .016, .34])
            colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[0, .5, 1])
            colorbar.ax.set_yticklabels(['0/2 失败', '1/2', '2/2 成功'])
            colorbar.set_label('两图颜色都表示实际操作结果')
            fig.text(.035, .055, '每个点都是实测位置；不插值成连续曲面。Z轴是评分或操作结果，不是相机高度。', fontsize=10)
            fig.text(.035, .025, '本图不使用 PPO 选位。每点仅2次操作；2/2不是已知的真实成功概率。', fontsize=10)
            filename = f'state-{number:02d}.png'
            fig.savefig(output/filename, dpi=150, bbox_inches='tight')
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
            index.append(dict(number=number, state_key=state, task_label=label, image=filename))
    return index


def native_panel_figures(comparisons, output, *, model_label):
    """Measured Pearson fields and piecewise-linear displays; no synthetic labels."""
    from pathlib import Path
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    names=list(comparisons)
    with PdfPages(output/'native-spatial-diagnostic.pdf') as pdf:
        fig,axes=plt.subplots(1,len(names),figsize=(6*len(names),4.8),squeeze=False)
        for ax,name in zip(axes[0],names):
            c=comparisons[name];x=np.asarray(c['mean']['score']);y=np.asarray(c['mean']['success'])*100
            ax.scatter(x[1:],y[1:],label='26 perturbed poses',s=36)
            ax.scatter(x[0],y[0],marker='*',s=180,color='darkorange',label='Canonical reference')
            scope=c['scopes']['nonreference'];xx=np.linspace(x[1:].min(),x[1:].max(),100)
            ax.plot(xx,100*(scope['raw_slope']*xx+scope['raw_intercept']),color='black',label='Nonreference linear fit')
            r=scope['favorable_r'];r='undefined' if r is None else f'{r:+.3f}'
            ax.set(xlabel=f'Task-equal mean Accel {name}',ylabel='Task-equal success (%)',
                   title=f'{model_label} / {name}\nPearson(-Accel, success) = {r}',ylim=(-3,103))
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(output/'pearson-scatter.png',dpi=160);pdf.savefig(fig);plt.close(fig)
        fig=plt.figure(figsize=(6*(len(names)+1),5.4))
        first=comparisons[names[0]];angles=np.asarray(first['angles_deg'])
        for i,name in enumerate(names+['success'],1):
            ax=fig.add_subplot(1,len(names)+1,i,projection='3d')
            z=np.asarray(first['mean']['success'])*100 if name=='success' else np.asarray(comparisons[name]['mean']['score'])
            limits=(0,100) if name=='success' else (float(z.min()),float(z.max()))
            surface=ax.plot_trisurf(angles[:,0],angles[:,1],z,cmap='viridis',vmin=limits[0],vmax=limits[1],alpha=.72,linewidth=.25,edgecolor='gray')
            ax.scatter(angles[:,0],angles[:,1],z,c='black',s=18,label='Measured poses')
            ax.scatter(*angles[0],z[0],c='darkorange',marker='*',s=140,label='Canonical reference')
            ax.set(xlabel='Yaw (degrees)',ylabel='Pitch (degrees)',zlabel='Success (%)' if name=='success' else 'Accel',title=name)
            ax.view_init(elev=25,azim=-65);ax.legend(fontsize=7,loc='upper left')
            fig.colorbar(surface,ax=ax,shrink=.5,pad=.08)
        fig.suptitle(model_label+' — task-equal measured response fields')
        fig.text(.5,.015,'Black points are measurements. Surfaces only join measured points; they are not additional evaluations.',ha='center',fontsize=10)
        fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(output/'accel-success-surfaces.png',dpi=160);pdf.savefig(fig);plt.close(fig)
