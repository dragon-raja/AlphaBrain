"""Model-neutral measured score/success surfaces and Pearson diagnostics."""
from pathlib import Path
import numpy as np


def figures(rows, comparisons, output, *, model_label, font_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font_path)).get_name(),
                         'axes.unicode_minus': False, 'font.size': 10})
    output = Path(output)
    scores = list(comparisons)
    reference = comparisons[scores[0]]
    xy = np.asarray(reference['angles_deg'])
    triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1])

    def surfaces(fields, title):
        fig = plt.figure(figsize=(18, 6.8))
        for i, (label, values) in enumerate(fields.items(), 1):
            ax = fig.add_subplot(1, len(fields), i, projection='3d', computed_zorder=False)
            z = np.asarray(values)
            surface = ax.plot_trisurf(triangulation, z, cmap='viridis', alpha=.72, vmin=float(z.min()), vmax=float(z.max()),
                                     edgecolor='#777777', linewidth=.3, zorder=1)
            ax.scatter(xy[:, 0], xy[:, 1], z, c='black', s=17, depthshade=False, zorder=3)
            ax.scatter(*xy[0], z[0], marker='*', c='#d62728', s=150, depthshade=False, zorder=4)
            ax.set(xlabel='Yaw 偏航角（度）', ylabel='Pitch 俯仰角（度）', zlabel=label, title=label)
            ax.view_init(elev=27, azim=-60)
            fig.colorbar(surface, ax=ax, shrink=.45, pad=.1)
            if label.startswith('成功'): ax.set_zlim(0, 100)
        fig.suptitle(title, fontsize=15)
        fig.legend(handles=[Line2D([], [], marker='o', color='black', linestyle='', label='黑点：实测位置'),
                            Line2D([], [], marker='*', color='#d62728', linestyle='', markersize=12, label='红星：参考位')],
                   loc='lower center', bbox_to_anchor=(.5, .075), ncol=2, frameon=False)
        fig.text(.5, .035, '颜色与第三轴一一对应（紫低黄高）；三角面仅为测点间线性插值，相关性只使用实测点。第三轴不是相机高度。', ha='center')
        fig.subplots_adjust(left=.01, right=.98, bottom=.18, top=.83, wspace=.12)
        return fig

    with PdfPages(output/'spatial-comparison.pdf') as pdf:
        fields = {f'Accel {p.upper()}': v['mean']['score'] for p, v in comparisons.items()}
        fields['成功率（%）'] = np.asarray(reference['mean']['success'])*100
        fig = surfaces(fields, f'{model_label}：任务等权平均空间场（{reference["tasks"]}任务 / {reference["states"]}初态 / 每视角2次操作）')
        fig.savefig(output/'task-average-surfaces.png', dpi=170, bbox_inches='tight')
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        fig, axes = plt.subplots(1, len(scores), figsize=(7*len(scores), 5.8), squeeze=False)
        for ax, (p, result) in zip(axes[0], comparisons.items()):
            x = np.asarray(result['mean']['score']); y = np.asarray(result['mean']['success'])*100
            stats = result['scopes']['nonreference']
            ax.scatter(x[1:], y[1:], s=45, c='#277da1', label='26个实测扰动点')
            ax.scatter(x[0], y[0], s=150, c='#d62728', marker='*', label='参考位：排除于拟合')
            grid = np.linspace(x[1:].min(), x[1:].max(), 100)
            ax.plot(grid, (stats['raw_slope']*grid+stats['raw_intercept'])*100, c='#da7b28', label='线性最小二乘拟合')
            r = stats['favorable_r']
            ax.set(xlabel=f'原始 Accel {p.upper()}', ylabel='任务等权平均成功率（%）', ylim=(-3, 103),
                   title=f'{p.upper()}：Pearson(−Accel, 成功率) = '+('未定义' if r is None else f'{r:+.3f}'))
            ax.ticklabel_format(axis='x', style='plain', useOffset=False)
            ax.grid(alpha=.2); ax.legend(fontsize=9)
        fig.suptitle(f'{model_label}：平均空间场的线性相关（不代表单一初态的选位效果）', fontsize=14)
        fig.text(.5, .025, '横轴为原始Accel，向右下降才符合低分/高成功率；标题以负Accel计算使预期方向为正。P4仅离线诊断。', ha='center')
        fig.subplots_adjust(top=.8, bottom=.17, wspace=.25)
        fig.savefig(output/'pearson-scatter.png', dpi=170, bbox_inches='tight'); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        for state in sorted({r['state_key'] for r in rows}):
            values = sorted((r for r in rows if r['state_key'] == state), key=lambda r: r['view_id'])
            fields = {f'Accel {p.upper()}': [np.mean(r[p]) for r in values] for p in scores}
            fields['成功率（%）'] = [100*np.mean(r['success']) for r in values]
            fig = surfaces(fields, model_label+'：'+state.split('::')[-2]+' / 初态 '+state.split('::')[-1])
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    return ['task-average-surfaces.png', 'pearson-scatter.png', 'spatial-comparison.pdf']
