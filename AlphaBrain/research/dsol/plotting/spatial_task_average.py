"""Explicit piecewise-linear display surfaces over measured two-axis poses."""
from pathlib import Path
import numpy as np


def linear_figures(comparisons, output, *, font_path):
    """Show actual-value fits and p3/p9/success surfaces without changing selection."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font_path)).get_name(),
                         'axes.unicode_minus': False, 'font.size': 10})
    output = Path(output)
    names = {'fastwam_p3': 'FastWAM · p3', 'fastwam_p9': 'FastWAM · p9', 'pi05_matched': 'π0.5 · 原评分（匹配16初态）'}
    with PdfPages(output/'pearson-p3-p9-comparison.pdf') as pdf:
        fig, axes = plt.subplots(1, len(comparisons), figsize=(6*len(comparisons), 5.6))
        for ax, (key, result) in zip(np.atleast_1d(axes), comparisons.items()):
            x = np.asarray(result['mean']['score']); y = np.asarray(result['mean']['success'])*100
            stats = result['scopes']['nonreference']
            ax.scatter(x[1:], y[1:], s=40, c='#277da1', label='26个扰动位置（实际测量）')
            ax.scatter(x[0], y[0], s=130, c='#d62728', marker='*', label='参考位：不参与本图拟合')
            grid = np.linspace(x[1:].min(), x[1:].max(), 100)
            ax.plot(grid, (stats['raw_slope']*grid+stats['raw_intercept'])*100, c='#da7b28', label='线性最小二乘拟合')
            ax.set(xlabel='原始 Accel（横轴没有取负）', ylabel='任务等权平均操作成功率（%）', ylim=(-3, 103),
                   title=names.get(key, key)+f"\nr(−Accel, 成功率)={stats['favorable_r']:+.3f}；R²={stats['r_squared']:.3f}")
            ax.grid(alpha=.2); ax.ticklabel_format(axis='x', style='plain', useOffset=False)
            ax.legend(loc='lower left', fontsize=8)
        fig.suptitle('Pearson 线性相关：同8任务 × 2初态 × 27角度；每点2次操作', fontsize=15)
        fig.text(.5, .025, '曲线向右下降才支持低Accel对应高成功率；标题将Accel取负，使预期相关方向为正。拟合只使用26个实测扰动点。', ha='center')
        fig.subplots_adjust(left=.05, right=.99, top=.78, bottom=.17, wspace=.28)
        fig.savefig(output/'pearson-scatter.png', dpi=160, bbox_inches='tight'); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        fig = plt.figure(figsize=(18, 6.5))
        for index, key in enumerate(('fastwam_p3', 'fastwam_p9', 'success'), 1):
            result = comparisons['fastwam_p3' if key == 'success' else key]
            xy = np.asarray(result['angles_deg'])
            z = np.asarray(result['mean']['success' if key == 'success' else 'score']) * (100 if key == 'success' else 1)
            ax = fig.add_subplot(1, 3, index, projection='3d', computed_zorder=False)
            ax.plot_trisurf(mtri.Triangulation(xy[:, 0], xy[:, 1]), z, cmap='viridis', edgecolor='#666666', linewidth=.3, alpha=.7, zorder=1)
            ax.scatter(xy[:, 0], xy[:, 1], z, c='black', s=18, depthshade=False, zorder=3)
            ax.scatter(*xy[0], z[0], marker='*', c='#d62728', s=140, depthshade=False, zorder=4)
            ax.set(xlabel='Yaw（度）', ylabel='Pitch（度）', zlabel='成功率（%）' if key == 'success' else 'Accel',
                   title='同一批操作的成功率' if key == 'success' else names[key])
            ax.view_init(elev=27, azim=-60)
            if key == 'success': ax.set_zlim(0, 100)
        fig.suptitle('FastWAM：p3 / p9 与同一批成功率的任务平均空间场', fontsize=15)
        fig.text(.5, .055, '黑点=实测位置；红星=参考位；紫低黄高，颜色对应各面第三轴。三角面仅为线性插值。', ha='center')
        fig.text(.5, .02, 'p9由已存的同一10步去噪轨迹离线重算；无新增模型调用，无PPO或评分配置切换。', ha='center')
        fig.subplots_adjust(left=.01, right=.95, top=.82, bottom=.18, wspace=.16)
        fig.savefig(output/'p3-p9-success-surfaces.png', dpi=160, bbox_inches='tight'); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def figures(summary, output, *, font_path, task_labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({'font.family': family, 'axes.unicode_minus': False, 'font.size': 10})
    output = Path(output)
    xy = np.asarray(summary['angles_deg'])
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])

    def surfaces(data, title, rho, *, rank=False):
        fig = plt.figure(figsize=(14, 6.4))
        for index, key in enumerate(('rank' if rank else 'score', 'success'), 1):
            ax = fig.add_subplot(1, 2, index, projection='3d', computed_zorder=False)
            z = np.asarray(data[key]) * (100 if key == 'success' else 1)
            ax.plot_trisurf(tri, z, cmap='viridis', edgecolor='#666666', linewidth=.35,
                            alpha=.72, antialiased=True, zorder=1)
            ax.scatter(xy[:, 0], xy[:, 1], z, color='black', s=22, depthshade=False, zorder=3)
            ax.scatter(*xy[0], z[0], marker='*', color='#d62728', s=170, depthshade=False, zorder=4)
            ax.set(xlabel='Yaw 偏航角（度）', ylabel='Pitch 俯仰角（度）',
                   zlabel={'score': '原始 Accel p3 均值', 'rank': '初态内评分秩的均值（0低—1高）',
                           'success': '实测操作成功率（%）'}[key],
                   title='Accel 空间分布' if index == 1 else '完整操作成功率空间分布')
            ax.view_init(elev=27, azim=-60)
            if key == 'success': ax.set_zlim(0, 100)
            if key == 'rank': ax.set_zlim(0, 1)
        fig.suptitle(title + f'\n平均面相关：排除参考位 Spearman ρ = {rho:+.3f}；负值才支持低分对应高成功率', y=.98)
        fig.legend(handles=[Line2D([], [], marker='o', color='black', linestyle='', label='黑点：实际测量的27个位置'),
                            Line2D([], [], marker='*', color='#d62728', linestyle='', markersize=11, label='红星：参考位（0°, 0°）')],
                   loc='lower center', bbox_to_anchor=(.5, .065), ncol=2, frameon=False)
        fig.text(.5, .025, '三角面仅为测点间线性插值，无额外测量、无平滑拟合；第三轴为评分/成功率，不是相机高度。', ha='center')
        fig.subplots_adjust(left=.025, right=.94, bottom=.19, top=.80, wspace=.18)
        return fig

    with PdfPages(output/'task-average-spatial.pdf') as pdf:
        fig = surfaces(summary['mean'], f"FastWAM：{summary['tasks']}任务等权平均（{summary['states']}初态；每个角度合计{2*summary['states']}条操作）",
                       summary['rho_average_surface_nonreference'])
        fig.savefig(output/'task-average-surface.png', dpi=170, bbox_inches='tight'); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        fig = surfaces(summary['mean'], '辅助读图：先在各初态内排名，再任务等权平均（只用于展示，不改变算法）',
                       summary['rho_average_rank_surface_nonreference'], rank=True)
        fig.savefig(output/'task-average-rank-surface.png', dpi=170, bbox_inches='tight'); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        score, success = np.asarray(summary['mean']['score']), np.asarray(summary['mean']['success']) * 100
        axes[0].scatter(score[1:], success[1:], color='#277da1', s=42)
        axes[0].scatter(score[0], success[0], color='#d62728', marker='*', s=150, label='参考位')
        for i, (x, y) in enumerate(zip(score, success)): axes[0].annotate(str(i), (x, y), xytext=(4, 3), textcoords='offset points', fontsize=8)
        axes[0].set(xlabel='任务等权平均 Accel p3', ylabel='任务等权平均成功率（%）',
                    title=f"27个实测位置（数字为视角编号）\n排除参考位 ρ={summary['rho_average_surface_nonreference']:+.3f}")
        axes[0].legend(); axes[0].grid(alpha=.2)
        tasks = summary['per_task']
        vals = [t['rho_nonreference'] for t in tasks]
        axes[1].barh(range(len(tasks)), vals, color=['#278c75' if x < 0 else '#c56a49' for x in vals])
        axes[1].set_yticks(range(len(tasks)), [task_labels.get(t['task'], t['task']) for t in tasks], fontsize=9)
        axes[1].set(xlim=(-1, 1), xlabel='Spearman ρ：负值才支持低分对应高成功率', title='各任务：先平均2个初态，再计算26个扰动点相关')
        axes[1].axvline(0, color='black', lw=.8)
        for i, x in enumerate(vals): axes[1].text(x + (.025 if x >= 0 else -.025), i, f'{x:+.3f}', va='center', ha='left' if x >= 0 else 'right', fontsize=9)
        fig.suptitle('平均图与逐任务结论必须分开：平均曲面不能代表任一具体场景', fontsize=14)
        fig.text(.5, .025, '每任务每角度仅4条操作；相关均使用实测点，未把插值点当成新样本。无新增模型调用。', ha='center')
        fig.subplots_adjust(left=.07, right=.98, top=.83, bottom=.17, wspace=.85)
        fig.savefig(output/'task-correlations.png', dpi=170, bbox_inches='tight'); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        for i, task in enumerate(tasks, 1):
            fig = surfaces(task, task_labels.get(task['task'], task['task']) + f"（{task['states']}初态平均；每角度{2*task['states']}条操作）", task['rho_nonreference'])
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
