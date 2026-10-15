"""Small figure primitives for audited window comparisons."""
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def curves(ax, include, exclude=None, title='', limits=None):
    a = 100*np.asarray(include)
    ax.plot(range(1, 6), a[2:], 'o-', color='#245DA9', label='允许规范位', lw=1.8, ms=4)
    if exclude is not None:
        ax.plot(range(1, 6), 100*np.asarray(exclude)[2:], 's--', color='#198778',
                label='排除规范位', lw=1.5, ms=3)
    ax.axhline(a[0], color='#B8882B', ls='--', lw=1.2, label=f'不调整 {a[0]:.2f}%')
    ax.axhline(a[1], color='#8996A2', ls=':', lw=1.5, label=f'规范 {a[1]:.2f}%')
    for k, value in enumerate(a[2:], 1):
        ax.annotate(f'{value:.1f}', (k, value), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=8)
    ax.set_title(title, fontsize=12, pad=12); ax.set_xticks(range(1, 6))
    ax.set_xlim(.7, 5.3); ax.set_ylim(*(limits or (0, 105)))
    ax.tick_params(labelsize=8); ax.set_xlabel('累计评分窗口 H', fontsize=8)
    ax.set_ylabel('完整任务成功率（%）', fontsize=8)
    ax.grid(axis='y', color='#DEE5ED', lw=.6); ax.spines[['right', 'top']].set_visible(False)


def task_heatmap(ax, matrix, labels, columns, limit):
    a = np.asarray(matrix)
    im = ax.imshow(a, cmap='RdBu', norm=TwoSlopeNorm(0, -limit, limit), aspect='auto')
    ax.set_yticks(range(len(labels)), labels, fontsize=7.3)
    ax.set_xticks(range(len(columns)), columns, fontsize=7.3)
    ax.tick_params(length=0)
    ax.axvline(2.5, color='white', lw=2)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ax.text(j, i, f'{a[i,j]:+.1f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(a[i,j]) > .62*limit else '#142A42')
    return im


def transition_heatmap(ax, panel, labels, title):
    """Task-paired changes; consumes an audit, never recomputes choices."""
    columns = ('W1_W3', 'W3_W4', 'W3_W5')
    tasks = list(panel[columns[0]]['all']['per_task'])
    matrix = np.array([[100*panel[c]['all']['per_task'][t]['delta']
                        for c in columns] for t in tasks])
    limit = max(1., np.abs(matrix).max())
    im = ax.imshow(matrix, cmap='RdBu', norm=TwoSlopeNorm(0, -limit, limit), aspect='auto')
    ax.set_yticks(range(len(tasks)), [labels.get(t, t).rstrip('*') for t in tasks], fontsize=9)
    ax.set_xticks(range(3), ['W1→W3', 'W3→W4', 'W3→W5'], fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)
    for i in range(len(tasks)):
        for j in range(3):
            ax.text(j, i, f'{matrix[i,j]:+.2f}', ha='center', va='center', fontsize=9,
                    color='white' if abs(matrix[i,j])>.62*limit else '#142A42')
    return im


def paired_stage_margin(ax, row, title):
    """Positive margin favors the newly selected view, not necessarily success."""
    values = np.asarray(row['stage_old_minus_new'])[:row['end']]
    stages = range(1, row['end']+1)
    ax.bar(stages, values, color=['#198778' if v >= 0 else '#B8882B' for v in values])
    ax.axhline(0, color='#142A42', lw=.8)
    ax.axvline(row['start']+.5, color='#245DA9', ls='--', lw=1)
    ax.set_xticks(stages); ax.set_xlabel('单次评分阶段（不是累计窗口）', fontsize=9)
    ax.set_ylabel('旧候选 Accel − 新候选 Accel', fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.text(.02, .97, f"完整成功率：{100*row['old_success']:.2f}% → {100*row['new_success']:.2f}%",
            va='top', transform=ax.transAxes, fontsize=9)
    ax.spines[['right', 'top']].set_visible(False)
