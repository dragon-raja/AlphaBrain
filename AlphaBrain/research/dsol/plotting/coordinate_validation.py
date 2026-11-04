"""Display the complete frozen decision; never infer or change pass criteria."""
from pathlib import Path


def summary_figure(summary, output):
    """Write a labelled, static comparison suitable for the research receipt."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    methods = ('G', 'stay', 'home', 'local', 'reference')
    labels = ('G (3 policy seeds)', 'Stay at perturbed start',
              'Mechanical home', 'Local Accel search', 'Known original reference')
    colors = ('#2369bd', '#777777', '#e49a24', '#36a578', '#9b6bb3')
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), constrained_layout=True)
    x = np.arange(3)
    for j, (method, label, color) in enumerate(zip(methods, labels, colors)):
        values = [summary['per_offset'][str(o)]['success_percent'][method]
                  for o in (-15, 0, 15)]
        bars = axes[0].bar(x + (j - 2) * .15, values, width=.145,
                           label=label, color=color)
        axes[0].bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
    axes[0].set(xticks=x, xticklabels=('-15 degrees', '0 degrees', '+15 degrees'),
                ylim=(0, 112), ylabel='Complete manipulation task success (%)',
                xlabel='Installation yaw offset (not a policy target)',
                title='All predeclared installation groups; equal task weights')
    axes[0].legend(loc='upper center', bbox_to_anchor=(.5, 1.25), ncol=3, fontsize=8)
    others = ('stay', 'home', 'local')
    for i, other in enumerate(others):
        contrast = summary['contrasts'][other]
        mid = contrast['difference_pp']
        lo, hi = contrast['paired_task_95_interval_pp']
        axes[1].plot((lo, hi), (i, i), color=colors[0], linewidth=3)
        axes[1].scatter([mid], [i], color=colors[0], s=45, zorder=3)
        axes[1].annotate(f'{mid:+.1f} pp [{lo:+.1f}, {hi:+.1f}]',
                         (hi, i), xytext=(7, 0), textcoords='offset points',
                         va='center', fontsize=9)
    axes[1].axvline(0, color='black', linewidth=.8)
    axes[1].set(yticks=range(3), yticklabels=('G - stay', 'G - mechanical home',
                                             'G - local Accel'),
                ylim=(-.6, 2.6), xlabel='Success difference (percentage points)',
                title='Primary population: nonzero offsets; paired 95% task bootstrap')
    axes[1].invert_yaxis()
    axes[1].margins(x=.3)
    axes[1].grid(axis='x', alpha=.2)
    fig.suptitle(summary['decision'] + '\n12 tasks / 36 initial states / 4 starts / 3 frozen G seeds',
                 fontsize=12)
    paths = []
    for suffix in ('png', 'pdf'):
        path = output / ('mechanism-summary.' + suffix)
        fig.savefig(path, dpi=170)
        paths.append(str(path))
    plt.close(fig)
    return paths


def endpoint_figure(searches, output):
    """Show every G endpoint in the declared two-axis orientation space."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [s for s in searches if s['method'].startswith('G-seed-')]
    if len(rows) != 1296:
        raise ValueError('Complete three-seed endpoint panel required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    colors = ('#2369bd', '#e49a24', '#36a578')
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), constrained_layout=True)
    for axis, offset in zip(axes, (-15., 0., 15.)):
        for seed, color in enumerate(colors):
            selected = [s for s in rows if s['method'].startswith(f'G-seed-{seed}-')
                        and s['mount_yaw_offset_deg'] == offset]
            if len(selected) != 144:
                raise ValueError('Incomplete endpoint seed/installation group')
            q = np.asarray([s['selected_angles'] for s in selected], dtype=float)
            axis.scatter(q[:, 0], q[:, 1], s=11, alpha=.32, color=color,
                         label=f'G seed {seed} (all endpoints)')
        starts = np.asarray([(offset + y, p) for y in (-10, 10) for p in (-10, 10)])
        axis.scatter(starts[:, 0], starts[:, 1], s=95, facecolors='none',
                     edgecolors='black', linewidths=1.3, label='Perturbed starts', zorder=4)
        axis.scatter([offset], [0], s=145, color='black', marker='*',
                     label='Mechanical home', zorder=5)
        axis.scatter([0], [0], s=100, facecolors='none', edgecolors='#b32185',
                     linewidths=2, marker='D', label='Known original reference', zorder=6)
        axis.axvline(offset - 15, color='gray', linestyle=':', alpha=.7)
        axis.axvline(offset + 15, color='gray', linestyle=':', alpha=.7)
        axis.set(xlim=(-32, 32), ylim=(-22, 22), xlabel='Physical camera yaw (degrees)',
                 ylabel='Physical camera pitch (degrees)', title=f'Installation yaw {offset:+.0f} degrees')
        axis.set_aspect('equal', adjustable='box')
        axis.grid(alpha=.15)
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=3, fontsize=8)
    fig.suptitle('All 1,296 frozen-G search endpoints; no success-based filtering\n'
                 'Two-axis orientation, not 3-D translation; proximity does not prove a shortcut', fontsize=11)
    paths = []
    for suffix in ('png', 'pdf'):
        path = output / ('all-g-endpoints.' + suffix)
        fig.savefig(path, dpi=170)
        paths.append(str(path))
    plt.close(fig)
    return paths
