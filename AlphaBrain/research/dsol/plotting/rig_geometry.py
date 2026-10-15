"""RoboCasa installation geometry figures; pure plotting, not motion simulation."""
import numpy as np
from .geometry_visibility import frustum_corners
from ..analysis.rig_geometry import camera_opencv


def installation_figure(geometry, fovy, title):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    p, r, mid, pivot = (geometry[k] for k in ('positions', 'rotations', 'midpoint', 'pivot'))
    fig = plt.figure(figsize=(15, 8), layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])
    ax = fig.add_subplot(grid[:, 0], projection='3d')
    for c, color, name in [(0, '#2778b8', '左外部相机'), (1, '#e68a28', '右外部相机')]:
        ax.scatter(*p[1:, c].T, s=13, color=color, alpha=.45, label=name+'：96候选')
    # Fixed predeclared indices, unrelated to scores or success labels.
    chosen = [0, 1, 16, 49, 64]
    for i in range(len(p)):
        ax.plot(*p[i].T, color='#9eabb4', alpha=.14, lw=.5)
    for i in chosen:
        for c, color in [(0, '#2778b8'), (1, '#e68a28')]:
            pos, rot = p[i, c], r[i, c]
            focal = 128 / np.tan(np.deg2rad(fovy[c]/2))
            k = np.array([[focal, 0, 127.5], [0, focal, 127.5], [0, 0, 1.]])
            corners = frustum_corners(camera_opencv(pos, rot), k, 256, 256, depth=.15)
            ax.add_collection3d(Poly3DCollection([corners], facecolors=color, edgecolors=color, alpha=.12))
            for corner in corners:
                ax.plot(*np.stack([pos, corner]).T, color=color, lw=.9)
            ax.quiver(*pos, *(-rot[:, 2]*.25), color=color, lw=1.7, arrow_length_ratio=.25)
        ax.plot(*p[i].T, color='#526677', lw=1.5)
        if i == 0:
            ax.scatter(*p[i].T, color='#b8293a', marker='*', s=105, label='规范安装')
        ax.plot(*np.stack([mid[i], pivot]).T, '--', color='#62806c', lw=.9, alpha=.65)
    ax.scatter(*pivot, marker='x', s=75, color='black', label='相机推导的参考中心')
    points = np.concatenate([p.reshape(-1, 3), pivot[None]])
    center = (points.max(0)+points.min(0))/2; half = np.ptp(points, axis=0).max()*.59
    ax.set(xlim=(center[0]-half, center[0]+half), ylim=(center[1]-half, center[1]+half), zlim=(center[2]-half, center[2]+half),
           xlabel='支撑体 X (m)', ylabel='支撑体 Y (m)', zlabel='支撑体 Z (m)', title='97组真实安装位姿：双目间距不变')
    ax.set_box_aspect((1, 1, 1)); ax.view_init(24, -58); ax.legend(fontsize=9, loc='upper left')
    for row, (a, b, name) in enumerate([(0, 1, '俯视：左右光轴与相机架中心方向'), (0, 2, '侧视：高度、距离和朝向同时变化')]):
        side = fig.add_subplot(grid[row, 1])
        for c, color in [(0, '#2778b8'), (1, '#e68a28')]:
            side.scatter(p[:, c, a], p[:, c, b], s=10, color=color, alpha=.3)
            for i in chosen:
                f = -r[i, c, :, 2]*.32
                side.annotate('', xy=(p[i,c,a]+f[a], p[i,c,b]+f[b]), xytext=(p[i,c,a], p[i,c,b]), arrowprops=dict(arrowstyle='-|>', color=color, lw=1.5))
        for i in chosen:
            side.plot(p[i, :, a], p[i, :, b], color='#72808b', lw=1)
            side.plot([mid[i,a], pivot[a]], [mid[i,b], pivot[b]], '--', color='#62806c', alpha=.65)
        side.scatter(pivot[a], pivot[b], marker='x', color='black', s=60)
        side.set(title=name, xlabel=f'支撑体 {"XYZ"[a]} (m)', ylabel=f'支撑体 {"XYZ"[b]} (m)')
        side.set_aspect('equal', adjustable='box'); side.grid(alpha=.15)
    fig.suptitle(title+'\n安装坐标，不是全程世界轨迹；绿色虚线为双目平均方向，不能解读为每只相机都精确瞄准同一物体', fontsize=13)
    return fig


def coverage_figure(parameters, camera_displacement, optical_angles):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 6), layout='constrained')
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(*parameters[1:65].T, s=28, color='#2778b8', label='训练目录：4×4×4=64')
    ax.scatter(*parameters[65:].T, s=40, marker='^', color='#e68a28', label='留出目录：Sobol 32')
    ax.scatter(*parameters[0], s=100, marker='*', color='#b8293a', label='规范安装')
    ax.set(xlabel='方位参数 (°)', ylabel='仰角参数 (°)', zlabel='距离倍率', title='三参数范围：留出点仍在同一个范围内')
    ax.legend(fontsize=9); ax.view_init(22, -50)
    right = fig.add_subplot(1, 2, 2)
    # All initials, same candidate identities. Density is not independent trials.
    hist = right.hist2d(camera_displacement.ravel(), optical_angles.ravel(), bins=35, cmap='Blues', cmin=1)
    fig.colorbar(hist[3], ax=right, shrink=.72, label='相机记录数（非独立场景数）')
    right.set(xlabel='相对规范安装的相机平移 (m)', ylabel='光轴夹角 (°)', title='400初态 × 96候选 × 2相机：几何分布')
    right.grid(alpha=.15)
    fig.suptitle('位置与朝向耦合的三维候选族，不是任意SE(3)；点密度不代表可达性或任务信息覆盖', fontsize=14)
    return fig


def compact_candidate_figure(parameters, positions, forwards):
    """Report-sized parameter catalogue and actual paired camera mounts.

    Arrays use the same 97-view order: reference, 64 train, 32 held-out.
    Positions and forward vectors are in the camera support-body frame.
    This is a geometry illustration, not a reachability certificate.
    """
    import matplotlib.pyplot as plt
    parameters, positions, forwards = map(np.asarray, (parameters, positions, forwards))
    if parameters.shape != (97, 3) or positions.shape != (97, 2, 3) or forwards.shape != positions.shape:
        raise ValueError('Expected 97 parameters and paired camera positions/forward vectors')
    fig = plt.figure(figsize=(9, 3.4))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(*parameters[1:65].T, s=17, color='#2778b8', label='训练安装 64')
    ax.scatter(*parameters[65:].T, s=23, marker='^', color='#e68a28', label='留出安装 32')
    ax.scatter(*parameters[0], s=75, marker='*', color='#b8293a', label='规范安装 1')
    ax.set(title='候选参数空间', xlabel='方位角（°）', ylabel='仰角（°）', zlabel='距离倍率')
    ax.view_init(22, -50)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(-.10, 1.05))
    geo = fig.add_subplot(1, 2, 2, projection='3d')
    chosen = [0, 1, 16, 49, 64]
    for c, color, label in [(0, '#2778b8', '左相机'), (1, '#e68a28', '右相机')]:
        geo.scatter(*positions[1:, c].T, s=11, color=color, alpha=.4, label=label)
        geo.scatter(*positions[0, c], s=75, color='#b8293a', marker='*')
        for i in chosen:
            geo.quiver(*positions[i, c], *forwards[i, c], length=.19,
                       color=color, linewidth=1.2, arrow_length_ratio=.25)
    for i in chosen:
        geo.plot(*positions[i].T, color='#738490', linewidth=1.1)
    geo.set(title='实际双相机安装位置与朝向', xlabel='X（米）', ylabel='Y（米）', zlabel='Z（米）')
    geo.view_init(23, -58)
    geo.legend(fontsize=8, loc='upper left')
    for a in (ax, geo):
        a.tick_params(labelsize=8, pad=0)
        a.xaxis.label.set_size(9)
        a.yaxis.label.set_size(9)
        a.zaxis.label.set_size(9)
        a.title.set_size(11)
    fig.subplots_adjust(left=.01, right=.96, top=.90, bottom=.07, wspace=.13)
    return fig
