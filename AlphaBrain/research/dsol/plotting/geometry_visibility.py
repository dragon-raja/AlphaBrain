"""Measured camera extrinsics and explicitly labelled geometric illustrations."""
import numpy as np


def pivot_from_camera(camera, table_z):
    """Saved cameras use OpenCV +Z optical forward, not MuJoCo -Z."""
    p, forward = camera[:3, 3], camera[:3, 2]
    t = (table_z-p[2])/forward[2]
    if not np.isfinite(t) or t <= 0:
        raise ValueError('Reference ray must meet table in front of camera')
    return p + t*forward


def frustum_corners(camera, intrinsics, width, height, depth=.28):
    """Four image-boundary rays at a display depth, in world coordinates.

    The finite depth only shortens the drawing; it is not a sensor far plane.
    """
    if min(width, height, depth) <= 0:
        raise ValueError('Positive image dimensions and drawing depth required')
    pixels = np.array([[0, 0, 1], [width-1, 0, 1], [width-1, height-1, 1], [0, height-1, 1]], float)
    local = (pixels @ np.linalg.inv(intrinsics).T)*depth
    return local @ camera[:3, :3].T + camera[:3, 3]


def orientation_figure(camera, intrinsics, table_z, image_size=(224, 224)):
    """All measured centers, sparse true frusta and unambiguous optical rays."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    c = camera[0]; p = c[:, :3, 3]; pivot = pivot_from_camera(c[0], table_z)
    # Fixed index subsample, independent of images, scores and outcomes.
    chosen = np.r_[0, np.linspace(1, 96, 6, dtype=int)]
    colors = ['#bd3039', '#147ba8', '#339862', '#a766b1', '#d58a14', '#376cbd', '#aa6842']
    fig = plt.figure(figsize=(15, 8), layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1])
    ax = fig.add_subplot(grid[:, 0], projection='3d')
    ax.scatter(*p[1:].T, color='#b6c1c7', s=12, alpha=.45)
    for i, color in zip(chosen, colors):
        corners = frustum_corners(c[i], intrinsics[i], *image_size)
        label = '规范' if i == 0 else ('T' + str(i-1).zfill(2) if i < 65 else 'H' + str(i-65).zfill(2))
        ax.scatter(*p[i], color=color, s=50, marker='*' if i == 0 else 'o')
        ax.text(*(p[i]+[.02, .02, .08]), label, color=color, fontsize=10)
        ax.add_collection3d(Poly3DCollection([corners], facecolors=color, edgecolors=color, alpha=.18))
        for corner in corners:
            ax.plot(*np.stack([p[i], corner]).T, color=color, linewidth=1.3)
        ax.plot(*np.stack([p[i], pivot]).T, '--', color=color, linewidth=1.1, alpha=.75)
        ax.quiver(*p[i], *(c[i, :3, 2]*.42), color=color, linewidth=2.5, arrow_length_ratio=.25)
    ax.scatter(*pivot, color='black', marker='x', s=90)
    ax.text2D(.03, .06, '黑叉：几何参考中心\n不是逐个任务物体的定位结果', transform=ax.transAxes, fontsize=10)
    _axes3d(ax, np.concatenate([p, pivot[None]])); ax.set_title('实际相机视锥与朝向', fontsize=16)
    for row, (a, b, title) in enumerate([(0, 1, '俯视：光轴朝参考中心汇聚'), (0, 2, '侧视：不同高度也朝同一中心')]):
        side = fig.add_subplot(grid[row, 1]); side.scatter(p[1:, a], p[1:, b], s=9, color='#bec8cd', alpha=.6)
        for i, color in zip(chosen, colors):
            side.plot([p[i, a], pivot[a]], [p[i, b], pivot[b]], '--', color=color, alpha=.6)
            f = c[i, :3, 2]*.42
            side.annotate('', xy=(p[i, a]+f[a], p[i, b]+f[b]), xytext=(p[i, a], p[i, b]),
                          arrowprops=dict(arrowstyle='-|>', color=color, lw=2))
            side.scatter(p[i, a], p[i, b], s=35, color=color)
        side.scatter(pivot[a], pivot[b], color='black', marker='x', s=65)
        side.set(title=title, xlabel=f'世界 {"XYZ"[a]} (m)', ylabel=f'世界 {"XYZ"[b]} (m)')
        side.set_aspect('equal', adjustable='box'); side.grid(alpha=.15)
    fig.suptitle('朝向增强图：彩色线框是相机视锥，粗箭头是光轴，虚线延伸到参考中心\n全部97个位置保留；仅按固定索引显示7组方向，避免遮挡；视锥由真实内参/外参绘制，长度仅为显示截断', fontsize=14)
    return fig


def orbit_positions(reference, pivot, azimuth, elevation, scales):
    """Geometry illustration only: matches spherical orbit, no new outcomes."""
    r = reference-pivot
    radius = np.linalg.norm(r)
    az = np.arctan2(r[1], r[0])+np.deg2rad(azimuth)
    el = np.arcsin(r[2]/radius)+np.deg2rad(elevation)
    az, el, scale = np.broadcast_arrays(az, el, scales)
    return pivot + (radius*scale)[..., None]*np.stack([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)], axis=-1)


def _axes3d(ax, points):
    lo, hi = points.min(0), points.max(0)
    center = (lo+hi)/2
    half = max(float((hi-lo).max())*.56, .1)
    ax.set(xlim=(center[0]-half, center[0]+half), ylim=(center[1]-half, center[1]+half),
           zlim=(center[2]-half, center[2]+half), xlabel='世界 X (m)', ylabel='世界 Y (m)', zlabel='世界 Z (m)')
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=23, azim=-55)
    ax.tick_params(labelsize=8)


def _draw_camera(ax, camera, pivot, small=False):
    p, f = camera[:, :3, 3], camera[:, :3, 2]
    ax.scatter(*p[1:65].T, s=10 if small else 25, color='#2478b4', label='训练目录 64')
    ax.scatter(*p[65:].T, s=18 if small else 36, color='#e18b22', marker='^', label='留出目录 32')
    ax.quiver(*p[1:].T, *f[1:].T, length=.10, normalize=True, color='#607483', alpha=.5, linewidth=.65)
    ax.scatter(*p[0], s=85 if small else 140, marker='*', color='#b83235', label='规范相机')
    ax.quiver(*p[0], *f[0], length=.25, normalize=True, color='#b83235', linewidth=2)
    ax.scatter(*pivot, marker='x', color='black', s=40, label='参考中心')
    ax.plot(*np.stack([p[0], pivot]).T, '--', color='#b83235', linewidth=1, alpha=.7)
    _axes3d(ax, np.concatenate([p, pivot[None]]))


def geometry_figure(camera, table_z):
    import matplotlib.pyplot as plt
    c = camera[0]; pivot = pivot_from_camera(c[0], table_z)
    fig = plt.figure(figsize=(12, 10), layout='constrained')
    ax = fig.add_subplot(221, projection='3d'); _draw_camera(ax, c, pivot)
    ax.set_title('任务0 / 初态0：真实相机中心与光轴', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    for row, (a, b, title) in enumerate([(0, 1, '俯视 XY'), (0, 2, '侧视 XZ'), (1, 2, '侧视 YZ')]):
        ax = fig.add_subplot(2, 2, row+2); p = c[:, :3, 3]
        ax.scatter(p[1:65, a], p[1:65, b], color='#2478b4', s=14)
        ax.scatter(p[65:, a], p[65:, b], color='#e18b22', marker='^', s=22)
        ax.scatter(p[0, a], p[0, b], color='#b83235', marker='*', s=100)
        ax.scatter(pivot[a], pivot[b], color='black', marker='x')
        ax.set(xlabel=f'世界 {"XYZ"[a]} (m)', ylabel=f'世界 {"XYZ"[b]} (m)', title=title)
        ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.2)
    fig.suptitle('实测外参展示：不是参数空间示意，也不是原地变焦\n点＝相机中心；短箭头＝光轴方向；内参不变；不代表位置可达或路径无碰撞', fontsize=15)
    return fig


def all_tasks_figure(camera, table_z):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(17, 9), layout='constrained')
    for task in range(8):
        ax = fig.add_subplot(2, 4, task+1, projection='3d')
        c = camera[task*4]; _draw_camera(ax, c, pivot_from_camera(c[0], table_z), small=True)
        ax.set_title(f'任务 {task} / 初态0', fontsize=12)
    fig.suptitle('八任务的实际世界坐标：相同角度/距离参数，不等于相同世界位置\n蓝点：64训练目录；橙三角：32留出目录；红星：规范相机；黑叉：参考中心；各面板等比例、范围可不同', fontsize=15)
    return fig


def orbit_figure(camera, table_z):
    import matplotlib.pyplot as plt
    c = camera[0, 0]; p = c[:3, 3]; pivot = pivot_from_camera(c, table_z)
    fig = plt.figure(figsize=(15, 5.5), layout='constrained')
    settings = [(np.linspace(-35, 35, 15), 0, 1., '方位角：左右绕中心移动'),
                (0, np.linspace(-20, 20, 15), 1., '仰角：上下绕中心移动'),
                (0, 0, np.linspace(.8, 1.2, 15), '距离倍率：沿径向前后移动')]
    for i, (az, el, scale, title) in enumerate(settings):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        xyz = orbit_positions(p, pivot, az, el, scale)
        f = pivot-xyz; f /= np.linalg.norm(f, axis=1)[:, None]
        ax.plot(*xyz.T, color='#2478b4'); ax.scatter(*xyz.T, c=np.arange(len(xyz)), cmap='viridis', s=28)
        ax.quiver(*xyz.T, *f.T, length=.14, normalize=True, color='#2478b4', alpha=.7)
        ax.scatter(*p, marker='*', s=130, color='#b83235'); ax.scatter(*pivot, marker='x', color='black', s=40)
        ax.plot(*np.stack([p, pivot]).T, '--', color='gray', alpha=.5)
        _axes3d(ax, np.concatenate([xyz, pivot[None]])); ax.set_title(title, fontsize=13)
    fig.suptitle('单轴几何解释：三种操作都会改变相机中心\n由原始参考外参解析计算的示意轨迹，不是新增候选评测；原地俯仰则应保持红星位置不动', fontsize=15)
    return fig


def strata_figure(summary):
    import matplotlib.pyplot as plt
    names = ['stay', 'min_accel', 'visibility', 'accel10_visibility', 'local_accel_16', 'random_accel_16']
    labels = ['原位', '最低Accel', '最高可见性', 'Top10+可见性', 'Accel搜索16', '随机搜索16']
    colors = ['#8a969d', '#227bb3', '#9b6da9', '#1c9787', '#e18b22', '#c3a278']
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout='constrained')
    for m, arm in enumerate(['canonical', 'broad']):
        for col, (stratum, label) in enumerate([('all', '全部扰动起点'), ('visible', '所有任务实体像素 > 0'), ('missing', '至少一实体像素 = 0')]):
            ax = axes[m, col]
            y = [100*summary['models'][arm]['primary'][n][stratum]['success'] for n in names]
            ax.bar(range(len(y)), y, color=colors)
            for j, v in enumerate(y): ax.text(j, v+1, f'{v:.1f}', ha='center', fontsize=9)
            ax.set(ylim=(0, 103), xticks=range(len(y)), xticklabels=labels, ylabel='完整任务成功率 (%)')
            ax.tick_params(axis='x', rotation=30, labelsize=9)
            ax.set_title(('规范训练' if m == 0 else '多视角训练')+' | '+label+f"\n{summary['strata'][stratum]['starts']} 起点，16初态", fontsize=12)
            ax.grid(axis='y', alpha=.15)
    fig.suptitle('只筛起点，不改原方法的终点和搜索路径\n同一初态内先平均保留起点，再等权汇总任务；>0像素不等于完整可见或信息相同；均值图，配对区间见数据表', fontsize=15)
    return fig


def restricted_figure(summary):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    names = ['uniform_visible', 'visibility', 'min_accel', 'accel10_visibility']
    for ax, arm, title in zip(axes, ['canonical', 'broad'], ['规范训练', '多视角训练']):
        y = [summary['models'][arm]['restricted_destination'][n]['success']*100 for n in names]
        ax.bar(range(4), y, color=['#8a969d', '#9b6da9', '#227bb3', '#1c9787'])
        for j, v in enumerate(y): ax.text(j, v+1, f'{v:.2f}', ha='center')
        ax.set(xticks=range(4), xticklabels=['可见候选均匀选', '最高可见性', '最低Accel', 'Top10+可见性'],
               ylim=(0, 103), ylabel='完整任务成功率 (%)', title=title)
        ax.tick_params(axis='x', rotation=15); ax.grid(axis='y', alpha=.15)
    fig.suptitle('另一个问题：若起点和可选终点均要求所有任务实体像素 > 0\n使用仿真分割真值限制候选的事后诊断，不是新增可部署算法；仍复用已有闭环结果', fontsize=13)
    return fig
