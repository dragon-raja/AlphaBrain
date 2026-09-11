#!/usr/bin/env python3
"""Three-page CPU-only explanation of fixed metric rules on archived view data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_view_landscape_report_v1 import read_json, sha256, setup_plotting, write_json

ROOT = Path('/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908')
RULE_LABELS = {
    'canonical': '保持规范视角',
    'uniform97': '97 候选均匀选择',
    'min_accel': '最小平均 Accel',
    'max_visibility': '最大可见像素占比',
    'accel_top10_uniform': '低 Accel 前 10 内均匀选择',
    'accel_top10_max_visibility': '低 Accel 前 10 → 最大可见性',
    'visibility_top10_min_accel': '高可见性前 10 → 最小 Accel',
}


def check_completed(root: Path) -> dict:
    receipt = read_json(root / 'completion.json')
    if receipt['status'] != 'PASS_COMPLETE':
        raise ValueError('metric analysis is incomplete')
    for name, expected in receipt['outputs'].items():
        if sha256(root / name) != expected:
            raise ValueError(f'metric analysis output changed: {name}')
    return read_json(root / 'summary.json')


def points_with_intervals(ax, summaries: list[dict], positions: np.ndarray, *, scale: float = 1., color: str = '#2a6f97', offset: float = 0.) -> None:
    for y, item in zip(positions, summaries):
        if item['mean'] is None:
            continue
        mean = scale * item['mean']
        ci = item['ci95']
        if ci[0] is not None:
            ax.plot([scale*ci[0], scale*ci[1]], [y+offset, y+offset], color=color, lw=1.5)
        ax.scatter(mean, y+offset, c=color, s=35, zorder=3)


def camera_centers(state: dict, candidate_ids: list[str]) -> np.ndarray:
    assets = state['static_assets']
    path = Path(assets['policy_inputs'])
    if sha256(path) != assets['policy_inputs_sha256']:
        raise ValueError('cached image/calibration artifact checksum changed')
    with np.load(path, allow_pickle=False) as arrays:
        ids = list(map(str, arrays['candidate_ids']))
        transforms = arrays['camera_to_world_opencv']
        if transforms.shape != (97, 4, 4) or len(set(ids)) != 97:
            raise ValueError('expected 97 external camera transforms')
        index = {key: j for j, key in enumerate(ids)}
        if set(index) != set(candidate_ids):
            raise ValueError('camera calibration and metric candidate IDs differ')
        return np.asarray([transforms[index[key], :3, 3] for key in candidate_ids])


def build(args: argparse.Namespace) -> None:
    summary = check_completed(args.metrics_root)
    original = read_json(args.input_root / 'provenance.json')
    cache_path = args.input_root / 'verified_arrays.npz'
    if sha256(cache_path) != original['cache_sha256']:
        raise ValueError('verified metric source cache changed')
    analysis_manifest = read_json(args.metrics_root / 'manifest.json')
    if analysis_manifest['input_identity']['verified_arrays.npz']['sha256'] != original['cache_sha256']:
        raise ValueError('rule analysis and plot read different arrays')
    if sha256(args.input_root / 'states.json') != analysis_manifest['input_identity']['states.json']['sha256']:
        raise ValueError('state metadata differs from frozen metric analysis')
    contrasts = read_json(args.metrics_root / 'pairwise_rule_contrasts.json')
    if (contrasts['summary_sha256'] != sha256(args.metrics_root / 'summary.json') or
            contrasts['state_rule_metrics_sha256'] != sha256(args.metrics_root / 'state_rule_metrics.csv')):
        raise ValueError('paired contrasts and rule analysis differ')
    states = read_json(args.input_root / 'states.json')['states']
    with np.load(cache_path, allow_pickle=False) as values:
        candidate_ids = list(map(str, values['candidate_ids']))
        if len(candidate_ids) != 97 or len(set(candidate_ids)) != 97 or 'canonical' not in candidate_ids:
            raise ValueError('expected 97 unique candidates including canonical')
        canonical_index = candidate_ids.index('canonical')
        other_indices = [j for j in range(len(candidate_ids)) if j != canonical_index]
        if list(map(str, values['state_keys'])) != [s['pair_key'] for s in states]:
            raise ValueError('state metadata and array order mismatch')
        rates = values['success'].mean(2)
        accel = values['accel'].mean(2)
        visibility = values['visibility'].copy()
    setup_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.font_manager import fontManager
    fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'pdf.fonttype': 3, 'axes.unicode_minus': False, 'font.size': 10})
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    provenance = {'schema': 'dsol_view_metric_comparison_brief_v1',
        'status': 'PASS_DESCRIPTIVE_REPORT', 'metric_summary_sha256': sha256(args.metrics_root / 'summary.json'),
        'metric_manifest_sha256': sha256(args.metrics_root / 'manifest.json'),
        'paired_contrasts_sha256': sha256(args.metrics_root / 'pairwise_rule_contrasts.json'),
        'input_cache_sha256': sha256(cache_path), 'script_sha256': sha256(Path(__file__)),
        'new_gpu_rollouts': 0, 'frozen_rollout_protocol_modified': False,
        'source_scope': 'Archived Broad64 only; not new canonical model results.',
        'spatial_plot': 'True world camera centers from cached OpenCV extrinsics within each single state; rotations not drawn; no depth shading or interpolation.'}

    def heading(fig, text, sub):
        fig.suptitle(text, x=.035, y=.973, ha='left', fontsize=20, fontweight='bold', color='#173047')
        fig.text(.036, .910, sub, fontsize=10, color='#526477', va='top')
        fig.text(.035, .020, f"历史探索诊断  |  {summary['model_label']}  |  同一状态下外部视角变化  |  不证明主动相机获取收益", fontsize=8, color='#64748b')

    def save(fig, filename, pdf):
        fig.savefig(figures / (filename + '.png'), dpi=150)
        pdf.savefig(fig)
        plt.close(fig)

    with PdfPages(out / 'view_metric_comparison_brief_v1.pdf') as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(16, 9), gridspec_kw={'width_ratios': [1.1, 1, 1]})
        heading(fig, '指标是否有用：随机候选与规范视角是两个不同基线',
            f"{len(states)} 个状态 / {len(set(s['task_id'] for s in states))} 个任务；97 候选 × 32 次闭环；8 次 Accel 评分噪声。规则固定，k=10 不按结果调参。")
        order = summary['rule_order']
        y = np.arange(len(order))
        overall = summary['overall']['rules']
        for ax, field, scale, title in zip(axes, ('success', 'gain_vs_uniform_pp', 'gain_vs_canonical_pp'),
                                         (100, 1, 1), ('闭环成功率（%）', '相对均匀选择（百分点）', '相对规范视角（百分点）')):
            points_with_intervals(ax, [overall[r][field] for r in order], y, scale=scale)
            ax.set_yticks(y, [RULE_LABELS[r] for r in order] if ax is axes[0] else [])
            ax.set_ylim(len(order)-.4, -.6)
            ax.set_xlabel(title)
            ax.grid(axis='x', alpha=.2)
            if field != 'success':
                ax.axvline(0, color='#555', lw=1)
            else:
                ax.set_xlim(0, 100)
                for i, rule in enumerate(order):
                    ax.text(overall[rule][field]['mean']*100 + .4, i-.15,
                            f"{overall[rule][field]['mean']*100:.2f}", fontsize=8)
        fig.subplots_adjust(left=.245, right=.97, top=.80, bottom=.23, wspace=.22)
        fig.text(.05, .130, '均匀选择是已有完整矩阵下的期望值，不是新跑一次随机 rollout。横线为固定任务下的来源 bootstrap 描述性 95% 区间。', fontsize=10)
        gain = contrasts['contrasts']['visibility_refinement_within_accel_top10']['gain_pp']
        fig.text(.05, .090, f"低 Accel Top10 内再用可见性选取，相对集合内均匀选择的增量：{gain['mean']:+.2f} pp，95% 区间 [{gain['ci95'][0]:+.2f}, {gain['ci95'][1]:+.2f}]；净增益尚未确认。", fontsize=10)
        fig.text(.05, .050, '可见性使用仿真任务实体分割的整图像素占比，属于特权输入，不等于任务信息量。', fontsize=10)
        save(fig, '01_rule_comparison', pdf)

        fig = plt.figure(figsize=(16, 10))
        heading(fig, '同一物理状态下，直接对齐成功率、Accel 与可见性',
            '使用缓存标定中的真实相机中心 XYZ（米），不是全状态共用的绝对位置图；每行是独立状态，三列逐候选对齐。')
        chosen = []
        for task in ('goal_cream_cheese_bowl', 'goal_wine_rack'):
            candidates = [(i, s) for i, s in enumerate(states) if s['task_id'] == task and s['split'] == 'development']
            chosen.append(min(candidates, key=lambda item: (item[1]['source_group'], item[1]['source_state_index'])))
        provenance['representative_rule'] = 'Fixed task IDs cream-cheese-bowl and wine-rack; first lexicographic development source within task, not selected by rule outcomes.'
        provenance['representative_state_keys'] = [s['pair_key'] for _, s in chosen]
        for row, (i, state) in enumerate(chosen):
            xyz = camera_centers(state, candidate_ids)
            center = xyz.mean(0)
            extent = np.ptp(xyz, axis=0).max() / 2 * 1.1
            task_label = '奶酪入碗' if row == 0 else '酒瓶入架'
            for col, (vals, label, cmap, lim) in enumerate((
                (rates[i], '闭环成功率（越高越好）', 'viridis', (0, 1)),
                (accel[i], '平均 Accel（当前假设越低越好）', 'magma', (float(accel[i].min()), float(accel[i].max())+1e-12)),
                (visibility[i], '可见像素占比（越高占比越大）', 'cividis', (float(visibility[i].min()), float(visibility[i].max())+1e-12)),
            )):
                ax = fig.add_subplot(2, 3, 3*row+col+1, projection='3d')
                sc = ax.scatter(*xyz[other_indices].T, c=vals[other_indices], cmap=cmap, vmin=lim[0], vmax=lim[1], s=24, alpha=1, depthshade=False)
                ax.scatter(*xyz[canonical_index], c=[vals[canonical_index]], cmap=cmap, vmin=lim[0], vmax=lim[1], s=100, marker='*', edgecolor='black', depthshade=False)
                ax.set(xlabel='X（米）', ylabel='Y（米）', zlabel='Z',
                       title=f"{task_label} / {state['demo_name']} / frame {state['source_state_index']}\n{label}",
                       xlim=(center[0]-extent,center[0]+extent), ylim=(center[1]-extent,center[1]+extent), zlim=(center[2]-extent,center[2]+extent))
                ax.set_box_aspect((1,1,1)); ax.view_init(elev=20, azim=-60)
                ax.tick_params(labelsize=7)
                fig.colorbar(sc, ax=ax, shrink=.45, pad=.13).ax.tick_params(labelsize=8)
        fig.subplots_adjust(left=.02, right=.98, top=.80, bottom=.14, wspace=.08, hspace=.32)
        fig.text(.04, .080, '星号是规范位置。关闭深度明暗，颜色只由数值映射；Accel／可见性色标按当前状态范围显示，不能跨行只看颜色。', fontsize=9)
        fig.text(.04, .050, 'XYZ 仅展示相机中心，不包含朝向。视角变化同时影响遮挡与目标大小；更高像素占比不等于更多任务信息。', fontsize=9)
        save(fig, '02_same_state_three_metrics', pdf)

        fig, axes = plt.subplots(1, 3, figsize=(16, 9))
        heading(fig, '简单状态是否稀释了指标作用：分半噪声的分层诊断',
            '一半 O 噪声定义状态分组，另一半评价规则；然后反向执行。分组使用整个 97 候选空间的平均成功率。')
        groups = ('hard_floor', 'middle', 'easy')
        group_labels = ('低成功组：≤10%', '中间组：10%–90%', '近天花板组：≥90%')
        display_rules = [r for r in order if r not in ('canonical', 'uniform97')]
        positions = np.arange(len(display_rules))
        for ax, group, label in zip(axes, groups, group_labels):
            item = summary['difficulty_crossfit']['strata'][group]
            for field, offset, color in (('gain_vs_uniform_pp', -.13, '#2a6f97'), ('gain_vs_canonical_pp', .13, '#c26b33')):
                points_with_intervals(ax, [item['rules'][r][field] for r in display_rules], positions, color=color, offset=offset)
            ax.set_yticks(positions, [RULE_LABELS[r] for r in display_rules] if ax is axes[0] else [])
            ax.set_ylim(len(display_rules)-.4, -.6)
            ax.set(xlabel='成功率差（百分点）', title=f"{label}\n{item['state_count']} 状态贡献 / {item['task_count']} 任务")
            ax.axvline(0, color='#555', lw=1); ax.grid(axis='x', alpha=.2)
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([0],[0],marker='o',ls='',color='#2a6f97',label='相对均匀选择'),
                            Line2D([0],[0],marker='o',ls='',color='#c26b33',label='相对规范视角')], loc='upper right', bbox_to_anchor=(.97,.855), frameon=False, ncol=2)
        fig.subplots_adjust(left=.25, right=.965, top=.76, bottom=.26, wspace=.20)
        changed = summary['difficulty_crossfit']['changed_stratum_state_count']
        stability = summary['overall']['stability']['top10_pairwise_jaccard']['mean']
        fig.text(.05, .160, f'{changed} 个状态在两个方向的分组不同，可分别贡献到两个组；不是互斥的状态划分。Top10 跨初噪平均 Jaccard：{stability:.3f}。', fontsize=10)
        fig.text(.05, .110, '来源不足时不画置信区间。分组只描述观察到的表现，不证明任务简单、证据充分或训练覆盖是原因。', fontsize=10)
        fig.text(.05, .065, '判断顺序：候选初筛是否优于均匀选择 → 二次筛选是否增加收益 → 能否超过规范视角 → 再验证可执行获取与新来源泛化。', fontsize=10)
        save(fig, '03_crossfit_strata', pdf)
    provenance['outputs'] = {p.name: sha256(p) for p in sorted(figures.glob('*.png'))}
    provenance['outputs']['view_metric_comparison_brief_v1.pdf'] = sha256(out / 'view_metric_comparison_brief_v1.pdf')
    write_json(out / 'report_provenance.json', provenance)
    print(f'PASS_COMPLETE {out}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-root', type=Path, default=ROOT / 'metric-rules-v1')
    parser.add_argument('--input-root', type=Path, default=ROOT / 'broad-existing')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'metric-rules-v1/report')
    build(parser.parse_args())


if __name__ == '__main__':
    main()
