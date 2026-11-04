"""P3-only spatial explanation: direct frozen-model outcomes, no PPO fitting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

from AlphaBrain.research.dsol.analysis.spatial_recovery import analyze
from AlphaBrain.research.dsol.execution.manifest import file_digest
from AlphaBrain.research.dsol.plotting.spatial_score_validity import recovery_figures
from reports.paper1.target_spatial_panel import load_verified_panel


TASK_LABELS = {
    'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it': '开炉并放置摩卡壶',
    'KITCHEN_SCENE8_put_both_moka_pots_on_the_stove': '把两只摩卡壶放到炉上',
    'push_the_plate_to_the_front_of_the_stove': '把盘子推到炉前',
    'put_the_wine_bottle_on_top_of_the_cabinet': '把酒瓶放到柜顶',
    'pick_up_the_alphabet_soup_and_place_it_in_the_basket': '字母汤罐放入篮子',
    'pick_up_the_salad_dressing_and_place_it_in_the_basket': '沙拉酱放入篮子',
    'pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate': '饼干盒上的黑碗放到盘上',
    'pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate': '小碗上的黑碗放到盘上',
}


def audit_pairing(positions):
    """Check actual saved native noise and execution addresses across viewpoints."""
    score_noise, state_vectors, execution_noise, cameras = {}, {}, {}, {}
    query_hashes = {}
    for (state, view), point in sorted(positions.items()):
        path = Path(point['score_query'])
        query_hashes[str(path)] = file_digest(path)
        with np.load(path, allow_pickle=False) as z:
            robot = z['robot_state']
            if state in state_vectors and not np.array_equal(robot, state_vectors[state]):
                raise ValueError('Robot state differs across candidate score queries')
            state_vectors[state] = robot.copy()
            cameras.setdefault(state, set()).add(z['camera_to_world_opencv'].tobytes())
            for member in (0, 1):
                value = z[f'full_noise_{member}']
                key = state, member
                if key in score_noise and not np.array_equal(value, score_noise[key]):
                    raise ValueError('Score noise not paired across candidate poses')
                score_noise[key] = value.copy()
        for repeat, entry in point['executions'].items():
            result = entry['result']
            if result.get('method') not in (None, 'stay', 'reference'):
                raise ValueError('PPO/search-selected operation entered fixed-grid evidence')
            actual_angles = result.get('selected_angles', result['angles_deg'])
            if not np.array_equal(actual_angles, point['angles_deg']):
                raise ValueError('Operation endpoint differs from scored candidate')
            for replan, stamp in enumerate(result['execution_noise_sha256']):
                key = state, repeat, replan
                if key in execution_noise and execution_noise[key] != stamp:
                    raise ValueError('Operation noise not paired across candidate poses')
                execution_noise[key] = stamp
    if any(len(v) != 27 for v in cameras.values()):
        raise ValueError('Expected 27 distinct physical camera transforms per state')
    return dict(status='PASS', poses=len(positions), states=len(state_vectors),
                original_score_noise_exactly_shared_across_poses=True,
                operation_noise_prefixes_exactly_paired=True,
                robot_state_shared_across_score_poses=True,
                distinct_camera_transforms_per_state=27,
                reused_methods=['stay', 'reference'],
                query_sha256=query_hashes)


def build(root, output, *, font_path):
    root, output = Path(root), Path(output)
    rows, positions = load_verified_panel(root)
    # Preserve the old published table and verify this reconstruction agrees.
    published = root/'report/rows.json'
    if rows != json.loads(published.read_text()):
        raise ValueError('Reconstruction differs from published fixed-grid rows')
    audit = audit_pairing(positions)
    summary = analyze(rows)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('analysis.json', summary), ('pairing-audit.json', audit)]:
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (output/'per-state.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['state_key', 'reference_sr', 'perturbed_sr', 'argmin_view', 'yaw', 'pitch',
                         'argmin_p3', 'argmin_successes_out_of_2', 'best_observed_sr',
                         'low7_contains_2of2', 'rescued_endpoint_pairs', 'harmed_endpoint_pairs'])
        for x in summary['per_state']:
            a, pair = x['argmin'], x['paired_endpoint_reuse']
            writer.writerow([x['state_key'], x['reference_success'], x['perturbed_success'],
                             a['view_id'], *a['angles_deg'], a['p3'], a['successes'],
                             x['observed_best_success'], x['low_group_has_two_of_two'],
                             pair['rescued'], pair['harmed']])
    index = recovery_figures(rows, summary, output, font_path=font_path, task_labels=TASK_LABELS)
    (output/'figure-index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2)+'\n')
    pair = summary['paired_endpoint_reuse']
    lines = [
        '冻结FastWAM：Accel p3 与操作成功率的空间对照（解释补充，不覆盖原报告）',
        '本次只重新读取已完成的固定网格，不运行训练、推理、仿真或新增操作。',
        '不是PPO终点，也不是迁移后的搜索策略得分：每个候选直接对冻结操作模型评分并完整操作。',
        '复用96条仅为stay/reference；新增768条固定角度操作。源视觉特征有记录，但不用于选位或改变操作动作。',
        'p3来自同一原生速度轨迹的原评分公式，没有使用reward分母、G/B校准或PPO value。',
        '每个初态27点，候选间实际评分初始噪声逐值相同、操作噪声地址逐重规划一致，机器人状态一致；每点只有2次操作。',
        '坐标：横轴yaw、纵轴pitch，第三轴为Accel或观察成功比例，不是相机高度。紫/青/黄=0/2、1/2、2/2。',
        '红星=参考位，黑X=非参考候选中仅按两成员均值选出的最低p3点，蓝圈=最低7点。全部16初态按名字排序展示，不筛选有效例子。',
        f"参考位 {summary['reference_success']:.4%}；26个扰动点平均 {summary['perturbed_success']:.4%}。",
        f"仅按最低p3选位：{summary['selected_successes']}/{summary['selected_distinct_operations']}={summary['selected_success']:.4%}；相对扰动平均 {summary['argmin_net_gain_task_mean']*100:+.2f}个百分点。",
        f"非参考空间存在至少一个2/2点：{summary['states_with_observed_two_of_two']}/{summary['states']}个初态；最低7点内存在2/2点：{summary['states_with_low_group_two_of_two']}/{summary['states']}。",
        f"{summary['starts_with_lower_score_higher_observed_sr']}/{summary['distinct_perturbed_starts']}个起点，事后可找到评分更低且观察成功比例更高的候选。该选择使用了结果标签，不能称为可部署策略。",
        f"按相同操作重复编号重用端点结果配对：{pair['comparisons']}对中救回{pair['rescued']}、损害{pair['harmed']}。",
        '这些配对不是额外独立的恢复轨迹；选中端点的32个操作结果在不同起点下重复使用。未检验真实移动路径、回位误差及查询耗时。',
        '因此：不是没有低分好视角，而是仅取最低分不能可靠识别它们。不能仅凭低7点平均断言空间里没有恢复潜力，也不能靠事后挑好点宣称算法已恢复。',
        'p9只保留旧报告中的同轨迹离线分析；本补充不以p9替换主评分、不启动p9训练或评测。',
        'GR00T应做相同类型的固定空间验证：冻结官方任务匹配模型、无PPO/迁移选位，每点自身原生Accel与完整操作结果。原生4步不改成9步，预算和操作协议单独绑定。',
    ]
    (output/'README.txt').write_text('\n'.join(lines)+'\n')
    (output/'provenance.json').write_text(json.dumps(dict(
        original_rows_sha256=file_digest(published), original_summary_sha256=file_digest(root/'report/summary.json'),
        campaign_sha256=file_digest(root/'campaign.json'), model_calls=0, training_updates=0,
        font_sha256=file_digest(font_path), report_type='posthoc_descriptive_p3_explanation'), indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--font', required=True)
    args = parser.parse_args()
    build(args.root, args.output, font_path=args.font)
