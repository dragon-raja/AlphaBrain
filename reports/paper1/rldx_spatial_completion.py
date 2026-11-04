"""Verify immutable RLDX grid operations, compose descriptive CPU-only evidence."""
import argparse
import csv
from pathlib import Path
import time
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read, write
from AlphaBrain.research.dsol.data.spatial_completion import key
from AlphaBrain.research.dsol.analysis.spatial_completion import assemble, summarize
from AlphaBrain.research.dsol.execution.manifest import file_digest as sha
from AlphaBrain.research.dsol.plotting.spatial_score_comparison import figures
from benchmarks.LIBERO import rldx_spatial_panel as panel
from benchmarks.LIBERO import controlled_rldx_operation as operation
from benchmarks.LIBERO.rldx_runtime import accel_scores


def load_verified(root):
    root = Path(root)
    plan = read(root/'reuse-plan.json')
    campaign_path = root/'spatial-campaign.json'
    campaign = panel.validate_campaign(campaign_path)
    phase = root/'execution/spatial-real'
    done = read(phase/'completion.json')
    if (done['status'] != 'COMPLETE' or done['dispatch_enabled'] is not False
            or done['counts']['episodes'] != len(plan['missing'])):
        raise ValueError('Scientific completion not fully settled')
    provenance = {}

    def verified(path, expected):
        if sha(path) != expected: raise ValueError('Changed evidence: '+str(path))
        provenance[str(path)] = expected

    verified(plan['source_campaign'], plan['source_campaign_sha256'])
    verified(plan['source_scores'], plan['source_scores_sha256'])
    old_campaign = read(plan['source_campaign'])
    scores = read(plan['source_scores'])
    score_map = {(r['state_key'], r['view']): r for r in scores}
    reconstructed_scores = []; score_noise = {}
    for b in old_campaign['blocks']:
        folder = Path(plan['source_campaign']).parent/'execution/selection-real/results'/f"block-{b['index']:04d}"
        operation.validate_block(operation.request_for(plan['source_campaign'], b['index']), folder)
        verified(folder/'complete.json', plan['verified_receipts'][str(folder/'complete.json')])
        state = b['rows'][0]['state_key']
        for point in read(folder/'results.json')['scores']:
            reconstructed_scores.append(dict(state_key=state, **point))
            for member in (0, 1):
                with np.load(folder/f"grid-{point['view']:02d}/member-{member}.npz", allow_pickle=False) as trace:
                    if accel_scores(trace['velocities']) != point['scores'][member]:
                        raise ValueError('Stored native velocity/score mismatch')
                    noise_key = state, member
                    noise = trace['initial_noise']
                    if noise_key in score_noise and not np.array_equal(score_noise[noise_key], noise):
                        raise ValueError('Original scoring noise is not paired across viewpoints')
                    score_noise[noise_key] = noise.copy()
    if reconstructed_scores != scores:
        raise ValueError('Published score table does not match original receipts')
    outcomes = []

    def audit_trajectory(row, path, result):
        with np.load(path, allow_pickle=False) as z:
            if (z['actions'].shape != (result['steps'], 7)
                    or z['normalized_actions'].shape != (result['replans'], 1, 16, 64)
                    or not np.isfinite(z['actions']).all() or not np.isfinite(z['normalized_actions']).all()
                    or z['seeds'].tolist() != [operation.seed_for(row['state_key'], row['repeat'], i) for i in range(result['replans'])]
                    or str(z['inputs'][0]) != score_map[row['state_key'], row['view_id']]['input_sha256']):
                raise ValueError('Changed native actions, noise schedule or scored initial input')

    for row in plan['reuse']:
        prior = row['prior']
        verified(prior['source_result'], prior['source_result_sha256'])
        verified(prior['source_trajectory'], prior['source_trajectory_sha256'])
        result = read(prior['source_result'])
        if any(result[k] != prior[k] for k in ('success', 'steps', 'replans', 'repeat', 'state_key')):
            raise ValueError('Prior result identity changed')
        audit_trajectory(row, prior['source_trajectory'], result)
        outcomes.append(dict(row, success=result['success']))
    calls = 0; resident_reuses = 0; model_loads = 0
    for b in campaign['blocks']:
        folder = phase/'results'/f"block-{b['index']:04d}"
        receipt = panel.validate_block(panel.request_for(campaign_path, b['index']), folder)
        resident = read(folder/'resident.json')
        if resident['identity'] != campaign['runtime_contract_sha256'] or type(resident['reused']) is not bool:
            raise ValueError('Unbound resident runtime identity')
        resident_reuses += int(resident['reused']); model_loads += int(not resident['reused'])
        calls += receipt['calls']['execution']
        provenance[str(folder/'complete.json')] = receipt['receipt_sha256']
        for i, (row, result) in enumerate(zip(b['rows'], read(folder/'results.json')['tasks'])):
            if key(row) != key(result): raise ValueError('Result order/identity changed')
            audit_trajectory(row, folder/f'episode-{i:03d}/trajectory.npz', result)
            outcomes.append(dict(row, success=result['success']))
    engineering = read(root/'engineering-real-completion.json')['real_model_calls'] + read(root/'resident-replay-proof.json')['model_calls']
    approval = read(root/'authorization.json')
    if (calls != done['actual_model_calls'] or calls > approval['scientific_model_call_limit']
            or engineering > approval['engineering_model_call_limit']
            or calls+engineering > approval['total_model_call_limit']):
        raise ValueError('Model-call budget mismatch')
    return assemble(plan['expected'], scores, outcomes), dict(status='PASS_COMPLETE_COVERAGE_AND_PROVENANCE',
        expected_operations=len(plan['expected']), reused_operations=len(plan['reuse']), new_operations=len(plan['missing']),
        scientific_model_calls=calls, engineering_model_calls=engineering, total_new_model_calls=calls+engineering,
        approved_new_call_limit=approval['total_model_call_limit'], training_updates=0, new_scoring_calls=0,
        resident_model_loads=model_loads, resident_block_reuses=resident_reuses,
        operation_initial_inputs_match_score=True, native_execution_noise_addresses_verified=True,
        original_scoring_noise_exactly_paired_across_views=True,
        scores_recomputed_from_original_native_velocity_traces=True, evidence_sha256=provenance)


def build(root, output, *, font_path):
    rows, audit = load_verified(root)
    summary = summarize(rows)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    for name, value in [('rows.json', rows), ('analysis.json', summary), ('audit.json', audit)]:
        write(output/name, value)
    with (output/'measured-points.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['state_key', 'view_id', 'yaw_deg', 'pitch_deg', 'p3_member0', 'p3_member1',
                         'p4_member0', 'p4_member1', 'success_repeat0', 'success_repeat1'])
        for r in rows: writer.writerow([r['state_key'], r['view_id'], *r['angles_deg'], *r['p3'], *r['p4'], *r['success']])
    index = figures(rows, summary['pearson'], output, model_label='RLDX-1', font_path=font_path)
    write(output/'figure-index.json', index)
    r = summary['recovery']['p3']; p4 = summary['recovery']['p4']
    lines = ['# RLDX-1：完整固定视角操作与 Accel 空间对照', '',
        '同一批8开发任务×2初态×27位置×2操作重复，共864条唯一操作；复用126条，新增738条。无PPO训练、无新评分调用、无更换任务或评分。', '',
        '| 对照 | 完整操作成功率 |', '|---|---:|',
        f"| 参考位（原协议零角） | {r['reference_success']:.3%} |",
        f"| 全26个扰动点均匀平均 | {r['perturbed_success']:.3%} |",
        f"| 旧两个固定负yaw起点 | {summary['old_two_starts_success']:.3%} |",
        f"| 非参考26候选最低P3 | {r['selected_success']:.3%} |",
        f"| 非参考26候选最低P4（离线诊断） | {p4['selected_success']:.3%} |", '',
        f"最低P3相对全扰动平均：{100*r['argmin_net_gain_task_mean']:+.3f}pp，8任务聚类探索性95%区间 {np.asarray(r['argmin_net_gain_task_cluster_95'])*100} pp。",
        f"存在至少一个观察2/2成功的非参考视角：{r['states_with_observed_two_of_two']}/16初态；最低7分点中存在2/2视角：{r['states_with_low_group_two_of_two']}/16。这些是事后空间存在性诊断，不是可部署选位方法。", '',
        '| Pearson(−Accel, 成功率)，排除参考位 | P3 | P4诊断 |', '|---|---:|---:|']
    for field, label in [('favorable_r', '任务平均空间场'), ('pooled_favorable_r', '逐初态位置池化'), ('state_centered_favorable_r', '初态内中心化')]:
        vals = [summary['pearson'][p]['scopes']['nonreference'][field] for p in ('p3', 'p4')]
        lines.append('| '+label+' | '+' | '.join('未定义' if v is None else f'{v:+.4f}' for v in vals)+' |')
    lines.extend(['', '最低7分区域（排除参考位，选位不读成功标签）：'])
    for v in summary['score_validity_secondary']['aggregate']:
        if v['exclude_reference']:
            interval = [100*x for x in v['task_cluster_95_interval']]
            lines.append(f"- {v['score'].upper()}低7点相对均匀扰动的收益 {100*v['low_group_gain_task_mean']:+.3f}pp，8任务聚类探索性95%区间 {interval} pp；两评分成员最低点一致比例 {v['member_argmin_agreement']:.1%}。")
    lines.extend(['', '正相关支持低Accel/高成功率；平均空间场关联不等于逐初态可选中好位置。全部逐任务/逐初态相关与配对救回、损害保留在analysis.json。', '',
        '三维图横轴yaw、纵轴pitch，第三轴为评分或成功率。黑点是实测点、红星为参考位；三角面仅线性插值，统计只使用实际27点。P4来自已存原生四步轨迹，未换采样步数或新主评分。', '',
        '此处最低分选择查询整个固定候选池；有限子集曲线不是实际8查询PPO。端点复用的救回/损害配对不是独立恢复轨迹。2次操作不能可靠估计单点真实概率；8任务16初态为开发诊断，不支持全量留出泛化结论。', '',
        f"新增实际调用：科学{audit['scientific_model_calls']}＋工程{audit['engineering_model_calls']}＝{audit['total_new_model_calls']}，批准上限{audit['approved_new_call_limit']}。结果已结算，派发关闭。"])
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    write(output/'provenance.json', dict(checked_unix=time.time(), report_source_sha256=sha(__file__),
        row_sha256=sha(output/'rows.json'), font_sha256=sha(font_path),
        scientific_campaign_sha256=sha(Path(root)/'spatial-campaign.json'), dispatch_enabled=False))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'output', 'font'): parser.add_argument('--'+name, required=True)
    args = parser.parse_args(); build(args.root, args.output, font_path=args.font)
