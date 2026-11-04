"""Assemble a complete fixed spatial panel from verified new and reused records."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from AlphaBrain.research.dsol.analysis.spatial_score_validity import summarize
from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores
from AlphaBrain.research.dsol.execution.manifest import file_digest


def read(path):
    return json.loads(Path(path).read_text())


def load_verified_panel(root, *, prefixes=(3,9), expected_states=16):
    """Read measurements and provenance without regenerating published reports."""
    root = Path(root)
    if read(root/'completion.json')['status'] != 'COMPLETE':
        raise ValueError('Cannot report a partial campaign as complete')
    c = read(root/'campaign.json'); reuse = read(c['reuse_manifest'])
    for path, stamp in reuse['artifacts'].items():
        if file_digest(path) != stamp: raise ValueError('Reused result changed')
    positions = {(x['state_key'], x['view_id']): dict(x) for x in reuse['points']}
    binding_paths = list((root/'execution').glob('*-real/binding.json'))
    if len(binding_paths) != 1: raise ValueError('Explicit single real attempt required')
    from benchmarks.LIBERO.controlled_target_panel import validate_block
    binding = read(binding_paths[0])
    for spec in binding['tasks'].values():
        validate_block(spec['request'], spec['output'])
        result = read(Path(spec['output'])/'results.json')
        for score in result['scores']:
            point = positions[(score['state_key'], score['view_id'])]
            if point['score_query'] is not None: raise ValueError('Duplicate scoring result')
            point['score_query'] = score['query_path']
        for task in result['tasks']:
            point = positions[(task['state_key'], task['view_id'])]
            repeat = str(task['repeat'])
            if repeat in point['executions']: raise ValueError('Duplicate operation outcome')
            point['executions'][repeat] = dict(result=task)
    rows = []
    for (state, view), point in sorted(positions.items()):
        if set(point['executions']) != {'0', '1'} or point['score_query'] is None:
            raise ValueError('Missing spatial observation')
        scores = {p: [] for p in prefixes}
        with np.load(point['score_query'], allow_pickle=False) as z:
            for member in (0, 1):
                metrics = compute_accel_scores(z[f'trace_{member}'][None], prefixes=prefixes)
                if any(metrics[p]['degenerate'][0] for p in prefixes):
                    raise ValueError('Degenerate score requires explicit diagnostic handling')
                for p in prefixes: scores[p].append(float(metrics[p]['score'][0]))
                if abs(scores[3][-1]-float(z[f'score_{member}'])) > 1e-12:
                    raise ValueError('Stored p3 no longer matches original score')
        outcomes = [point['executions'][str(i)]['result']['success'] for i in (0, 1)]
        rows.append(dict(state_key=state, task=state.split('::')[2], view_id=view,
            angles_deg=point['angles_deg'], **{f'p{p}':scores[p] for p in prefixes},success=outcomes))
    if len(rows) != expected_states*27 or len({r['state_key'] for r in rows}) != expected_states:
        raise ValueError('Expected complete declared spatial panel')
    return rows, positions


def build(root, output):
    rows, _ = load_verified_panel(root)
    summary = summarize(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('rows.json', rows), ('summary.json', summary)]:
        (output/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    with (output/'spatial.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['state_key','view_id','yaw_deg','pitch_deg','p3','p9','successes','repeats'])
        for r in rows: writer.writerow([r['state_key'], r['view_id'], *r['angles_deg'],
                                        np.mean(r['p3']), np.mean(r['p9']), sum(r['success']), 2])
    text = ['FastWAM 固定候选空间完整诊断',
            '8开发任务×2初态×27固定视角；每点2次操作。全体样本保留，不是未见任务泛化结论。',
            '低分组为每个初态评分最低的7个位置；比较对象是该初态全部候选等概率选择。',
            '区间按8个任务聚类自举，重复操作/初态不冒充独立任务；分数绝对值不跨模型比较。']
    for a in summary['aggregate']:
        ci = a['task_cluster_95_interval']
        text.append(f"{a['score']} / 排除参考位={a['exclude_reference']}: 低分组净增"
                    f"{100*a['low_group_gain_task_mean']:.2f}个百分点，任务聚类95%区间"
                    f"[{100*ci[0]:.2f}, {100*ci[1]:.2f}]；最低分单点成功率"
                    f"{100*a['argmin_success_state_mean']:.2f}%。")
    (output/'README.txt').write_text('\n'.join(text)+'\n')
    from AlphaBrain.research.dsol.plotting.spatial_score_validity import figures
    figures(rows, output)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); build(args.root, args.output)
