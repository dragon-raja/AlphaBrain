"""Compose aligned A/B/G tables from verified reports without new model calls."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


def build(ab_report, g_report, output):
    ab_report, g_report, output = map(Path, (ab_report, g_report, output))
    if output.exists():
        raise ValueError('New output directory required')
    sources = [ab_report / 'metrics.json', ab_report / 'recovery-spatial.csv',
               g_report / 'summary.json', g_report / 'paired-outcomes.csv']
    ab = json.loads(sources[0].read_text())['recovery']
    g = json.loads(sources[2].read_text())
    metrics = {**ab, **g['per_method']}
    searches = defaultdict(dict)
    with sources[1].open() as stream:
        for row in csv.DictReader(stream):
            key = (row['state_key'], int(row['view_id']))
            group = searches[row['split'] + '/' + row['method']]
            if key in group:
                raise ValueError('Duplicate historical search')
            group[key] = (row['task'], int(row['success_repeats']), float(row['search_queries']))
    paired = defaultdict(dict)
    with sources[3].open() as stream:
        for row in csv.DictReader(stream):
            group = paired[row['split'] + '/' + row['method']]
            key = (row['state_key'], int(row['view_id']), int(row['repeat']))
            if key in group or row['success'] not in ('True', 'False'):
                raise ValueError('Duplicate or invalid G outcome')
            group[key] = row
    for name, group in paired.items():
        for state, view, repeat in group:
            if repeat not in (0, 1) or (state, view, 1-repeat) not in group:
                raise ValueError('Two execution repeats required')
            if repeat:
                continue
            a, b = group[state, view, 0], group[state, view, 1]
            if a['task'] != b['task'] or a['queries'] != b['queries']:
                raise ValueError('Search metadata differs across repeats')
            searches[name][state, view] = (a['task'], int(a['success']=='True') + int(b['success']=='True'), float(a['queries']))
    details = {}
    for name, rows in searches.items():
        m = metrics[name]
        tasks = defaultdict(list)
        for task, successes, queries in rows.values():
            if successes not in (0, 1, 2) or not 1 <= queries <= 8:
                raise ValueError('Invalid result or query count')
            tasks[task].append(successes / 2)
        if len(rows)*2 != m['pairs'] or len(tasks) != m['task_count']:
            raise ValueError('Coverage disagrees with audited report')
        if abs(mean(map(mean, tasks.values())) - m['task_macro']['success']) > 1e-12:
            raise ValueError('Success disagrees with audited report')
        details[name] = dict(m['task_macro'], mean_queries=mean(x[2] for x in rows.values()),
                             initial_only_fraction=mean(x[2]==1 for x in rows.values()),
                             search_count=len(rows), task_count=len(tasks))
    splits = ('seen_new_initials', 'validation', 'unseen_test')
    table = {}
    for split in splits:
        reference = searches[split+'/A-seed-0']
        for name, rows in searches.items():
            if not name.startswith(split+'/'):
                continue
            if rows.keys() != reference.keys() or any(rows[k][0] != reference[k][0] for k in rows):
                raise ValueError('Methods do not share exact task/initial/start coverage')
            for field in ('stay', 'nominal'):
                if abs(details[name][field] - details[split+'/A-seed-0'][field]) > 1e-12:
                    raise ValueError('Reference baselines differ')
        seeds = (0, 1, 2) if split == 'unseen_test' else (0,)
        result = {}
        for method in ('A', 'B', 'G'):
            members = [details[f'{split}/{method}-seed-{s}'] for s in seeds]
            result[method] = {key: mean(x[key] for x in members) for key in
                              ('success', 'stay', 'nominal', 'rescue', 'harm', 'net_recovery', 'mean_queries', 'initial_only_fraction')}
            result[method]['seeds'] = list(seeds)
        for method in ('local', 'random_from_local', 'old_ppo'):
            result[method] = details.get(split+'/'+method)
        table[split] = result
    output.mkdir(parents=True)
    result = dict(aligned=table, per_seed=details, g_minus_b=g['matched_G_minus_B'],
                  g_all_three_seeds=g['aggregates'], new_model_calls=0,
                  scope='Seen/validation use seed0 for A/B/G; unseen uses all three. Missing random seen data stays missing.')
    (output/'comparison.json').write_text(json.dumps(result, indent=2))
    (output/'input-hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2))
    with (output/'per-seed.csv').open('w') as stream:
        writer = csv.DictWriter(stream, ['split_method', *next(iter(details.values())).keys()])
        writer.writeheader()
        writer.writerows(dict(split_method=k, **v) for k, v in details.items())
    lines = ['# 原生模型 A/B/G 完整对照', '',
             '同一冻结π0.5；40任务×10评测初态×8扰动起点×2执行噪声。训练任务新初态20任务、验证8任务、PPO未见12任务。',
             'A/B/G各三种子训练2048回合，但A/B在训练任务新初态与验证集仅评测seed0，主表G也取seed0；未见集取三种子均值。', '',
             '| 方法 | 训练任务新初态（seed0） | 验证（seed0） | PPO未见任务（三种子） |',
             '|---|---:|---:|---:|']
    labels = [('规范/回参考位', 'nominal'), ('扰动不调整', 'stay'), ('A 原尺度', 'A'),
              ('B 固定模型内尺度', 'B'), ('G 在线尺度', 'G'), ('局部 Accel', 'local'),
              ('随机候选对照', 'random_from_local'), ('旧 PPO', 'old_ppo')]
    for label, method in labels:
        values = []
        for split in splits:
            entry = table[split]
            value = entry['A'][method] if method in ('nominal', 'stay') else (entry[method]['success'] if entry[method] is not None else None)
            values.append('未采集' if value is None else f'{100*value:.2f}%')
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines += ['', '## 未见任务的恢复与查询', '',
              '| 方法 | 成功率 | 净恢复百分点 | 救回占全部配对试验 | 额外损害占全部配对试验 | 平均搜索查询 |',
              '|---|---:|---:|---:|---:|---:|']
    for method in ('A','B','G','local','random_from_local'):
        x=table['unseen_test'][method]
        lines.append(f"| {method} | {100*x['success']:.2f}% | {100*x['net_recovery']:+.2f} | {100*x['rescue']:.2f}% | {100*x['harm']:.2f}% | {x['mean_queries']:.2f} |")
    lines += ['', '## 未见任务逐种子', '', '| 种子 | A | B | G | G−B百分点 |', '|---|---:|---:|---:|---:|']
    for seed in range(3):
        a,b,c=[details[f'unseen_test/{method}-seed-{seed}']['success'] for method in ('A','B','G')]
        lines.append(f'| {seed} | {100*a:.2f}% | {100*b:.2f}% | {100*c:.2f}% | {100*(c-b):+.2f} |')
    lines += ['', '## 未见任务逐任务（三种子均值）', '', '| 任务 | A/保持 | B | G | G−B百分点 |', '|---|---:|---:|---:|---:|']
    task_maps={method:[{t['task']:t for t in metrics[f'unseen_test/{method}-seed-{seed}']['per_task']} for seed in range(3)] for method in ('A','B','G')}
    for task in sorted(task_maps['A'][0]):
        a,b,c=[mean(m[task]['success'] for m in task_maps[method]) for method in ('A','B','G')]
        lines.append(f'| {task} | {100*a:.2f}% | {100*b:.2f}% | {100*c:.2f}% | {100*(c-b):+.2f} |')
    lines += ['', '搜索查询含初始观察和返回重查；每查询两条评分噪声，不含独立诊断评分与操作执行调用。随机对照复用局部候选集合及其搜索成本，不是免费随机看一次。',
              '局部/随机没有PPO训练种子，固定对照只报告一次，不复制成三份独立样本。规范结果仅在旧报告的物理/输入往返等价验收条件下复用；不能推广到未知规范位真机。',
              'G三种子全体成功率仍为79.70%/68.85%/77.78%；前两项与主表seed0口径不同，不是结果更正。',
              'A评测全部仅做初始查询，成功率等于扰动保持；B/G都保留负结果。G−B未见集探索性任务/种子重采样区间跨零，不能声称等效、优于或稳定劣于B。',
              'PPO未见任务不等于操作VLA未见任务；测试结果此前已查看，是冻结确认而非新盲测。G仍保留原观察归一化与中心坐标先验。',
              '先核查冷启动训练轨迹/尺度微小差异、STOP和终点分布，再决定是否需要新实验；不因本表自动重训、换种子或选更优检查点。']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('ab-report','g-report','output'):
        parser.add_argument('--'+name, required=True)
    args=parser.parse_args()
    build(args.ab_report,args.g_report,args.output)
