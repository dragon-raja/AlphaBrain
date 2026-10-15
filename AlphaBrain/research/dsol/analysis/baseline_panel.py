"""Audit and summarize the bounded baseline panel against matching P0 cells."""
from pathlib import Path
import argparse
import math

import numpy as np

from ..data.artifacts import read, sha, write
from ..data.diagnostic_panel import validate_diagnostic
from ..data.evaluation_matrix import flow_noise
from ..data.rollout_replay import array_identity
from .robocasa_statistics import interval


def summarize(root):
    root = Path(root); plan = read(root/'plan.json'); p0, reference = validate_diagnostic(plan)
    job_map = {j['initial_id']:j for j in reference['jobs']}
    jobs = [job_map[i] for i in plan['initial_ids']]
    extension = plan['format'] == 'robocasa_official_view_space_v1'
    views = plan['all_views'] if extension else plan['panel_views']
    intermediate = plan['format'] == 'robocasa_checkpoint_4000_panel_v1'
    names = ['official','canonical_4000','broad_4000','canonical','broad'] if intermediate else ['official','canonical','broad']
    baseline = read(plan['baseline_plan']) if intermediate or extension else plan
    outcomes = np.zeros((len(names),40,len(views),4), dtype=np.uint8)
    accel = np.zeros((len(names),40,len(views)))
    p0_root = Path(plan['p0_plan']).parent
    noise_cache = {}; audited = 0; broad_first_input_differences = 0
    source_records = []
    for ii, job in enumerate(jobs):
        initial = job['initial_id']
        worker = next(w for w in plan['workers'] if initial in w['initials'])
        for vi, view in enumerate(views):
            old_can = next(w for w in p0['workers'] if w['arm']=='canonical' and initial in w['initials'])
            can_cell = p0_root/'workers'/old_can['worker_id']/'cells'/initial/view
            old_first = read(can_cell/'execution_0.json')
            can_scores = read(can_cell/'scores.json')
            for mi, name in enumerate(names):
                is_new = name in plan['models'] and not (extension and view in plan['reuse_views'])
                if is_new:
                    cell = root/'workers'/worker['worker_id']/name/'cells'/initial/view
                elif name == 'official':
                    old_w = next(w for w in baseline['workers'] if initial in w['initials'])
                    cell = Path(plan['baseline_plan']).parent/'workers'/old_w['worker_id']/name/'cells'/initial/view
                else:
                    old_w = next(w for w in p0['workers'] if w['arm']==name and initial in w['initials'])
                    cell = p0_root/'workers'/old_w['worker_id']/'cells'/initial/view
                complete = read(cell/'complete.json'); scores = read(cell/'scores.json')
                if not complete['complete'] or len(complete['trials']) != 4:
                    raise ValueError('Incomplete cell: '+str(cell))
                if sha(cell/'scores.npz') != scores['arrays_sha256']:
                    raise ValueError('Scoring payload hash mismatch')
                if is_new:
                    if complete['provenance']['plan_sha256'] != sha(root/'plan.json'):
                        raise ValueError('Baseline plan provenance mismatch')
                    if scores['input_arrays'] != can_scores['input_arrays'] or scores['prompt'] != can_scores['prompt']:
                        raise ValueError('Baseline first input differs from frozen P0 canonical')
                    for check in ('canonical_pixels','candidate_pixels'):
                        if not all(v['pass'] for v in complete[check].values()):
                            raise ValueError('Render admission failed')
                if name == 'broad' and scores['input_arrays'] != can_scores['input_arrays']:
                    broad_first_input_differences += 1
                # All eight scoring noises, original prefix-three metric.
                if len(scores['rows']) != 8:
                    raise ValueError('Eight score repeats required')
                accel[mi,ii,vi] = np.mean([r['metrics']['3']['score'] for r in scores['rows']])
                for repeat, record in enumerate(complete['trials']):
                    file = cell/f'execution_{repeat}.json'; trial = read(file)
                    if sha(file) != record['sha256'] or trial['repeat'] != repeat or not trial['complete_task_attempt']:
                        raise ValueError('Trial identity or completion mismatch')
                    if trial['initial_physical_sha256'] != old_first['initial_physical_sha256']:
                        raise ValueError('Physical initial state mismatch')
                    if trial['plans'][0]['input_arrays'] != scores['input_arrays']:
                        raise ValueError('Scoring/execution first input mismatch')
                    if trial['horizon'] != p0['task_horizons'][job['task']] or len(trial['plans']) != math.ceil(trial['steps']/25):
                        raise ValueError('Execution horizon/replanning changed')
                    if is_new:
                        for n, query in enumerate(trial['plans']):
                            address = (initial,repeat,n)
                            if address not in noise_cache:
                                noise_cache[address] = array_identity(flow_noise(initial,'execution',repeat,n))
                            if query['noise'] != noise_cache[address]:
                                raise ValueError('Flow noise address changed')
                        audited += 1
                    outcomes[mi,ii,vi,repeat] = bool(trial['success'])
                    source_records.append(dict(path=str(file),sha256=record['sha256']))
    if audited != plan['rollout_count']:
        raise ValueError('New rollout count differs from frozen plan')
    tasks = np.array([j['task'] for j in jobs]); splits = np.array([j['split'] for j in jobs])
    q = outcomes.mean(-1)
    groups = {'canonical':[0], 'mild':[1,2], 'strong':[3,4], 'four_perturbed':[1,2,3,4]}
    if extension:
        catalog = {v['view_id']:v for v in reference['catalog']['views']}
        groups = {'canonical':[0], 'train_views':[i for i,v in enumerate(views) if catalog[v]['role']=='train'],
                  'extra_views':[i for i,v in enumerate(views) if catalog[v]['role']=='extra_eval'],
                  'all_perturbed':list(range(1,len(views)))}
    summary = dict(scope=plan['scope'],plan_sha256=sha(root/'plan.json'),
        baseline_rollouts=audited,reused_final_rollouts=2*40*len(views)*4, models=names,
        reused_official_rollouts=800 if intermediate or extension else 0, groups=list(groups),
        broad_vs_canonical_first_input_different_cells=broad_first_input_differences,
        first_input_note='New checkpoint trials reuse canonical P0 first input exactly; historical broad inputs retain native-render differences if present.',
        bootstrap='Paired task/initial hierarchical bootstrap; fixed four measured execution repeats, not exact population uncertainty.',
        views=read(root/'selection.json')['views'],splits={}, per_task=[])
    for split in ('pretrain','target'):
        ix = np.flatnonzero(splits==split); summary['splits'][split] = {}
        for mi,name in enumerate(names):
            row = {g:interval(q[mi,ix][:,vs].mean(1),tasks[ix]) for g,vs in groups.items()}
            row['drop_from_reference'] = interval(q[mi,ix,0]-q[mi,ix,1:].mean(1),tasks[ix])
            if mi:
                row['gain_over_official'] = {g:interval((q[mi,ix][:,vs]-q[0,ix][:,vs]).mean(1),tasks[ix]) for g,vs in groups.items()}
            if intermediate and name in ('canonical','broad'):
                earlier = names.index(name + '_4000')
                row['gain_8250_over_4000'] = {g:interval((q[mi,ix][:,vs]-q[earlier,ix][:,vs]).mean(1),tasks[ix]) for g,vs in groups.items()}
            # A five-view diagnostic, never mislabeled full-catalog selector.
            chosen = np.argmin(accel[mi,ix],axis=1)
            row['candidate_min_accel' if extension else 'five_view_min_accel'] = interval(q[mi,ix,chosen],tasks[ix])
            if extension:
                row['min_accel_gain_vs_extra'] = interval(q[mi,ix,chosen]-q[mi,ix][:,groups['extra_views']].mean(1),tasks[ix])
                row['empirical_max_same_noise'] = interval(q[mi,ix].max(1),tasks[ix])
            summary['splits'][split][name]=row
            for task in sorted(set(tasks[ix])):
                js=ix[tasks[ix]==task]
                summary['per_task'].append(dict(split=split,task=task,model=name,
                    **{g:float(q[mi,js][:,vs].mean()) for g,vs in groups.items()}))
    destination=root/'analysis';destination.mkdir(exist_ok=True)
    write(destination/'audit.json',dict(status='PASS',rollouts=audited,source_records=source_records,
        analyzer_sha256=sha(__file__),plan_sha256=sha(root/'plan.json')))
    write(destination/'summary.json',summary)
    np.savez_compressed(destination/'panel_arrays.npz', success=outcomes, accel=accel,
                        models=np.asarray(names), initials=np.asarray(plan['initial_ids']), views=np.asarray(views))
    lines=[f'# RoboCasa 检查点对照：{audited} 次新增诊断','',
        f'10 个已见任务；每任务 2 个 seen 和 2 个 heldout 初态；{len(views)}个候选；每格 4 次。',
        ('两份4000步权重各新跑800次；官方起点800次与8250步端点1600次复用。' if intermediate else '官方起点新跑800次；最终模型1600次复用。') +
        '不是全候选平均、Oracle或新方法泛化证明。','']
    if extension:
        lines = [f'# RoboCasa 官方模型全候选诊断：{audited} 次新增闭环','',
            '固定原40初态（seen/heldout各20），每初态全97候选、4执行噪声、8评分噪声。',
            '官方新增14720次，复用原800次；两份8250步模型复用完全对应的31040次。',
            '这是40初态诊断，不是原400初态总体。候选先对4噪声取平均再取最大；同批选优为乐观经验值，不是已独立确认的Oracle。','']
    for split in ('pretrain','target'):
        columns = '规范 | 64训练候选 | 32额外候选 | 96扰动候选' if extension else '规范 | 较轻扰动 | 较强扰动 | 四扰动平均'
        lines += ['## '+split,'','| 模型 | '+columns+' |','|---|---:|---:|---:|---:|']
        for name,row in summary['splits'][split].items():
            values=[100*row[g]['mean'] for g in groups]
            lines.append('| '+name+' | '+' | '.join(f'{x:.2f}%' for x in values)+' |')
        lines.append('')
        if extension:
            lines += ['| 模型 | 最低Accel选位 | 相对32额外视角平均 | 同批经验最大值（乐观） |', '|---|---:|---:|---:|']
            for name,row in summary['splits'][split].items():
                lines.append(f"| {name} | {100*row['candidate_min_accel']['mean']:.2f}% | {100*row['min_accel_gain_vs_extra']['mean']:+.2f} pp | {100*row['empirical_max_same_noise']['mean']:.2f}% |")
            lines.append('')
    lines += ['原始 native renderer 风险保留；置信区间见 summary.json。',
              '候选及初态在评测前冻结，不按新增结果删除困难任务或挑选视角。',
              '完整推论需结合任务分层和配对区间，不能只用均值认定训练充分或不足。']
    (destination/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    result=summarize(args.root)
    print({'status':'PASS','baseline_rollouts':result['baseline_rollouts']})


if __name__=='__main__':
    main()
