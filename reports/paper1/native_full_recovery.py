"""Compose every native baseline/recovery outcome, including failures and harm."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,json
import numpy as np
from AlphaBrain.research.dsol.analysis.recovery_summary import paired_summary
from AlphaBrain.research.dsol.execution.manifest import file_digest
from AlphaBrain.research.dsol.data.artifacts import read,write


def csv_rows(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def build(root,output):
    root=Path(root);output=Path(output);output.mkdir(exist_ok=False)
    refs=read(root/'protocol/recovery/baseline-references.json')['views']
    baseline=read(root/'protocol/campaign.json');states={r['state_key']:r for r in baseline['states']}
    original=read(root/'protocol/evaluation-inputs.json')['records']
    baseline_rows=[];outcomes={}
    for key,ref in refs.items():
        for p,h in ref['artifacts'].items():
            if file_digest(p)!=h:raise ValueError('Baseline reference changed')
        state,view=key.rsplit('::',1);view=int(view);identity=states[state]
        case=Path(ref['query']).parent;scores=read(case/'scores.json')
        result=[read(p) for p in ref['episode_results']]
        for value in result:outcomes[(state,view,value['repeat'])]=value
        with np.load(ref['query'],allow_pickle=False) as z:pose=z['camera_to_world_opencv'].reshape(-1).tolist()
        row=dict(task=identity['task'],state_key=state,split=original[identity['initial']]['state']['split'],
            view_id=view,yaw_deg=scores[0]['angles_deg'][0],pitch_deg=scores[0]['angles_deg'][1],
            accel_selection=float(np.mean([s['score'] for s in scores if s['member'] in (0,1)])),
            accel_independent=float(np.mean([s['score'] for s in scores if s['member'] in (2,3)])),
            success_repeats=sum(t['success'] for t in result),execution_repeats=2,
            camera_to_world_opencv=json.dumps(pose))
        baseline_rows.append(row)
    assert len(baseline_rows)==10800 and len(outcomes)==21600
    allocation=read(root/'protocol/recovery/allocation.json');paired=defaultdict(list);spatial=[];searches=[]
    from benchmarks.LIBERO.controlled_recovery import validate_block
    for phase in allocation['phases']:
        directory=root/'execution-recovery'/(phase['phase']+'-real')
        assert read(directory/'completion.json')['status']=='COMPLETE'
        for spec in read(directory/'binding.json')['tasks'].values():
            folder=Path(spec['output']);validate_block(spec['request'],folder);results=read(folder/'results.json')
            for i,s in enumerate(results['searches']):
                assert not s['validation_used_for_selection']
                proof=read(folder/f'search-{i:03d}/nominal-return.json')
                assert proof['status']=='PASS_EXACT_INPUT_AND_PHYSICAL_STATE'
                with np.load(folder/f'search-{i:03d}/independent-validation/query-01.npz',allow_pickle=False) as z:
                    pose=z['camera_to_world_opencv'].reshape(-1).tolist()
                spatial.append(dict(task=s['task'],state_key=s['state_key'],split=s['split'],view_id=s['view_id'],method=s['method'],
                    selected_yaw_deg=s['selected_angles'][0],selected_pitch_deg=s['selected_angles'][1],
                    initial_accel=s['initial_accel'],minimum_visited_accel=s['best_accel'],
                    independent_selected_accel=float(np.mean(s['validation_scores'])),search_queries=s['queries'],
                    validation_queries=s['validation_queries'],camera_to_world_opencv=json.dumps(pose),
                    success_repeats=sum(t['success'] for t in results['tasks'] if t['state_key']==s['state_key'] and t['view_id']==s['view_id'] and t['method']==s['method'])))
                searches.append(s)
            for t in results['tasks']:
                stay=outcomes[(t['state_key'],t['view_id'],t['repeat'])]
                nominal=outcomes[(t['state_key'],0,t['repeat'])]
                row={k:t[k] for k in ['task','state_key','view_id','repeat','method','success']}
                row.update(stay_success=stay['success'],nominal_success=nominal['success'])
                paired[(t['split'],t['method'])].append(row)
    assert len(spatial)==18240 and sum(len(v) for v in paired.values())==36480
    metrics={f'{split}/{method}':paired_summary(rows) for (split,method),rows in sorted(paired.items())}
    # Dense perturbation loss uses all 26 noncanonical points, separately from
    # the eight-start recovery population. Never substitute one denominator.
    dense=[]
    for (state,view,repeat),value in outcomes.items():
        if view:
            dense.append(dict(task=states[state]['task'],state_key=state,view_id=view,repeat=repeat,method='perturbed_stay',
                success=value['success'],stay_success=value['success'],nominal_success=outcomes[(state,0,repeat)]['success']))
    write(output/'metrics.json',dict(status='COMPLETE_ALL_OUTCOMES',baseline_dense=paired_summary(dense),recovery=metrics,
        baseline_episodes=21600,recovery_episodes=36480,recovery_searches=18240,
        nominal_return='Same canonical outcome reused only after exact round-trip physical/input proof; same model/noise/horizon.',
        independent_score_is_selection_free=True,operation_model_updates=0,
        limitations=['Virtual fixed-optical-center pan/tilt, not physical head dynamics.',
          '40 LIBERO tasks; using LIBERO-plus runtime does not constitute the full LIBERO-plus benchmark.',
          'Seen means task membership; evaluation initial states are disjoint from training.',
          'Fixed-panel task-cluster intervals do not establish transfer outside the sampled domain.',
          'No reward/hyperparameter tuning using final task outcomes.']))
    csv_rows(output/'baseline-spatial.csv',baseline_rows);csv_rows(output/'recovery-spatial.csv',spatial)
    lines=['# 原生模型完整评测与 A/B 恢复对照','',
        '完整保留负结果；成功率按任务等权。规范视角、扰动不调整和恢复后的成功率使用各自注明的配对总体。',
        '', '| 划分/方法 | 任务数 | 规范视角 | 扰动不调整 | 恢复后 | 净救回百分点 | 救回/额外损害 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,value in metrics.items():
        m=value['task_macro'];lines.append(f"| {name} | {value['task_count']} | {100*m['nominal']:.2f}% | {100*m['stay']:.2f}% | {100*m['success']:.2f}% | {100*m['net_recovery']:+.2f} | {value['rescue_count']}/{value['harm_count']} |")
    lines+=['','置信区间、逐任务结果见 metrics.json；三维相机位姿、独立 Accel、真实成功率见两个 spatial.csv。',
        '随机对照的 minimum_visited_accel 是候选集合最低值，不是随机选中位置的分数；选中位置看 independent_selected_accel。',
        '独立评分没有参与最终选位；不把降 Accel、增 reward 或增加查询当作任务恢复。',
        '本轮是虚拟固定光心视角变化与 LIBERO 40 任务，不代表真实机器人头部运动或完整 LIBERO-plus 泛化。','']
    (output/'README.md').write_text('\n'.join(lines))
    return dict(status='COMPLETE',report=str(output/'README.md'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(build(a.run,a.output)))
