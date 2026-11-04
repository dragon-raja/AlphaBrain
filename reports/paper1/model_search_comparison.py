"""Compose audited grid-selection and PPO results; never launch experiments."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from AlphaBrain.research.dsol.analysis.native_baseline_diagnostic import analyze
from AlphaBrain.research.dsol.execution.manifest import file_digest


def build(config, output):
    cfg=json.loads(Path(config).read_text());out=Path(output)
    if out.exists():raise ValueError('New report directory required')
    for path,stamp in cfg['hashes'].items():
        if file_digest(path)!=stamp:raise ValueError('Changed report input: '+path)
    def read(key):return json.loads(Path(cfg['inputs'][key]).read_text())
    native=read('native_panel');fw=read('fastwam_panel');gr=read('groot_panel')
    identity={(r['state_key'],r['view_id']):r['angles_deg'] for r in fw}
    if len(fw)!=432 or len(identity)!=432:raise ValueError('Expected complete FastWAM development panel')
    matched=[r for r in native if (r['state_key'],r['view_id']) in identity]
    for records,angles in [(matched,lambda r:[r['yaw_deg'],r['pitch_deg']]),(gr,lambda r:r['angles_deg'])]:
        if len(records)!=432 or {(r['state_key'],r['view_id']) for r in records}!=set(identity):raise ValueError('Model panel identity mismatch')
        if any(not np.allclose(angles(r),identity[r['state_key'],r['view_id']],rtol=0,atol=1e-12) for r in records):raise ValueError('Model panel geometry mismatch')
    # Selection and scientific coverage validation use the existing core analyzer.
    def panel_summary(rows,score):
        converted=[dict(r,accel=float(np.mean(r[score])),independent_accel=float(np.mean(r[score])),suite=r['state_key'].split('::')[1],split='development',yaw_deg=identity[r['state_key'],r['view_id']][0],pitch_deg=identity[r['state_key'],r['view_id']][1]) for r in rows]
        a=analyze(converted)
        return dict(reference_successes=a['nominal_successes'],reference_episodes=a['nominal_episodes'],perturbed_successes=a['perturbed_successes'],perturbed_episodes=a['perturbed_episodes'],min26_success=a['offline_selectors']['min_accel_26']['task_macro']['success'],min27_success=a['offline_selectors']['min_accel_27']['task_macro']['success'],selected_distinct_episodes=2*a['states'])
    pi=[dict(r,p3=[r['accel']]) for r in matched]
    grid={'pi05':{'archived':panel_summary(pi,'p3')},'fastwam':{n:panel_summary(fw,n) for n in ('p3','p9')},'groot':{n:panel_summary(gr,n) for n in ('p3','p4')}}
    abg=read('abg_comparison')['aligned'];full={}
    for panel_split,split in [('train','seen_new_initials'),('validation','validation'),('test_unseen_task','unseen_test')]:
        rows=[r for r in native if r['split']==panel_split];a=analyze(rows);entry=abg[split]
        minimum=a['offline_selectors']['min_accel_26']['task_macro']['success']
        full[split]=dict(tasks=a['tasks'],states=a['states'],offline_min26_success=minimum,offline_distinct_episodes=2*a['states'],stay_recovery=entry['A']['stay'],reference=entry['A']['nominal'],B_success=entry['B']['success'],G_success=entry['G']['success'],B_minus_grid_pp=100*(entry['B']['success']-minimum),G_minus_grid_pp=100*(entry['G']['success']-minimum),B_queries=entry['B']['mean_queries'],G_queries=entry['G']['mean_queries'],policy_seeds=entry['G']['seeds'])
    states={k[0] for k in identity};g_outcomes=list(csv.DictReader(Path(cfg['inputs']['g_outcomes']).open()));ab_outcomes=list(csv.DictReader(Path(cfg['inputs']['ab_outcomes']).open()))
    subset=[r for r in g_outcomes if r['state_key'] in states and r['method']=='G-seed-0']
    if len(subset)!=256 or len({(r['state_key'],r['view_id'],r['repeat']) for r in subset})!=256:raise ValueError('G matched cohort incomplete')
    b=[r for r in ab_outcomes if r['state_key'] in states and r['method']=='B-seed-0']
    if len(b)!=128 or {(r['state_key'],r['view_id']) for r in b}!={(r['state_key'],r['view_id']) for r in subset}:raise ValueError('B/G starting cohorts differ')
    matched_ppo=dict(states=16,start_trials=128,operation_episodes=256,B_success=sum(int(r['success_repeats']) for r in b)/256,G_success=sum(r['success']=='True' for r in subset)/256,stay=sum(r['stay_success']=='True' for r in subset)/256,offline_min26=grid['pi05']['archived']['min26_success'])
    # Endpoints are measured continuous angles; this does not attribute gains to continuity.
    searches=read('g_searches');angles=np.array([identity[next(iter(identity))[0],i] for i in range(27)])
    for scope,records in [('full',searches),('matched_seed0',[r for r in searches if r['state_key'] in states and r['method']=='G-seed-0'])]:
        end=np.array([r['selected_angles'] for r in records]);distance=np.linalg.norm(end[:,None,:]-angles[None,:,:],axis=-1).min(axis=1)
        matched_ppo[scope+'_continuous_endpoint']=dict(searches=len(records),not_equal_to_any_grid_point=int((distance>1e-6).sum()),tolerance_degrees=1e-6)
    f=read('fastwam_summary');methods=f['methods'];curve={}
    for kind in ('source-first_batch_fixed','source-window','random-first_batch_fixed','random-window'):
        curve[kind]={}
        for u in (0,1,4,8):
            names=[f'actor:{kind}-seed-{s}:u{u}' for s in range(3)]
            if not all(n in methods for n in names):continue
            rr=[methods[n] for n in names]
            if any(r['episodes']!=64 for r in rr):raise ValueError('FastWAM endpoint cohort changed')
            curve[kind][str(8*u)]=dict(success=float(np.mean([r['task_equal_success_rate'] for r in rr])),mean_queries=float(np.mean([r['mean_queries'] for r in rr])),policy_seeds=[0,1,2],episodes_per_seed=64)
    result=dict(grid=grid,pi05_full=full,pi05_matched_ppo=matched_ppo,fastwam_training_curves=curve,groot_ppo='NOT_RUN',new_model_calls=0,training_updates=0,limits=['Grid selection reuses 32 selected endpoints; repeated endpoints across starts/seeds are not independent samples.','Across models, task/initial/angle IDs match but native execution contracts and noise spaces differ.','Pi05 full comparison matches task/initial populations; grid uses 26 candidates, PPO at most eight including initial and return.','FastWAM PPO uses two predefined starts and shared pi05 visual features; 64 target episodes versus 2048 for pi05.','GR00T panel contains no search-policy transfer or PPO training.','Observed numerical differences do not establish statistical significance or an effect caused by continuous actions.'])
    out.mkdir(parents=True);(out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (out/'input-hashes.json').write_text(json.dumps(cfg['hashes'],indent=2)+'\n')
    lines=['三模型：固定候选与PPO核对（CPU分析，零新模型调用）','','一、同8开发任务/16初态/27位置；最低分均排除规范参考位。', '|模型/评分|规范|26扰动不调整|26候选最低分|','|---|---:|---:|---:|']
    for model,values in grid.items():
        for name,a in values.items():lines.append(f"|{model}/{name}|{a['reference_successes']/a['reference_episodes']:.2%}|{a['perturbed_successes']/a['perturbed_episodes']:.2%}|{a['min26_success']:.2%}|")
    lines+=['','二、π0.5完整40任务/400初态：候选选择与PPO。seen/dev为共同seed0，12未见任务为三种子均值。','|总体|候选最低分|B|G|B−候选(pp)|G−候选(pp)|','|---|---:|---:|---:|---:|---:|']
    for s,a in full.items():lines.append(f"|{s}|{a['offline_min26_success']:.2%}|{a['B_success']:.2%}|{a['G_success']:.2%}|{a['B_minus_grid_pp']:+.2f}|{a['G_minus_grid_pp']:+.2f}|")
    lines+=['','三、与三模型小面板严格匹配16初态的π0.5结果：',f"26候选最低分{matched_ppo['offline_min26']:.2%}；B{matched_ppo['B_success']:.2%}；G{matched_ppo['G_success']:.2%}；同PPO起点保持{matched_ppo['stay']:.2%}。B/G各256操作，网格选中端点32操作。",'','四、FastWAM保留全部四条件及32/64端点，不挑最好种子或训练时刻：',json.dumps(curve,ensure_ascii=False,indent=2),'','边界：',*result['limits']]
    (out/'README.txt').write_text('\n'.join(lines)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(build(a.config,a.output)))
