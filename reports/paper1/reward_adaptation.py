"""CPU report from every completed pilot outcome, with hash-bound baselines."""
from pathlib import Path
import argparse
import csv
import json
import datetime
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.execution.manifest import file_digest,digest
from AlphaBrain.research.dsol.analysis.reward_adaptation import summarize


def build(root, output, *, config_path=None):
    root=Path(root);output=Path(output)
    if output.exists():raise ValueError('Report output must be new')
    current=read(root/'current.json')
    if current['status']!='ALL_SCIENTIFIC_WORK_COMPLETE' or current.get('dispatch_enabled') is not False:
        raise ValueError('All scientific work must finish and dispatch close before the complete report')
    config=read(config_path or root/'execution-config.json')
    if file_digest(config['baseline_references'])!=config['input_hashes'][config['baseline_references']]:
        raise ValueError('Baseline reference manifest changed')
    refs=read(config['baseline_references'])['views']
    allocation=read(Path(config.get('evaluation_root',str(root/'evaluation')))/'allocation.json')
    all_episodes=[];all_searches=[];evidence={};baseline_cache={}
    def checked(path,expected):
        if file_digest(path)!=expected:raise ValueError('Report input hash changed: '+str(path))
        evidence[str(path)]=expected
        return read(path)
    def base(state,view,repeat):
        key=(state,view,repeat)
        if key not in baseline_cache:
            ref=refs[f'{state}::{view}'];path=ref['episode_results'][repeat]
            value=checked(path,ref['artifacts'][path])
            if (value['state_key'],value['view_id'],value['repeat'])!=key:raise ValueError('Baseline identity differs')
            baseline_cache[key]=value
        return baseline_cache[key]
    calls=0
    for phase in allocation['phases']:
        campaign=checked(phase['path'],phase['sha256'])
        directory=root/'execution'/(phase['phase']+'-real')
        complete=read(directory/'completion.json')
        if complete['status']!='COMPLETE' or complete['dispatch_enabled']:raise ValueError('Unsettled evaluation phase')
        calls+=complete['actual_model_calls']
        binding=read(directory/'binding.json')
        for spec in binding['tasks'].values():
            out=Path(spec['output']);receipt=read(out/'complete.json')
            if receipt['status']!='COMPLETE' or receipt['request_sha256']!=digest(spec['request']):raise ValueError('Unbound result receipt')
            result=checked(out/'results.json',receipt['artifacts']['results.json'])
            expected={(r['state_key'],r['view_id'],r['method']) for r in spec['request']['block']['rows']}
            if {(r['state_key'],r['view_id'],r['method']) for r in result['searches']}!=expected:raise ValueError('Changed result panel')
            all_searches.extend(result['searches'])
            for task in result['tasks']:
                stay=base(task['state_key'],task['view_id'],task['repeat']);nominal=base(task['state_key'],0,task['repeat'])
                n=min(task['replans'],stay['replans'])
                if task['execution_noise_sha256'][:n]!=stay['execution_noise_sha256'][:n]:raise ValueError('Operation noises differ')
                all_episodes.append(dict(task,stay_success=stay['success'],nominal_success=nominal['success']))
    if len(all_episodes)!=1152 or len(all_searches)!=576 or calls!=current['actual_model_calls']['evaluation']:
        raise ValueError('Final coverage or accounting differs')
    summary=summarize(all_episodes,all_searches)
    if summary['baseline']['tasks']!=8 or summary['baseline']['unique_initials']!=16:
        raise ValueError('Pilot task/initial coverage differs')
    training={}
    for name in summary['per_method']:
        directory=root/'training/runs'/name
        episodes=[read(p)['metrics'] for p in sorted((directory/'episodes').glob('*.json'))]
        updates=[read(directory/'updates'/f'{i:03d}.json') for i in range(1,9)]
        if len(episodes)!=256 or updates[-1]['completed_episodes']!=256:raise ValueError('Incomplete training branch')
        curves=[]
        for i,u in enumerate(updates):
            rows=episodes[i*32:(i+1)*32];reward=u['reward_settlement']
            if reward['count']!=(i+1)*32 or reward['update']!=i+1 or reward['kind']!=name.split('-')[0]:
                raise ValueError('Online statistic identity/count differs')
            curves.append(dict(update=i+1,scale=reward['scale'],tau=reward['tau'],noise_q75=reward['disagreement_q75'],
                reward_mean=float(np.mean(reward['rewards'])),mean_queries=float(np.mean([r['queries'] for r in rows])),
                immediate_stop_rate=float(np.mean([r['queries']==1 and r['reason']=='stop' for r in rows])),ppo_means=u['means']))
        training[name]=dict(episodes=256,updates=8,calls=2*sum(r['queries'] for r in episodes),curves=curves)
    recovery_accounting={}
    if config.get('continuation_recovery'):
        path=config['continuation_recovery']
        proof=checked(path,config['input_hashes'][path])
        recovery_accounting=dict(interrupted_attempts=proof['interrupted_attempts'],
            reused_calls=proof['reused_calls'],reused_complete_episodes=len(proof['records']),
            discarded_attempts=proof['interrupted_attempts']-proof['reused_calls'],
            held_failed_phase_cap=proof['held_failed_phase_cap'])
    if sum(t['calls'] for t in training.values())+recovery_accounting.get('discarded_attempts',0)!=current['actual_model_calls']['training']:
        raise ValueError('Training model-call settlement differs')
    summary.update(training=training,recovery_accounting=recovery_accounting,actual_model_calls=current['actual_model_calls'],
        report_created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol_sha256=file_digest(root/'protocol/adaptation.json'),
        report_scope='Full approved development pilot, no dropped scientific failures',
        source_manifest=config['source_manifest'])
    output.mkdir(parents=True,exist_ok=False)
    write(output/'summary.json',summary);write(output/'input-hashes.json',evidence)
    fields=['method','task','state_key','view_id','repeat','success','stay_success','nominal_success','queries','selected_angles']
    with (output/'paired-outcomes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader();writer.writerows(all_episodes)
    write(output/'search-diagnostics.json',all_searches)
    figure(summary,output)
    b=summary['baseline'];lines=['# B/G/L在线奖励小规模验证','',
        '完整已批准开发面板：8任务 × 2初态 × 4扰动起点；三奖励 × 三种子。每策略64搜索、128操作轨迹；全部576搜索/1152轨迹纳入。',
        '',f"同面板扰动保持成功率：{b['stay_success_rate']:.2%}；直接回规范参考：{b['nominal_success_rate']:.2%}。基线从已验收原始结果按初态/视角/执行噪声复用，没有重新执行或将共享基线当作九次独立观测。",'',
        '| 奖励 | 三种子平均成功率 | 相对扰动保持净增(pp) | 救回/损害(三种子合计) | 平均查询 | 三种子成功率 |',
        '|---|---:|---:|---:|---:|---|']
    for arm,a in summary['arms'].items():
        lines.append(f"| {arm} | {a['success_rate']:.2%} | {a['net_recovery_pp']:+.2f} | {a['rescue']}/{a['harm']} | {a['mean_queries']:.3f} | "+', '.join(f'{v:.2%}' for v in a['seed_success_rates'])+' |')
    lines+=['','相同种子与任务配对，相对继续训练B的差异：']
    for name,c in summary['contrasts'].items():
        lo,hi=c['exploratory_crossed_task_seed_interval_pp']
        lines.append(f"- {name}：{c['success_difference_pp']:+.2f} pp；任务/种子交叉重采样的探索性95%区间[{lo:+.2f}, {hi:+.2f}] pp。")
    lines+=['','该区间只有8任务、3训练种子，不能当大样本确认性结论。视角与执行重复在任务内相关。',
        '', 'B固定原分母；G从正式交互初始分累计估计S；L用带tau的log相对改善，tau由同批正式数据确定。每批采集结束后统一结算reward/GAE，优化期间固定。G/L没有额外预校准调用，但三组均从此前B检查点暖启动，不能声称已证明从零无校准训练。',
        '', '独立候选/参考复查只用于诊断，使用相同两噪声成员，不参与选位。两成员一致不是统计置信保证；低Accel不等于成功。各奖励单位不同，不能直接比较reward大小。',
        '',f"实际模型调用：训练{summary['actual_model_calls']['training']}，评测{summary['actual_model_calls']['evaluation']}，工程{summary['actual_model_calls']['engineering']}；总上限169216。",'',
        '![结果与训练诊断](results.png)','',
        '训练端点固定8更新/256回合，没有按评测效果选择检查点。失败与显式接续证据保留在run内recovery-*；完整回合校验后复用，中断调用仍计费且失败阶段预留不退款；原始实验、冻结操作模型、科学输入及原预算未覆盖。']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    write(output/'completion.json',dict(status='COMPLETE_CPU_REPORT',model_calls=0,
        searches=576,operation_episodes=1152,input_json_files=len(evidence),summary_sha256=file_digest(output/'summary.json')))
    return summary


def figure(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(B='#666666',G='#277da8',L='#e07a24')
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for index,(arm,a) in enumerate(summary['arms'].items()):
        axs[0,0].bar(index,100*a['success_rate'],color=colors[arm],alpha=.55)
        axs[0,0].scatter([index-.12,index,index+.12],100*np.array(a['seed_success_rates']),color=colors[arm],zorder=3)
        axs[0,1].bar(index,a['mean_queries'],color=colors[arm])
        for seed in range(3):
            curve=summary['training'][f'{arm}-seed-{seed}']['curves'];x=[r['update'] for r in curve]
            axs[1,0].plot(x,[r['mean_queries'] for r in curve],color=colors[arm],alpha=.65,label=arm if seed==0 else None)
            axs[1,1].plot(x,[r['scale'] for r in curve],color=colors[arm],alpha=.65,label=arm if seed==0 else None)
    axs[0,0].axhline(100*summary['baseline']['stay_success_rate'],color='black',ls='--',label='Perturbed stay')
    axs[0,0].axhline(100*summary['baseline']['nominal_success_rate'],color='black',ls=':',label='Nominal return')
    axs[0,0].set(title='Operation success (dots: training seeds)',ylabel='Success (%)',ylim=(0,105));axs[0,0].legend(fontsize=8)
    axs[0,1].set(title='Evaluation search cost',ylabel='Queries per search',ylim=(0,8.5))
    for ax in axs[0]:ax.set_xticks(range(3),['B: fixed','G: online','L: relative'])
    axs[1,0].set(title='Formal training query usage',xlabel='Adaptation update',ylabel='Queries per search',ylim=(0,8.5));axs[1,0].legend()
    axs[1,1].set(title='Online initial-score median S',xlabel='Adaptation update',ylabel='Accel');axs[1,1].legend()
    fig.suptitle('Warm-start development pilot: 8 tasks, 16 initials, 3 seeds per arm')
    fig.savefig(output/'results.png',dpi=160);fig.savefig(output/'results.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config')
    args=p.parse_args();build(args.run,args.output,config_path=args.config)
