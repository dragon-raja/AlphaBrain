"""Completed development-only U/T screen: paired outcomes and audited short curves."""
from pathlib import Path
import argparse,csv
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.analysis.prefix_health import paired_diagnostic,physical_key,training_health
from AlphaBrain.research.dsol.plotting.core_screen import figures
from reports.paper1.core_selection import load_outcomes
from reports.paper1.prefix_health import Report


def build(config):
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=False)
    outcomes,sources=load_outcomes(config['roots'])
    baseline=Report(config['baseline_root'],out/'baseline-audit')
    baseline.pi_refs=baseline.load(config['pi_references'])
    seeds=config.get('expected_seeds',[0,1])
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Explicit unique policy seeds required')
    summary={};curves={};flat=[];contrasts={}
    for variant,rows in outcomes.items():
        if set(r['seed'] for r in rows)!=set(seeds):raise ValueError('Incomplete policy seed matrix')
        for r in rows:
            stay=baseline.pi_baseline(r);reference=baseline.pi_baseline(r,reference=True)
            baseline.check_noise(r,stay);baseline.check_noise(r,reference)
            if stay['horizon']!=r['horizon'] or reference['horizon']!=r['horizon']:raise ValueError('Baseline horizon differs')
            r.update(stay_success=stay['success'],nominal_success=reference['success'])
        diag=paired_diagnostic(rows)
        diag.update(status='COMPLETE_DEVELOPMENT_SCREEN',per_seed={str(seed):paired_diagnostic([r for r in rows if r['seed']==seed]) for seed in seeds},
                    mean_initial_accel=float(np.mean([r['search_initial_accel'] for r in rows])),mean_best_accel=float(np.mean([r['search_best_accel'] for r in rows])))
        summary[variant]=diag
        flat.extend(dict(variant=variant,**{k:r[k] for k in ['seed','suite','task','state_key','view_id','repeat','success','stay_success','nominal_success','queries','search_initial_accel','search_best_accel']}) for r in rows)
    maps={v:{(r['seed'],*physical_key(r)):r for r in rows} for v,rows in outcomes.items()}
    pairs=[(v,'U0') for v in outcomes if v!='U0']+[('U12','U3'),('U123','U12')]
    pairs.extend((v,v[:-1]) for v in outcomes if v.endswith('T'))
    for a,b in pairs:
        if a not in maps or b not in maps:continue
        if maps[a].keys()!=maps[b].keys():raise ValueError('Compared panels differ')
        rows=[dict(r,stay_success=maps[b][key]['success']) for key,r in maps[a].items()]
        contrasts[a+'-minus-'+b]=dict(comparator=b,**paired_diagnostic(rows))
    for root in map(Path,config['roots']):
        p=baseline.load(root/'protocol.json')
        for name,cell in p['cells'].items():
            folder=root/'training'/name
            updates=[baseline.load(folder/'updates'/f'{i:03d}.json') for i in range(1,p['updates']+1)]
            episodes=[baseline.load(folder/'episodes'/f'{i:04d}.json')['metrics'] for i in range(p['updates']*32)]
            curves[name]=training_health(updates,episodes,expected_updates=p['updates'])
    sources.update(baseline.sources)
    write(out/'summary.json',dict(status='COMPLETE_DEVELOPMENT_ONLY',variants=summary,contrasts=contrasts,
        unique_physical_operations_per_variant=64,policy_seed_outcomes_per_variant=64*len(seeds),independent_tasks=8,
        warning=f'Same 64 physical noise-repeat cases reused for {len(seeds)} policy seeds, not {64*len(seeds)} independent scenes. Task-cluster intervals exploratory; structures selected on this panel; no confirmatory significance or convergence claim.',model_calls=0))
    write(out/'learning-curves.json',curves);write(out/'input-hashes.json',sources)
    with (out/'paired-outcomes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    figures(summary,curves,out)
    lines=['π0.5 P10/G — 统一核心开发集筛选（完整端点，非最终主表）','',
           f'每结构{len(seeds)}种子×512训练回合，8开发任务×2初态×2起点×2执行噪声=每种子64操作。策略种子复用同一物理面板，不是{64*len(seeds)}个独立场景。',
           '任务编号/标签不用于模型输入；本分析不发起模型调用，不改在途T。','',
           '结构 | 各种子成功率 | 合并成功 | 相对Stay净提升pp | 救回/损害 | 平均查询']
    for v,d in summary.items():lines.append(f"{v} | {'; '.join(str(seed)+': '+format(100*d['per_seed'][str(seed)]['policy_success'],'.2f')+'%' for seed in seeds)} | {100*d['policy_success']:.2f}% | {d['net_pp']:+.2f} | {d['rescues']}/{d['harms']} | {d['mean_queries']:.3f}")
    first=next(iter(summary.values()));lines.extend(['',f"匹配Stay={100*first['stay_success']:.2f}%，参考位={100*first['reference_success']:.2f}%；不与40任务主表直接比百分点。",'U0基准；U1状态前融合；U2 Context加入Tanh；U3辅助腕部移入记录融合。组合不是保证单调收益。',
        '学习曲线以reward_settlement中的正式G奖励重算，核验累计训练初始分中位数、查询收费和有限损失。没有中途完整操作成功率，不能从奖励画出虚构的成功率收敛曲线。',
        '同面板开发候选排序仅用于既定T第三组；不凭探索性区间宣称已消除选择偏差或所有任务无损害。所有低结果保留。'])
    (out/'README.txt').write_text('\n'.join(lines)+'\n')
    return summary


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);args=a.parse_args();build(read(args.config))
