#!/usr/bin/env python3
"""Add explicit optimistic and cross-noise search references to the seven rules."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import yaml

from analyze_view_metric_rules_v1 import analyze, source_bootstrap, RULES
from build_view_landscape_report_v1 import read_json, write_json, write_csv, sha256, require, finite_json, setup_plotting
from compare_matched_view_landscapes_v1 import CHECKPOINTS, CONFIGS, SOURCE_FIELDS, ASSET_HASH_FIELDS, GEOMETRY_FIELDS, BANK_MANIFEST_SHA256
from build_view_metric_comparison_brief_v1 import RULE_LABELS

ORACLES=('statewise_empirical_max_O32','statewise_crossfit_O16')
LABELS={**RULE_LABELS,ORACLES[0]:'逐状态经验 Oracle（同批 O32，乐观）',ORACLES[1]:'逐状态搜索（O16↔O16，交叉验证）'}


def load_verified(root,label):
    provenance=read_json(root/'provenance.json')
    require(provenance['schema']=='dsol_view_landscape_provenance_v1' and provenance['status'].startswith('PASS_'),'Upstream report not verified')
    require(provenance['checkpoint_sha256']==CHECKPOINTS[label],'Wrong model identity')
    require(sha256(root/'verified_arrays.npz')==provenance['cache_sha256'],'Array hash mismatch')
    require(provenance['dense']['bank_manifest_sha256']==BANK_MANIFEST_SHA256,'Not shared O32 bank')
    require(provenance['static_assets']['physics_and_image_join']=='PASS_ALL','Physics/image join not PASS')
    for item in provenance['file_stats']:
        st=Path(item['path']).stat()
        require((st.st_size,st.st_mtime_ns)==(item['size'],item['mtime_ns']),'Upstream input changed: '+item['path'])
    for item in provenance['dense']['inputs']:
        p=Path(item['path'])
        if p.name in ('run_manifest.json','audit.json'):
            require(sha256(p)==item['sha256'],'Upstream manifest/audit hash changed')
            r=read_json(p)
            if p.name=='audit.json':
                require(r['status']=='PASS_COMPLETE' and all(r[k] for k in ('every_policy_call_matches_frozen_noise_bank','paired_noise_identity_at_common_replan_indices','physics_hash_constant_within_state','environment_seed_constant_within_state')),'Invalid dense audit')
            else:
                require(r['checkpoint_sha256']==CHECKPOINTS[label] and r['replan_steps']==5 and r['wait_steps']==0 and r['require_explicit_noise'],'Dense controls differ')
    config=Path(provenance['checkpoint'])/'framework_config.yaml'
    require(sha256(config)==CONFIGS[label],'Model config differs')
    require(yaml.safe_load(config.read_text())['framework']['action_model']['num_inference_steps']==10,'Denoising grid differs')
    states=read_json(root/'states.json')['states']; poses=read_json(root/'candidate_geometry.json')['poses']
    with np.load(root/'verified_arrays.npz',allow_pickle=False) as data:
        result={k:data[k].copy() for k in ('success','accel','visibility')}
        ids=list(map(str,data['candidate_ids'])); keys=list(map(str,data['state_keys']))
    require(len(states)==len(set(keys)) and keys==[s['pair_key'] for s in states],'State order mismatch')
    require(len(ids)==len(set(ids))==97 and ids==[p['pose_id'] for p in poses] and ids[0]=='canonical','Candidate order mismatch')
    require(result['success'].shape==(len(states),97,32) and np.isin(result['success'],[0,1]).all(),'Missing/nonbinary O32 cell')
    require(result['accel'].shape==(len(states),97,8) and np.isfinite(result['accel']).all(),'Invalid Accel tensor')
    result.update(states=states,poses=poses,candidate_ids=ids,provenance=provenance,
        input_identity={name:{'path':str(root/name),'sha256':sha256(root/name)} for name in ('verified_arrays.npz','provenance.json','states.json','candidate_geometry.json')})
    return result


def oracle_rows(report):
    rows=[]
    for state,y in zip(report['states'],report['success']):
        identity={k:state[k] for k in ('pair_key','task_id','source_group','split')}
        rates=y.mean(1); full=int(np.argmax(rates))
        first=int(np.argmax(y[:,:16].mean(1))); last=int(np.argmax(y[:,16:].mean(1)))
        values={ORACLES[0]:float(rates[full]),ORACLES[1]:float((y[first,16:].mean()+y[last,:16].mean())/2)}
        for rule,value in values.items():
            rows.append({**identity,'rule':rule,'success':value,'gain_vs_canonical_pp':100*(value-rates[0]),
                'gain_vs_uniform_pp':100*(value-rates.mean()),'full32_selected_candidate':report['candidate_ids'][full],
                'first16_selected_candidate':report['candidate_ids'][first],'last16_selected_candidate':report['candidate_ids'][last]})
    return rows


def paired_training_rows(canonical,broad):
    require(len(canonical['states'])==len(broad['states'])==64,'Full comparison needs the same complete 64 states')
    require(canonical['candidate_ids']==broad['candidate_ids'],'Candidate order differs')
    for c,b in zip(canonical['states'],broad['states']):
        require(all(c[k]==b[k] for k in SOURCE_FIELDS),'Physical/source pairing differs')
        require(all(c['static_assets'][k]==b['static_assets'][k] for k in ASSET_HASH_FIELDS),'Image/physics pairing differs')
        require(c['accel_ensemble_seeds']==b['accel_ensemble_seeds'],'Accel noise differs')
    for c,b in zip(canonical['poses'],broad['poses']): require(all(c[k]==b[k] for k in GEOMETRY_FIELDS),'Candidate geometry differs')
    require(np.allclose(canonical['visibility'],broad['visibility'],rtol=0,atol=1e-12),'Visibility differs')


def analysis(report):
    result=analyze(report); oracle=oracle_rows(report)
    fields=('pair_key','task_id','source_group','split','rule','success','gain_vs_canonical_pp','gain_vs_uniform_pp')
    rows=[{k:r[k] for k in fields} for r in result['state_rule_metrics']+oracle]
    summary={**result['summary'],'schema':'dsol_view_metric_with_oracle_v2','rule_order':list(RULES+ORACLES)}
    summary['overall']['rules'].update({rule:{field:source_bootstrap([r for r in oracle if r['rule']==rule],field) for field in ('success','gain_vs_canonical_pp','gain_vs_uniform_pp')} for rule in ORACLES})
    summary['oracle_interpretation']={ORACLES[0]:'Per state choose one view maximizing O32 average and report the same average: optimistic empirical reference, not a population upper bound and not a per-noise clairvoyant selector.',
        ORACLES[1]:'Choose on first O16, test on last O16; reverse and average. Disjoint-noise search performance; not exact oracle, not independent-source selector generalization.'}
    summary['by_historical_split_rules']={split:{rule:{f:source_bootstrap([r for r in rows if r['split']==split and r['rule']==rule],f) for f in ('success','gain_vs_canonical_pp','gain_vs_uniform_pp')} for rule in RULES+ORACLES} for split in ('development','test')}
    summary['oracle_tie_break']='Frozen geometry order, canonical first; same as original landscape diagnostic. Metric rules retain their separately frozen historical ID ties.'
    return summary,rows,oracle


def plot(summaries,output,append_dir=None):
    setup_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager
    from matplotlib.backends.backend_pdf import PdfPages
    fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','pdf.fonttype':3,'axes.unicode_minus':False})
    with PdfPages(output/'view_metric_with_oracle_v2.pdf') as pdf:
        fig,axes=plt.subplots(len(summaries),3,figsize=(17,9 if len(summaries)==1 else 14),squeeze=False)
        fig.suptitle('指标筛选能兑现多少视角收益：加入经验 Oracle 与跨噪声搜索',x=.035,y=.98,ha='left',fontsize=20,fontweight='bold')
        fig.text(.035,.93,'同批最大值仅作乐观参照；跨噪声搜索用不重叠的两半结果选取与评价。二者都不是可直接部署的指标选择器。',fontsize=10)
        for row,(label,summary) in enumerate(summaries.items()):
            for col,(field,scale,title) in enumerate((('success',100,'成功率（%）'),('gain_vs_uniform_pp',1,'相对均匀候选（百分点）'),('gain_vs_canonical_pp',1,'相对规范视角（百分点）'))):
                ax=axes[row,col]
                for j,rule in enumerate(RULES+ORACLES):
                    value=summary['overall']['rules'][rule][field]; mean=scale*value['mean']; lo,hi=value['ci95']
                    color='#b66a20' if rule in ORACLES else '#2a6f97'
                    if lo is not None: ax.plot([scale*lo,scale*hi],[j,j],color=color,lw=1.4)
                    ax.scatter(mean,j,color=color,marker='D' if rule in ORACLES else 'o',s=32)
                    if col==0: ax.text(mean+.7,j-.12,f'{mean:.2f}',fontsize=8)
                ax.axhline(6.5,color='#999',lw=.8,ls='--')
                if col: ax.axvline(0,color='#777',lw=.8)
                else: ax.set_xlim(0,100)
                ax.set_yticks(range(9),[LABELS[r] for r in RULES+ORACLES] if col==0 else [])
                ax.set_ylim(8.6,-.7); ax.set_xlabel(title); ax.grid(axis='x',alpha=.15)
                ax.set_title(('多视角训练' if label=='broad' else '规范视角训练')+f" / {summary['state_count']} 状态",fontsize=11)
        fig.subplots_adjust(left=.27,right=.97,top=.85,bottom=.20,wspace=.21,hspace=.32)
        fig.text(.035,.125,'横线：固定任务下的来源 bootstrap 描述性 95% 区间。Oracle 参考使用成功标签，不能与只读图像/指标的规则混称为同类方法。',fontsize=10)
        fig.text(.035,.085,'逐状态 O32 最大值不等于真实期望成功率的无偏上限；O16 搜索失败也不证明真实 Oracle 不存在。保留原 48/16 历史来源划分。',fontsize=10)
        fig.text(.035,.045,'历史探索性分析；候选图像预先可用；外部相机在单次闭环中固定；不证明主动获取或未见任务泛化。',fontsize=9,color='#64748b')
        fig.savefig(output/'metric_with_oracle.png',dpi=150); pdf.savefig(fig); plt.close(fig)
        if append_dir:
            for name in ('02_same_state_three_metrics.png','03_crossfit_strata.png'):
                fig=plt.figure(figsize=(16,10 if name.startswith('02') else 9))
                ax=fig.add_axes((0,0,1,1)); ax.imshow(plt.imread(append_dir/name)); ax.axis('off'); pdf.savefig(fig); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--broad-root',type=Path,required=True); parser.add_argument('--canonical-root',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True); parser.add_argument('--append-brief-dir',type=Path)
    args=parser.parse_args(); require(not args.output_dir.exists(),'Output exists; refusing overwrite')
    reports={'broad':load_verified(args.broad_root,'broad')}
    if args.canonical_root:
        reports['canonical']=load_verified(args.canonical_root,'canonical'); paired_training_rows(reports['canonical'],reports['broad'])
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/'manifest.json',dict(schema='dsol_metric_oracle_manifest_v2',status='FROZEN_BEFORE_ORACLE_AGGREGATION',
        inputs={label:report['input_identity'] for label,report in reports.items()},rules=list(RULES+ORACLES),
        script_sha256=sha256(Path(__file__)),metric_core_sha256=sha256(Path(__file__).with_name('analyze_view_metric_rules_v1.py')),
        gpu_used=False,new_rollouts=0,noise_split=[list(range(16)),list(range(16,32))],tie_break='frozen geometry order; canonical first',
        appended_figures={name:sha256(args.append_brief_dir/name) for name in ('02_same_state_three_metrics.png','03_crossfit_strata.png')} if args.append_brief_dir else {}))
    summaries={}; all_rows={}
    for label,report in reports.items():
        summary,rows,oracle=analysis(report); summaries[label]=summary; all_rows[label]=rows
        write_json(args.output_dir/(label+'_summary.json'),summary)
        write_csv(args.output_dir/(label+'_state_rule_metrics.csv'),finite_json(rows)); write_csv(args.output_dir/(label+'_oracle_selections.csv'),finite_json(oracle))
    if 'canonical' in all_rows:
        canonical={(r['pair_key'],r['rule']):r for r in all_rows['canonical']}; paired=[]
        for b in all_rows['broad']:
            c=canonical[b['pair_key'],b['rule']]
            paired.append({**{k:b[k] for k in ('pair_key','task_id','source_group','split','rule')},
                'broad_minus_canonical_success_pp':100*(b['success']-c['success']),
                'training_interaction_vs_uniform_pp':b['gain_vs_uniform_pp']-c['gain_vs_uniform_pp'],
                'training_interaction_vs_canonical_view_pp':b['gain_vs_canonical_pp']-c['gain_vs_canonical_pp']})
        write_csv(args.output_dir/'paired_training_rule_metrics.csv',finite_json(paired))
        write_json(args.output_dir/'paired_training_summary.json',{rule:{f:source_bootstrap([r for r in paired if r['rule']==rule],f) for f in ('broad_minus_canonical_success_pp','training_interaction_vs_uniform_pp','training_interaction_vs_canonical_view_pp')} for rule in RULES+ORACLES})
    plot(summaries,args.output_dir,args.append_brief_dir)
    write_json(args.output_dir/'completion.json',dict(status='PASS_COMPLETE',outputs={p.name:sha256(p) for p in args.output_dir.iterdir() if p.is_file()},new_rollouts=0))
    print('PASS_COMPLETE',args.output_dir)


if __name__=='__main__': main()
