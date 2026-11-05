"""Read-only fixed-snapshot health report for independently owned prefix runs."""
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
import argparse,csv,json,time,datetime

from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.execution.manifest import file_digest as sha,digest
from AlphaBrain.research.dsol.analysis.prefix_health import training_health,physical_key,paired_diagnostic,complete_three_seed_tasks,development_progress
from AlphaBrain.research.dsol.plotting.prefix_health import learning_figures


class Report:
    def __init__(self, root, output):
        self.root=Path(root);self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.sources={};self.p=read(self.root/'protocol.json');self.bindings=read(self.root/'run-bindings.json')
        self.final=defaultdict(list);self.development=defaultdict(list);self.baseline=defaultdict(dict)
        self.p3=defaultdict(dict);self.searches={};self.snapshot=[];self.noise_checks=0
        self.hashes={};self.binding_audit=dict(status='PASS',settled_evaluation_phases=0,final_actor_64_phases=0,
            known_model_calls_in_these_phases=0,actual_gpu_reexecution=False)
        self.cached= lru_cache(maxsize=192)(self.load)

    def load(self,path):
        path=Path(path);data=path.read_bytes()
        import hashlib
        self.sources[str(path)]=hashlib.sha256(data).hexdigest()
        return json.loads(data)

    def progress(self,stage,**values):
        write(self.output/'analysis-progress.json',dict(stage=stage,updated_unix=time.time(),**values))
        print(stage,json.dumps(values),flush=True)

    def training(self):
        curves={};outliers={}
        for name,root in self.bindings['cells'].items():
            folder=Path(root)/'training'/name
            updates=[self.load(folder/'updates'/f'{i:03d}.json') for i in range(1,65)]
            episodes=[self.load(folder/'episodes'/f'{i:04d}.json')['metrics'] for i in range(2048)]
            curves[name]=training_health(updates,episodes)
            top=sorted(curves[name]['points'],key=lambda p:p['value_loss'],reverse=True)[:3]
            cases=[]
            for point in top:
                update=updates[point['update']-1];rewards=update['reward_settlement']['rewards'];index=max(range(32),key=lambda i:rewards[i])
                ep=episodes[32*(point['update']-1)+index]
                cases.append(dict(update=point['update'],value_loss=point['value_loss'],maximum_episode_settled_reward=rewards[index],
                    queries=ep['queries'],initial_accel=ep['initial_accel'],best_accel=ep['best_accel'],scale=update['reward_settlement']['denominator']))
            outliers[name]=dict(maximum_gradient_norm=curves[name]['maximum_gradient_norm'],top_value_loss_updates=cases)
            publication=self.load(Path(root)/'publications'/name/'0064.json')
            if publication['episodes']!=2048 or sha(publication['policy'])!=publication['policy_sha256']:raise ValueError('Changed final actor')
            self.progress('training-audited',cell=name,reward_error=curves[name]['max_reward_error'],
                          first_reward=curves[name]['windows']['first8']['mean_reward'],last_reward=curves[name]['windows']['last8']['mean_reward'])
        write(self.output/'critic-outlier-diagnostic.json',outliers)
        write(self.output/'learning-curves.json',curves);learning_figures(curves,self.output)
        return curves

    def freeze_phases(self):
        if self.config.get('phase_snapshot'):
            old=self.load(self.config['phase_snapshot']);self.snapshot=old['phases']
            for spec in self.snapshot:
                row=spec['receipt'];campaign=self.cached(row['campaign'])
                if (row['status']!='COMPLETE' or self.load(spec['ledger'])!=row
                        or campaign.get('kind')!='evaluation' or self.sources[row['campaign']]!=row['campaign_sha256']):
                    raise ValueError('Frozen analysis snapshot changed')
            write(self.output/'settled-phase-snapshot.json',old)
            return
        roots=[self.root,self.root/'recovery-r3',self.root/'recovery-r4',self.root/'evaluation-r8/p1',self.root/'evaluation-r8/p3']
        for root in roots:
            cfg=self.load(root/'coordinator-config.json')
            for path in sorted((Path(cfg['local_root'])/'budget').glob('*.json')):
                row=self.load(path)
                if row['status']!='COMPLETE' or row['key'].endswith('-allocation'):continue
                c=self.cached(row['campaign'])
                if c.get('kind')!='evaluation':continue
                if self.sources[row['campaign']]!=row['campaign_sha256']:raise ValueError('Changed settled campaign')
                self.snapshot.append(dict(ledger=str(path),receipt=row))
        write(self.output/'settled-phase-snapshot.json',dict(checked_unix=time.time(),phases=self.snapshot,unfinished_phases_excluded=True))

    def results(self,folder,expected=None,campaign=None,block=None):
        proof=self.cached(str(folder/'complete.json'));value=self.load(folder/'results.json')
        if proof['status']!='COMPLETE' or self.sources[str(folder/'results.json')]!=proof['artifacts']['results.json']:
            raise ValueError('Changed or incomplete operation result')
        if campaign is not None:
            request=dict(campaign=campaign,campaign_sha256=self.sources[campaign],block=block,host=block['host'],stage='formal')
            if proof['request_sha256']!=digest(request):raise ValueError('Changed execution request')
        rows=value['tasks']
        if expected is not None:
            wanted={(r['trial_id'],rep):r for r in expected for rep in r['execution_repeats']}
            observed={(r['trial_id'],r['repeat']):r for r in rows}
            if len(observed)!=len(rows) or wanted.keys()!=observed.keys():raise ValueError('Wrong result coverage')
            for key,r in observed.items():
                if type(r['success']) is not bool or any(r.get(k)!=v for k,v in wanted[key].items()):raise ValueError('Changed result metadata')
        return value

    def consume_current(self):
        for index,spec in enumerate(self.snapshot):
            receipt=spec['receipt'];cp=receipt['campaign'];c=self.cached(cp)
            cell=self.cached(c['protocol'])['cells'][c['cell']];meta=cell['evaluation_schedules'][c['schedule']]
            if c['evaluation_version']!=meta['version'] or c['native_policy']!=cell['native_policy']:raise ValueError('Changed checkpoint or native feature contract')
            for field in ['score_noise','execution_noise']:
                path=c[field]
                if path not in self.hashes:self.hashes[path]=sha(path)
                if path!=cell[field] or c['input_hashes'][path]!=self.hashes[path]:raise ValueError('Changed noise manifest identity')
            if meta['purpose']=='final':
                accepted=self.cached(str(Path(self.bindings['cells'][c['cell']])/'publications'/c['cell']/'0064.json'))
                if c['actor']['path']!=accepted['policy'] or c['actor']['sha256']!=accepted['policy_sha256']:raise ValueError('Final actor publication differs')
                self.binding_audit['final_actor_64_phases']+=1
            if receipt['actual_model_calls']>receipt['reserved_model_calls']:raise ValueError('Settled call budget exceeded')
            self.binding_audit['known_model_calls_in_these_phases']+=receipt['actual_model_calls']
            self.binding_audit['settled_evaluation_phases']+=1
            family=cell['family'];label=Path(cp).stem;output=Path(cp).parents[2]
            folder=output/'execution'/c['cell']/'execution'/(label+'-real')
            completion=self.cached(str(folder/'completion.json'))
            if completion['status']!='COMPLETE' or completion['dispatch_enabled'] is not False:raise ValueError('Unsettled phase owner')
            count=0
            for b in c['blocks']:
                sim=self.cached(b['sim_config']);model=sim['models'][sim['operation_model']]
                if model['weights_sha256']!=cell['model_hashes'][b['rows'][0]['suite']]:raise ValueError('Operation checkpoint identity differs')
                result=self.results(folder/'results'/f"block-{b['index']:04d}",b['rows'],cp,b)
                for row in result['tasks']:
                    count+=1
                    keep=['success','repeat','state_key','suite','task','initial','view_id','start','split','method','trial_id','queries','selected_angles','angles_deg','steps','replans','max_steps','settle_steps','replan_steps','execution_noise_sha256']
                    r={k:row[k] for k in keep};r['seed']=cell['seed']
                    if meta['purpose']=='baseline':
                        k=(meta['method'],*physical_key(r))
                        if k in self.baseline[family]:raise ValueError('Repeated baseline outcome')
                        self.baseline[family][k]=r
                    else:
                        r['checkpoint_version']=meta['version']
                        (self.final if meta['purpose']=='final' else self.development)[family].append(r)
                for search in result.get('searches',[]):
                    key=(family,c['cell'],label,search['trial_id'])
                    if key in self.searches:raise ValueError('Repeated search')
                    self.searches[key]=search
            if count!=receipt['settlement']['actual_episodes']:raise ValueError('Settled operation count differs')
            if index%20==0:self.progress('evaluation-read',phases=index+1,total_phases=len(self.snapshot))
        write(self.output/'runtime-binding-audit.json',self.binding_audit)

    def historical(self):
        audit=self.load(self.root/'reuse-audit.json')
        for family in ['pi05','groot_n15']:
            if not audit['models'][family]['status'].startswith('PASS_'):raise ValueError('Unaudited baseline reuse')
            path=Path(self.config['historical_csv'][family])
            self.sources[str(path)]=sha(path)
            if family=='pi05':
                pinned=self.load(self.config['pi_csv_pin'])
                if pinned[str(path)]!=self.sources[str(path)]:raise ValueError('Changed audited P3 table')
            with path.open() as stream:
                for raw in csv.DictReader(stream):
                    seed=int(raw['method'].rsplit('-seed-',1)[1]);key=(raw['state_key'],int(raw['view_id']),int(raw['repeat']))
                    values={k:raw[k]=='True' for k in ['success','stay_success','nominal_success']}
                    if (seed,*key) in self.p3[family]:raise ValueError('Repeated historical PPO outcome')
                    self.p3[family][(seed,*key)]=values

    def pi_baseline(self,row,*,reference=False):
        view=0 if reference else row['view_id'];refs=self.pi_refs['views'][row['state_key']+'::'+str(view)]
        path=refs['episode_results'][row['repeat']];value=self.cached(path)
        if self.sources[path]!=refs['artifacts'][path]:raise ValueError('Changed raw pi baseline')
        if type(value['success']) is not bool:raise ValueError('Invalid raw pi success')
        return value

    def gr_baselines(self):
        needed={physical_key(r) for r in self.final['groot_n15']}
        states={r['state_key'] for r in self.final['groot_n15']}
        root=Path(self.config['groot_baseline'])
        for settlement in sorted((root/'settlements').glob('baseline-*.json')):
            s=self.load(settlement);cp=root/'campaigns'/settlement.name;c=self.cached(str(cp))
            if s['status']!='COMPLETE':raise ValueError('Unsettled historical baseline')
            for b in c['blocks']:
                if not any((r['method']=='reference' and r['state_key'] in states)
                           or any((r['state_key'],r['view_id'],rep) in needed for rep in r['execution_repeats']) for r in b['rows']):continue
                folder=root/'execution'/(settlement.stem+'-real')/'results'/f"block-{b['index']:04d}"
                for row in self.results(folder,b['rows'])['tasks']:
                    k=(row['method'],*physical_key(row))
                    if k in self.baseline['groot_n15']:raise ValueError('Repeated historical baseline')
                    self.baseline['groot_n15'][k]=row

    def check_noise(self,a,b):
        aa,bb=a['execution_noise_sha256'],b['execution_noise_sha256'];n=min(len(aa),len(bb))
        if not n or aa[:n]!=bb[:n]:raise ValueError('Execution noise pairing changed')
        self.noise_checks+=1

    def paired(self,family):
        rows=[];missing_stay=0
        for r in self.final[family]:
            key=physical_key(r);old=self.p3[family].get((r['seed'],*key));reference=None
            if family=='pi05':stay=self.pi_baseline(r);reference=self.pi_baseline(r,reference=True)
            else:
                stay=self.baseline[family].get(('stay',*key))
                reference=self.baseline[family].get(('reference',r['state_key'],0,r['repeat']))
            if stay is None:missing_stay+=1;continue
            self.check_noise(r,stay)
            if reference is not None:self.check_noise(r,reference)
            if old and (old['stay_success']!=stay['success'] or (reference is not None and old['nominal_success']!=reference['success'])):
                raise ValueError('Historical paired labels disagree with original baseline')
            row={k:v for k,v in r.items() if k!='execution_noise_sha256'}
            row.update(stay_success=stay['success'],nominal_success=reference['success'] if reference is not None else None,
                       p3_success=old['success'] if old else None)
            rows.append(row)
        expected=[]
        prefix=10 if family=='pi05' else 4
        for seed in range(3):
            c=self.p['cells'][f'{family}-p{prefix}-seed-{seed}']
            for schedule,meta in c['evaluation_schedules'].items():
                if meta['purpose']=='final':
                    for r in self.cached(schedule):
                        for repeat in r['execution_repeats']:expected.append(dict(r,repeat=repeat,seed=seed))
        complete,tasks=complete_three_seed_tasks(rows,expected)
        by_seed={str(seed):paired_diagnostic([r for r in rows if r['seed']==seed]) for seed in range(3)}
        splits={split:paired_diagnostic([r for r in complete if r['split']==split]) for split in sorted({r['split'] for r in complete})}
        return rows,dict(final_operations=len(self.final[family]),missing_stay_excluded=missing_stay,
            complete_three_seed_tasks=tasks,balanced=paired_diagnostic(complete),all_available=paired_diagnostic(rows),
            per_seed=by_seed,balanced_splits=splits,excluded_partial_task_operations=len(rows)-len(complete),
            caution='Fixed-order partial coverage; task populations differ between models. Not an all-40-task conclusion. Shared baseline labels are reused across seeds, not independent repeats.')

    def build(self,config):
        self.config=config;self.pi_refs=self.load(config['pi_references'])
        self.freeze_phases();curves=self.training();self.consume_current();self.historical();self.gr_baselines()
        write(self.output/'development-outcomes.json',self.development)
        progress={family:development_progress(rows) for family,rows in self.development.items()}
        for values in self.development.values():
            for seed in range(3):
                arms={v:{physical_key(r):r for r in values if r['seed']==seed and r['checkpoint_version']==v} for v in (16,32)}
                for key in arms[16]:self.check_noise(arms[16][key],arms[32][key])
        write(self.output/'development-progress.json',progress)
        from AlphaBrain.research.dsol.plotting.prefix_health import development_figure
        development_figure(progress,self.output)
        summaries={};paired=[]
        for family in ['pi05','groot_n15','rldx']:
            rows,summary=self.paired(family);summaries[family]=summary;paired.extend(dict(model=family,**r) for r in rows)
            self.progress('paired',family=family,balanced=summary['balanced'],per_seed=summary['per_seed'])
        with (self.output/'paired-outcomes.csv').open('w') as stream:
            fields=list(paired[0]) if paired else []
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(paired)
        write(self.output/'summary.json',dict(checked_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            models=summaries,training={k:{x:v for x,v in c.items() if x!='points'} for k,c in curves.items()},
            execution_noise_prefix_checks=self.noise_checks,new_model_calls=0,optimizer_updates_performed=0,
            evaluation_unchanged=True,scope='Descriptive snapshot; no checkpoint selection, scope change or stopping based on outcomes.'))
        self.sources.update(self.hashes);write(self.output/'input-hashes.json',self.sources)
        render_report(self.output)
        compile_diagnostic_pdf(self.output)
        self.progress('COMPLETE',noise_checks=self.noise_checks)
        return summaries


def render_report(output):
    output=Path(output);s=read(output/'summary.json');curves=read(output/'learning-curves.json')
    percent=lambda x:'待齐全' if x is None else f'{100*x:.2f}%'
    lines=['# 完整前缀主线：训练健康与部分评测诊断','',
        '固定已结算阶段快照，零新模型调用、零优化更新，不修改在途评测。三种子均完整覆盖的任务用于主对比；任务按原清单完成顺序纳入，不按成功与否筛选。','',
        '| 模型 | 完整任务 | PPO操作数 | 扰动不调整 | 原P3/G | 完整前缀/G | 参考位 | 净恢复 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for family,value in s['models'].items():
        b=value['balanced'];lines.append(f"| {family} | {b.get('tasks',0)} | {b['operations']} | {percent(b.get('stay_success'))} | {percent(b.get('p3_success'))} | {percent(b.get('policy_success'))} | {percent(b.get('reference_success'))} | {b.get('net_pp',0):+.2f} pp |")
    lines+=['','各模型面板不同，不横向排名。参考位是同任务样本的已知参考对照，非搜索网络的目标。RLDX完整P3按原批准范围未跑；没有完整匹配的参考或P3结果显示待齐全，不记零。','']
    for family,value in s['models'].items():
        splits=value['balanced_splits'];unseen=splits.get('unseen_test',{}).get('tasks',0)
        lines.append(f"{family}完整三种子面板中已有{unseen}个PPO未见任务；泛化结论仅限该已覆盖子集，分层结果见summary.json。")
        b=value['balanced'];a=value['all_available']
        if b.get('pfull_minus_p3_pp') is not None:
            lines.append(f"{family}完整任务面板full−P3为{b['pfull_minus_p3_pp']:+.2f}pp；较宽但种子覆盖不均的面板为{a['pfull_minus_p3_pp']:+.2f}pp。两者均保留，不择正向子集。")
    lines+=['',
        '| 模型 | 救回 / 额外损害 | 平均查询 | 初始即结束 |','|---|---:|---:|---:|']
    for family,value in s['models'].items():
        b=value['balanced'];lines.append(f"| {family} | {b.get('rescues',0)} / {b.get('harms',0)} | {b.get('mean_queries',0):.3f} | {percent(b.get('initial_only_fraction'))} |")
    lines+=['','基线在三个策略种子间配对复用，不是三份独立样本；救回/损害计数含各策略种子及两个执行重复。区间按任务聚类，少于5个任务不展示bootstrap区间。','',
        '## 训练曲线','',
        '| 模型 | G奖励：前8→后8更新 | 查询数：前8→后8 | 初始结束率：前8→后8 | Value loss：前8→后8 |','|---|---:|---:|---:|---:|']
    for family in ['pi05','groot_n15','rldx']:
        group=[r for n,r in curves.items() if n.startswith(family+'-')];pairs=[]
        for metric in ['mean_reward','queries','initial_stop_rate','value_loss']:
            a=sum(r['windows']['first8'][metric] for r in group)/3;b=sum(r['windows']['last8'][metric] for r in group)/3
            pairs.append(f'{a:.4f}→{b:.4f}' if metric not in ['initial_stop_rate'] else f'{100*a:.2f}%→{100*b:.2f}%')
        lines.append('| '+family+' | '+' | '.join(pairs)+' |')
    error=max(c['max_reward_error'] for c in curves.values());kl=max(c['maximum_kl'] for c in curves.values())
    clips=[c['windows']['last8']['clip_fraction'] for c in curves.values()]
    lines+=['',f'奖励使用PPO真正采用的`reward_settlement.rewards`，不是采集器按临时尺度记录的`total_reward`。576次更新的G尺度均精确复现，18432回合的奖励最大重算误差{error:.3g}；未重新加载全部轨迹张量、重做GAE或梯度。不同前缀/模型的分数和损失量纲不同，不能拿奖励数值直接排名。','',
        'Value loss尖峰需要结合单回合回报长尾解释，细节见critic-outlier-diagnostic.json；不能把有限的尖峰直接称为NaN/数值爆炸，也不能仅凭均值下降称critic稳定收敛。','',
        f'九组最大approximate KL为{kl:.3f}；末8次更新平均clip fraction范围{min(clips):.3f}–{max(clips):.3f}。梯度范数图为clip_grad_norm_返回的裁剪前范数，本轮实际阈值0.5；日志超过0.5不等于裁剪失效。当前未记录explained variance，value loss下降不能单独证明critic拟合充分。','',
        '训练后段查询接近上限，尤其GR00T；这说明没有全STOP，但尚不能证明停止策略达到最优成本效率。当前结果不支持自动增加训练、换奖励或改查询成本。','',
        '[奖励与行为曲线](learning-behavior.pdf) · [损失与梯度](optimizer-losses.pdf) · [KL/裁剪/STOP](optimizer-stability.pdf) · [收益与查询成本](gain-and-cost.pdf)','',
        '[完整统计](summary.json) · [逐对操作](paired-outcomes.csv) · [来源哈希](input-hashes.json) · [运行绑定核验](runtime-binding-audit.json)','',
        '## 固定开发检查点','',
        '开发评测只比较相同64条操作在512/1024回合检查点的表现。2048回合的相同开发面板尚未全部覆盖，不以别的任务子集填补该点。',
        '[开发评测曲线](development-success.pdf) · [开发统计](development-progress.json)']
    (output/'README.md').write_text('\n'.join(lines)+'\n')


def compile_diagnostic_pdf(output):
    """Portable review copy; individual plot PDFs retain their vector originals."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    output=Path(output);target=output/'full-diagnostics.pdf'
    if target.exists():raise ValueError('Published diagnostic PDF must not be overwritten')
    summary=read(output/'summary.json');snapshot=read(output/'settled-phase-snapshot.json')
    stamp=datetime.datetime.fromtimestamp(snapshot['checked_unix'],datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M CST')
    percent=lambda x:'pending' if x is None else f'{100*x:.2f}'
    with PdfPages(target) as pdf:
        fig,ax=plt.subplots(figsize=(14,8));ax.axis('off')
        ax.text(.04,.93,'Full-prefix G/PPO: interim outcomes and training health',fontsize=19,transform=ax.transAxes)
        ax.text(.04,.86,f'Fixed settled-phase snapshot: {stamp}. Nine actors: 2,048 episodes / 64 updates each.',fontsize=12,transform=ax.transAxes)
        data=[]
        for family,v in summary['models'].items():
            b=v['balanced'];data.append([family,str(b.get('tasks',0)),str(b['operations']),percent(b.get('stay_success')),
                percent(b.get('p3_success')),percent(b.get('policy_success')),percent(b.get('reference_success'))])
        table=ax.table(cellText=data,colLabels=['Model','Complete tasks','PPO operations','Stay (%)','P3/G (%)','Full/G (%)','Reference (%)'],bbox=[.04,.49,.92,.27])
        table.auto_set_font_size(False);table.set_fontsize(11)
        notes=['Matched task/initial/view/execution noise; only complete three-seed tasks enter the table.',
            'Models cover different partial task populations. These are not final all-40-task results or model rankings.',
            'Missing references are not failures. Baseline labels reused across seeds are not independent repeats.',
            'Do not choose prefixes/checkpoints or stop evaluation based on these partial results.',
            'Batch-frozen G rewards and cumulative initial-score medians were recomputed from the complete logs.',
            'Value-loss spikes, query saturation and seed variability require separate interpretation.',
            'No new model calls, optimizer updates, or changes to running evaluation.',
            'Next pages: behavior; losses; KL/clipping/STOP; gain and cost; fixed development checkpoints.']
        for i,line in enumerate(notes):ax.text(.04,.40-i*.044,line,fontsize=10,transform=ax.transAxes)
        pdf.savefig(fig);plt.close(fig)
        for name in ['learning-behavior','optimizer-losses','optimizer-stability','gain-and-cost','development-success']:
            fig=plt.figure(figsize=(15,8.5));ax=fig.add_axes([0,0,1,1]);ax.axis('off');ax.imshow(plt.imread(output/(name+'.png')))
            pdf.savefig(fig,dpi=160);plt.close(fig)
        if pdf.get_pagecount()!=6:raise ValueError('Incomplete diagnostic PDF')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=read(a.config)
    Report(cfg['root'],cfg['output']).build(cfg)

if __name__=='__main__':main()
