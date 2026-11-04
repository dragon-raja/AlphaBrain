"""Audit complete paired mechanism data and apply its predeclared decision."""
import json
from pathlib import Path
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read
from AlphaBrain.research.dsol.analysis.coordinate_validation import summarize
from AlphaBrain.research.dsol.execution.manifest import file_digest
from benchmarks.LIBERO.controlled_recovery import validate_block


def build(config_path,output):
    c=read(config_path);root=Path(c['output']);p=read(c['protocol'])
    expected={(r['state_key'],r['view_id'],r['method']):r for r in p['rows']}
    searches={};episodes={};first_queries={};proofs=[]
    for index in range(len(c['campaigns'])):
        phase=root/'execution'/f'probe-{index:02d}-real';complete=read(phase/'completion.json')
        if complete['status']!='COMPLETE' or complete['dispatch_enabled']:raise ValueError('Unsettled phase')
        binding=read(phase/'binding.json')
        for spec in binding['tasks'].values():
            proof=validate_block(spec['request'],spec['output']);proofs.append(proof)
            result=read(Path(spec['output'])/'results.json')
            for j,s in enumerate(result['searches']):
                key=(s['state_key'],s['view_id'],s['method'])
                if key in searches or key not in expected:raise ValueError('Duplicate or unplanned search')
                searches[key]=s
                condition=key[:2];query=Path(spec['output'])/f'search-{j:03d}'/'query-01.npz'
                if condition in first_queries:
                    with np.load(query,allow_pickle=False) as a,np.load(first_queries[condition],allow_pickle=False) as b:
                        if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):
                            raise ValueError('Methods did not start at identical physical/model input')
                else:first_queries[condition]=str(query)
            for t in result['tasks']:
                key=(t['state_key'],t['view_id'],t['method'],t['repeat'])
                if key in episodes:raise ValueError('Duplicate execution')
                episodes[key]=t
    if (len(expected)!=2268 or set(searches)!=set(expected)
            or set(episodes)!={(*key,rep) for key in expected for rep in (0,1)}):
        raise ValueError('Incomplete full mechanism coverage')
    refs=read(c['baseline_references'])['views'];canonical={};lookup={}
    for key,row in expected.items():
        for repeat in (0,1):
            lookup[(row['state_key'],row['mount_yaw_offset_deg'],row['start'],row['method_family'],repeat)]=episodes[(*key,repeat)]
            canonkey=(row['state_key'],repeat)
            path=refs[f"{row['state_key']}::0"]['episode_results'][repeat]
            if file_digest(path)!=refs[f"{row['state_key']}::0"]['artifacts'][path]:raise ValueError('Changed reference outcome')
            canonical[canonkey]=read(path)
            if row['method_family']=='home' and row['mount_yaw_offset_deg']==0.:
                actual=episodes[(*key,repeat)];reference=canonical[canonkey]
                if any(actual[k]!=reference[k] for k in ('input_sha256','success','steps','replans','execution_noise_sha256')):
                    raise ValueError('Zero-installation home differs from original reference execution')
    pairs=[]
    for key,row in expected.items():
        if not row['actor_key']:continue
        state,offset,start=row['state_key'],row['mount_yaw_offset_deg'],row['start']
        local_key=(state,row['view_id'],f'local-mount-{int(offset):+d}')
        for repeat in (0,1):
            outcome=episodes[(*key,repeat)];home=lookup[(state,offset,0,'home',repeat)]
            controls={m:lookup[(state,offset,start,m,repeat)] for m in ('stay','local')}
            controls['home']=home;controls['reference']=canonical[(state,repeat)]
            for other in controls.values():
                n=min(outcome['replans'],other['replans'])
                if outcome['execution_noise_sha256'][:n]!=other['execution_noise_sha256'][:n]:
                    raise ValueError('Unpaired manipulation noise')
            pairs.append(dict(state_key=state,task=row['task'],offset=offset,start=start,seed=int(row['actor_key'][-1]),
                repeat=repeat,G=outcome['success'],**{m:t['success'] for m,t in controls.items()},
                G_queries=searches[key]['queries'],local_queries=searches[local_key]['queries']))
    summary=summarize(pairs,criteria=p['criteria']);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    for name,value in [('summary.json',summary),('paired-comparisons.json',pairs),('verification.json',proofs)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    lines=['# 完整坐标机制验证','',f"固定标准判定：**{summary['decision']}**",'',
        '12个PPO未见任务、每任务3初态、4起点、3安装偏移；2268次搜索、4536条操作，G三种子。规范参考按36初态×2执行噪声复用，不作为策略目标。','',
        '| 偏移 | G | 保持 | 机械回零 | 局部Accel | 原参考 |','| --- | --- | --- | --- | --- | --- |']
    for offset,s in summary['per_offset'].items():
        lines.append('| '+offset+'° | '+' | '.join(f"{s['success_percent'][m]:.2f}%" for m in ('G','stay','home','local','reference'))+' |')
    lines+=['','## 非零偏移主判定','',json.dumps(dict(gates=summary['gates'],ppo_cost_effectiveness_gate=summary['ppo_cost_effectiveness_gate'],contrasts=summary['contrasts']),ensure_ascii=False,indent=2),
        '', '任务为聚类单位，不把重复起点、噪声或三个策略种子当独立任务。完整逐任务/种子统计在summary.json。',
        '机械零位在当前条件代表安装方向；物理相机模型不改。共同±15°操作范围令各偏移仍在原±30°物理范围内；可达集合随安装变化，不能称为纯单字段因果消融。',
        '本轮检验的是冻结G部署稳健性；不证明对象布局变化后旧参考失效，不证明已经训练出相对起点编码的新G，也不等于真机验收。',
        'PASS仅对预先列明的主实验范围有效；HOLD不自动增加样本、改门槛或挑种子。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    return summary
