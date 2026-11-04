"""Complete paired mounting-probe report; no runtime or outcome selection."""
import json
from pathlib import Path
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read
from benchmarks.LIBERO.controlled_recovery import validate_block


def build(config_path,output):
    cfg=read(config_path);root=Path(cfg['output']);p=read(cfg['protocol'])
    expected={(r['state_key'],r['view_id'],r['method']):r for r in p['rows']}
    searches={};tasks={};proofs=[]
    for number in range(len(cfg['campaigns'])):
        folder=root/'execution'/f'probe-{number:02d}-real';binding=read(folder/'binding.json')
        completion=read(folder/'completion.json')
        if completion['status']!='COMPLETE' or completion['dispatch_enabled']:raise ValueError('Unsettled phase')
        for spec in binding['tasks'].values():
            proofs.append(validate_block(spec['request'],spec['output']))
            result=read(Path(spec['output'])/'results.json')
            for s in result['searches']:
                key=(s['state_key'],s['view_id'],s['method'])
                if key in searches or key not in expected:raise ValueError('Duplicate or unexpected search')
                searches[key]=s
            for t in result['tasks']:
                key=(t['state_key'],t['view_id'],t['method'],t['repeat'])
                if key in tasks:raise ValueError('Duplicate operation outcome')
                tasks[key]=t
    if set(searches)!=set(expected) or set(tasks)!={(*k,r) for k in expected for r in (0,1)}:
        raise ValueError('Incomplete scientific coverage')
    refs=read(cfg['baseline_references'])['views'];baseline={}
    for state,view,_ in expected:
        for v in (0,view):
            ref=refs[f'{state}::{v}']
            for repeat,path in enumerate(ref['episode_results']):
                baseline[(state,v,repeat)]=read(path)['success']
    summary=[]
    for offset in (-5.,0.,5.):
        selected=[(key,row) for key,row in expected.items() if row['mount_yaw_offset_deg']==offset]
        outcomes=[];rescue=[];harm=[];deltas=[]
        for key,row in selected:
            for repeat in (0,1):
                success=tasks[(*key,repeat)]['success'];stay=baseline[(key[0],key[1],repeat)]
                zero=(key[0],key[1],f"{row['actor_key']}-mount-+0",repeat)
                outcomes.append(success);rescue.append(success and not stay);harm.append(stay and not success)
                deltas.append(int(success)-int(tasks[zero]['success']))
        summary.append(dict(mount_yaw_deg=offset,searches=len(selected),operation_episodes=len(outcomes),
            success_percent=float(100*np.mean(outcomes)),rescue_percent=float(100*np.mean(rescue)),
            harm_percent=float(100*np.mean(harm)),paired_change_vs_zero_pp=float(100*np.mean(deltas)),
            mean_queries=float(np.mean([searches[k]['queries'] for k,_ in selected]))))
    root_out=Path(output);root_out.mkdir(parents=True,exist_ok=False)
    baseline_keys={(r['state_key'],r['view_id'],repeat) for r in p['rows'] for repeat in (0,1)}
    result=dict(status='COMPLETE_SMALL_DEVELOPMENT_DIAGNOSTIC',searches=144,episodes=288,offsets=summary,
        stay_success_percent=float(100*np.mean([baseline[k] for k in sorted(baseline_keys)])),
        nominal_success_percent=float(100*np.mean([baseline[(s,0,r)] for s,_,r in sorted(baseline_keys)])),
        boundaries=['Four previously inspected development tasks, two initials, two yaw starts, three G seeds; not broad generalization.',
                    'Frozen G, no training or reward changes. This tests a shifted installation relationship, not a newly repaired policy.',
                    'All arms use +/-25 operational yaw; physical camera kinematics and +/-30 physical envelope unchanged.',
                    'Mount changes affect the longer-horizon reachable set; paired CPU first-step probe isolates coordinate input effects separately.'])
    (root_out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (root_out/'paired-outcomes.json').write_text(json.dumps(list(tasks.values()),indent=2)+'\n')
    (root_out/'verification.json').write_text(json.dumps(proofs,indent=2)+'\n')
    lines=['# G 安装零位小规模闭环检查','', '冻结 G；不训练、不改 URDF 或相机运动学。144 次搜索、288 条操作轨迹。', '',
           '| 安装偏移 | 成功率 | 相对零偏移 | 查询均值 |','| --- | --- | --- | --- |']
    lines += [f"| {r['mount_yaw_deg']:+g}° | {r['success_percent']:.2f}% | {r['paired_change_vs_zero_pp']:+.2f} pp | {r['mean_queries']:.2f} |" for r in summary]
    lines += ['',f"同面板起点保持 {result['stay_success_percent']:.2f}%，原始参考位 {result['nominal_success_percent']:.2f}%。",'',*result['boundaries']]
    (root_out/'README.md').write_text('\n'.join(lines)+'\n')
    return result
