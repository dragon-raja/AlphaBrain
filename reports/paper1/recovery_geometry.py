"""Read-only geometry report from hash-bound, complete recovery result JSONs."""
from pathlib import Path
import argparse, collections, hashlib, json
import numpy as np
from AlphaBrain.research.dsol.analysis.recovery_geometry import summarize
from AlphaBrain.research.dsol.data.native_view_panel import views
from AlphaBrain.research.dsol.plotting.recovery_geometry import figures


def build(run, baseline_panel, verified_hashes, output):
    run, output = Path(run), Path(output)
    output.mkdir(exist_ok=False, parents=True)
    hashes = json.loads(Path(verified_hashes).read_text())
    base = {(r['state_key'],r['view_id']):r for r in json.loads(Path(baseline_panel).read_text())}
    panel = views(); records=[]; consumed={}
    paths = sorted(p for p in hashes if p.endswith('/results.json') and '/execution-recovery/' in p)
    for n, name in enumerate(paths):
        path=Path(name)
        if not path.is_relative_to(run/'execution-recovery'): raise ValueError('Unexpected source')
        raw=path.read_bytes(); actual=hashlib.sha256(raw).hexdigest()
        if actual!=hashes[name]: raise ValueError('Changed result JSON')
        consumed[name]=actual; result=json.loads(raw)
        for s in result['searches']:
            key=(s['state_key'],s['view_id']); b=base[key]
            outcomes=sorted([t for t in result['tasks'] if (t['state_key'],t['view_id'],t['method'])==(*key,s['method'])],key=lambda t:t['repeat'])
            if [t['repeat'] for t in outcomes]!=[0,1]: raise ValueError('Missing paired execution')
            records.append(dict(state_key=s['state_key'],task=s['task'],split=s['split'],method=s['method'],
                view_id=s['view_id'],start_angles=panel[s['view_id']],selected_angles=s['selected_angles'],
                queries=s['queries'],initial_accel=s['initial_accel'],minimum_accel=s['best_accel'],
                independent_accel=float(np.mean(s['validation_scores'])),
                success=[t['success'] for t in outcomes],stay_success=b['success'],
                nominal_success=base[s['state_key'],0]['success'],
                query_path=[e['angles'] for e in s['events'] if e['kind'] in ('initial','search')]))
        if n%100==0:print(json.dumps(dict(verified_blocks=n+1,total=len(paths))),flush=True)
    if len(records)!=18240: raise ValueError('Incomplete recovery panel')
    summary=summarize(records,panel)
    summary['provenance']=dict(consumed_result_files=len(consumed),scope='Result JSON hashes rechecked; full binary audit separately running.',
        baseline_panel_sha256=hashlib.sha256(Path(baseline_panel).read_bytes()).hexdigest(),
        verified_hashes_sha256=hashlib.sha256(Path(verified_hashes).read_bytes()).hexdigest())
    for name,value in [('records.json',records),('geometry.json',summary),('consumed-hashes.json',consumed)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    figures(records,list(base.values()),output)
    lines=['# 完整恢复搜索的连续视角分布','','仅复算既有数据，无新增训练、模型调用或仿真。三维方向图是相机光轴在相对坐标中的单位方向，不是相机XYZ位置；相机光心固定。',
        '每个初态从8个预定扰动点出发，方向覆盖8种，扰动强度轮换；27点含规范1点、分级扰动24点及旧扰动2点。恢复起点不含规范点与旧扰动2点。',
        '', '| 划分/方法 | 搜索数 | 面板外终点 | 起点/终点距规范中位数 | 更靠近规范 | 规范5°内终点 |', '|---|---:|---:|---:|---:|---:|']
    for name,g in summary['groups'].items():
        lines.append(f"| {name} | {g['searches']} | {100*g['off_panel_fraction']:.1f}% | {g['start_distance_median_deg']:.2f}° / {g['final_distance_median_deg']:.2f}° | {100*g['closer_fraction']:.1f}% | {100*g['final_within_5deg_fraction']:.1f}% |")
    lines+=['',f"规范视角起始的搜索数：{summary['nominal_start_searches']}。因此当前不能估计从规范点启动PPO的额外损害；扰动起点搜索与规范执行的对比不能替代此实验。",'',
        '[三维光轴分布](orientation-3d.png) · [yaw/pitch/独立Accel三维图](score-3d.png) · [二维角度分布](angle-density.png) · [全部图及逐任务示例PDF](geometry.pdf)', '',
        '三种子未见任务均展示；逐任务图按任务名称排序取每任务第一个初态，选择不看成败。没有拟合未测位置的成功率曲面。详细逐搜索轨迹见records.json，所有分组统计见geometry.json。']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='COMPLETE',report=str(output/'README.md'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','baseline-panel','verified-hashes','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();build(a.run,a.baseline_panel,a.verified_hashes,a.output)
