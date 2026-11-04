"""Read-only baseline diagnostics from verified, already completed artifacts."""
from pathlib import Path
import argparse, csv, datetime, hashlib, json, time
import numpy as np
from AlphaBrain.research.dsol.analysis.native_baseline_diagnostic import analyze
from AlphaBrain.research.dsol.data.native_view_panel import views
from AlphaBrain.research.dsol.execution.manifest import file_digest


def build(run, output):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    references_path = run/'protocol/recovery/baseline-references.json'
    refs = json.loads(references_path.read_text())
    proof = json.loads((run/'recovery-optimization-20261026/baseline-preparation.json').read_text())
    if file_digest(references_path) != proof['sha256']:
        raise ValueError('Baseline reference manifest changed')
    baseline = json.loads((run/'protocol/campaign.json').read_text())
    if file_digest(run/'protocol/campaign.json') != refs['baseline_campaign_sha256']:
        raise ValueError('Baseline campaign identity changed')
    inputs_path = run/'protocol/evaluation-inputs.json'
    if file_digest(inputs_path) != baseline['input_hashes'][str(inputs_path)]:
        raise ValueError('Input split identity changed')
    records = json.loads(inputs_path.read_text())['records']
    states = {s['state_key']:s for s in baseline['states']}
    if len(states) != 400 or len(refs['views']) != 10800:
        raise ValueError('Expected complete forty-task panel')
    used = {}; panel = []; noise = {}; expected_angles = views()
    def read_bound(path, expected):
        raw = Path(path).read_bytes(); actual = hashlib.sha256(raw).hexdigest()
        if actual != expected: raise ValueError('Changed baseline result: '+str(path))
        used[str(path)] = actual
        return json.loads(raw)
    for index,(key,ref) in enumerate(refs['views'].items()):
        state,view = key.rsplit('::',1); view=int(view); s=states[state]
        score_path = str(Path(ref['query']).parent/'scores.json')
        scores = sorted(read_bound(score_path, ref['artifacts'][score_path]), key=lambda x:x['member'])
        if [x['member'] for x in scores] != [0,1,2,3]: raise ValueError('Missing score member')
        outcomes = sorted([read_bound(p,ref['artifacts'][p]) for p in ref['episode_results']], key=lambda x:x['repeat'])
        if [x['repeat'] for x in outcomes] != [0,1]: raise ValueError('Missing execution repeat')
        for item in scores+outcomes:
            if (item['state_key']!=state or item['view_id']!=view or item['task']!=s['task']
                    or item['initial']!=s['initial'] or item['angles_deg']!=expected_angles[view]):
                raise ValueError('Mixed scientific identity')
        for item in outcomes:
            identity=(state,item['repeat']); stream=item['execution_noise_sha256']
            if identity in noise:
                n=min(len(stream),len(noise[identity]))
                if stream[:n]!=noise[identity][:n]:raise ValueError('Unpaired execution noise')
                if len(stream)>len(noise[identity]):noise[identity]=stream
            else:noise[identity]=stream
        if any(x['input_sha256']!=ref['input_sha256'] for x in scores):raise ValueError('Score input mismatch')
        panel.append(dict(state_key=state, task=s['task'], suite=s['suite'],
            split=records[s['initial']]['state']['split'], view_id=view,
            yaw_deg=expected_angles[view][0],pitch_deg=expected_angles[view][1],
            success=[x['success'] for x in outcomes],accel=float(np.mean([x['score'] for x in scores[:2]])),
            independent_accel=float(np.mean([x['score'] for x in scores[2:]]))))
        if index%100==0:
            print(json.dumps(dict(verified_views=index+1,total=10800)),flush=True)
            time.sleep(.01)
    result=analyze(panel)
    result['provenance']=dict(run=str(run),reference_manifest_sha256=file_digest(references_path),
        verified_json_files=len(used),created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        verification_scope='Exact consumed score/result JSON hashes, complete identities and paired execution noise. Prior baseline receipts retained; this diagnostic does not rehash trajectory/image arrays or rerun GPU.')
    for name,value in [('metrics.json',result),('consumed-artifacts.json',used),('panel.json',panel)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    for name,rows in [('per-task.csv',result['per_task']),('per-view.csv',result['per_view'])]:
        with (output/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    m=result['dense']['task_macro'];ci=result['dense']['task_cluster_bootstrap_95']['perturbation_loss']
    lines=['# 原生模型完整基础评测：只读诊断','',
        '40任务×10初态×27视角×2执行种子。规范视角是[0°,0°]；扰动不调整为其余26个固定视角。不是官方标准分数复现，也不代表完整LIBERO-plus或真机。',
        '本报告只读取已完成数据，不改变正在执行的PPO、恢复评测、预算或选位规则。','',
        f"规范视角：{result['nominal_successes']}/{result['nominal_episodes']} = {100*m['nominal']:.2f}%；扰动不调整：{result['perturbed_successes']}/{result['perturbed_episodes']} = {100*m['stay']:.2f}%。",
        f"配对平均损失：{100*m['perturbation_loss']:.2f}个百分点；任务聚类bootstrap 95%区间 [{100*ci[0]:.2f}, {100*ci[1]:.2f}]。区间只描述该固定40任务面板。",'',
        '| 分组 | 任务数 | 规范成功率 | 扰动不调整 | 损失百分点 |','|---|---:|---:|---:|---:|']
    for name,g in result['groups'].items():
        x=g['task_macro'];lines.append(f"| {name} | {g['task_count']} | {100*x['nominal']:.2f}% | {100*x['stay']:.2f}% | {100*x['perturbation_loss']:.2f} |")
    lines += ['', '扰动强度不是单一旋转角：small/medium/large分别为8个轴向或对角方向，yaw幅度10/20/30°、pitch幅度6.67/13.33/20°；另有两个旧扰动点。',
        '',f"规范成功→扰动失败：{result['perturbation_harm_pairs']}对；规范失败→扰动成功：{result['perturbation_improvement_pairs']}对。规范结果按26个视角复用配对，不能把20800对当成20800个独立初态。",'',
        '## Accel诊断', '',json.dumps(result['correlation'],ensure_ascii=False), '',
        '| 离线选位规则 | 成功率 | 相对规范净变化 | 救回 / 损害（相对规范） |','|---|---:|---:|---:|']
    for name,g in result['offline_selectors'].items():
        x=g['task_macro'];lines.append(f"| {name} | {100*x['success']:.2f}% | {100*x['net_recovery']:+.2f}个百分点 | {g['rescue_count']} / {g['harm_count']} |")
    lines += ['',result['offline_selector_scope'],
        '这是全候选面板的诊断，不能充当8查询在线PPO/局部搜索成绩；未用任务成败选择视角。后续不能据此在同一测试集调规则后仍声称独立验证。',
        '', '逐任务和全部27个视角见CSV；原始可重算面板及输入哈希见panel.json、consumed-artifacts.json。恢复评测仍未全部完成，本报告不宣称A/B优劣。']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],report=str(output/'README.md'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();build(a.run,a.output)
