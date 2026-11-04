"""Hash-verified CPU report for a completed RLDX bounded selection screen."""
import argparse
import json
from pathlib import Path
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.execution.manifest import file_digest,digest
from AlphaBrain.research.dsol.analysis.selection_screen import summarize
from benchmarks.LIBERO.controlled_rldx_operation import request_for,validate_block


def build(root,output):
    root=Path(root);output=Path(output)
    c=read(root/'campaign-selection.json');done=read(root/'execution/selection-real/completion.json')
    if done['status']!='COMPLETE' or done['dispatch_enabled'] is not False:raise ValueError('Unsettled selection campaign')
    tasks=[];score_rows=[];evidence={};noise_consistency=[]
    for b in c['blocks']:
        folder=root/'execution/selection-real/results'/f"block-{b['index']:04d}"
        validate_block(request_for(root/'campaign-selection.json',b['index']),folder)
        result=read(folder/'results.json');selection=read(folder/'selection.json');row=b['rows'][0]
        expected=min(range(1,27),key=lambda i:(np.mean([x['3'] for x in selection['scores'][i]['scores']]),i))
        if selection['selected']!=expected:raise ValueError('Selection uses another rule')
        for t in result['tasks']:
            expected_view={'reference':0,'stay-23':23,'stay-24':24,'minimum-p3':expected}[t['variant']]
            if t['view']!=expected_view or t['state_key']!=row['state_key']:raise ValueError('Wrong executed selected view')
        tasks.extend(result['tasks']);evidence[str(folder/'complete.json')]=file_digest(folder/'complete.json')
        scores=selection['scores'];a=np.array([[x['3'] for x in s['scores']] for s in scores[1:]])
        corr=None if min(a[:,0].std(),a[:,1].std())==0 else float(np.corrcoef(a.T)[0,1])
        noise_consistency.append(dict(state_key=row['state_key'],p3_member_pearson=corr,
            p3_range=float(np.ptp(a.mean(axis=1))),selected_view=expected,selected_angles=row['views'][expected]))
        score_rows.extend(dict(state_key=row['state_key'],**s) for s in scores)
    if done['actual_episodes']!=len(tasks):raise ValueError('Settled episode count differs')
    value=summarize(tasks,expected_states=16,expected_tasks=8)
    value.update(model='RLDX-1-FT-LIBERO',sampler='Native4 denoise / 16 predicted / 8 executed; P3 on first8x7',
        accounting=done,noise_consistency=noise_consistency,verified_receipts=evidence,optimizer_updates=0)
    output.mkdir(parents=True,exist_ok=False);write(output/'summary.json',value);write(output/'score-grid.json',score_rows)
    rates=value['rates'];lo,hi=value['task_cluster_interval_pp']
    recommendation=('值得进入目标原生特征＋G的小规模训练验证；当前仅开发集选位证据，不授权全量PPO。' if value['net_pp']>0 else
                    '先暂停大规模PPO；当前P3选位未见正净恢复，应继续诊断评分/执行与更宽成功率面板，不能据此淘汰整个模型。')
    (output/'README.md').write_text(f'''# RLDX 基础选位验证\n\n8个既定开发任务、16初态、27评分视角；128条操作，不训练。\n\n| 方式 | 实际操作条数 | 成功率 |\n|---|---:|---:|\n| 参考位 |32|{rates['reference']:.2%}|\n| 扰动不调整(-30,+20) |32|{rates['stay-23']:.2%}|\n| 扰动不调整(-30,-20) |32|{rates['stay-24']:.2%}|\n| 两起点平均 |64|{rates['pooled_stay']:.2%}|\n| 排除参考位、26候选最低P3 |32|{rates['minimum-p3']:.2%}|\n\n配对净变化{value['net_pp']:+.2f}个百分点；救回{value['rescued']}、损害{value['harmed']}，64个配对比较复用32个选定终点结果，绝非新增64次选位操作。按8任务聚类的95% bootstrap区间[{lo:+.2f},{hi:+.2f}]个百分点。\n\n{recommendation}\n\n这是27个实测评分点下的候选效用检查，评分预算大于8查询PPO；没有训练搜索网络。成功标签只采集两个固定扰动起点、参考位与预先按P3选出的点，因此不是完整27点成功率曲面，不能据此报告全空间Accel/成功率线性相关性。起点都在负yaw边缘，不能扩大为全部扰动分布。P4只保留原始诊断，不按结果切换。\n\nG可保留正式交互估计尺度、每个PPO更新内固定分母的机制；必须重置目标尺度状态，接目标自身联合cognition tokens，不直接装载异构模型旧actor。G不保证低Accel等于高成功率；也未在本实验验证其训练效果。GR1尚未实际准入。\n\n真实调用{done['actual_model_calls']}，所有任务和预算已结算，派发关闭。完整逐初态、噪声一致性、输入/结果哈希与预算在summary.json；27评分点在score-grid.json。\n''')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=build(a.root,a.output);print(json.dumps({k:result[k] for k in ('rates','net_pp','rescued','harmed','task_cluster_interval_pp')}))

if __name__=='__main__':main()
