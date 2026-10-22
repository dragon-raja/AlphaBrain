"""Compose an offline channel/phase audit without replacing previous reports."""
import argparse
from html import escape
from pathlib import Path
from AlphaBrain.research.dsol.data.artifacts import read,write,sha
from AlphaBrain.research.dsol.plotting.channel_phase import plot_physical,plot_view_channels,plot_recovery
from .flow_structure import table,NAMES


def render(root,interpretation):
    root=Path(root);a=root/'analysis';out=root/'report';out.mkdir(exist_ok=False)
    s=read(a/'evidence-summary.json');coverage=read(a/'summary.json');notes=read(interpretation)
    plot_physical(read(a/'physical-timelines.json'),out/'physical.png')
    plot_view_channels(s['same_state_view_effects'],out/'channels.png')
    plot_recovery(s['libero_plus_groups'],out/'recovery.png')
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>通道、操作阶段与视角反转</title><style>body{max-width:1200px;margin:35px auto;font:16px/1.7 system-ui;color:#222;padding:0 20px}table{border-collapse:collapse;font-size:13px;width:100%}th,td{border:1px solid #ddd;padding:7px}img{max-width:100%}.note{background:#edf4f8;padding:16px}</style>',
        '<h1>已有遥测：通道、操作阶段与视角评分反转</h1>',
        '<p class="note">上一轮两例同状态诊断已完成；本轮仅做CPU分析。新增模型调用、渲染、仿真步进和动作均为0。没有重新估计权重或选位规则，没有覆盖原报告或原实验结果。</p>',
        '<h2>结论与下一步</h2>']
    parts.extend('<p>'+escape(t)+'</p>' for t in notes['conclusions'])
    parts.append('<h2>覆盖与时间口径</h2>'+table(['项','数量'],[[k,coverage[k]] for k in ('branches','physical_frames','original_nodes','same_state_cells','view_contrasts','libero_plus_initial_records','new_model_calls')]))
    parts.append('<p>实际开度代理取两个手指关节坐标差（毫米），另存观测开度和渲染时刻指尖中心距。它不是标定的指腹净间隙。距离取末端site到物体body中心，不是表面距离。接触采用原MuJoCo非正间距；RoboCasa闭合命令阈值沿用原包装器0.5，LIBERO沿用原控制器正号语义。没有根据结果调距离或开度阈值。</p>')
    parts.append('<p>同状态结果固定了各重复的评分噪声；旧前缀各窗口使用原有不同噪声地址。两种分数分别展示，不拼接成一条曲线。评分时只使用截至当时已有的物理事件；三个采样点之间的反转只报告区间，不插值指定某一帧。</p>')
    parts.append('<h2>实际距离、开度、接触与执行命令</h2><img src="physical.png"><p>色带为8条分支的最小—最大范围，不是置信区间；contact包括掌部接触，close cmd是实际历史执行命令。</p>')
    rows=[]
    for b in s['branches']:
        target='obj_main' if b['case'].startswith('xr1') else 'wooden_cabinet_1_cabinet_top';c=b['contacts'][target]
        rows.append([b['case'],b['view'],b['closing_execution_repeats'],c['repeats'],c['palm_repeats'],c['any_finger_repeats'],c['bilateral_same_frame_repeats'],f"{1000*b['final_joint_separation_m']['mean']:.2f}"])
    parts.append(table(['案例','历史分支','出现闭合命令/8','目标接触/8','掌部/8','任一手指/8','同帧双指/8','末帧关节间距均值mm'],rows))
    parts.append('<h2>同状态视角差及通道贡献</h2><img src="channels.png"><p>负值表示原历史高成功率视角的Accel更低。该历史标签不等于这个中间状态下两个新观察的真实任务价值；本轮没有新增这些干预的闭环标签。</p>')
    rows=[]
    for r in s['same_state_view_effects']:
        rows.append([r['initial'][:8],r['origin'],r['step'],f"{r['channels']['baseline__score']['mean']:+.6f}",f"{r['good_view_lower']}/8",r['good_view_predicted_closing_counts'],r['bad_view_predicted_closing_counts']])
    parts.append(table(['案例','状态/历史分支','评分步','总分差均值','历史好视角较低','好视角预测闭合数/16（逐重复）','坏视角预测闭合数/16（逐重复）'],rows))
    parts.append('<h2>反转发生在哪些区间</h2>'+table(['案例','分支','区间','反转重复数','区间无闭合命令','截至末端无目标接触'],[[r['initial'][:8],r['origin'],f"({r['after']},{r['by']}]",r['n'],r['without_closing_execution'],r['without_target_contact_by_end']] for r in s['observed_flip_intervals']]))
    parts.append('<h2>LIBERO/Plus：保住已有正结果与负结果</h2><img src="recovery.png"><p>恢复基线严格沿用原协议：LIBERO为heldout32均匀扰动，Plus为各初态所有非规范视角均匀扰动。另保留规范视角作为第二比较，不能把相对规范视角的下降误说成“没有扰动恢复收益”。所有原收益已与冻结汇总逐项核对。</p>')
    rows=[]
    for r in s['libero_plus_groups']:
        counts=r['outcomes'];rows.append([NAMES[r['panel']],r['baseline'],r['mode'],r['window'],f"{100*r['gain']:+.2f}",'/'.join(str(counts.get(k,0)) for k in ('improved','degraded','equal','unknown')),r['channels_by_outcome']['improved'],r['channels_by_outcome']['degraded']])
    parts.append(table(['面板','基线','规范候选','窗口','收益pp','改善/退化/相同/未知','改善时最主要通道','退化时最主要通道'],rows))
    parts.append('<h2>证据边界</h2><ul>'+''.join('<li>'+escape(t)+'</li>' for t in notes['limitations'])+'</ul>')
    parts.append('<h2>复核入口</h2><p>'+ ' · '.join('<a href="../analysis/'+f+'">'+label+'</a>' for f,label in [('summary.json','覆盖'),('physical-timelines.json','逐帧遥测'),('original-nodes.json','原评分节点'),('same-state-alignment.json','同状态逐格对齐'),('flip-intervals.json','逐重复反转'),('positive-controls.json','全部正反对照'),('sources.json','输入哈希')])+'</p></html>')
    (out/'report.html').write_text('\n'.join(parts))
    write(out/'provenance.json',dict(inputs={str(p):sha(p) for p in [a/'evidence-summary.json',a/'summary.json',Path(interpretation)]},source_sha256=sha(__file__),report_sha256=sha(out/'report.html')))
    return out/'report.html'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--interpretation',required=True);x=p.parse_args();print(render(x.root,x.interpretation))
