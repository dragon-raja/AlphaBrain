"""Render the saved structural audit; does not rescore or run experiments."""
import argparse
from html import escape
from pathlib import Path

from AlphaBrain.research.dsol.data.artifacts import read, sha, write


NAMES={'libero_canonical':'LIBERO 规范模型','libero_broad':'LIBERO 多视角模型',
       'plus_canonical':'Plus 规范模型','plus_broad':'Plus 多视角模型',
       'robocasa_official':'RoboCasa π0.5 官方模型','robocasa_xr1':'RoboCasa XR-1'}
RULES={'original':'原 Accel','near':'近期执行部分','near_radial':'近期幅度项','near_angular':'近期方向项'}


def pct(value):
    return '未知' if value is None else f'{100*value:.2f}%'


def interval(values):
    return '['+', '.join(f'{100*v:+.2f}' for v in values)+'] pp'


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def render(root):
    root=Path(root);target=root/'report.html'
    if target.exists():raise ValueError('Refuse to replace report')
    main=read(root/'summary.json');diag=read(root/'diagnostics/summary.json');phase=read(root/'phase-alignment/alignment.json')
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Accel 结构分解与阶段对齐</title><style>body{max-width:1280px;margin:36px auto;padding:0 22px;font:16px/1.7 system-ui;color:#202733}h1,h2,h3{line-height:1.3}table{border-collapse:collapse;width:100%;margin:18px 0;font-size:14px}th,td{padding:8px 10px;border:1px solid #d8dee7;text-align:left}th{background:#eef3f8}img{width:100%;margin:12px 0}code{background:#f3f5f7}a{color:#125a9d}.note{background:#f4f7fb;padding:16px;border-left:4px solid #426d99}</style><body>',
        '<h1>Accel 结构分解与阶段对齐：已有数据分析</h1>',
        '<p>2026-10-15。结论：本轮固定的近期、幅度、方向版本均不足以替换原指标。结构分解提供了更具体的解释：LIBERO 抽屉反转主要涉及旋转通道，XR-1 入柜反转主要涉及夹爪通道；相同通道也参与改善，不能据此直接删去或减权。</p>',
        '<p class="note">仅 CPU 读取已有数组。没有新增推理、渲染、仿真、训练或旧队列，没有新增成功标签，没有覆盖旧结果或展示 PDF。全部数据此前已被研究者查看，属于开发证据；描述性区间不构成独立确认或多重比较校正。</p>',
        '<h2>覆盖与数值验收</h2>']
    rows=[]
    for p,c in main['coverage'].items():
        rows.append([NAMES[p],c['initials'],c['candidates'],c['labeled_candidates'],c['candidates']-c['labeled_candidates'],c['scores'],'逐值一致',f"{c['max_identity_error']:.2e}"])
    parts.append(table(['面板','初态记录','初态×候选','有闭环标签','缺失标签','评分数','原分数','最大分解重构误差'],rows))
    parts.append('<p>六面板合计 288 条初态记录、24,520 个初态×候选组合、980,800 个分数；不同面板及评分重复不视为独立初态。四例短轨迹另外核验 64 条分支、320 个评分节点。原分数及原候选选择均精确重现；各诊断子空间未出现近零范数导致的无效方向。</p>')
    parts.append('<h2>固定口径</h2><p>去噪前缀固定为 3，沿用原 8 套评分噪声及先噪声平均、再累计窗口平均的顺序。只比较 W1/W3/W5、含规范与去规范候选。LIBERO/Plus 近期为 5×7，π0.5 为 25×12；XR-1 原指标已经是 16×12，近期版本与原指标完全相同。</p>')
    parts.append('<p>令 r=||v||、u=v/r。逐对核验 ||Δv||²=(Δr)²+rⱼrⱼ₋₁||Δu||²，随后开平方重构原分数。幅度规则为 3Σ|Δr|/Σr；方向规则为 3Σ√(rⱼrⱼ₋₁||Δu||²)/Σr，均在近期子空间计算。两者不是可相加的 Accel 分量。方向均为模型归一化动作坐标中的方向，不能直接解释为物理运动方向。</p>')
    parts.append('<p>通道贡献为 3Σ(||Δv通道||²/||Δv全体||)/Σ||v全体||，因此通道贡献可严格加回原 Accel。RoboCasa 保留平移、旋转、夹爪、底座及控制模式，π0.5 额外保留内部未输出维度。范数≤1e-12时方向无效；若一个候选所需的任何评分重复/窗口无效，就不参与对应规则选位，无有效候选则记未知。本轮未触发。</p>')
    parts.append('<h2>W3 总览：完整候选集合</h2>')
    rows=[]
    for p,scopes in main['panels'].items():
        r=scopes['all']['include']
        rows.append([NAMES[p],*[pct(r[k][1]['success'])+(f"（标签{r[k][1]['label_coverage']}/{r[k][1]['n']}）" if r[k][1]['success'] is None else '') for k in RULES]])
    parts.append(table(['面板',*RULES.values()],rows))
    parts.append('<p>LIBERO 规范 W3：近期版本比原指标低 10.74 pp，近期幅度低 9.47 pp，近期方向低 10.25 pp。去掉规范候选后分别低 7.71、13.67、7.71 pp，因此损失不能仅归于规范候选。近期版本的逐任务差值没有正值；不能因为 W5 某些版本略升，就忽略 W3 原有恢复收益受损。</p>')
    parts.append('<h2>全部固定比较：负结果、缺失标签与候选排除均保留</h2><p>成功率缺失时给出将未知候选成绩限定在 [0,1] 的总体界限；差值界限保留相同未知赢家的配对约束。界限不是置信区间。只有完整标签时给出配对描述性 95% bootstrap 区间，沿用任务→划分内初态→执行重复的统计实现。改善/退化计数仅指有配对标签的初态。</p>')
    for p,scopes in main['panels'].items():
        parts.append('<h3>'+NAMES[p]+'</h3>')
        for mode,title in [('include','含规范候选'),('exclude','去掉规范候选')]:
            rows=[]
            for rule,rr in scopes['all'][mode].items():
                for r in rr:
                    value=pct(r['success']) if r['success'] is not None else '未知；界限 '+str([round(100*v,2) for v in r['success_bounds']])+'%'
                    delta=interval(r['delta_ci95']) if 'delta_ci95' in r else '界限 '+interval(r['paired_delta_bounds'])
                    rows.append([RULES[rule],r['window'],value,f"{r['label_coverage']}/{r['n']}",f"{r['improved']}/{r['degraded']}/{r['equal']}",delta])
            parts.append('<p>'+title+'</p>'+table(['规则','窗口','成功率','标签覆盖','改善/退化/相同','相对原规则差值区间'],rows))
    parts.append('<h2>RoboCasa 固定 32 候选补充诊断</h2><p>只使用历史预先固定的 extra32 集合，每个初态都有相同 32 个候选及完整标签。该子集不按本轮成功标签选择。这里重新在 32 个候选内选位，是独立的统计范围，不是完整 97 候选的成功率，也不替代缺失标签。</p>')
    rows=[]
    for p,scopes in diag['fixed32'].items():
        for rule,rr in scopes['all'].items():
            for r in rr:rows.append([NAMES[p],RULES[rule],r['window'],pct(r['success']),f"{100*r['delta']:+.2f} pp",interval(r['delta_ci95']),f"{r['improved']}/{r['degraded']}"])
    parts.append(table(['模型','规则','窗口','成功率','相对原规则','配对95%区间','改善/退化'],rows))
    parts.append('<p>π0.5 三种变体在 W1/W3/W5 的点估计均低于原规则。XR-1 幅度项 W5 有 +3.75 pp 的描述性改善，但区间跨零、W1 为负且入柜案例未修复；方向项 W3 的 +1.25 pp 来自一个改善初态。均不足以升级为可靠新规则。</p>')
    parts.append('<h2>四例短轨迹：分数、通道与实际事件</h2><p>负的“较好候选−较差候选”评分差表示偏好历史较好候选。曲线的好坏标签来自独立执行重复，并非这些短轨迹的最终成功判定。末节点只生成了新动作块，没有继续执行。距离为末端到物体/柜体中心，仅用于解释。</p>')
    for case,c in phase['cases'].items():
        parts.append('<h3>'+escape(c['title'])+'</h3>')
        rows=[]
        for rule,r in c['comparisons'].items():
            rows.append([RULES[rule],*[f"{r['good_preferred_repeats'][i]}/8" for i in (0,2,4)]])
        parts.append(table(['规则','W1偏好较好候选','W3偏好较好候选','W5偏好较好候选'],rows))
        parts.append(f'<a href="phase-figures/{case}.pdf">下载矢量 PDF</a><img src="phase-figures/{case}.png" alt="{escape(c["title"])}">')
    parts.append('<p>入柜 W5 总评分差 +0.02735，其中夹爪通道贡献 +0.02906，其余通道净抵消一部分。均值排序在第48步反转，好候选首次目标接触在第57–64步，7/8发生；因此接触不能解释更早的反转。较好候选第64步的夹爪预测块发生明显变化，但这个最终动作块未执行，不能当作已经抓住。</p>')
    parts.append('<p>两例 LIBERO 的后期差值主要来自旋转通道；近期截断可推迟部分噪声重复的反转，却未消除。XR-1 入柜中，原指标、幅度项、方向项在 W5 都有 8/8 重复偏好较差候选。水槽案例的幅度项 W5 为 8/8 偏好较好候选，原指标为 6/8；这是局部正结果，不能抵消其他反例。</p>')
    parts.append('<h2>跨初态核验：通道主导不等于错误原因</h2>')
    rows=[]
    for p,g in diag['transition_groups'].items():
        for outcome in ('improved','degraded'):
            r=g['W3_W5'][outcome];counts={k.replace('baseline_','').replace('__accel_contribution',''):v for k,v in r['strongest_channel_counts'].items()}
            rows.append([NAMES[p],'改善' if outcome=='improved' else '退化',r['n'],counts])
    parts.append(table(['面板','原W3→W5结果','切换初态数','最推动新候选排序的通道计数'],rows))
    parts.append('<p>LIBERO 规范的6次退化均由旋转贡献主导，但1次改善也一样；多视角模型的7次改善和7次退化也都如此。XR-1 的夹爪通道主导4/7次退化，也主导5/14次改善。因此“降低旋转/夹爪权重”仍没有足够依据。</p>')
    parts.append('<h2>值得继续什么</h2><ul><li>保留原 Accel 与远期诊断，暂不采用近期截断、纯幅度或纯方向替代。它们没有同时保住 LIBERO/Plus 收益并改善 RoboCasa。</li><li>保留“通道变化×操作阶段”的机制方向，重点是接近目标/准备抓取时夹爪生成结构如何改变，以及旋转变化为何既伴随改善也伴随退化。当前证据仍是分支状态不同下的描述，尚未解释网络内部原因。</li><li>XR-1 幅度项的局部改善只作为弱假设保留，不选最佳窗口、不加权追分、不因这个结果扩大实验。</li><li>真正区分状态阶段与观察视角，需要同物理状态跨视角数据。现有中间状态资料缺少这种对照；本轮没有补采。若下一步开展，应先另行明确最小对照范围和判据。</li></ul>')
    parts.append('<h2>复核入口</h2><p><a href="summary.json">六面板汇总</a> · <a href="coverage.json">覆盖核对</a> · <a href="sources.json">历史输入哈希</a> · <a href="frozen-plan.json">固定分析方案</a> · <a href="diagnostics/summary.json">固定32、逐任务与零成功候选变化</a> · <a href="diagnostics/transitions.json">所有切换及通道贡献</a> · <a href="phase-alignment/alignment.json">逐重复时间对齐</a> · <a href="validation.json">验证记录</a></p><p>每面板目录中的编号 JSON 保留逐初态各规则选择、标签与缺失状态，NPZ 保留完整结构分解；summary.json 同时包含初态划分、去规范候选及逐任务/逐一去任务结果。所有文件为本轮独立产物。</p></body></html>')
    target.write_text('\n'.join(parts),encoding='utf-8')
    write(root/'report-provenance.json',dict(source_sha256=sha(__file__),inputs={str(root/p):sha(root/p) for p in ['summary.json','diagnostics/summary.json','phase-alignment/alignment.json']},output_sha256=sha(target)))
    return target


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);print(render(p.parse_args().root))
