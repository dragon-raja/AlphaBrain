"""Publish a complete, read-only review of the audited shared-core decision.

Usage: python -m reports.paper1.core_review --source AUDITED_REPORT --output NEW_DIR
Requires NumPy, Matplotlib and PyMuPDF. Does not evaluate models or alter queues.
"""
import argparse
import csv
import hashlib
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import shutil

from AlphaBrain.research.dsol.analysis.core_review import verify_presentation
from AlphaBrain.research.dsol.plotting.core_review import figures


def table(headers, rows):
    def cell(value):
        return html.escape(str(value))
    return '<table><thead><tr>' + ''.join(f'<th>{cell(h)}</th>' for h in headers) + (
        '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{cell(v)}</td>' for v in row)
        + '</tr>' for row in rows) + '</tbody></table>')


def para(text):
    return f'<p>{html.escape(text)}</p>'


def image(name):
    return f'<img src="{name}.png" style="width:100%">'


def export_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


CSS = '''
body { font-family: sans-serif; font-size: 10pt; line-height: 1.45; color: #172c40; }
h1 { font-size: 20pt; color: #145577; margin-bottom: 15pt; }
h2 { font-size: 13pt; margin-top: 14pt; }
p { margin: 8pt 0; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 12pt 0; }
td, th { border-bottom: 1px solid #cbd5df; padding: 5pt 4pt; text-align: left; }
th { background: #e8f0f5; }
img { margin: 8pt 0; }
'''


def publish_pdf(sections, output):
    import pymupdf as fitz
    draft = output / '.review-layout.pdf'
    writer = fitz.DocumentWriter(str(draft))
    count, toc = 0, []
    for title, body in sections:
        toc.append([1, title, count+1])
        story = fitz.Story(html=f'<h1>{html.escape(title)}</h1>{body}', user_css=CSS,
                           archive=fitz.Archive(str(output)))
        more = True
        while more:
            device = writer.begin_page(fitz.Rect(0, 0, 595, 842))
            more, _ = story.place(fitz.Rect(35, 35, 560, 795))
            story.draw(device)
            writer.end_page()
            count += 1
    writer.close()
    doc = fitz.open(draft)
    doc.set_toc(toc)
    for index, page in enumerate(doc):
        page.insert_text((35, 820), f'Development evidence review | {index+1} / {len(doc)}',
                         fontsize=8, color=(.35, .4, .45))
    doc.set_metadata(dict(title='Shared-core ablation, mechanism and continuation review',
                          subject='Audited development evidence; not final held-out results'))
    doc.subset_fonts()
    doc.save(output / 'review.pdf', garbage=4, deflate=True)
    (output / 'review.txt').write_text('\n\n'.join(page.get_text() for page in doc))
    pages = len(doc)
    doc.close()
    draft.unlink()
    return pages


def build(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    names = ['summary.json', 'learning-curves.json', 'paired-outcomes.csv', 'config.json',
             'input-hashes.json', 'development-curves.png', 'development-curves.pdf',
             'training-diagnostics.png', 'training-diagnostics.pdf', 'query-budget.png', 'query-budget.pdf']
    hashes = {name: digest(source/name) for name in names}
    summary = json.loads((source/'summary.json').read_text())
    curves = json.loads((source/'learning-curves.json').read_text())
    with (source/'paired-outcomes.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['split'] = 'validation'
        for key in ['success', 'stay_success', 'nominal_success']:
            if row[key] not in ('True', 'False'):
                raise ValueError('Invalid CSV boolean')
            row[key] = row[key] == 'True'
        for key in ['seed', 'queries', 'query_cap']:
            row[key] = int(row[key])
        for key in ['search_initial_accel', 'search_best_accel']:
            row[key] = float(row[key])
    verification = verify_presentation(rows, summary)
    output.mkdir(parents=True, exist_ok=False)
    for name in names:
        shutil.copyfile(source/name, output/name)
    figures(summary['summaries'], curves, output)
    s, comparisons = summary['summaries'], summary['comparisons']
    pct = lambda x: f'{100*x:.2f}'
    groups = ['U0', 'U1', 'U2', 'U3', 'U12', 'U13', 'U23', 'U123', 'U0T', 'U123T', 'U12T']
    sections = []
    add = lambda title, body: sections.append((title, body))
    add('完整审查：结构、机制、预算与续训',
        para('这份报告展示已完成的全部开发分析，而不只给配置选择结论。数值取自已核验的 shared-core-decision/report-r2；本次重新核对逐条配对 CSV 与汇总，新增图表与可审查表格，零模型调用，不修改在途主实验。')
        + table(['范围', '实际完成'], [
            ['结构 / 任务输入', '11组 × 3种子，各512回合；全部列出'],
            ['长训 / 早期评测', 'U12、U3各3种子，0–4096回合，11个实际评测点'],
            ['部署查询上限', '512点Q4/6/8；3072点Q4/6/8/16；训练均Q8'],
            ['机制 / 成本', 'λ=.005/.01/.02；无显式评分输入；无学习STOP'],
            ['总覆盖', '40面板 × 192条 = 7680条已有操作统计行；27项配对比较']])
        + para('全部是 π0.5 开发集实验：3策略种子 × 8任务 × 2初态 × 2扰动起点 × 2执行噪声。三个种子共享64条物理条件；192条不是192个独立场景。8个开发任务不属于20个当前PPO训练任务，也没有用最终12个留出任务选结构。')
        + para('Stay（扰动不调整）=50.00%，可靠参考位=90.625%，是这批开发面板的基线，不是全LIBERO成功率。不要与不同组成的在途主表直接相减。既有编码器初始化谱系仍须披露；任务划分核验不能替代早期预训练谱系审计。')
        + para('核心判断：U12在短训与多数长训点较好，但三种子区间较宽；2048点优于后期点，不能称严格收敛；Q16增加查询未显示收益；显式评分输入有开发集支持，STOP和额外T的收益没有被这批数据充分证明。')
        + para('阅读顺序：结构定义与全表 → 续训成功率/Accel → PPO日志 → 查询预算 → 机制 → 配对区间 → 完整明细与证据边界。当前三模型主实验另有冻结配置与状态，本报告不重写历史决策时间线。'))
    add('1. 结构定义与全部11组结果',
        para('U0：当前角度/限位在历史聚合后加入，Context线性映射，真实腕部进入Context。U1：状态改为History前融合。U2：Context改为Linear+Tanh。U3：辅助视图改进查询记录，并引入可学习的双视图融合，参数随之变化。组合按编号叠加。T：额外独立任务信息支路。本组不是逐项“删掉创新模块”的最终长训消融，而是匹配512回合的结构筛选。')
        + para('Actor与Critic使用对应同结构、独立参数；History的2层4头128宽与已访问记录mean聚合保持一致。T只补了U0/U123/U12三组，不是所有U组合都加T。')
        + table(['组', '成功率%', '种子0 / 1 / 2 (%)', '救回/损害', '平均查询'], [
            [v, pct(s[v+'@512']['success_rate']), ' / '.join(pct(x) for x in s[v+'@512']['seed_rates']),
             f"{s[v+'@512']['rescues']}/{s[v+'@512']['harms']}", f"{s[v+'@512']['mean_queries']:.3f}"] for v in groups])
        + para('救回：Stay失败、搜索成功；损害：Stay成功、搜索失败。此表每组分母192，净增益=(救回−损害)/192。U12领先U0约10.94个百分点，但本表不能把组合差值全部解释成单一模块的独立因果收益。')
        + para('U12+T比U12低8.33个百分点，三个配对种子均下降；只能说明这次连接方式/512回合下没有支持把T设为默认，不能得出任务信息普遍无用。U0/U13的种子跨度很大，不能隐藏低分种子。'))
    add('2. 结构图表：逐种子与查询成本', image('structure-all-seeds')
        + para('点：每个训练种子的成功率；柱：三种子均值，不是置信区间。所有组共享开发任务、初态、起点与执行噪声；短训几乎都用满预算，所以成功率差异不能用“谁多问了很多次”简单解释。')
        + para('U12与U3是后续长训比较对象。短训排名不等于最终收敛排名，也不等于在GR00T、RLDX上全部组合都完成了验证。统一的是后续核心合同，而不是把π0.5结构筛选冒充跨模型结构消融。'))
    episodes = [0, 128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]
    add('3. 续训完整成功率曲线与原始测点',
        table(['回合', '更新数', 'U12成功率%', 'U3成功率%', 'U12查询', 'U3查询'], [
            [n, n//32, pct(s[f'U12@{n}']['success_rate']), pct(s[f'U3@{n}']['success_rate']),
             f"{s[f'U12@{n}']['mean_queries']:.3f}", f"{s[f'U3@{n}']['mean_queries']:.3f}"] for n in episodes])
        + para('每32个训练回合更新一次PPO；512/2048/4096回合分别是16/64/128次更新。每个测点运行完整操作得到成功率，不是训练reward换算的成功率。没有每回合都运行完整操作评测，图上的线只连接实际测点。')
        + para('U12@2048：种子57/64、57/64、56/64；总170/192=88.54%，救回76、额外损害2。3072为87.50%，4096为82.81%。继续训练并未单调改善，不能默认最后检查点最好。')
        + para('U3后期查询约5.1–5.5，低于U12约6.6–7.3，但成功率多数略低。U12在512–4096的8个对应点中6次较高、1次较低、1次相同；这些是相关检查点，不能当8次独立显著性试验。')
        + para('不满足严格平台判据：后期连续512回合点仍有超过2个百分点的变化。更准确的描述是收益趋缓且波动，而不是“已收敛”。开发集挑选2048存在选择偏差，冻结后完整独立评测才是主结果。'))
    add('4. 操作成功率、最低Accel与停止行为', image('development-curves')
        + para('蓝=U12，橙=U3；成功率图细线=各种子，粗线=均值；灰虚线=Stay，黑点线=可靠参考位。其他三图为部署平均查询、访问过的最低Accel、只观察起点的比例。')
        + para('最低Accel降低不保证操作成功率单调提高；它是搜索代理指标。图中仅有开发评测测点，没有用平滑曲线隐藏波动。0回合确定性策略只留起点，不代表随机训练采样也完全不探索。'))
    add('5. PPO训练日志：奖励、尺度与更新稳定性', image('training-diagnostics')
        + para('每条流128次真实更新；细线为各种子、粗线为均值，不平滑。reward取在线G尺度结算后的值，不能用采集阶段尚未结算的total_reward替代。')
        + para('没有NaN/Inf，G奖励重算最大误差约4.45e-16、累计尺度误差0，支持日志/奖励记账一致。它不等于重放所有梯度、证明策略最优或证明Accel与操作收益完全一致。近似KL与clip后期仍波动，价值损失低不能证明策略收敛。'))
    add('6. PPO补充诊断：损失、熵与收益分解', image('training-extra')
        + para('policy_loss / loss为优化日志；normalized_gain与query_cost分开呈现；stop_entropy和motion_unit_entropy分别反映停止与运动分布。initial_stop_rate是训练随机采样的起点停止频率，不能直接等同于确定性部署的initial-only。')
        + para('训练查询数和部署查询数可以不同。继续训练后reward仍在波动，但操作成功率并未同步提高，不能仅凭reward走高决定无限续训。'))
    add('7. 六条训练流的后段审计',
        table(['训练流', '后16更新奖励', '后16尺度', '后16 KL', '后16 clip', '全程max KL'], [
            [k.replace('pi05-', ''), f"{v['windows']['last16']['mean_reward']:.4f}",
             f"{v['windows']['last16']['scale']:.4f}", f"{v['windows']['last16']['approximate_kl']:.4f}",
             f"{v['windows']['last16']['clip_fraction']:.4f}", f"{v['maximum_kl']:.4f}"] for k, v in curves.items()])
        + para('后16更新对应最后512训练回合，是同一预定窗口，不逐种子挑选好看的区间。全程maximum KL是单次峰值，不是均值或失败阈值；没有预先规定的停止阈值时不能事后将其当作某组无效的理由。')
        + para('本次报告附training-updates.csv：6条流×128更新，完整保存policy/value loss、总loss、KL、clip、gradient norm、熵、尺度、查询、初始/最佳Accel及结算收益。若只看成功率曲线无法解释的行为，可以沿更新索引回查这些日志。')
        + para('三种子是有限证据。现阶段能说“未见数值发散、开发性能后期波动”，不能说三种子足以保证稳定收敛。扩大种子会提高精度，但本次整理不新增实验或改变既定预算。'))
    budget_rows = []
    for n in [512, 3072]:
        for q in ([4, 6, 8] if n == 512 else [4, 6, 8, 16]):
            d = s[f'U12{("-Q"+str(q)) if q != 8 else ""}@{n}']
            budget_rows.append([n, q, pct(d['success_rate']), f"{d['rescues']}/{d['harms']}", f"{d['mean_queries']:.3f}"])
    add('8. 查询预算：完整对照',
        table(['检查点回合', '部署Q上限', '成功率%', '救回/损害', '平均查询'], budget_rows)
        + image('query-budget')
        + para('同一训练检查点仅改部署上限，训练均为Q8。3072点Q16比Q8多约77.2%查询，但成功率少3.125个百分点；区间跨0，不能说显著更差，也不能称扩大预算有效。它不是Q16训练实验。')
        + para('Q包括初始与返回重查，每查询使用既定两个评分噪声成员；Q8不等于8次模型调用或8个新位置。Q4/Q6代价更低但恢复率下降，支持当前Q8折中，不支持所有任务/模型都最优。'))
    mech = ['U12', 'U12-cost005', 'U12-cost020', 'U12-no_score', 'U12-fixed_query']
    add('9. 查询成本、评分反馈与STOP',
        table(['条件@512', '成功率%', '三种子成功率%', '平均查询'], [
            [v, pct(s[v+'@512']['success_rate']), ' / '.join(pct(x) for x in s[v+'@512']['seed_rates']),
             f"{s[v+'@512']['mean_queries']:.3f}"] for v in mech])
        + image('mechanisms')
        + para('cost005/.020分别是λ=.005/.02；默认λ=.01。只做了三个值的短训筛查，不能包装为完整超参数稳健性证明。')
        + para('no_score只移除网络的显式评分输入，奖励与最终选位仍用Accel，必须写w/o score input，不能写w/o Accel。fixed_query移除学习STOP，仍有查询/安全终止；已在最终位时可能无需返回重查，因此不是每条都恰好8次。')
        + para('完整81.25% vs无学习STOP79.69%，仅差1.56个百分点，不能声称STOP显著提高成功率。无评分输入下降14.58个百分点，是较明确的开发集信号；仍只适用于本次512训练量与输入定义。'))
    comparison_rows = []
    for name, c in comparisons.items():
        lo, hi = c['crossed_seed_task_initial_ci95_sensitivity']
        comparison_rows.append([name.replace(' minus ', ' − '), f"{100*c['mean_difference']:+.2f}",
                                f'[{100*lo:.2f}, {100*hi:.2f}]', f"{c['query_difference']:+.3f}"])
    for part in range(2):
        add(f'10.{part+1} 全部配对比较与不确定性',
            para('差异=左−右，成功率单位为百分点。区间为种子×任务×初态交叉bootstrap的95%敏感性区间，20,000次重采样。三个策略种子复用物理面板；未校正结构/检查点多次选择，不能用于宣称确认性显著性，跨0也不代表等价。')
            + table(['比较', '差异pp', '交叉区间', '查询差'], comparison_rows[part*14:(part+1)*14])
            + para('固定已观测策略的task/initial区间、逐种子/逐任务差异均保留在comparisons.csv和summary.json。当前报告沿用已审计的bootstrap，不重新选随机种子或重算到满意为止。'))
    allrows = []
    flat = []
    for name, d in s.items():
        allrows.append([name, f"{d['successes']}/{d['operations']}", ' / '.join(pct(x) for x in d['seed_rates']),
                        f"{d['mean_queries']:.3f}", f"{d['mean_best_accel']:.5f}"])
        flat.append(dict(panel=name, **{k: v for k, v in d.items() if not isinstance(v, (dict, list))},
                         **{f'seed{i}_success_rate': v for i, v in enumerate(d['seed_rates'])}))
    for part in range(2):
        add(f'11.{part+1} 全部40个面板明细',
            para('不只列最佳检查点。成功数/192中包含相同物理面板的三策略种子；各种子分母64。Accel为同一开发总体上平均的访问最低分，不是成功率，越低也不自动代表更好操作。')
            + table(['面板', '成功数', '种子0 / 1 / 2 (%)', '平均查询', '最佳Accel均值'], allrows[part*20:(part+1)*20]))
    add('12. 能写进论文的结论与尚未完成的证据',
        table(['可以据此报告', '不能据此声称'], [
            ['11组匹配512回合的开发结构筛选', '最终长训模型每个组件都已完成消融'],
            ['U12比U3多数开发点成功率高', 'U12在所有模型上统计最优'],
            ['完整0–4096测点与PPO日志', '4096已严格收敛或2048普遍最优'],
            ['同权重Q16未改善该面板', '按Q16重训也必定无效'],
            ['显式评分输入有开发集作用', '去掉整个Accel的方法已被评测'],
            ['当前T接法短训无收益', '任务语义本身没有价值'],
            ['Stay/Ours完整系统的恢复比较', '已证明PPO获取优于独立随机搜索']])
        + para('独立随机获取＋同样最低Accel终选，没有包含在本次已完成包中；Random-from-local只能解释同集合选位，不能替代独立随机获取。完整w/o Accel也未完成。若正文强调这些机制命题，需要相应证据，不能用别的实验改名补位。')
        + para('建议正文放冻结后的三模型独立测试主表、统一核心图及简洁关键机制证据；附录放11组筛选、逐种子曲线、完整成本/预算/STOP/T表和选择规则。512消融必须标训练量，不能与2048/3072最终结果混算组件收益。')
        + para('审查包新增CSV：all-panels.csv、comparisons.csv、training-updates.csv、task-map.csv；原summary.json、learning-curves.json、paired-outcomes.csv、input-hashes.json原字节保留。所有图同时提供PNG及矢量PDF。')
        + para('溯源：本次仅重新验证导出CSV与40组汇总、27项配对均值/救回/损害一致。此前原报告的请求/结果/噪声/账本核验及输入哈希继续作为上游证据，本轮没有声称重做全部原始GPU轨迹审计。源码与输入SHA256见provenance.json。'))
    export_csv(output/'all-panels.csv', flat)
    export_csv(output/'comparisons.csv', [dict(comparison=k, **{key: json.dumps(val) if isinstance(val, (list, dict)) else val
                                                             for key, val in v.items()}) for k, v in comparisons.items()])
    export_csv(output/'training-updates.csv', [dict(stream=k, **p) for k, v in curves.items() for p in v['points']])
    tasks = sorted({(r['suite'], r['task']) for r in rows})
    export_csv(output/'task-map.csv', [dict(index=i+1, suite=a, task=b) for i, (a, b) in enumerate(tasks)])
    content = '<html><head><meta charset="utf-8"><title>完整开发分析审查</title><style>'+CSS+(
        'body{max-width:1000px;margin:30px auto;padding:0 20px} section{margin:55px 0} ')+ '</style></head><body>'
    content += '<nav>' + ''.join(f'<p><a href="#s{i}">{html.escape(t)}</a></p>' for i, (t, _) in enumerate(sections)) + '</nav>'
    content += ''.join(f'<section id="s{i}"><h1>{html.escape(t)}</h1>{b}</section>' for i, (t, b) in enumerate(sections))
    (output/'review.html').write_text(content+'</body></html>')
    pages = publish_pdf(sections, output)
    if hashes != {name: digest(source/name) for name in names}:
        raise RuntimeError('Audited inputs changed during presentation')
    code = [Path(__file__), Path(verify_presentation.__code__.co_filename), Path(figures.__code__.co_filename)]
    provenance = dict(created_at=datetime.now(timezone.utc).isoformat(), source=str(source), input_sha256=hashes,
                      source_sha256={str(p): digest(p) for p in code}, pdf_pages=pages, verification=verification,
                      outputs_sha256={p.name: digest(p) for p in output.iterdir() if p.is_file()})
    (output/'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), pages=pages, verification=verification), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == '__main__':
    main()
