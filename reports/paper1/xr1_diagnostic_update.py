"""Integrate admitted XR-1 evidence, preserving the previous report body."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from AlphaBrain.research.dsol.analysis.view_landscape import load_panel
from AlphaBrain.research.dsol.data.artifacts import read, write, sha
from AlphaBrain.research.dsol.plotting.robocasa_results import space_axes
from .robocasa_diagnostic_update import DiagnosticBrief, publish, fitz, PdfPages
from .robocasa_results import BLUE, TEAL, GRAY, INK, MUTED


def percentage(value):
    """Keep .125 percentage ties consistent with the user-facing summary."""
    return str(Decimal(str(value*100)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))


class XR1Brief(DiagnosticBrief):
    def stronger_model(self, pdf):
        r = self.summary['splits']['all']['xr1']
        pi = self.summary['splits']['all']['official']
        self.new('更强模型仍未恢复：空间关系不等于最低点决策',
                 'RoboCasa · XR-1 官方权重对照',
                 '相同10任务、40初态、97候选；4次完整闭环＋8次独立Accel评分；未做我们的后训练',
                 '模型能力提高，但视角损失仍存在；按最低 Accel 选位未显示平均恢复。',
                 'XR-1与π0.5使用各自原生动作窗口、历史和流采样；比较恢复与排名，不直接比较Accel绝对尺度。\nOracle先平均4次执行再按初态取最大，仅为有限样本事后参照，不是已部署方法或独立确认的真实上限。',
                 'XR-1规范66.875%，额外32视角平均53.515625%，最低Accel48.125%，经验Oracle88.125%。'
                 '规范视角比π0.5高18.75pp，但初态最低Accel相对不调整为-5.390625pp，'
                 '任务/初态bootstrap区间[-14.8457,3.5552]pp，不能断言稳定有害。'
                 'seen和heldout恢复分别-5.3125/-5.46875pp。空间图相关0.8345，初态内合并排名相关0.1696；'
                 '不能将平均相关作为逐初态准确识别能力。安装每次执行固定，随原生本体运动，不是在线调相机。')
        ax = self.fig.add_axes([.065, .685, .87, .092]); ax.axis('off')
        values = [[name, percentage(x['reference']), percentage(x['stay']),
                   percentage(x['selected']), percentage(x['empirical_hierarchy']['initial_oracle'])]
                  for name, x in [('π0.5 官方', pi), ('XR-1 官方', r)]]
        table = ax.table(cellText=values, colLabels=['模型', '规范视角%', '扰动不调整%', '最低Accel%', '经验Oracle%'],
                         loc='center', cellLoc='center', colWidths=[.2]*5)
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.32)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor('#DCE3EB')
            if row == 0: cell.set_facecolor('#EDF3FA')
        m = self.panel['models'].index('xr1')
        for col, (values, is_percentage, title) in enumerate([
                (self.panel['accel'][m, :, 1:].mean(0), False, 'XR-1 · 初态Accel'),
                (self.panel['y'][m, :, 1:].mean((0, 2)), True, 'XR-1 · 闭环成功率')]):
            ax = space_axes(self.fig, [.065+.475*col, .31, .36, .32],
                            self.panel['coordinates'], values, title, percentage=is_percentage)
            ax.set_zlabel('')
        corr = r['association']
        self.text(.075, .263, f"平均空间图相关 {corr['spatial_mean_spearman']:.3f}  ｜  初态内排名相关 {corr['pooled_within_initial_rank_correlation']:.3f}", 12, INK, True)
        self.save(pdf)

    def implications(self, pdf):
        r = self.summary['splits']['all']['xr1']
        self.new('当前瓶颈：初态 Accel 未可靠识别整任务好视角',
                 '研究结论与下一步 · 不作单因素归因',
                 '低值区域、最低点与可实现决策分开；保持已有LIBERO正结果和RoboCasa负结果',
                 '下一步先检验阶段信息与任务差异，不直接训练追逐有限样本 Oracle。',
                 '低值组：96个扰动候选按每初态Accel排名，各24点；最低点允许规范候选，不调整来自额外32点。\n这些分布并不相同，不能将区域均值写成新选择器成绩。只有4次执行，Oracle排名与差异仍需独立确认。',
                 'XR-1四分位成功率52.42/53.57/50.10/40.49%，最低点48.13%，'
                 '比低值24点平均更差，但不能据此直接证明新区域规则会恢复。'
                 '咖啡放杯+11.91pp、打开抽屉+6.25pp、关闭冰箱-6.64pp、导航-16.21pp，说明任务依赖，'
                 '不是证明唯一任务根因。更强模型也不是每任务都好：搅拌杯盖规范为0。'
                 '原四步为评分定义核验、更强模型对照、多时刻诊断与LIBERO-Plus Camera、最后搜索与主动相机闭环。'
                 '前1/3/5次重规划已存档，可先分析已有数据；自己的执行轨迹分数与自己的成败相关不能直接作为初态选择器成绩。'
                 '如做短前缀分支试探，应固定校准与执行噪声分离，明确需要动作、回滚与查询成本。')
        labels = ['最低组', '次低组', '次高组', '最高组']
        ax = self.axes([.075, .36, .41, .37])
        ax.plot(range(4), 100*np.array(r['quartiles']), 'o-', color=BLUE, lw=2)
        ax.axhline(r['stay']*100, color=GRAY, ls='--', label='32扰动视角不调整')
        ax.axhline(r['selected']*100, color=TEAL, ls=':', label='最低Accel单点')
        ax.set_xticks(range(4), labels, fontsize=10); ax.set_ylim(30, 65)
        ax.set_ylabel('完整任务成功率（%）', fontsize=11)
        ax.set_xlabel('初态内按Accel从低到高分组', fontsize=10)
        ax.legend(fontsize=8, loc='lower left'); ax.grid(alpha=.2)
        self.text(.54, .73, '支持的结论', 15, BLUE, True)
        self.text(.54, .67, '① 任务之间，指标作用方向不同。\n② 空间存在经验收益，但最低点没有提取。\n③ 更强模型并未自动解决这个问题。', 12, INK)
        calibration=self.summary.get('seen_rules',{}).get('models',{}).get('xr1')
        if calibration:
            self.text(.54, .525, '简单校准对照（探索性）', 15, BLUE, True)
            self.text(.54, .465, '仅seen20初态选定一个固定位置：\n'
                      f"heldout20：{calibration['stay']*100:.2f}% → {calibration['global_fixed']*100:.2f}%。\n"
                      '已知任务／目录；非盲测，非主动搜索。', 12, INK)
        else:
            self.text(.54, .525, '尚不能确定的根因', 15, BLUE, True)
            self.text(.54, .465, '任务瓶颈、阶段信息与评分误选各占多少？\n低Accel可能对应错误但平稳的动作生成。\n这仍是待验证解释，不是机制结论。', 12, INK)
        self.text(.075, .265, '下一步：先用已存档前1/3/5次重规划做诊断；再独立验证时序规则和Camera扰动。', 11, INK, True)
        self.save(pdf)


def build(public_dir, analysis_root, panel_root, plot_root, output_root, seen_rule_root=None):
    public_dir, analysis_root, plot_root, output_root = map(Path, (public_dir, analysis_root, plot_root, output_root))
    completion = read(plot_root/'completion.json')
    if completion['status'] != 'COMPLETE' or completion['statistics_recomputed']:
        raise ValueError('Completed unchanged statistics and figures required')
    sources = dict(completion['sources'])
    sources.update({str(plot_root/'completion.json'):sha(plot_root/'completion.json')})
    sources.update({str(plot_root/name):value for name,value in completion['figures'].items()})
    for path,value in sources.items():
        if sha(path) != value: raise ValueError('Input artifact changed')
    summary = read(analysis_root/'summary.json')
    if seen_rule_root is not None:
        path=Path(seen_rule_root)/'summary.json'
        seen_rules=read(path)
        if seen_rules['sources'][str(analysis_root/'panel_arrays.npz')] != sha(analysis_root/'panel_arrays.npz'):
            raise ValueError('Seen-calibration outcomes differ from the admitted panel')
        summary['seen_rules']=seen_rules
        sources[str(path)]=sha(path)
    panel = load_panel(panel_root)
    with np.load(analysis_root/'panel_arrays.npz', allow_pickle=False) as data:
        panel.update(y=data['success'].copy(), accel=data['accel'].copy(), models=data['models'].tolist())
    output = output_root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    previous = output/'previous-publication'; previous.mkdir()
    for name in ('vla_view_research_progress_zh.pdf','vla_view_research_progress_zh.md','publication.json'):
        shutil.copy2(public_dir/name, previous/name)
    before = {p.name:sha(p) for p in previous.iterdir()}
    brief = XR1Brief(summary, panel)
    with PdfPages(output/'xr1-pages.pdf') as pdf:
        brief.stronger_model(pdf); brief.implications(pdf)
    with fitz.open(previous/'vla_view_research_progress_zh.pdf') as old, fitz.open(output/'xr1-pages.pdf') as new:
        if len(old) not in (18,19) or len(new) != 2:
            raise ValueError('Expected original18/19 + replacement2')
        result = fitz.open(); result.insert_pdf(old, from_page=0, to_page=16); result.insert_pdf(new)
        for i,page in enumerate(result):
            page.add_redact_annot(fitz.Rect(0,524,960,540), fill=(1,1,1))
            page.apply_redactions(images=0,graphics=0,text=0)
            page.insert_text((34,531),'VLA 研究进展 · 2026-09-30 · XR-1同初态验证；经验Oracle不等于可实现方法',fontname='china-s',fontsize=7,color=(.39,.45,.51))
            page.insert_text((901,531),f'{i+1:02d} / 19',fontsize=8,color=(.39,.45,.51))
        toc = [row for row in old.get_toc() if row[2] <= 17]
        toc.extend([[1,n['title'],18+i] for i,n in enumerate(brief.notes)])
        result.set_toc(toc); result.set_metadata(dict(title='VLA研究进展｜LIBERO、RoboCasa与XR-1恢复验证',author='VLA View Research'))
        target = output/'vla_view_research_progress_zh.pdf'; result.save(target,garbage=4,deflate=True); result.close()
    preservation=[]
    with fitz.open(previous/'vla_view_research_progress_zh.pdf') as old, fitz.open(target) as current:
        if len(current) != 19: raise ValueError('Unexpected page count')
        for i in range(17):
            clip=fitz.Rect(0,40,959.976,518)
            same=old[i].get_pixmap(clip=clip).samples==current[i].get_pixmap(clip=clip).samples
            if not same: raise ValueError('Retained page body changed')
            preservation.append(dict(page=i+1,body_pixels_identical=same))
        previews=output/'previews';previews.mkdir()
        for i in (17,18):
            pix=current[i].get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False)
            Image.frombytes('RGB',(pix.width,pix.height),pix.samples).save(previews/f'page-{i+1:02d}.png')
    write(output/'preservation.json',preservation)
    notes=(previous/'vla_view_research_progress_zh.md').read_text()
    notes=notes.split('\n\n## 2026-09-30 XR-1更新')[0]
    notes=notes.replace('（2026-09-29）','（2026-09-30）',1)
    notes=notes.replace('本版18页。保留旧版第1–14页正文与所有LIBERO图表；旧第15–16页由四页最新诊断与结论替换。',
                        '本版19页。保留此前第1–17页正文及全部LIBERO图表；此前第18页结论被两页XR-1证据与结论替换。',1)
    notes=notes.replace('没有启动新训练、推理或闭环实验。新增官方诊断本身已完成14,720次闭环，复用800次；两后训练模型复用原结果。',
                        '本次绘图和PDF处理未启动训练或新闭环。π0.5官方诊断为此前14,720次＋复用800次；本轮XR-1独立完成15,520次闭环，不能用π0.5结果替代。三份π0.5模型对照只读复用原数组。',1)
    notes=notes.replace('17页：最低点、低值区域和任务差异；18页：当前问题与证据边界。',
                        '17页：最低点、低值区域和任务差异；18页：XR-1同初态对照；19页：决策瓶颈、校准对照与下一步。',1)
    begin=notes.find('### 第18页：')
    end=notes.find('## 来源与归档',begin)
    if begin>=0 and end>begin:
        notes=notes[:begin]+notes[end:]
    notes += '\n\n## 2026-09-30 XR-1更新（本版19页）\n\n保留此前1–17页正文与图表；原第18页结论被两页更强模型结果与下一步替换。此前第18页讲解仅代表旧版，旧版完整归档。\n'
    notes += '\n| 同40初态模型 | 规范% | 扰动不调整% | 最低Accel% | 收益pp | 经验Oracle% |\n|---|---:|---:|---:|---:|---:|\n'
    for name,label in [('official','π0.5官方'),('xr1','XR-1官方')]:
        r=summary['splits']['all'][name]
        notes += f"| {label} | {percentage(r['reference'])} | {percentage(r['stay'])} | {percentage(r['selected'])} | {r['gain']['mean']*100:+.2f} | {percentage(r['empirical_hierarchy']['initial_oracle'])} |\n"
    for i,n in enumerate(brief.notes,18):
        notes += f"\n### 新第{i}页：{n['title']}\n\n{n['takeaway']}\n\n{n['speaker_note']}\n"
    if 'seen_rules' in summary:
        r=summary['seen_rules']['models']['xr1']
        notes += ('\n### 规则是否一定无法恢复？\n\n'
                  f"只用seen20初态标签选定全局固定视角{r['global_view']}，在heldout20初态上：不调整{r['stay']*100:.2f}%，固定位置{r['global_fixed']*100:.2f}%，恢复{r['global_gain']['mean']*100:+.2f}pp；每任务固定查表{r['task_fixed']*100:.2f}%。\n"
                  '这是探索性校准对照：选择没有使用heldout标签，但heldout结果此前已被查看，不声称独立盲测；任务和候选目录已知，不证明任意场景泛化或真实主动相机搜索。\n')
    notes += f'\n原始统计：{analysis_root}/summary.json\n绘图补齐与哈希：{plot_root}/completion.json\n旧版归档：{previous}\n'
    (output/'vla_view_research_progress_zh.md').write_text(notes)
    artifacts={name:sha(output/name) for name in ('vla_view_research_progress_zh.pdf','vla_view_research_progress_zh.md')}
    write(output/'provenance.json',dict(before=before,sources=sources,artifacts=artifacts,pages=19,
          builder={str(Path(__file__).resolve()):sha(__file__)},retained_body_pages=list(range(1,18)),new_pages=[18,19],
          new_experiments=0,added_closed_loop_results=15520,
          scope='Unified LIBERO/RoboCasa including matched XR-1 official recovery diagnostic',
          layout_pass=not any(q['issues'] for q in brief.qa)))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-dir',type=Path,required=True)
    parser.add_argument('--analysis-root',type=Path)
    parser.add_argument('--panel-root',type=Path)
    parser.add_argument('--plot-root',type=Path)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--seen-rule-root',type=Path)
    parser.add_argument('--publish-built',type=Path)
    args=parser.parse_args()
    if args.publish_built:
        publish(args.publish_built,args.public_dir)
    else:
        if not all((args.analysis_root,args.panel_root,args.plot_root,args.output_root)):
            parser.error('Build requires analysis/panel/plot/output roots')
        print(build(args.public_dir,args.analysis_root,args.panel_root,args.plot_root,args.output_root,args.seen_rule_root))


if __name__=='__main__':main()
