"""Update the integrated report from admitted diagnostic artifacts, without experiments.

Keep pages 1–14 (including all LIBERO evidence) and replace the old conclusion
with four figure-led diagnostic pages. Build and publication are separate.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import shutil

import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw

from AlphaBrain.research.dsol.analysis.view_landscape import load_panel
from AlphaBrain.research.dsol.data.artifacts import read, write, sha
from AlphaBrain.research.dsol.plotting.robocasa_results import space_axes
from reports.paper1.robocasa_results import (
    IntegratedBrief, fitz, plt, INK, MUTED, BLUE, TEAL, GRAY, AMBER, TASK_NAMES,
)

MODELS = ('official', 'canonical', 'broad')
NAMES = ('官方原始权重', '规范视角后训练', '多视角后训练')


def evidence_rows(summary, population=False):
    """Keep the two populations explicit; never replace one with the other."""
    source = summary['population_sensitivity'] if population else summary
    models = MODELS[1:] if population else MODELS
    return [(split, model, source['splits'][split][model])
            for split in ('pretrain', 'target') for model in models]


class DiagnosticBrief(IntegratedBrief):
    def __init__(self, summary, panel):
        for weight in ('Regular', 'Bold'):
            font_manager.fontManager.addfont(f'/usr/share/fonts/opentype/noto/NotoSansCJK-{weight}.ttc')
        # CJK TTC files here contain CFF outlines. Type42 output extracts text
        # but corrupts displayed glyphs; Type3 embeds the correct outlines.
        plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'pdf.fonttype': 3,
                             'axes.unicode_minus': False, 'font.size': 11})
        self.summary, self.panel = summary, panel
        self.notes, self.qa, self.figure_inputs = [], [], set()

    def official_space(self, pdf):
        self.new('官方模型补测：平均空间规律不等于逐初态选位能力',
                 'RoboCasa · 官方起点与同初态对照',
                 '共同40初态：20 seen＋20 heldout；97候选，每格4次完整闭环，初态8套评分噪声',
                 '官方模型存在平均空间关系，但最低 Accel 选位未显示恢复收益。',
                 '96个实测点；坐标＝方位角／仰角／半径倍率；不插值，各图独立色条，上黄低Accel、下黄高成功率。\n官方新增14,720次闭环＋复用800次；两份后训练模型复用同初态旧结果。不是400初态主矩阵。',
                 '官方为后训练前的RoboCasa Human300权重，不是通用pi05 base。两份后训练均为8250步。'
                 '图为40初态合并，不是heldout200总体。空间相关先平均初态，再对96个候选计算Spearman；'
                 '初态内相关先对每初态两变量排名并中心化，再合并，未使用逐初态中位数。'
                 '官方seen/heldout的空间相关为0.508/0.471，初态内合并排名相关为0.036/0.050。')
        for m, name in enumerate(MODELS):
            x = .035 + m*.325
            self.text(x+.145, .765, NAMES[m], 14, BLUE, True, ha='center')
            for row in range(2):
                values = (self.panel['accel'][m, :, 1:].mean(0) if row == 0 else
                          self.panel['y'][m, :, 1:].mean((0, 2)))
                ax = space_axes(self.fig, [x, .515 if row == 0 else .28, .29, .23],
                                self.panel['coordinates'], values, '', percentage=row == 1)
                ax.set_zlabel('')
                if row == 0:
                    ax.set_xlabel(''); ax.set_ylabel('')
            r = self.summary['splits']['all'][name]['association']
            self.text(x+.015, .258,
                      f"空间相关 {r['spatial_mean_spearman']:.3f}  ｜  初态内 {r['pooled_within_initial_rank_correlation']:.3f}",
                      10, INK)
        self.text(.05, .715, 'Accel', 9, INK, True)
        self.text(.05, .475, '成功率', 9, INK, True)
        self.save(pdf)

    def scopes(self, pdf):
        self.new('完整评测与小诊断分开报告：不能用子集替换主结果',
                 'RoboCasa · 场景划分与恢复收益',
                 '恢复收益＝全97候选最低Accel选位 − 额外32扰动视角均值；单位：百分点（pp）',
                 '规范模型 heldout 主结果仍为 +3.24 pp；小面板差异不是重新运行后的退化。',
                 'seen/heldout指布局与风格分组，不是新任务；两组均含同样10个已知任务。\n40初态诊断的后训练模型直接复用原单元，成功结果逐值一致；不能据此断言跨场景稳定恢复。',
                 '规范模型heldout200：40.2578%→43.5000%，+3.2422pp；诊断heldout20：'
                 '45.3906%→46.2500%，+0.8594pp；合并40初态：45.2930%→45.6250%，+0.3320pp。'
                 '总体与子集不同，不是新旧噪声重跑。主结果任务/初态bootstrap区间[-1.06,7.72]pp，'
                 '条件于已有4次执行均值，不含训练seed变异。官方仅40初态，不外推400初态成绩。')
        ax = self.axes([.12, .405, .35, .285], True, False)
        rows = evidence_rows(self.summary, population=True)
        vals = [100*r['gain']['mean'] for _, _, r in rows]
        ax.barh(range(4), vals, color=[BLUE, TEAL, BLUE, TEAL], height=.56)
        ax.set_yticks(range(4), ['seen · 规范', 'seen · 多视角', 'heldout · 规范', 'heldout · 多视角'], fontsize=10)
        ax.invert_yaxis(); ax.axvline(0, color=GRAY, lw=1); ax.set_xlim(-5.2, 5.2)
        ax.set_title('完整主矩阵｜每组200初态', fontsize=13, pad=16)
        ax.set_xlabel('恢复收益（pp）', fontsize=10)
        for i, v in enumerate(vals):
            ax.text(v+(.13 if v >= 0 else -.13), i, f'{v:+.2f}',
                    ha='left' if v >= 0 else 'right', va='center', fontsize=11)
        ax = self.fig.add_axes([.55, .40, .405, .33]); ax.set_axis_off()
        rows = evidence_rows(self.summary)
        cells = [[('seen' if s == 'pretrain' else 'heldout'), NAMES[MODELS.index(m)],
                  f"{r['stay']*100:.2f}", f"{r['selected']*100:.2f}", f"{r['gain']['mean']*100:+.2f}"]
                 for s, m, r in rows]
        table = ax.table(cellText=cells, colLabels=['场景', '模型', '不调整%', '选位后%', '收益pp'],
                         colWidths=[.16,.32,.17,.17,.18], loc='center', cellLoc='center')
        table.auto_set_font_size(False); table.set_fontsize(8.8); table.scale(1, 1.55)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('#DCE3EB')
            if row == 0: cell.set_facecolor('#EDF3FA')
        self.text(.56, .753, '共同小面板｜每组20初态', 13, BLUE, True)
        self.text(.12, .295, '规范模型：heldout 200初态 +3.24  →  子集20初态 +0.86  →  混合40初态 +0.33', 12, INK, True)
        self.text(.12, .255, '初态改变的是布局、物体与机器人状态等任务条件；4次执行均值也仍带有估计噪声。', 10.5, MUTED)
        self.save(pdf)

    def failure_structure(self, pdf):
        self.new('恢复失败并非单一原因：最低点、区域与任务都需区分',
                 'RoboCasa · 现象定位，不作单因素归因',
                 '左：固定初态内按Accel分组；右：完整200个heldout初态，每任务20初态',
                 '部分组存在极小值误选迹象；多视角模型的主要损失集中在两个任务。',
                 '低值区域为96扰动候选中排名最低的24点之平均，不是已实现的新选择器；最低点允许规范视角。\n右图为最低Accel相对不调整的收益；不按结果删除任务。官方小面板每任务仅2个heldout初态。',
                 '官方heldout低值区域41.82%、最低点33.75%、不调整42.89%；规范seen低值区域41.52%、'
                 '最低点39.38%、不调整40.74%。但规范heldout最低点优于低值区域，多视角heldout区域也没有超过基线。'
                 '多视角heldout关闭冰箱-11.56pp、水槽到台面-16.76pp，合计对十任务平均贡献约-2.83pp，'
                 '整体-2.89pp。不能因此删去两任务，也不能仅凭均值判定训练损坏。'
                 '相关性覆盖96位置，扰动基线平均32额外位置；两集合分布不同，正相关不保证超过该基线。')
        specs = [('官方\nheldout20', self.summary['splits']['target']['official']),
                 ('规范\nseen200', self.summary['population_sensitivity']['splits']['pretrain']['canonical']),
                 ('规范\nheldout200', self.summary['population_sensitivity']['splits']['target']['canonical']),
                 ('多视角\nheldout200', self.summary['population_sensitivity']['splits']['target']['broad'])]
        ax = self.axes([.075, .38, .40, .31])
        for offset, key, label, color in [(-.23, 'stay', '扰动不调整', GRAY),
                                          (0, 'region', '低值区域平均', BLUE),
                                          (.23, 'selected', '最低点', TEAL)]:
            vals = [100*(r['quartiles'][0] if key == 'region' else r[key]) for _,r in specs]
            ax.bar(np.arange(4)+offset, vals, .22, color=color, label=label)
            for i,v in enumerate(vals):
                ax.text(i+offset, v+.22, f'{v:.1f}', ha='center', fontsize=8, rotation=90, va='bottom')
        ax.set_xticks(range(4), [x[0] for x in specs], fontsize=9); ax.set_ylim(0, 53)
        ax.set_ylabel('完整任务成功率（%）', fontsize=10)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.23), ncol=3, fontsize=8, frameon=False)
        pop = self.summary['population_sensitivity']['splits']['target']
        tasks = [r['task'] for r in pop['canonical']['per_task']]
        vals = np.array([[r['gain']['mean'] if isinstance(r['gain'],dict) else r['gain']
                          for r in pop[m]['per_task']] for m in MODELS[1:]]).T*100
        ax = self.fig.add_axes([.665, .335, .225, .40])
        im = ax.imshow(vals, cmap='RdBu', vmin=-20, vmax=20, aspect='auto')
        ax.set_yticks(range(10), [TASK_NAMES[t] for t in tasks], fontsize=8)
        ax.set_xticks([0,1], ['规范后训练', '多视角后训练'], fontsize=9)
        for i in range(10):
            for j in range(2):
                ax.text(j, i, f'{vals[i,j]:+.2f}', ha='center', va='center', fontsize=9,
                        color='white' if abs(vals[i,j])>12 else INK)
        self.fig.colorbar(im, ax=ax, fraction=.05, pad=.05).set_label('恢复收益（pp）', fontsize=9)
        self.text(.075, .27, '低值区域较好 ≠ 最低点最好；低值区域平均仍不是可部署方法成绩。', 11, INK)
        self.save(pdf)

    def conclusions(self, pdf):
        self.new('阶段结论：保留正结果，明确恢复方法的适用边界',
                 '研究进展 · 本次仅更新分析，不引入新方法',
                 '训练覆盖 → 视角价值 → 指标选择 → 实际相机获取；后三者不能由平均相关性直接推出',
                 '当前缺口是逐初态选位的可靠性，不是继续证明平均空间存在相关性。',
                 '所有新增页只使用已完成闭环及存档评分；未重训、未启动新评测、未修改原始结果。\n原PDF已逐字节归档；LIBERO及完整RoboCasa主体图表保留。主动移动与新决策规则均未验证。',
                 '完整heldout规范模型的+3.24pp不能用诊断+0.33pp替换；也不能只报告heldout而忽略seen-1.36pp。'
                 '官方40初态未显示最低Accel恢复，故多视角后训练不是失效的必要条件。'
                 '模型输入、初始化、训练实现、任务执行和相机结构同时变化，不能识别唯一跨环境根因。'
                 '后续经确认再做失败轨迹与评分稳定性定位；任何新方法应冻结后用独立初态/噪声验证。')
        blocks = [
            ('已有证据', 'LIBERO：多视角覆盖与扰动恢复的正结果保留。\nRoboCasa：规范后训练 heldout 恢复 +3.24 pp；\nseen 为 −1.36 pp，尚非跨场景稳定收益。'),
            ('本次定位', '官方模型也出现选位失效，不能全归因于多视角训练。\n平均空间关系不能替代具体初态的排序能力。\n最低点、低值区域和任务差异需要分别检验。'),
            ('仍待解释', '尚不能确定：评分噪声、任务执行瓶颈或训练机制\n分别贡献多少。4次执行不足以精确排列单初态视角。\n固定安装干预不是已完成的主动相机闭环。'),
        ]
        for y,(title,body) in zip([.735,.565,.395],blocks):
            self.text(.075,y,title,15,BLUE,True)
            self.text(.235,y,body,13,INK)
        self.save(pdf)


def build(public_dir, diagnostic_root, analysis_dir, output_root):
    public_dir, diagnostic_root, analysis_dir, output_root = map(Path, (public_dir, diagnostic_root, analysis_dir, output_root))
    summary = read(analysis_dir/'summary.json')
    if not summary['population_sensitivity']['matched_panel_verified']:
        raise ValueError('Matched panel identity has not been verified')
    data_sources = {str(diagnostic_root/'selection.json'): sha(diagnostic_root/'selection.json'),
                    str(diagnostic_root/'analysis/panel_arrays.npz'): sha(diagnostic_root/'analysis/panel_arrays.npz'),
                    str(analysis_dir/'summary.json'): sha(analysis_dir/'summary.json')}
    for path in list(data_sources)[:2]:
        if summary['sources'][path] != data_sources[path]:
            raise ValueError('Diagnostic input changed since analysis')
    for path,digest in summary['population_sensitivity']['sources'].items():
        if sha(path) != digest: raise ValueError('Population source changed')
        data_sources[path] = digest
    panel = load_panel(diagnostic_root)
    output = output_root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    previous = output/'previous-publication'; previous.mkdir()
    for name in ('vla_view_research_progress_zh.pdf','vla_view_research_progress_zh.md','publication.json'):
        shutil.copy2(public_dir/name, previous/name)
    before = {p.name: sha(p) for p in previous.iterdir()}
    brief = DiagnosticBrief(summary,panel)
    with PdfPages(output/'diagnostic-pages.pdf') as pdf:
        for method in ('official_space','scopes','failure_structure','conclusions'):
            getattr(brief, method)(pdf)
    write(output/'layout.json', brief.qa)
    if any(p['issues'] for p in brief.qa): raise ValueError('Layout overflow')
    old = fitz.open(previous/'vla_view_research_progress_zh.pdf')
    if len(old) != 16: raise ValueError('Expected the frozen 16-page report; refuse recursive append')
    new = fitz.open(output/'diagnostic-pages.pdf'); result = fitz.open()
    result.insert_pdf(old,from_page=0,to_page=13); result.insert_pdf(new)
    toc = [row for row in old.get_toc() if row[2] <= 14]
    toc += [[1,n['title'],15+i] for i,n in enumerate(brief.notes)]
    for i,page in enumerate(result):
        body = fitz.Rect(0,40,959.976,518)
        expected = Counter(c for c in page.get_text(clip=body) if not c.isspace())
        page.add_redact_annot(fitz.Rect(0,524,960,540),fill=(1,1,1))
        page.apply_redactions(images=0,graphics=0,text=0)
        page.insert_text((34,531),'VLA 研究进展 · 2026-09-29 · 完整矩阵与官方小诊断分开；实际主动获取待验证',fontname='china-s',fontsize=7,color=(.39,.45,.51))
        page.insert_text((901,531),f'{i+1:02d} / 18',fontsize=8,color=(.39,.45,.51))
        if expected-Counter(c for c in page.get_text(clip=body) if not c.isspace()):
            raise ValueError('Source body text lost')
    result.set_toc(toc); result.set_metadata(dict(title='VLA研究进展｜LIBERO与RoboCasa：视角恢复及指标边界',author='VLA View Research',subject='2026-09-29；18页；原完整结果与官方40初态诊断分开'))
    target=output/'vla_view_research_progress_zh.pdf';result.save(target,garbage=4,deflate=True)
    result.close();old.close();new.close()
    preservation=[]
    with fitz.open(previous/'vla_view_research_progress_zh.pdf') as old, fitz.open(target) as current:
        for i in range(14):
            clip=fitz.Rect(0,40,959.976,518)
            identical=old[i].get_pixmap(clip=clip).samples==current[i].get_pixmap(clip=clip).samples
            preservation.append(dict(page=i+1,body_pixels_identical=identical))
            if not identical: raise ValueError('Retained page body pixels changed')
    write(output/'preservation.json',preservation)
    previews=output/'previews';previews.mkdir()
    contact=Image.new('RGB',(1600,4*470),'#E8EDF3');draw=ImageDraw.Draw(contact)
    with fitz.open(target) as doc:
        if len(doc)!=18: raise ValueError('Unexpected report length')
        for i in range(14,18):
            pix=doc[i].get_pixmap(matrix=fitz.Matrix(1.7,1.7),alpha=False)
            image=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
            image.save(previews/f'page-{i+1:02d}.png')
            image.thumbnail((1560,440));contact.paste(image,((1600-image.width)//2,(i-14)*470+25))
            draw.text((10,(i-14)*470+5),f'Page {i+1}',fill='black')
    contact.save(output/'updated-pages-contact.png')
    notes=['# VLA研究进展｜LIBERO与RoboCasa统一报告（2026-09-29）','',
           '本版18页。保留旧版第1–14页正文与所有LIBERO图表；旧第15–16页由四页最新诊断与结论替换。',
           '没有启动新训练、推理或闭环实验。新增官方诊断本身已完成14,720次闭环，复用800次；两后训练模型复用原结果。','',
           '## 阅读顺序','', '1–10页：LIBERO已有证据；11–14页：RoboCasa完整400初态结果。',
           '15页：官方与两后训练模型共同40初态三维图；16页：seen/heldout及总体/子集口径；',
           '17页：最低点、低值区域和任务差异；18页：当前问题与证据边界。','',
           '## 完整结果与诊断结果（全查询最低Accel，不是16查询搜索）','',
           '| 范围 | 场景 | 模型 | 不调整% | 选位后% | 收益pp | 空间相关 | 初态内排名相关 |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for population in (True,False):
        for split,m,r in evidence_rows(summary,population):
            corr=r['association'];notes.append(f"| {'完整200/组' if population else '诊断20/组'} | {'seen' if split=='pretrain' else 'heldout'} | {NAMES[MODELS.index(m)]} | {r['stay']*100:.2f} | {r['selected']*100:.2f} | {r['gain']['mean']*100:+.2f} | {corr['spatial_mean_spearman']:.3f} | {corr['pooled_within_initial_rank_correlation']:.3f} |")
    notes += ['', '空间相关对应平均三维图；初态内相关为各初态内两变量排名并中心化后合并，非中位数。',
              '规范完整heldout +3.24pp的任务/初态bootstrap 95%区间为[-1.06,7.72]pp；固定已观测执行均值，不等于跨噪声、跨训练seed充分确认。',
              '原生渲染微差保留，不能由统计分析宣称逐像素或闭环逐位确定。','', '## 新增页讲解','']
    for i,n in enumerate(brief.notes,15):
        notes += [f"### 第{i}页：{n['title']}",'',n['takeaway'],'',n['speaker_note'],'']
    notes += ['## 来源与归档','',f'- [完整原说明与旧版PDF]({previous})：训练配置、原选择器、执行噪声、旧图及逐页解释均保留。',
              f'- [本轮统计]({analysis_dir/"summary.json"})',f'- [出版来源与哈希]({output/"provenance.json"})',
              '未将尚未验证的新决策规则、TopK诊断或保持原位机制写成方法成果。',
              '新增页以嵌入字形保障中文显示；当前绘图环境的Type3中文文字提取有限，完整中文讲解见本文。']
    (output/'vla_view_research_progress_zh.md').write_text('\n'.join(notes)+'\n')
    artifacts={name:sha(output/name) for name in ('vla_view_research_progress_zh.pdf','vla_view_research_progress_zh.md')}
    write(output/'provenance.json',dict(before=before,sources=data_sources,builder={str(Path(__file__).resolve()):sha(__file__)},
          artifacts=artifacts,retained_body_pages=list(range(1,15)),new_pages=list(range(15,19)),new_experiments=0,layout_pass=True))
    return output


def publish(output, public_dir):
    output,public_dir=Path(output),Path(public_dir);receipt=read(output/'provenance.json')
    for name,digest in receipt['before'].items():
        if sha(public_dir/name)!=digest: raise ValueError('Current publication changed; refuse overwrite')
    for name,digest in receipt['artifacts'].items():
        if sha(output/name)!=digest: raise ValueError('Built artifact changed')
    previous_record=read(output/'previous-publication/publication.json')
    record=dict(scope=receipt.get('scope', 'Unified LIBERO/RoboCasa; original matrix + official matched diagnostic'),
                original_pdf=previous_record['original_pdf'],
                current=dict(archive=str(output),pdf_sha256=receipt['artifacts']['vla_view_research_progress_zh.pdf'],
                             notes_sha256=receipt['artifacts']['vla_view_research_progress_zh.md'],
                             pages=receipt.get('pages', 18), new_closed_loops=0,
                             added_closed_loop_results=receipt.get('added_closed_loop_results', 0)),
                previous_publication=str(output/'previous-publication'),provenance=str(output/'provenance.json'))
    for name in receipt['artifacts']:
        temp=public_dir/(name+'.tmp');shutil.copyfile(output/name,temp);os.replace(temp,public_dir/name)
    write(public_dir/'publication.json',record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-dir',type=Path,required=True)
    parser.add_argument('--diagnostic-root',type=Path)
    parser.add_argument('--analysis-dir',type=Path)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--publish-built',type=Path)
    args=parser.parse_args()
    if args.publish_built:
        publish(args.publish_built,args.public_dir);print(args.public_dir)
    else:
        if not all((args.diagnostic_root,args.analysis_dir,args.output_root)):parser.error('Build requires diagnostic, analysis and output paths')
        print(build(args.public_dir,args.diagnostic_root,args.analysis_dir,args.output_root))


if __name__=='__main__':main()
