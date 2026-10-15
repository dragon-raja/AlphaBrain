"""Replace the old outlook with two audited confirmation pages in the one PDF."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import shutil

import numpy as np
from matplotlib.colors import TwoSlopeNorm

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from .robocasa_diagnostic_update import DiagnosticBrief, publish, fitz, PdfPages
from .robocasa_results import BLUE, TEAL, GRAY, AMBER, INK, MUTED, TASK_NAMES

LABELS={'official':'π0.5 官方','xr1':'XR-1 官方'}
GROUPS={'capable_view_sensitive':'高能力／视角敏感',
        'capable_low_view_loss':'高能力／低视角损失','lower_reference_capacity':'较低规范能力'}


def number(x, digits=2, signed=False):
    value=Decimal(str(round(float(x),10))).quantize(Decimal(10)**-digits,rounding=ROUND_HALF_UP)
    return format(value,('+' if signed else '')+f'.{digits}f')


def pp(x): return number(100*x,signed=True)


class ConfirmationBrief(DiagnosticBrief):
    def comparison(self,pdf):
        self.new('独立确认：窗口3没有跨模型一致恢复',
            '新增证据 · 2026-10-12',
            '每模型80个新初态；97候选×8评分噪声；选位冻结后，8套独立执行噪声完整评测',
            '预定窗口3未显示一致恢复；不把事后较好的窗口替换为主结论。',
            '先在候选试执行，再复位至原初态执行完整任务；非执行途中换相机。窗口1严格退化为初态评分。\n扰动不调整＝原32个额外视角均值；两模型原生重规划步距不同。曲线为80初态总体，表格分开报告两组场景。',
            '新初态属于相同10个已知任务与既有官方场景划分，不是新任务泛化。规范安装与每个窗口均为每模型640次完整任务；扰动基线是80×32×8=20480次。相同选位共用执行结果，总计44376条去重完整闭环。评分分支和执行分支分离，不使用成败标签选位。主窗口3预先冻结，其他窗口为描述性次要对照。')
        for col,model in enumerate(LABELS):
            r=self.summary['models'][model]['populations']['all']['methods']
            ax=self.axes([.075+col*.48,.475,.385,.285])
            vals=np.array([r[f'window{k}']['success']['mean']*100 for k in range(1,6)])
            baseline=r['disturbance_uniform']['success']['mean']*100;canonical=r['canonical']['success']['mean']*100
            ax.plot(range(1,6),vals,'o-',color=BLUE,lw=2,label='窗口选位')
            ax.scatter([3],[vals[2]],s=72,color=TEAL,zorder=5)
            ax.axhline(baseline,color=AMBER,ls='--',label=f'扰动不调整 {number(baseline)}%')
            ax.axhline(canonical,color=GRAY,ls=':',label=f'规范视角 {number(canonical)}%')
            for k,v in enumerate(vals,1):ax.annotate(number(v),(k,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
            low=min(*vals,baseline,canonical);high=max(*vals,baseline,canonical)
            ax.set_ylim(max(0,low-8),min(100,high+12));ax.set_xlim(.7,5.3);ax.set_xticks(range(1,6))
            ax.set_title(LABELS[model],fontsize=13);ax.set_ylabel('完整任务成功率（%）',fontsize=10)
            ax.tick_params(labelsize=10);ax.legend(fontsize=8,loc='best')
            gain=r['window3']['gain_vs_stay'];lo,hi=gain['ci95']
            self.text(.075+col*.48,.432,f'窗口3恢复 {pp(gain["mean"])} pp；95% CI [{pp(lo)}, {pp(hi)}]',10,INK,True)
        ax=self.fig.add_axes([.075,.26,.87,.125]);ax.axis('off')
        table_rows=[]
        for model in LABELS:
            for split,label in [('seen','seen'),('heldout','heldout')]:
                r=self.summary['models'][model]['populations'][split]['methods']
                table_rows.append([LABELS[model]+' · '+label,*[number(100*r[k]['success']['mean']) for k in
                    ['canonical','disturbance_uniform','window1','window2','window3','window4','window5']]])
        table=ax.table(cellText=table_rows,colLabels=['场景组（各40初态）','规范%','不调整%','窗口1%','窗口2%','窗口3%','窗口4%','窗口5%'],
            colWidths=[.25]+[.107]*7,cellLoc='center',loc='center')
        table.auto_set_font_size(False);table.set_fontsize(8.6);table.scale(1,1.35)
        for (row,_),cell in table.get_celld().items():
            cell.set_edgecolor('#DCE3EB')
            if row==0:cell.set_facecolor('#EDF3FA')
        self.save(pdf)

    def heterogeneity(self,pdf):
        self.new('整体恢复与任务差异分开判断，不能仅选有利任务',
            '独立确认 · 任务结构与研究边界',
            '左：相对扰动不调整的收益；右：只用旧40初态定义的分组，在新80初态中检查',
            '局部增益可以保留，但尚不能推出多步 Accel 是稳定的恢复规则。',
            '收益单位为百分点；所有10任务保留。分组门槛来自开发数据：规范成功率≥80%、视角损失≥10 pp。\n分组与窗口1/5属描述性分析；区间包含初态与配对执行噪声，另核验共享布局的敏感性。训练seed不在区间内。',
            '即使多步窗口有局部改善，也不能说明窗口越长越好、Accel是普适成功概率或模型足够好就是充分条件。97候选短分支有试执行与复位成本，真实机器人不能直接免费复位；主动获取方法仍须另作设计。未评测未选候选的独立完整闭环，所以不能给出新的97视角Oracle。')
        tasks=[r['task'] for r in self.summary['models']['official']['populations']['all']['per_task']]
        mat=[]
        for task in tasks:
            row=[]
            for model in LABELS:
                r=next(x for x in self.summary['models'][model]['populations']['all']['per_task'] if x['task']==task)['methods']
                row.extend([100*(r[f'window{k}']-r['disturbance_uniform']) for k in (1,3,5)])
            mat.append(row)
        mat=np.array(mat);limit=max(5,float(np.abs(mat).max()))
        ax=self.fig.add_axes([.185,.30,.39,.435]);im=ax.imshow(mat,cmap='RdBu',norm=TwoSlopeNorm(0,-limit,limit),aspect='auto')
        ax.set_xticks(range(6),['π·1','π·3','π·5','XR·1','XR·3','XR·5'],fontsize=9)
        ax.set_yticks(range(10),[TASK_NAMES.get(t,t) for t in tasks],fontsize=8.5);ax.tick_params(length=0)
        for i in range(10):
            for j in range(6):ax.text(j,i,number(mat[i,j],1,True),ha='center',va='center',fontsize=8,color='white' if abs(mat[i,j])>.62*limit else INK)
        ax.axvline(2.5,color='white',lw=2)
        cb=self.fig.colorbar(im,cax=self.fig.add_axes([.585,.30,.011,.435]));cb.ax.tick_params(labelsize=8);cb.set_label('收益（pp）',fontsize=9)
        y=.75
        for model in LABELS:
            self.text(.66,y,LABELS[model]+' · 冻结分组窗口3',12,BLUE,True);y-=.048
            for group,g in self.summary['models'][model]['frozen_strata'].items():
                v=g['methods']['window3']['gain_vs_stay']['mean']
                self.text(.66,y,f'{GROUPS[group]}（n={g["initials"]}）',9.5,INK)
                canonical=g['methods']['canonical']['success']['mean']
                self.text(.66,y-.023,f'新规范 {number(100*canonical)}%；恢复 {pp(v)} pp',9.5,INK);y-=.061
            y-=.009
        gaps=[self.summary['models'][m]['populations']['all']['methods']['canonical']['gain_vs_stay']['mean'] for m in LABELS]
        self.text(.075,.252,f'规范安装相对不调整：π0.5 {pp(gaps[0])} pp；XR-1 {pp(gaps[1])} pp。规则失败不等于没有视角收益。',10,INK,True)
        self.save(pdf)


def build(public_dir,analysis,output_root):
    public_dir,analysis,output_root=map(Path,(public_dir,analysis,output_root))
    summary=read(analysis/'summary.json');audit=read(analysis/'audit.json');metric=read(analysis/'metric-audit.json')
    if audit['status']!='PASS' or metric['status']!='PASS':raise ValueError('Final audit required')
    for path,digest in summary['sources'].items():
        if sha(path)!=digest:raise ValueError('Summary evidence changed')
    out=output_root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    previous=out/'previous-publication';previous.mkdir()
    for name in ('vla_view_research_progress_zh.pdf','vla_view_research_progress_zh.md','publication.json'):
        shutil.copy2(public_dir/name,previous/name)
    before={p.name:sha(p) for p in previous.iterdir()}
    brief=ConfirmationBrief(summary,{})
    with PdfPages(out/'confirmation-pages.pdf') as pdf:
        brief.comparison(pdf);brief.heterogeneity(pdf)
    if any(q['issues'] for q in brief.qa):raise ValueError('Layout overflow')
    with fitz.open(previous/'vla_view_research_progress_zh.pdf') as old,fitz.open(out/'confirmation-pages.pdf') as new:
        if len(old)!=19:raise ValueError('Expected the current 19-page report')
        result=fitz.open();result.insert_pdf(old,from_page=0,to_page=17);result.insert_pdf(new)
        for i,page in enumerate(result):
            page.add_redact_annot(fitz.Rect(0,524,960,540),fill=(1,1,1));page.apply_redactions(images=0,graphics=0,text=0)
            label='原实验与开发证据（保留原口径）' if i<18 else '新初态／独立执行噪声确认'
            page.insert_text((34,531),'VLA 研究进展 · 2026-10-12 · '+label,fontname='china-s',fontsize=7,color=(.39,.45,.51))
            page.insert_text((901,531),f'{i+1:02d} / 20',fontsize=8,color=(.39,.45,.51))
        result.set_toc([r for r in old.get_toc() if r[2]<=18]+[[1,n['title'],19+i] for i,n in enumerate(brief.notes)])
        result.set_metadata(dict(title='VLA研究进展｜训练覆盖、视角恢复与独立确认',author='VLA View Research'))
        target=out/'vla_view_research_progress_zh.pdf';result.save(target,garbage=4,deflate=True);result.close()
    preservation=[];previews=out/'previews';previews.mkdir()
    with fitz.open(previous/'vla_view_research_progress_zh.pdf') as old,fitz.open(target) as new:
        for i in range(18):
            clip=fitz.Rect(0,40,959.976,518)
            same=old[i].get_pixmap(clip=clip).samples==new[i].get_pixmap(clip=clip).samples
            if not same:raise ValueError('Historical page body changed')
            preservation.append(dict(page=i+1,body_pixels_identical=True))
        for i in (18,19):new[i].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(previews/f'page-{i+1:02d}.png')
    write(out/'preservation.json',preservation);write(out/'layout-qa.json',brief.qa)
    notes=(previous/'vla_view_research_progress_zh.md').read_text()
    notes='# 当前版本说明：2026-10-12\n\n本版20页，保留旧版1–18页正文与图表（历史／开发证据），用新19–20页替换原末页展望。下方旧版逐页说明保留作来源，新结果见本文末节。\n\n'+notes
    notes+='\n\n## 2026-10-12 独立恢复确认：第19–20页\n\n'
    notes+='两模型各80个新初态、10个已知任务，各场景组40初态；8套评分噪声与8套执行噪声独立。每初态97候选短探测，冻结窗口1–5选位后，从同一初态重新完整执行。主窗口3事先固定。不是在线换相机，不是新任务泛化，不计算新的97视角Oracle。\n\n'
    notes+='| 模型／组 | 规范% | 不调整% | W1% | W2% | W3% | W4% | W5% | W3恢复pp（95% CI） |\n|---|---:|---:|---:|---:|---:|---:|---:|---|\n'
    for model in LABELS:
        for split,p in summary['models'][model]['populations'].items():
            r=p['methods'];v=[r[k]['success']['mean']*100 for k in ['canonical','disturbance_uniform','window1','window2','window3','window4','window5']]
            g=r['window3']['gain_vs_stay'];lo,hi=g['ci95']
            notes+='| '+LABELS[model]+'/'+split+' | '+' | '.join(number(x) for x in v)+f' | {pp(g["mean"])} [{pp(lo)}, {pp(hi)}] |\n'
    notes+='\n### 统计与限制\n\n'+summary['uncertainty']+'\n\n'
    for model in LABELS:
        r=summary['models'][model]['populations']['all']['methods']['window3']
        notes+=f'\n{LABELS[model]}：主窗口3相对扰动不调整的两模型族校正97.5%区间为{r["gain_vs_stay"]["family2_ci975"]}；共享布局敏感性95%区间为{r["gain_vs_stay"]["shared_layout_ci95"]}（这里数组单位为成功率比例）。相对窗口1增益为{pp(r["gain_vs_window1"]["mean"])} pp，95%区间{r["gain_vs_window1"]["ci95"]}。\n'
    for n in brief.notes:notes+=f'\n### {n["title"]}\n\n{n["speaker_note"]}\n'
    notes+='\n冻结分组不是用本轮结果重分：π0.5的“开发高能力／视角敏感”组仅4个新初态，本轮规范成功率已降至53.13%，不能继续当作可靠高能力组解释；XR-1同类组有32个初态，新规范87.11%、扰动72.39%、窗口3为70.70%，说明即便这一组保持较高规范能力和视角损失，当前规则也未自动恢复（描述性证据，不宣称充分条件定理被普遍证伪）。\n'
    notes+=f'\n全量验收与统计：{analysis}\n旧版完整归档：{previous}\n'
    (out/'vla_view_research_progress_zh.md').write_text(notes)
    write(out/'provenance.json',dict(before=before,
        artifacts={name:sha(out/name) for name in ('vla_view_research_progress_zh.pdf','vla_view_research_progress_zh.md')},
        sources={str(analysis/name):sha(analysis/name) for name in ('summary.json','audit.json','metric-audit.json')},
        pages=20,added_closed_loop_results=44376,new_experiments=0,scope='Unified LIBERO/RoboCasa with fresh-initial two-model five-window confirmation',
        builder={str(Path(__file__).resolve()):sha(__file__)},retained_body_pages=list(range(1,19)),new_pages=[19,20],layout_pass=True))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--public-dir',required=True)
    p.add_argument('--analysis');p.add_argument('--output-root');p.add_argument('--publish-built')
    a=p.parse_args()
    if a.publish_built:publish(a.publish_built,a.public_dir)
    else:print(build(a.public_dir,a.analysis,a.output_root))
