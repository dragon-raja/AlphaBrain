#!/usr/bin/env python3
"""Twelve-page research narrative, combining legacy evidence and current analysis.

Only presentation outputs are changed. Every input PDF and replaced public PDF
is archived with hashes; the two former public filenames become identical copies
of one report, so existing user bookmarks do not show conflicting versions.
"""
from datetime import datetime, timezone
import json
import os
import fcntl
from pathlib import Path
import shutil
import sys

try:
    import pymupdf as fitz
except ImportError:
    # Existing local dependency; no environment installation or network access.
    sys.path.insert(0,'/root/.cache/uv/archive-v0/XO8_y4zBEt33DOY7wfN64')
    import pymupdf as fitz
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw
from build_view_research_focus_brief import Brief, INK, MUTED, BLUE, TEAL, GRAY, AMBER
from build_view_landscape_report_v1 import read_json, write_json, sha256, require
from analyze_view_metric_rules_v1 import RULES
from build_view_metric_comparison_brief_v1 import RULE_LABELS

REPO=Path(__file__).resolve().parents[2]
DOCS=REPO/'docs/dsol_paper1'
ROOT=Path('/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908')
BRIDGE=ROOT/'render-protocol-bridge-v1'
PUBLIC=DOCS/'reports/current/vla_view_research_progress_zh.pdf'
LEGACY_PUBLIC=DOCS/'vla_view_research_focus_20260907_zh.pdf'
COMPAT=DOCS/'vla_view_landscape_and_metrics_20260908_zh.pdf'
OUTPUT=ROOT/'unified-research-progress-v1'

SCOPE_CONTRACT={
    'primary_start':'standard task initial condition, not a restored intermediate demonstration frame',
    'outcome':'complete task success or failure at the common execution horizon',
    'oracle_order':'average successes over noise first; maximize over views second',
    'selection_unit':'one view per initial condition; may differ across initial conditions',
    'external_view_fixed_within_rollout':True,
    'per_noise_hindsight_selection':False,
    'dynamic_view_sequence':False,
    'historical_snapshot_results':'auxiliary continuation evidence, not primary initial-condition results',
    'independent_retest':'stability of frozen selections, not the definition of empirical Oracle',
    'execution_changes_authorized_by_report_build':False,
}

LANDSCAPE_HEADERS={
    0:('辅助证据：恢复快照下的候选空间与指标分布',
       '历史 Broad 模型；64 恢复快照 × 97 候选 × 32 次续跑。含中间状态，不是从标准任务初态开始的主评测。'),
    1:('辅助证据：恢复快照—候选视角矩阵',
       '每行是独立恢复的快照，不是同一次任务的连续决策时刻；成功率指从该快照续跑至任务终点。'),
    5:('辅助证据：评分初噪改变候选排序',
       '历史恢复快照诊断；评分噪声、排序稳定性与续跑收益分别评价，不直接外推到标准任务初态。'),
}


def validate_scope_texts(texts):
    """Prevent a future automatic report refresh from restoring old claim scope."""
    require(len(texts)==12,'Expected twelve report pages')
    require('每个初态先对噪声求平均' in texts[0],'Noise-averaged Oracle definition missing')
    require('从任务初态到最终成败' in texts[1] and '独立复评' in texts[1], 'Primary protocol missing')
    for i in range(4,11):
        require('辅助证据' in texts[i] and ('恢复' in texts[i] or '快照' in texts[i]),f'Historical page {i+1} not scoped')
    require('75.88%' in texts[7] and '不是标准任务初态' in texts[7], 'Snapshot Oracle mislabeled')
    require('主实验尚缺' in texts[11], 'Primary result presented as complete')
    combined='\n'.join(texts)
    require('62.99' not in combined and 'O/P/Q' not in combined,'Old protocol comparison returned to main report')


class ResearchBrief(Brief):
    def text(self,x,y,s,*args,**kwargs):
        s={'主动观测收益':'后续主动观测','能否补偿获取成本？':'后续：能否补偿获取成本？'}.get(s,s)
        return super().text(x,y,s,*args,**kwargs)

    def new(self,title,stage,subtitle,takeaway,source,note):
        if title=='外部多视角后训练改善被动视角鲁棒性':
            title='历史开发对照：视角覆盖改善被动鲁棒性'
            takeaway='开发设置内覆盖收益明显；配对与一致性的额外收益尚未确认。'
        elif title=='宽视角后训练的鲁棒性与原始任务性能权衡':
            title='历史整体基准：相机鲁棒性与原始性能存在权衡'
            source=source.replace('\nCamera任务等权增益','\n非逐初态候选 Oracle；Camera任务等权增益')
        elif title=='信息视角呈现局部收益，机制解释仍需对照':
            title='辅助证据：恢复状态中的可见性视角对照'
            takeaway='恢复状态中的局部收益用于提出假设，不替代任务初态的完整执行验证。'
        super().new(title,stage,subtitle,takeaway,source,note)

    def page1(self,pdf):
        self.new('VLA 视角利用：训练覆盖与整任务候选收益','研究问题与主线定义',
                 '主问题：从任务初态完整执行时，多视角训练后还剩多少可利用的视角收益，指标能否识别它？',
                 '每个初态先对噪声求平均，再选最佳视角；不同初态可选不同视角。',
                 '主评测采用标准任务初态；现有恢复快照矩阵仅作辅助证据。主动相机运动、腕部耦合与真机验证不属于当前已证实范围。',
                 '候选视角在一次任务执行中固定。经验 Oracle 按初态分别取多噪声平均最优候选，不逐噪声挑赢家，不选择动态视角序列。当前没有完成与此定义对应的两训练条件完整初态候选矩阵。')
        for x,title,body in [(.065,'训练覆盖','规范位姿后训练\n多视角后训练\n匹配其他训练条件'),
                             (.38,'候选收益','同一任务初态\n各候选分别完整执行\n多噪声平均后比较'),
                             (.695,'视角利用','规范与任务固定基线\n逐初态经验 Oracle\n指标／选择器的收益')]:
            self.box(x,.475,.25,.265);self.text(x+.018,.708,title,20,BLUE,True);self.text(x+.018,.64,body,13,MUTED)
        self.arrow((.323,.60),(.364,.60));self.arrow((.638,.60),(.68,.60))
        self.text(.084,.39,'初态 1 → 最佳视角 A',18,TEAL,True)
        self.text(.52,.39,'初态 2 → 最佳视角 B',18,BLUE,True)
        self.text(.085,.315,'允许跨初态改变选择；同一初态不根据某一次噪声的成败临时更换赢家。',14,INK)
        self.save(pdf)

    def primary_protocol_page(self,pdf):
        self.new('主评测定义：从任务初态到最终成败','主线实验设计 · 尚未完成本版主矩阵',
                 '固定模型、任务初态、候选域与执行预算；对每个候选使用配对的完整推理噪声序列。',
                 '经验 Oracle 是候选矩阵的上限参照；独立复评另行检验选中视角的收益稳定性。',
                 '候选包含规范视角。初态数量与噪声预算在后续评测规划中冻结；本轮报告修订不启动、停止或修改任何评测。',
                 '每次都从相同标准任务初态重置，固定一个候选相机位姿，运行至最终目标成功或统一超时。先平均噪声，再取视角最大；独立复评锁定赢家，不根据新成绩重选。该静态视角 Oracle 不要求在线移动相机。')
        steps=[('任务初态','标准环境重置'),('候选视角','单次执行内固定'),('完整闭环','成功或统一超时'),('噪声平均','每视角多次重复'),('初态 Oracle','取平均最高视角')]
        for i,(title,body) in enumerate(steps):
            x=.055+i*.185;self.box(x,.565,.153,.165);self.text(x+.012,.705,title,16,BLUE,True);self.text(x+.012,.643,body,10.8,MUTED)
            if i<4:self.arrow((x+.158,.635),(x+.18,.635))
        self.text(.075,.493,'经验上限比较（同一矩阵）',17,INK,True)
        self.text(.078,.435,'规范视角：所有初态使用规范位姿\n每任务最佳固定：取该任务跨初态平均最高的候选\n逐初态 Oracle：每个初态分别选择噪声平均最佳候选',13,INK)
        self.text(.63,.493,'后续收益验证',17,BLUE,True)
        self.text(.63,.435,'冻结选定视角\n换一批噪声，重新完整执行\n检验收益是否保持',13,MUTED)
        self.text(.078,.275,'不允许：逐噪声事后挑成功视角；用中间状态续跑冒充完整初态任务；将经验 Oracle 当成已部署算法。',11.5,MUTED)
        self.save(pdf)

    def selector_page(self,pdf,q):
        self.new('辅助证据：恢复快照上的两种选择器','辅助研究 · 不替代初态主评测',
                 '历史 Broad 模型；48 个恢复快照开发、16 个快照评价；每个评价起点使用 64 套独立噪声续跑。',
                 '历史选择器未显示可靠续跑收益，不能据此判断初态视角价值不可预测。',
                 '来源：v2 final-analysis；旧 AA4 协议。每起点选一次，外部位姿保持到结束；没有选择动态视角序列。',
                 'Q 对已经冻结的规则有独立评价意义；零附近的增益不能证明所有方法不可学习。当前历史来源已被分析，后续不能重新宣称是未接触的留出集。')
        keys=['canonical','geometry_context_ridge','image_queryable_ridge']
        ax=self.axes([.09,.38,.48,.33]); values=[100*q['population'][k]['success'] for k in keys]
        ax.bar(range(3),values,color=[GRAY,AMBER,BLUE]); ax.set_ylim(0,100)
        ax.set_xticks(range(3),['规范','几何／上下文','候选图可查询'],fontsize=11)
        ax.set_ylabel('恢复后续跑成功率（%）',fontsize=13)
        for i,v in enumerate(values):ax.text(i,v+2,f'{v:.2f}',ha='center',fontsize=17,fontweight='bold')
        self.text(.62,.73,'选择时能使用什么？',18,INK,True)
        self.text(.62,.65,'几何／上下文：协议允许的状态特征\n候选图版本：额外获得候选图像\n两者均不使用测试闭环成败来选视角',13,MUTED)
        self.text(.62,.46,'独立噪声评价：增益及 95% 区间',14,INK,True)
        for y,key,label in ((.405,'geometry_context_ridge','几何／上下文'),(.345,'image_queryable_ridge','候选图可查询')):
            c=q['comparisons'][key+'_minus_canonical']
            self.text(.62,y,f"{label}：{100*c['estimate']:+.2f} pp  [{100*c['ci95'][0]:+.2f}, {100*c['ci95'][1]:+.2f}]",12,MUTED)
        self.text(.092,.282,'迁移到主实验时须使用任务初态；模型、候选图访问权限与基线须一致。',13,INK)
        self.save(pdf)

    def headroom_page(self,pdf,h):
        self.new('辅助证据：恢复快照下存在经验视角收益','辅助研究 · 不替代初态主评测',
                 '历史 Broad 模型；64 恢复快照 × 97 候选 × 32 次续跑。包含中间状态，各柱使用同一批快照与噪声。',
                 '75.88% 是恢复快照下的经验最佳续跑率，不是标准任务初态的 Oracle 成绩。',
                 '先对每视角的 32 次噪声结果求平均，再按快照取最大；不是随机选视角，也不是逐噪声挑赢家。',
                 '此图用于回答完整有限候选矩阵中的经验收益结构，不把经验最大值当作真实期望上限。正式选择效果应在冻结选择后用独立噪声闭环评价。旧 O/P/Q 结果及其口径比较保留在技术归档，不再作为主报告叙事。')
        keys=['canonical','global_fixed','task_fixed','statewise']
        values=[100*h[k] for k in keys]
        ax=self.axes([.085,.34,.52,.37]);ax.bar(range(4),values,color=[GRAY,'#7594ae',BLUE,TEAL]);ax.set_ylim(0,100)
        ax.set_xticks(range(4),['规范视角','全局最佳固定','每任务最佳固定','每快照经验最佳'],fontsize=11)
        ax.set_ylabel('恢复后续跑成功率（%）',fontsize=13)
        for i,v in enumerate(values):ax.text(i,v+2,f'{v:.2f}',ha='center',fontsize=19,fontweight='bold')
        self.text(.66,.73,'相对规范视角',16,INK,True)
        self.text(.66,.67,f"+{100*(h['statewise']-h['canonical']):.2f} pp",27,TEAL,True)
        self.text(.66,.56,'相对每任务最佳固定',16,INK,True)
        self.text(.66,.50,f"+{100*(h['statewise']-h['task_fixed']):.2f} pp",25,BLUE,True)
        self.text(.66,.385,'64.70% 同样来自这 64 个快照\n不是另一个规范训练模型的成绩',11.5,MUTED)
        self.text(.087,.265,'这些数据用于提出视角收益假设；初态主实验尚待完成，不能直接沿用此处的绝对成功率。',12,INK)
        self.save(pdf)

    def metric_page(self,pdf,s,contrasts):
        self.new('辅助证据：指标筛选与恢复后续跑收益','辅助研究 · 不替代初态主评测',
                 '历史 Broad 模型；同 64 恢复快照、97 候选、32 次续跑。七条规则固定，不按成功标签为每个快照挑规则。',
                 '低 Accel 有部分初筛信号；是否提高完整初态任务成功率，尚未由本实验验证。',
                 '均匀候选与 Top10 均匀规则使用候选结果的期望平均；线段为来源 bootstrap 95% 区间。指标可访问候选图。',
                 '把候选空间经验 Oracle 与规则成功率分开成页。这里不再混入噪声留半搜索或旧 O/P/Q；这些是技术诊断，不是提出的指标方法。')
        for col,(field,scale,label) in enumerate([('success',100,'恢复后续跑成功率（%）'),('gain_vs_canonical_pp',1,'相对规范视角（百分点）')]):
            ax=self.axes([.305+col*.355,.32,.285,.405],True,False)
            for i,rule in enumerate(RULES):
                item=s['overall']['rules'][rule][field];v=scale*item['mean'];lo,hi=[scale*x for x in item['ci95']]
                ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=BLUE,markersize=6,capsize=2,lw=1.5)
                if col==0:ax.text(v+1,i-.13,f'{v:.2f}',fontsize=9)
            ax.set_yticks(range(7),[RULE_LABELS[r] for r in RULES] if col==0 else [],fontsize=10)
            ax.set_ylim(6.7,-.7);ax.set_xlabel(label,fontsize=12)
            if col:ax.axvline(0,color=GRAY,lw=1)
            else:ax.set_xlim(0,100)
        g=contrasts['contrasts']['visibility_refinement_within_accel_top10']['gain_pp']
        self.text(.071,.248,f"Top10 内加入可见性二次筛选：净增益 {g['mean']:+.2f} pp，95% 区间 [{g['ci95'][0]:+.2f}, {g['ci95'][1]:+.2f}]；额外收益未确认。",11.5,MUTED)
        self.save(pdf)

    def audit_page(self,pdf,a):
        passed=a['noaa_repeatability_pass']
        self.new('一致性复核：区分同协议重复性与更换渲染器的影响','04 · 评测可信度',
                 '2 模型 × 4 执行条件 × 192 闭环 = 1,536 次；每条件为 8 状态 × 3 视角 × 8 噪声。',
                 '确定性修复不等于历史数值已获认证；新旧协议结果不能直接混合。',
                 'AA4：原多重采样抗锯齿；AA0：关闭该抗锯齿。逐次检查输入、动作、物理轨迹和成败；静态指标另覆盖 97 候选。',
                 'GPU 评测已完成。末尾 JSON 写出失败已通过仅转换 NumPy 标量恢复；冻结代码、原始账本及失败记录保留。统计区间使用 8 个来源簇。')
        rows=[['模型／渲染','完整轨迹不一致','成败翻转']]
        for model,label in [('canonical','规范训练'),('broad','多视角训练')]:
            for aa in ['aa4','aa0']:
                r=a['repeatability'][model+'/'+aa]
                rows.append([label+' / '+aa.upper(),f"{len(r['full_signature_mismatches'])} / 192",f"{len(r['success_flip_indices'])} / 192"])
        ax=self.fig.add_axes([.055,.43,.47,.28]); ax.axis('off')
        table=ax.table(cellText=rows,cellLoc='center',loc='center',colWidths=[.48,.28,.24]);table.auto_set_font_size(False);table.set_fontsize(12);table.scale(1,2.0)
        for (r,c),cell in table.get_celld().items():
            cell.set_edgecolor('#DFE6EE');cell.set_facecolor('#EFF4FA' if r==0 else 'white')
        self.text(.061,.36,f"AA0 完整重复性：{'通过' if passed else '未通过'}",18,TEAL if passed else AMBER,True)
        self.text(.061,.30,'AA0 对比 32／48 并发；AA4 为两次 32 并发。',11,MUTED)
        ax=self.axes([.725,.405,.235,.30],True,False)
        effects=[('canonical_renderer_change','规范模型变化'),('broad_renderer_change','多视角模型变化'),('training_renderer_interaction','两模型差距变化')]
        for y,(key,label) in enumerate(effects):
            r=a['effects'][key];v=r['mean_pp'];lo,hi=r['ci95_pp']
            ax.errorbar(v,2-y,xerr=[[v-lo],[hi-v]],fmt='o',color=BLUE,lw=2,capsize=3)
            ax.text(v,2-y+.18,f'{v:+.2f}',ha='center',fontsize=11)
        ax.axvline(0,color=GRAY,lw=1);ax.set_yticks([2,1,0],[r[1] for r in effects],fontsize=11);ax.set_ylim(-.5,2.5)
        ax.set_xlabel('AA0 − AA4（百分点，95% 区间）',fontsize=10)
        self.text(.596,.30,'仅 8 状态／3 视角，不能认证 97 视角 Oracle。',11,MUTED)
        changed=sum(not r['renderer_change']['top1_same'] for r in a['metrics']['broad'])
        self.text(.061,.255,f'97 候选指标重算：AA0 同协议分数完全一致；更换协议使 Broad 的最小 Accel 视角在 {changed}/8 状态改变。',11,MUTED)
        self.save(pdf)

    def synthesis_page(self,pdf):
        self.new('阶段结论：已有证据与主问题的验证边界','论文收敛与后续工作',
                 '论文拟研究训练覆盖如何改变整任务视角收益，以及指标或选择器能恢复其中多少收益。当前不将研究假设写成已成立贡献。',
                 '初态整任务 Oracle 是主线；恢复快照分析是辅助；主动感知属于后续扩展。',
                 '原 64 个恢复快照不等于 64 个任务初态；其 75.88% 不替代主实验结果。此次仅修订材料，评测安排待讨论。',
                 '历史覆盖训练实验提供动机，恢复状态实验提供局部价值和指标诊断。新主矩阵须从标准任务初态执行到最终成败，先平均噪声再选择视角。经验上限、独立复评和可部署选择器分别报告，不以任何一项失败否定其他问题。')
        heads=['已有证据','主实验尚缺','后续方法验证']
        texts=[
            '历史训练覆盖与性能权衡\n恢复快照中的视角价值差异\nAccel 与可见性筛选诊断\n两种历史选择器的局限\n均保留各自评价范围',
            '真正的任务初态集合\n两训练条件的完整候选对照\n多噪声平均后的初态 Oracle\n规范与任务固定基线\n不以中间状态结果补齐',
            '预先定义可用输入与规则\n与 Oracle 的收益差距\n冻结选择后的新噪声复评\n来源泛化另行检验\n相机运动与真机暂不扩展']
        for x,head,text in zip([.06,.375,.69],heads,texts):
            self.box(x,.32,.255,.40);self.text(x+.016,.69,head,19,BLUE,True);self.text(x+.016,.62,text,12.5,MUTED)
        self.save(pdf)


def build():
    latest=read_json(ROOT/'consolidated-analysis-v1/latest.json')
    landscape=Path(latest['archive'])/'consolidated_view_analysis.pdf'
    require(sha256(landscape)==latest['sha256'],'Archived landscape input changed')
    bridge_path=BRIDGE/'analysis-recovered-v1.json'
    repair=read_json(BRIDGE/'analysis-recovery-v1.json')
    require(sha256(bridge_path)==repair['analysis_sha256'],'Recovered audit changed')
    bridge=read_json(bridge_path)
    qpath=Path('/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/final-analysis/analysis.json')
    q=next(r for r in read_json(qpath)['checkpoint_results'] if r['checkpoint_seed']==41)
    stamp=datetime.now(timezone.utc);out=OUTPUT/'history'/stamp.strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    old={}
    previous_delivery=None
    if (OUTPUT/'latest.json').exists():
        previous_delivery=Path(read_json(OUTPUT/'latest.json')['archive'])/'vla_research_progress.pdf'
    for p in [PUBLIC,PUBLIC.with_suffix('.md')]:
        if p.exists():
            dest=out/('previous-'+p.name);h=sha256(p)
            if p.suffix=='.pdf' and previous_delivery and previous_delivery.is_file() and sha256(previous_delivery)==h:
                dest.symlink_to(os.path.relpath(previous_delivery,dest.parent))
            else:
                shutil.copyfile(p,dest)
            old[str(p)]={'sha256':h,'archive':str(dest)}
    brief=ResearchBrief()
    headroom=read_json(Path(latest['archive'])/'empirical_headroom.json')
    metric_summary=read_json(ROOT/'metric-oracle-comparison-v2/broad_summary.json')
    contrasts=read_json(ROOT/'metric-rules-v1/pairwise_rule_contrasts.json')
    with PdfPages(out/'research-panels.pdf') as pdf:
        for i in [1,2,3,4]:getattr(brief,f'page{i}')(pdf)
        brief.selector_page(pdf,q)
        brief.primary_protocol_page(pdf)
        brief.synthesis_page(pdf)
        brief.headroom_page(pdf,headroom['scopes']['all64'])
        brief.metric_page(pdf,metric_summary,contrasts)
    # Curated sequence, not a concatenation: remove superseded Oracle claims,
    # duplicate geometry/difficulty pages and premature active-method diagrams.
    order=[('research',0),('research',5),('research',1),('research',2),('research',3),
           ('landscape',0),('landscape',1),('research',7),('research',8),('landscape',5),
           ('research',4),('research',6)]
    sources={'research':fitz.open(out/'research-panels.pdf'),'landscape':fitz.open(landscape)}
    result=fitz.open();toc=[]
    titles=[brief.notes[index]['title'] if name=='research' else LANDSCAPE_HEADERS[index][0]
            for name,index in order]
    for i,(name,index) in enumerate(order):
        src=sources[name];page=result.new_page(width=960,height=540)
        page.show_pdf_page(page.rect,src,index)
        if name=='landscape':
            page.add_redact_annot(fitz.Rect(15,8,950,65),fill=(1,1,1))
        # Remove obsolete page counters/footer labels; keep all scientific notes.
        if name=='research':
            page.add_redact_annot(fitz.Rect(840,13,952,35),fill=(1,1,1))
        page.add_redact_annot(fitz.Rect(0,518,960,540),fill=(1,1,1));page.apply_redactions(images=0,graphics=0,text=0)
        page.insert_font(fontname='reportcjk',fontfile='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
        if name=='landscape':
            title,subtitle=LANDSCAPE_HEADERS[index]
            page.insert_text((34,31),title,fontname='reportcjk',fontsize=19,color=(.10,.20,.29))
            page.insert_text((35,54),subtitle,fontname='reportcjk',fontsize=8,color=(.33,.40,.47))
        page.insert_text((34,531),'VLA 研究进展｜主线：任务初态完整执行；恢复快照仅作辅助｜历史渲染复核由独立分支处理',fontname='reportcjk',fontsize=7,color=(.39,.45,.51))
        page.insert_text((911,531),f'{i+1:02d} / 12',fontsize=8,color=(.39,.45,.51))
        toc.append([1,titles[i],i+1])
    result.set_toc(toc);result.set_metadata({'title':'VLA 研究进展｜训练覆盖、候选收益与视角选择','author':'VLA View Research','subject':'统一十二页图表汇报；历史与当前实验同一论证主线'})
    target=out/'vla_research_progress.pdf';result.save(target,garbage=4,deflate=True);result.close()
    for doc in sources.values():doc.close()
    # QA is on the actual merged PDF, not only source figures.
    qa=out/'pdf-qa';qa.mkdir();contact=Image.new('RGB',(1600,6*470),'#DDE5ED');draw=ImageDraw.Draw(contact)
    with fitz.open(target) as doc:
        require(len(doc)==12 and len(doc.get_toc())==12,'Wrong merged page structure')
        validate_scope_texts([page.get_text() for page in doc])
        for i,page in enumerate(doc):
            text=page.get_text();require(len(text)>150 and '\ufffd' not in text,'Missing or invalid PDF text')
            png=qa/f'page-{i+1:02d}.png';page.get_pixmap(matrix=fitz.Matrix(1.2,1.2),alpha=False).save(png)
            im=Image.open(png).convert('RGB');im.thumbnail((780,439));x=i%2*800+10;y=i//2*470+25;contact.paste(im,(x,y));draw.text((x,y-18),str(i+1),fill=INK)
    contact.save(qa/'contact.jpg',quality=93)
    provenance=dict(status='PASS_UNIFIED_RESEARCH_PROGRESS_12_PAGES',utc=stamp.isoformat(),pages=12,pdf_sha256=sha256(target),
        public=str(PUBLIC),compatibility_copy=str(COMPAT),archive=str(out),previous_outputs=old,
        inputs={str(p):sha256(p) for p in [landscape,bridge_path,BRIDGE/'analysis-recovery-v1.json',qpath,Path(__file__),
             Path(latest['archive'])/'empirical_headroom.json',ROOT/'metric-oracle-comparison-v2/broad_summary.json',ROOT/'metric-rules-v1/pairwise_rule_contrasts.json',
             REPO/'scripts/dsol_paper1/build_view_research_focus_brief.py',
             DOCS/'report_sources_20260907/legacy_evidence.json',DOCS/'report_sources_20260907/new_evidence.json',*sorted(brief.figure_inputs)]},
        outline=titles,source_page_map=order,new_gpu_calls=0,layout_checks=brief.qa,scope_contract=SCOPE_CONTRACT,
        review='Structural and text checks passed; rendered pages available for visual inspection')
    write_json(out/'provenance.json',provenance)
    PUBLIC.parent.mkdir(parents=True,exist_ok=True)
    temp=PUBLIC.with_suffix('.unified-building.pdf');shutil.copyfile(target,temp);os.replace(temp,PUBLIC)
    for p in [LEGACY_PUBLIC,COMPAT]:
        alias=p.with_suffix('.alias-building');alias.symlink_to(os.path.relpath(PUBLIC,p.parent));os.replace(alias,p)
    write_json(OUTPUT/'latest.json',provenance)
    print(json.dumps({'status':provenance['status'],'pdf':str(PUBLIC),'archive':str(out),'sha256':provenance['pdf_sha256']},ensure_ascii=False))
    return provenance


if __name__=='__main__':
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'.report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        build()
