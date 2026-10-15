"""One integrated, figure-led LIBERO/RoboCasa report; archive before publish.

Consumes admitted analysis artifacts only. Does not run simulation, infer VLA
actions, tune methods, or recompute scientific estimates during layout.
"""
from datetime import datetime, timezone
from io import BytesIO
from collections import Counter
from pathlib import Path
import argparse
import json
import os
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw

try:
    import pymupdf as fitz
except ImportError:
    sys.path.insert(0,'/root/.cache/uv/archive-v0/XO8_y4zBEt33DOY7wfN64')
    import pymupdf as fitz

from AlphaBrain.research.dsol.data.artifacts import read, write, sha
from AlphaBrain.research.dsol.plotting.robocasa_results import (
    space_axes, grouped_success, success_heatmap, budget_curves,
)
from reports.paper1.legacy_style import Brief, INK, MUTED, BLUE, TEAL, AMBER, GRAY, LINE

MODEL_NAMES=['规范训练','多视角训练']
TASK_NAMES={'CloseBlenderLid':'关闭搅拌杯盖','CloseFridge':'关闭冰箱','CoffeeSetupMug':'咖啡机放杯',
            'NavigateKitchen':'厨房导航*','OpenCabinet':'打开柜门','OpenDrawer':'打开抽屉',
            'PickPlaceCounterToCabinet':'台面物体入柜','PickPlaceSinkToCounter':'水槽物体到台面',
            'TurnOffStove':'关闭炉灶','TurnOnMicrowave':'开启微波炉'}


def space_rank_captions(spatial_summary):
    """Correlation of the actual plotted mean fields; no initial median."""
    result={}
    for name in ('libero','robocasa'):
        item=spatial_summary['benchmarks'][name]
        r=[item['models'][m]['rho'] for m in ('canonical','broad')]
        result[name]=[
            f'平均空间相关系数（Spearman）：规范训练 {r[0]:.3f}；多视角训练 {r[1]:.3f}。正值表示平均 Accel 较低的位置，平均成功率更高。',
            f"口径与图一致：先平均噪声与本页{item['initials']}个初态，再在96个非规范位置上计算 −平均Accel 与平均成功率的秩相关；不是逐初态相关的中位数。",
        ]
    return result


def annotate_space_page(page, lines):
    """Add embedded-font captions in whitespace; never redact source content."""
    if len(lines)!=2:raise ValueError('Exactly two concise additive lines required')
    font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font_path)
    prop=font_manager.FontProperties(fname=font_path,size=9)
    buffer=BytesIO()
    with plt.rc_context({'pdf.fonttype':3}):
        fig=plt.figure(figsize=(page.rect.width/72,page.rect.height/72))
        fig.patch.set_alpha(0)
        try:
            for i,line in enumerate(lines):
                fig.text(64/page.rect.width,1-(465+13*i)/page.rect.height,line,
                         fontproperties=prop,color=MUTED,va='baseline')
            fig.canvas.draw()
            for artist in fig.texts:
                if artist.get_window_extent().x1>fig.bbox.width-20:
                    raise ValueError('Spatial rank caption overflow')
            fig.savefig(buffer,format='pdf',transparent=True)
        finally:
            plt.close(fig)
    with fitz.open(stream=buffer.getvalue(),filetype='pdf') as overlay:
        for font in overlay[0].get_fonts():
            embedded_glyphs=(font[2]=='Type3' and
                             overlay.xref_get_key(font[0],'CharProcs')[0] in ('xref','dict'))
            if not embedded_glyphs and not overlay.extract_font(font[0])[3]:
                raise ValueError('Caption font must be embedded')
        page.show_pdf_page(page.rect,overlay,0,overlay=True)


class IntegratedBrief(Brief):
    def __init__(self, root):
        for weight in ('Regular','Bold'):
            font_manager.fontManager.addfont(f'/usr/share/fonts/opentype/noto/NotoSansCJK-{weight}.ttc')
        plt.rcParams.update({'font.family':'Noto Sans CJK JP','pdf.fonttype':3,'axes.unicode_minus':False,'font.size':12})
        self.root=Path(root);self.summary=read(self.root/'summary.json');self.search=read(self.root/'search-summary.json')
        self.population=read(self.root/'population.json');self.audit=read(self.root/'audit.json')
        with np.load(self.root/'matrix.npz') as data:
            self.q=data['success'].mean(-1);self.acc=data['accel_prefixes'][...,1].mean(-1)
        self.notes=[];self.qa=[];self.figure_inputs=set()

    def new(self,title,stage,subtitle,takeaway,source,note):
        self.fig=plt.figure(figsize=(13.333,7.5),facecolor='white')
        self.canvas=self.fig.add_axes([0,0,1,1]);self.canvas.set_axis_off();self.artists=[]
        self.canvas.add_patch(Rectangle((.045,.941),.039,.009,color=BLUE,transform=self.canvas.transAxes))
        self.text(.096,.956,stage,10.5,MUTED)
        self.text(.045,.895,title,25,INK,True)
        self.text(.048,.82,subtitle,11,MUTED)
        self.canvas.add_patch(Rectangle((.046,.13),.005,.065,color=TEAL,transform=self.canvas.transAxes))
        self.text(.067,.19,takeaway,16,INK,True)
        self.text(.048,.098,source,8.2,MUTED)
        self.notes.append(dict(title=title,takeaway=takeaway,speaker_note=note,source=source))

    def scope(self,m,split='target'):
        return self.summary['models'][['canonical','broad'][m]]['scopes'][split]

    def overview(self,pdf):
        gain=self.summary['scopes']['target']['training_gain']['stay_extra32']['mean']*100
        self.new('如何利用 VLA 视角：训练覆盖与扰动恢复','研究问题与证据链',
            'LIBERO：198,656 次闭环  ｜  RoboCasa：310,400 次闭环，完整矩阵已核验',
            '区分训练覆盖、静态视角恢复与实际主动获取，不以经验上限替代方法收益。',
            'RoboCasa 为 10 个已知任务的 seen／layout+style heldout；当前两套主矩阵均从任务初态完整执行。',
            '多视角训练是主线，不只是历史背景。本报告依次说明训练改变了什么、指标能否识别有效视角、方法比不调整和同预算随机改善多少。实际相机移动与任务中主动观察仍未验证。')
        titles=['训练覆盖','视角价值','可执行的方法']
        bodies=['同预算规范／多视角训练\n比较规范与额外视角表现',
                '噪声平均后比较候选\n区分经验上限与事前选择',
                '已访问观测驱动决策\n实际移动与闭环仍待补齐']
        for x,title,body in zip([.065,.375,.685],titles,bodies):
            self.box(x,.40,.25,.30);self.text(x+.014,.67,title,20,BLUE,True);self.text(x+.014,.59,body,13,MUTED)
        self.text(.075,.32,f'RoboCasa heldout：多视角训练对额外32视角的平均变化为 {gain:+.2f} pp。',16,INK,True)
        self.text(.075,.265,'中心问题：哪些视角失效能由训练覆盖解决，哪些仍可通过观察调整恢复？',13,MUTED)
        self.save(pdf)

    def protocol(self,pdf):
        self.new('两套证据对齐研究问题，不混用绝对成功率','协议与相机定义',
            '任务初态 × 候选安装 × 独立执行噪声；不是任务中途逐状态切换相机',
            'RoboCasa 固定的是安装偏置，相机仍随底盘／躯干自然运动。',
            'RoboCasa：64训练目录＋32额外目录＋规范安装；外部左右联动，腕部安装不变。\n97点为受限三参数轨道，不是完整SE(3)，也未认证真实相机机构可达。',
            'LIBERO当前主矩阵是32个标准初态，无人为遮挡；历史恢复快照与遮挡百分比不再混入。RoboCasa两臂从同一官方Human300初始化，在300任务的33000源窗口上各曝光两次，各8250步、全局batch8；使用原all_lora配置（含视觉编码器及小头更新），不是仅更新LoRA。广域臂每源episode分配两处训练安装，不是每帧渲染64处。正式评价取10个已知任务。LIBERO为AlphaBrain/PyTorch，RoboCasa为OpenPI/JAX；各环境内部匹配对照，不能把跨环境差异只归因于场景。Heldout同时留出layout/style，并非未见任务；目录留出与场景留出是两个维度。')
        rows=[['项目','LIBERO','RoboCasa'],['初态','8任务 × 4 = 32','10任务 × 40 = 400'],
              ['场景划分','初态0/1开发；2/3评价','200 seen；200 heldout'],['候选 / 执行重复','97 / 32','97 / 4'],
              ['初态评分噪声','8','8（不运行环境）'],['相机安装','外部世界固定＋腕部','左右外部随本体＋腕部'],
              ['训练配置','既有规范／多视角权重','同初始化，300任务，各8250步'],['渲染','AA0冻结协议','原生AA4，保留微小波动']]
        ax=self.fig.add_axes([.065,.32,.87,.42]);ax.axis('off')
        table=ax.table(cellText=rows,cellLoc='center',loc='center',colWidths=[.25,.34,.41]);table.auto_set_font_size(False);table.set_fontsize(12);table.scale(1,1.9)
        for (r,c),cell in table.get_celld().items():cell.set_edgecolor(LINE);cell.set_facecolor('#EFF4FA' if r==0 else 'white')
        self.text(.075,.25,'共同边界：标准初态整任务、单次执行不主动改安装；不能称为物理主动相机闭环。',12,INK)
        self.save(pdf)

    def coverage(self,pdf):
        gain=self.summary['scopes']['target']['training_gain']
        self.new('RoboCasa：本轮多视角训练未改善留出视角均值','跨环境验证 · 训练覆盖',
            '每列：10任务 × 20初态 × 4执行；候选先在初态内平均，任务等权',
            f"Heldout 额外视角变化 {gain['stay_extra32']['mean']*100:+.2f} pp；参考安装变化 {gain['canonical']['mean']*100:+.2f} pp。",
            'Seen 与 heldout 使用不同场景，不将二者差值解释为单独的训练效应。训练仅一个seed，不代表收敛或最优预算。',
            '训练因果比较在每个场景组内部对同初态、同候选、同噪声比较。规范安装可能不是最优安装；额外32视角来自同参数范围的留出目录，不是范围外SE(3)泛化。')
        for col,split in enumerate(('pretrain','target')):
            ax=self.axes([.075+.48*col,.35,.39,.37])
            values=[[self.scope(m,split)['coverage'][k]['mean'] for k in ('canonical','train64','extra32')] for m in range(2)]
            grouped_success(ax,values,['参考安装','训练目录64','额外目录32'])
            ax.set_title('Seen 场景' if col==0 else 'Heldout 布局／风格',fontsize=16,color=INK,pad=13)
        self.text(.075,.27,'含导航任务的10任务全表为主；逐任务结果与去除导航的敏感性分析另列，不静默删任务。',11,MUTED)
        self.save(pdf)

    def space(self,pdf):
        self.new('RoboCasa 候选空间：分开观察成功率与 Accel','跨环境验证 · 空间结构',
            'Heldout 200初态等权；点为非规范96安装，星号为规范安装；色条各自适应实际范围',
            '亮色分别表示较高成功率、较低Accel；空间结构必须用初态内选择结果检验。',
            '坐标：方位角／仰角／半径倍率；相机中心与朝向均变化。相同颜色跨图不代表同值，请读色条。\nAccel是初态8次评分均值（去噪前3步），不是整条任务动作轨迹的加速度。',
            'RoboCasa三维图保留独立原始色条；每点先平均执行噪声再平均初态。相机架固定安装后的本体运动不在这张参数空间示意图中。完整矩阵、查询预算曲线、逐任务辅助图与CSV在分析目录。')
        ix=np.array([j['split']=='target' for j in self.population['jobs']]);p=np.array([v['parameters'] for v in self.population['views']])
        for m in range(2):
            q=self.q[m,ix].mean(0);a=self.acc[m,ix].mean(0)
            upper=space_axes(self.fig,[.04+.48*m,.515,.39,.255],p[1:],q[1:],MODEL_NAMES[m],percentage=True,canonical=q[0])
            lower=space_axes(self.fig,[.04+.48*m,.255,.39,.255],p[1:],a[1:],'',canonical=a[0])
            upper.set_xlabel('');upper.set_ylabel('');upper.set_zlabel('');lower.set_zlabel('')
        self.text(.10,.685,'成功率',11,INK,True);self.text(.10,.42,'Accel',11,INK,True)
        self.save(pdf)

    def headroom(self,pdf):
        self.new('RoboCasa：候选经验收益不等于可实现方法收益','跨环境验证 · 上限与固定选择',
            'Heldout同一200初态、97候选、4执行；经验Oracle先平均4次，再按初态取最大',
            '每格仅4次，经验最大值有乐观选择偏差；不把它称为稳定真实上限。',
            'Seen全局／每任务固定：只用seen标签选定，再在heldout评价。\n经验Oracle：使用当前heldout成败选赢家，仅作事后诊断，不是部署方法。',
            '两种固定选择器与每初态经验最大值的标签权限不同。两半执行噪声交叉选择仅在详细说明中作偏差诊断，不恢复O/P/Q主叙事，也不是新闭环或动态状态Oracle。')
        keys=['canonical','seen_global_fixed','seen_task_fixed','empirical_oracle'];labels=['参考安装','Seen全局固定','Seen每任务固定','每初态经验Oracle']
        for m in range(2):
            ax=self.axes([.075+m*.48,.36,.39,.35]);vals=[round(self.scope(m)['methods'][k]['success']['mean']*100,10) for k in keys]
            ax.bar(range(4),vals,color=[GRAY,BLUE,'#7099CC',TEAL]);ax.set_xticks(range(4),labels,fontsize=9,rotation=10)
            ax.set_ylim(0,100);ax.set_title(MODEL_NAMES[m],fontsize=16,color=INK);ax.set_ylabel('闭环成功率（%）',fontsize=11)
            for i,v in enumerate(vals):ax.text(i,v+1.2,f'{v:.2f}',ha='center',fontsize=12,fontweight='bold')
        self.text(.075,.26,'单次任务内仍固定所选安装；不同初态可选不同安装，不能按每次执行噪声更换赢家。',12,INK)
        self.save(pdf)

    def methods(self,pdf):
        self.new('RoboCasa：扰动恢复、学习选择与同预算随机','跨环境验证 · 方法收益',
            'Heldout场景 × 额外32扰动起点；选择器仅在seen场景拟合，超参数按seen布局交叉验证',
            '超过“不调整”说明存在恢复；超过同预算随机才支持搜索策略的额外作用。',
            '全查询访问97候选；16查询只暴露已访问点。学习器为冻结RGB特征＋岭回归，不更新VLA。\nRGB＋Accel使用每个查询点8次评分；候选图为缓存，未计真实获取耗时或验证可达性。',
            '所有比较使用同一heldout初态总体。RGB学习器不读规范图、规范标记或候选ID；左右外部特征取平均，腕部特征独立，任务ID已知。Seen训练内成绩不作为泛化证据。可见性分割在RoboCasa未采集，因此不存在对应组合规则结果。')
        definitions=[('stay_extra32','扰动不调整'),('canonical','已知参考安装'),('min_accel','全查询 Accel'),
                     ('rgb','全查询 RGB'),('rgb_accel','全查询 RGB＋Accel'),
                     ('local_accel','16查询 Accel'),('local_rgb','16查询 RGB'),
                     ('random_rgb','16随机＋RGB'),('local_rgb_accel','16查询 RGB＋Accel'),('random_rgb_accel','16随机＋RGB/Accel')]
        for m,model in enumerate(('canonical','broad')):
            ax=self.axes([.24+m*.47,.30,.235,.43],True,False)
            vals=[]
            for key,label in definitions:
                entry=self.scope(m)['methods'][key]['success'] if key in self.scope(m)['methods'] else self.search['models'][model][key]['16']['success']
                vals.append(round(100*entry['mean'],10))
            ax.barh(np.arange(len(vals)),vals,color=[GRAY,GRAY,BLUE,BLUE,BLUE,TEAL,TEAL,AMBER,TEAL,AMBER],height=.65)
            ax.set_yticks(np.arange(len(vals)),[x[1] for x in definitions],fontsize=9);ax.invert_yaxis();ax.set_xlim(0,105)
            ax.set_xlabel('完整任务成功率（%）',fontsize=10);ax.set_title(MODEL_NAMES[m],fontsize=16,pad=12,color=INK)
            for i,v in enumerate(vals):ax.text(v+1,i,f'{v:.1f}',va='center',fontsize=10)
        self.save(pdf)

    def diagnostics(self,pdf):
        self.new('RoboCasa：指标分区与任务差异一起看','跨环境验证 · 指标解释',
            '左：每初态内部按Accel排序的四分位；右：多视角模型的heldout任务差异',
            '规范模型呈低值区域优势；多视角模型中关系较弱，且四分位不再单调。',
            '四分位：仅非规范96候选，各24点；不跨任务按绝对Accel混排。右图单位为百分点。\n*厨房导航保留且单列；任务均值、区间、评分噪声稳定性、几何条件化诊断见配套说明。',
            '成功率全为0或全为1的初态不定义Spearman，单独计数，不能填0拉低相关性。几何残差相关只控制已知参数偏移，不是可见性控制或因果证据。高成功候选的共同特征是事后诊断，未反馈选择器训练。')
        ax=self.axes([.075,.37,.33,.34])
        for m,color in enumerate([BLUE,TEAL]):
            values=[r['mean']*100 for r in self.scope(m)['quartiles']]
            ax.plot(range(4),values,'o-',color=color,label=MODEL_NAMES[m])
        ax.set_xticks(range(4),['最低25%','次低','次高','最高25%'],fontsize=10);ax.set_ylim(0,100)
        ax.set_ylabel('组内闭环成功率（%）',fontsize=11);ax.legend(frameon=False,fontsize=10)
        rows=self.scope(1)['tasks'];matrix=[]
        canonical_tasks={r['task']:r for r in self.scope(0)['tasks']}
        for r in rows:
            d=r['methods'];matrix.append([d['stay_extra32']-canonical_tasks[r['task']]['methods']['stay_extra32'],
                                        d['min_accel']-d['stay_extra32'],d['rgb_accel']-d['stay_extra32'],d['empirical_oracle']-d['canonical']])
        ax=self.fig.add_axes([.61,.28,.33,.45]);success_heatmap(ax,matrix,[TASK_NAMES[r['task']] for r in rows],['训练增益','Accel恢复','学习恢复','经验余量'],difference=True)
        r0,r1=self.scope(0),self.scope(1)
        text=lambda x:'未定义' if x is None else f'{x:.2f}'
        self.text(.075,.29,f"初态内 Spearman 中位数：{text(r0['spearman_median'])} / {text(r1['spearman_median'])}\n可定义初态数：{r0['spearman_defined']} / {r1['spearman_defined']}（各200）",11,MUTED)
        self.save(pdf)

    def synthesis(self,pdf):
        g=self.summary['scopes']['target']['training_gain']['stay_extra32']
        a=self.scope(1)['methods']['min_accel']['gain_vs_stay']
        s=self.search['models']['broad']['local_rgb_accel']['16']['gain_vs_random']
        c=self.search['models']['canonical']['local_rgb_accel']['16']['gain_vs_random']
        self.new('阶段结论：跨环境限制已明确，主动获取仍待验证','结论与下一步',
            '主线不变：训练覆盖 → 剩余视角价值 → 可获得反馈驱动的恢复 → 实际相机验证',
            '先独立确认可重复的恢复，再验证可执行相机获取；不急于增加复杂模块。',
            '区间为任务—初态配对bootstrap，条件于当前实测执行噪声；不代表所有VLA／本体的普遍规律。\n完整审计、原始表、额外图及每页解释均由当前PDF旁的Markdown指向同一归档。',
            '若同预算随机已能解释收益，就不能把搜索器复杂度写成贡献。若新视角不能被操作模型利用，再考虑联合训练。实际头部闭环需先证明操作策略适配；不得把固定安装干预等同于可执行主动动作。')
        fmt=lambda r:f"{r['mean']*100:+.2f} pp\n95% CI [{r['ci95'][0]*100:+.2f}, {r['ci95'][1]*100:+.2f}]"
        for x,title,result in zip([.065,.375],['多视角覆盖训练增益','多视角全查询Accel恢复'],[g,a]):
            self.box(x,.43,.25,.28);self.text(x+.013,.69,title,15,BLUE,True);self.text(x+.013,.57,fmt(result),15,INK)
        self.box(.685,.43,.25,.28);self.text(.698,.69,'16查询 RGB＋Accel\n相对同评分随机',14,BLUE,True)
        self.text(.698,.585,f"规范：{c['mean']*100:+.2f} pp [{c['ci95'][0]*100:+.2f}, {c['ci95'][1]*100:+.2f}]\n多视角：{s['mean']*100:+.2f} pp [{s['ci95'][0]*100:+.2f}, {s['ci95'][1]*100:+.2f}]\n（95% CI；仅离线重放）",10.5,INK)
        self.text(.075,.35,'已完成：两环境静态证据、RoboCasa全矩阵核验、heldout轻量选择器与查询重放。',12,INK)
        self.text(.075,.29,'待补齐：任务相关可见性标签、物理相机获取、冻结方法独立复评；动态观察作为后续。',12,MUTED)
        self.save(pdf)


def make_supporting_figures(root, brief):
    """Separate high-resolution plots, not an ever-growing presentation deck."""
    output=root/'figures';output.mkdir(exist_ok=True)
    ix=np.array([j['split']=='target' for j in brief.population['jobs']])
    fig,axes=plt.subplots(2,2,figsize=(15,12),layout='constrained')
    for m in range(2):
        for col,data,title in [(0,brief.q[m,ix]*100,'完整任务成功率（%）'),(1,brief.acc[m,ix],'初态Accel均值')]:
            ax=axes[m,col]
            im=ax.imshow(data,aspect='auto',cmap='viridis' if col==0 else 'viridis_r',
                         vmin=0 if col==0 else None,vmax=100 if col==0 else None)
            ax.set_title(MODEL_NAMES[m]+' · '+title);ax.set_xlabel('候选序号：0规范，1–64训练，65–96额外')
            ax.set_ylabel('Heldout初态（按任务分组）')
            ax.set_yticks(np.arange(10)*20+9.5,[TASK_NAMES[t] for t in brief.summary['task_names']],fontsize=9)
            for row in np.arange(1,10)*20-.5:ax.axhline(row,color='white',lw=.4)
            for edge in (.5,64.5):ax.axvline(edge,color='white',lw=.7)
            fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('RoboCasa heldout 完整矩阵：不按结果排序或筛除初态',fontsize=17)
    fig.savefig(output/'01_full_matrix.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),layout='constrained')
    for m,model in enumerate(('canonical','broad')):
        budget_curves(axes[m],brief.search,model);axes[m].set_title(MODEL_NAMES[m]);axes[m].grid(alpha=.2)
    fig.suptitle('有限目录查询重放：各评分器均配同预算随机前沿对照（非真实移动）',fontsize=14)
    fig.savefig(output/'02_query_budget.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(16,7),layout='constrained')
    keys=['stay_extra32','canonical','min_accel','rgb','rgb_accel','empirical_oracle']
    columns=['扰动不调整','参考安装','Accel全查询','RGB全查询','RGB+Accel','经验Oracle*']
    for m in range(2):
        rows=brief.scope(m)['tasks']
        artist=success_heatmap(axes[m],[[r['methods'][k] for k in keys] for r in rows],
                              [TASK_NAMES[r['task']] for r in rows],columns)
        axes[m].set_title(MODEL_NAMES[m]);axes[m].tick_params(axis='x',labelrotation=25)
        fig.colorbar(artist,ax=axes[m],shrink=.8,label='闭环成功率（%）')
    fig.suptitle('逐任务 heldout 成绩：*经验Oracle用评价标签选最大，仅作乐观事后参照',fontsize=15)
    fig.savefig(output/'03_task_methods.png',dpi=170);plt.close(fig)
    # This slice plot displays only the actual 4x4x4 training grid. Extra
    # Sobol points are not interpolated into invented observations.
    views=brief.population['views'];params=np.array([v['parameters'] for v in views])
    train=np.array([i for i,v in enumerate(views) if v['role']=='train'])
    radii=np.unique(params[train,2]);az=np.unique(params[train,0]);elev=np.unique(params[train,1])
    fig,axes=plt.subplots(4,4,figsize=(14,12),layout='constrained')
    for row in range(4):
        m=row//2;metric=row%2;mean=(brief.q if metric==0 else brief.acc)[m,ix].mean(0)
        for col,radius in enumerate(radii):
            grid=np.full((len(elev),len(az)),np.nan)
            for v in train[np.isclose(params[train,2],radius)]:
                grid[np.flatnonzero(elev==params[v,1])[0],np.flatnonzero(az==params[v,0])[0]]=mean[v]*(100 if metric==0 else 1)
            ax=axes[row,col];im=ax.imshow(grid,origin='lower',cmap='viridis' if metric==0 else 'viridis_r',aspect='auto')
            ax.set_title(f'{MODEL_NAMES[m]} · '+('成功率' if metric==0 else 'Accel')+f' · r={radius:.3f}',fontsize=10)
            ax.set_xticks(range(len(az)),[f'{v:g}' for v in az]);ax.set_yticks(range(len(elev)),[f'{v:g}' for v in elev])
            ax.set_xlabel('方位角（°）',fontsize=9);ax.set_ylabel('仰角（°）',fontsize=9)
            fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('规则训练目录的真实切片：每格均有实测；独立色条，不插值，不包含额外Sobol点',fontsize=15)
    fig.savefig(output/'04_measured_slices.png',dpi=160);plt.close(fig)
    write(output/'manifest.json',dict(files={p.name:sha(p) for p in sorted(output.glob('*.png'))},
          source_summary_sha256=sha(root/'summary.json'),source_search_sha256=sha(root/'search-summary.json')))


def build(config, *, publish=False):
    cfg=read(config);root=Path(cfg['output']);audit=read(root/'audit.json')
    if audit['status']!='PASS':raise ValueError('Matrix not admitted')
    for name,digest in audit['artifacts'].items():
        if sha(root/name)!=digest:raise ValueError('Admitted input changed')
    code_identity=sha(__file__)
    repo=Path(__file__).resolve().parents[2];public=repo/'docs/dsol_paper1/reports/current/vla_view_research_progress_zh.pdf'
    previous=root/'previous-publication';previous.mkdir(exist_ok=True)
    for p in (public,public.with_suffix('.md'),public.parent/'publication.json'):
        target=previous/p.name
        if not target.exists():shutil.copyfile(p,target)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=root/'reports'/stamp;output.mkdir(parents=True)
    brief=IntegratedBrief(root)
    structure_path=Path(cfg['libero_space_structure'])
    if sha(structure_path)!=cfg['libero_space_structure_sha256']:
        raise ValueError('LIBERO spatial-caption source changed')
    spatial_path=Path(cfg['spatial_correlation_summary'])
    if sha(spatial_path)!=cfg['spatial_correlation_summary_sha256']:
        raise ValueError('Spatial correlation source changed')
    spatial_summary=read(spatial_path)
    for name,digest in spatial_summary['sources'].items():
        if sha(name)!=digest:raise ValueError('Spatial mean-field input changed')
    captions=space_rank_captions(spatial_summary)
    make_supporting_figures(root, brief)
    with PdfPages(output/'new-panels.pdf') as pdf:
        for method in ('overview','protocol','coverage','space','headroom','methods','diagnostics','synthesis'):
            getattr(brief,method)(pdf)
    write(output/'layout.json',brief.qa)
    if any(p['issues'] for p in brief.qa):raise ValueError('Layout text overflow; see layout.json')
    old=fitz.open(previous/public.name);new=fitz.open(output/'new-panels.pdf')
    # Preserve the eight substantive historical/LIBERO pages unchanged except
    # counters/footer. Replace superseded overview, protocol and next-stage page.
    order=[('new',0),('new',1),*[('old',p) for p in (2,3,4,5,7,8,9,10)],
           *[('new',p) for p in (2,3,4,5,6,7)]]
    result=fitz.open();toc=[]
    for i,(kind,index) in enumerate(order):
        source=old if kind=='old' else new
        result.insert_pdf(source,from_page=index,to_page=index)
        page=result[-1]
        page.add_redact_annot(fitz.Rect(840,10,952,34),fill=(1,1,1))
        page.add_redact_annot(fitz.Rect(0,524,960,540),fill=(1,1,1))
        page.apply_redactions(images=0,graphics=0,text=0)
        if kind=='old' and index==5:
            annotate_space_page(page,captions['libero'])
        elif kind=='new' and index==3:
            annotate_space_page(page,captions['robocasa'])
        body=fitz.Rect(0,40,959.976,518)
        original_chars=Counter(c for c in source[index].get_text(clip=body) if not c.isspace())
        current_chars=Counter(c for c in page.get_text(clip=body) if not c.isspace())
        if original_chars-current_chars:
            raise ValueError('Original page text lost while adding caption')
        label=('LIBERO · 历史证据' if index in (2,3) else 'LIBERO · 标准初态') if kind=='old' else ('统一研究口径' if index<2 else 'RoboCasa · P0')
        # Built-in CJK font avoids rewriting historical fonts and preserves
        # correct CID mappings (a separately subset CFF can render wrong glyphs).
        page.insert_text((793,27),label,fontname='china-s',fontsize=9,color=(.39,.45,.51))
        page.insert_text((34,531),'VLA 研究进展 · LIBERO / RoboCasa · 2026-09-23 · 初态整任务；实际主动获取待验证',fontname='china-s',fontsize=7,color=(.39,.45,.51))
        page.insert_text((901,531),f'{i+1:02d} / 16',fontsize=8,color=(.39,.45,.51))
        title=brief.notes[index]['title'] if kind=='new' else next((row[1] for row in old.get_toc() if row[2]==index+1),f'LIBERO 证据 {index+1}')
        toc.append([1,title,i+1])
    result.set_toc(toc);result.set_metadata(dict(title='VLA 研究进展｜训练覆盖、视角价值与恢复',author='VLA View Research',subject='LIBERO冻结证据与RoboCasa P0完整矩阵统一分析；16页'))
    target=output/'vla_view_research_progress_zh.pdf';result.save(target,garbage=4,deflate=True);result.close();old.close();new.close()
    qa=output/'previews';qa.mkdir();contact=Image.new('RGB',(1600,8*465),'#E8EDF3');draw=ImageDraw.Draw(contact)
    with fitz.open(target) as doc:
        if len(doc)!=16:raise ValueError('Wrong report length')
        for i,page in enumerate(doc):
            if len(page.get_text())<100 or '\ufffd' in page.get_text():raise ValueError('Missing report text')
            png=qa/f'page-{i+1:02d}.png';page.get_pixmap(matrix=fitz.Matrix(1.25,1.25),alpha=False).save(png)
            im=Image.open(png);im.thumbnail((780,439));x=i%2*800+10;y=i//2*465+23;contact.paste(im,(x,y));draw.text((x,y-18),str(i+1),fill=INK)
        preservation=[]
        with fitz.open(previous/public.name) as historical:
            for page,(kind,index) in enumerate(order):
                if kind!='old':continue
                clip=fitz.Rect(0,40,959.976,420 if index==5 else 518)
                old_pix=historical[index].get_pixmap(matrix=fitz.Matrix(1,1),clip=clip,alpha=False)
                new_pix=doc[page].get_pixmap(matrix=fitz.Matrix(1,1),clip=clip,alpha=False)
                identical=old_pix.samples==new_pix.samples
                preservation.append(dict(old_page=index+1,new_page=page+1,body_pixels_identical=identical,
                    caption_only_revision=index==5,comparison_clip=list(clip)))
                if not identical:raise ValueError('Historical scientific body changed during merge')
        write(output/'preservation.json',preservation)
    contact.save(qa/'contact.jpg',quality=92)
    notes=make_notes(root,brief,order,previous,output)
    notes+='\n## 三维图旁的相对排序说明（本次文字修订）\n\n'
    for label,key in [('第6页 LIBERO','libero'),('第12页 RoboCasa','robocasa')]:
        notes+='### '+label+'\n\n'+'\n\n'.join(captions[key])+'\n\n'
    notes+='本次仅将两页图旁指标改为与三维图直接对应的平均空间秩相关：先平均，再在96个位置上计算相关；不使用逐初态中位数。无新增表格、页数或评测。LIBERO为32初态，RoboCasa为200个heldout初态；不与另页逐初态选择能力诊断混称。\n'
    (output/'vla_view_research_progress_zh.md').write_text(notes)
    if sha(__file__)!=code_identity:raise ValueError('Report source changed during build; rebuild for correct provenance')
    sources=[repo/'AlphaBrain/research/dsol'/p for p in (
        'analysis/robocasa_results.py','analysis/robocasa_features.py','analysis/robocasa_statistics.py',
        'selectors/query_ranker.py','acquisition/discrete_search.py','plotting/robocasa_results.py')]
    inputs={str(p):sha(p) for p in [root/'audit.json',root/'summary.json',root/'search-summary.json',root/'matrix.npz',root/'method-arrays.npz',Path(config),previous/public.name,Path(__file__),structure_path,spatial_path,*sources]}
    receipt=dict(status='BUILT_PENDING_VISUAL_REVIEW',utc=stamp,pages=16,pdf_sha256=sha(target),archive=str(output),
                 inputs=inputs,previous_publication=str(previous),page_map=order,new_scientific_rollouts=0,
                 newly_analyzed_rollouts=310400,source='Frozen native-render P0 evaluation; LIBERO figures preserved',
                 public=str(public),notes_sha256=sha(output/'vla_view_research_progress_zh.md'))
    write(output/'provenance.json',receipt);write(root/'report-latest.json',receipt)
    if publish:
        publish_report(output,public)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
    return receipt


def make_notes(root,brief,order,previous,output):
    s=brief.summary;a=brief.audit
    lines=['# VLA 研究进展｜LIBERO 与 RoboCasa 统一报告（2026-09-23）','',
           '唯一当前PDF：[打开报告]('+str(Path(__file__).resolve().parents[2]/'docs/dsol_paper1/reports/current/vla_view_research_progress_zh.pdf')+')。',
           '',f'RoboCasa正式矩阵：{a["complete_rollouts"]:,}次完整任务、77,600个模型—初态—候选格、620,800次初态评分。32个分片全部完成；逐文件和评分重算验收通过。',
           '',f'本次新增科学闭环0次；分析新增轻量岭回归选择器，不更新VLA。详细产物：[分析目录]({root})。',
           '', '## 核心结果（完整任务成功率，%）','',
           '| 场景组 | 训练 | 参考安装 | 额外32扰动不调整 | 最低平均Accel，全查询 | RGB+Accel，全查询 | 每初态经验Oracle |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for split,label in [('pretrain','Seen'),('target','Heldout')]:
        for m,model in enumerate(('canonical','broad')):
            d=s['models'][model]['scopes'][split]['methods']
            vals=[round(d[k]['success']['mean']*100,10) for k in ('canonical','stay_extra32','min_accel','rgb_accel','empirical_oracle')]
            lines.append('| '+label+' | '+MODEL_NAMES[m]+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
    lines+=['','Seen学习器成绩为训练内拟合值，不作为泛化证据。Heldout同时留出布局与风格，但任务已知。',
            '经验Oracle先平均每候选4次执行再取最大，未做噪声选择偏差校正；不代表可部署选择器或任务中动态选视角。',
            'RoboCasa两臂以同一官方Human300权重初始化，在300任务、7498段示范的33000源窗口上各曝光两次：各66000样本、8250步、全局batch8、seed41、FSDP8。沿用all_lora配置，含Gemma LoRA、视觉编码器和部分小头更新；不是只训练LoRA。正式评价取其中10个已知任务，分seen/heldout场景；不是完整Human300从base重新训练。不能与另一台机器的50任务官方17.20%直接比较。',
            'LIBERO使用AlphaBrain/PyTorch，RoboCasa使用原OpenPI/JAX链路，初始化、后训练预算、本体及数据分布也不同。这里只能在各环境内部做匹配模型对照；跨环境结果差异不能单独归因于场景复杂度。',
            '', '## 同预算查询重放（Heldout，32个额外起点均匀平均）','',
            '| 训练 | 方法 | 成功率% | 相对不调整pp | 相对同评分随机pp（95% CI） |',
            '| --- | --- | ---: | ---: | --- |']
    for m,model in enumerate(('canonical','broad')):
        for method in ('local_accel','local_rgb','local_rgb_accel','random_accel','random_rgb','random_rgb_accel'):
            d=brief.search['models'][model][method]['16'];g=d.get('gain_vs_random')
            text='—' if g is None else f"{g['mean']*100:+.2f} [{g['ci95'][0]*100:+.2f}, {g['ci95'][1]*100:+.2f}]"
            lines.append(f"| {MODEL_NAMES[m]} | {method} | {d['success']['mean']*100:.2f} | {d['gain_vs_stay']['mean']*100:+.2f} | {text} |")
    lines+=['','这是离线缓存查询与终点成绩重放：世界不随查询推进，路径未做动力学/碰撞认证。不能报告为真实相机移动后的收益。',
            '相同查询数不等于相同推理成本：RGB规则不需VLA评分，Accel规则每点8次评分；随机对照与其对应方法使用相同评分器和预算。',
            '搜索规则：起点计1次查询；在已访问点的邻接边界中，从当前评分最低的锚点扩展相近新点。只能读已查询点的分数；同预算随机对照随机选邻接边界边，再用同一个评分器选已访问点中的最低分。不是对连续SE(3)求全局最优。',
            '', '## 对论文主线的判断','',
            '1. **训练覆盖的收益具有条件性。** LIBERO的强改善仍成立；RoboCasa本轮heldout额外32视角没有平均改善，参考安装反而下降6.38个百分点。不能合并成普遍有效的结论，也不能由一个8250步/单seed配置反证所有多视角训练无效。候选分布、每源示范的增强密度、预算和模型适配仍需区别。',
            '2. **较低的流指标与较好的闭环并非同一件事。** RoboCasa多视角模型的Accel绝对值下降，但相对排序的指示性变弱；在规范模型上可见区域趋势，却不足以保证最低点最优。这为“评价指标必须经下游任务校验”提供了分析证据，不是新的通用指标已成立。',
            '3. **经验余量尚未变成可实现余量。** 经验Oracle很高，但同一格只有4次执行；不能把有限噪声最大值当作稳定真实成功率。Seen选定的固定视角、事前指标与heldout学习器才对应不读取测试成败的可实现选择。当前两种固定选择都没有超过各自参考安装的heldout均值。',
            '4. **学习打分是基线，不等于完整主动方法。** RGB/Accel的小模型保留了部分恢复均值，但图像获取权限、随机查询对照和新噪声确认仍关键。当前冻结ResNet18+线性特征模型失败或收益有限，不代表所有选择器学不到。',
            '5. **论文缺口在方法与可执行验证，不在单纯扩大表格。** 静态全矩阵能分析训练、指标和视角价值；尚不能证明移动相机改善任务，更不能说明动态遮挡处理或自进化能力。应保留分析主线，同时用最小可执行获取实验验证方法，而非先搭建复杂预测/生成/RL模块。',
            '', '以上是当前证据支持的阶段判断，不是宣称已经形成五项独立算法创新。',
            '', '## 指标分析：强度、排序与评分稳定性','',
            'Accel定义：前p个去噪速度场的相邻变化范数之和乘p，除以前p个速度场的范数之和；主指标p=3。每个初态/候选计算8次初噪的均值。这里不是物理动作加速度，也不是整条rollout的平均指标。全部前缀已从保存的速度张量独立重算。',
            '', '| Heldout训练 | Accel均值（非规范96） | 初态内Spearman中位数 | 可定义初态 | 最低/次低/次高/最高四分位成功率% | 单噪声赢家与8噪声赢家一致率% |',
            '| --- | ---: | ---: | ---: | --- | ---: |']
    for m in range(2):
        d=brief.scope(m)
        lines.append(f"| {MODEL_NAMES[m]} | {d['accel_mean']:.4f} | {d['spearman_median']:.3f} | {d['spearman_defined']}/200 | "+
                     ' / '.join(f"{v['mean']*100:.2f}" for v in d['quartiles'])+f" | {d['noise_winner_agreement']*100:.2f} |")
    lines+=['','Spearman使用每个初态内部的−Accel与成功率，因此正值表示低Accel对应较高成功率。全零/全一等常量成功率不定义相关性，未强填为0。',
            '绝对Accel整体下降不能自动解释排序相关性下降：单纯正比例缩放不会改变Spearman。规范模型的区域趋势不代表最低点总是最佳；多视角模型四分位关系非单调，不能用统一绝对阈值保证成功。',
            '', '| Heldout训练 | 逐评分噪声最低点的平均结果% | 8评分均值最低点% | 低Accel前10均匀选% | RGB全查询% |',
            '| --- | ---: | ---: | ---: | ---: |']
    for m in range(2):
        d=brief.scope(m)['methods']
        lines.append('| '+MODEL_NAMES[m]+' | '+' | '.join(f"{d[k]['success']['mean']*100:.2f}" for k in ('single_accel','min_accel','top10_uniform','rgb'))+' |')
    lines+=['','前10均匀选择并未使用目标可见性。RoboCasa没有这份标签，因此不把这个结果冒充“top10+可见性”。前缀2–10的结果只作敏感性诊断，未按heldout结果重新挑主指标。',
            '', '## 任务与场景敏感性','',
            '| Heldout训练 | 不含导航：扰动不调整% | 不含导航：Accel% | 不含导航：RGB+Accel% | 全10任务：Accel恢复，布局簇95%CI（pp） |',
            '| --- | ---: | ---: | ---: | --- |']
    for m in range(2):
        d=brief.scope(m);n=d['without_navigation'];ci=d['layout_cluster_sensitivity']['min_accel']['ci95']
        lines.append('| '+MODEL_NAMES[m]+' | '+' | '.join(f"{n[k]['success']['mean']*100:.2f}" for k in ('stay_extra32','min_accel','rgb_accel'))+f' | [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] |')
    lines+=['','主结果仍含全部10任务。补充的布局簇bootstrap固定当前任务集，对共享布局成簇重采样，和主文的任务—初态bootstrap回答不同的不确定性问题。不得从两个区间中挑更有利者。',
            '每任务只有20个heldout初态，每候选每初态4次执行；大量候选不等于大量独立场景。任务级成绩与所有候选的均值见CSV，优于参考安装的候选特征见summary.json中的better_view_diagnostics（事后描述，不是训练输入）。',
            '', '## 验收与统计边界','',
            '- 两份8250步最终权重逐文件重算SHA256；冻结执行源码与计划身份核对。',
            '- 逐格检查完整性、四次完整任务、原生时限、首帧、重规划索引和每次Flow噪声地址。',
            '- 逐格校验scores.npz，并从保存的速度轨迹重算2–10步所有Accel前缀，主指标仍固定prefix=3。',
            '- 初始numeric fingerprint包含相机安装，因此只要求同候选的重复/模型之间相同；不能要求不同候选相同。本轮审计初稿误用了跨候选等同检查，已依据冻结实现纠正，旧审计失败回执保留，不是实验数据修复。',
            f'- 两模型初态输入哈希有差异的候选格：{a["paired_input_different_cells"]} / 38,800。原生渲染微差保留，不宣称像素或闭环逐位确定。',
            '- 原运行记录中相对统一渲染参考的最差检查：单通道最大差2/255、平均绝对差0.00467/255、大于1级的通道比例0.00356%。这不是“对成败绝无影响”的保证；评测保留这一原生数值分布。',
            '- 置信区间：任务—初态配对分层bootstrap；先平均候选起点/搜索随机数，不能把它们当独立场景。区间条件于已有执行噪声均值，未计训练seed变异，也不消除经验最大值偏差。',
            '- RoboCasa未保存目标语义分割，因此可见性与Accel+可见性组合无法仅靠这些缓存复现。保留LIBERO该项；RoboCasa明确缺项。',
            '- 相机候选为左右刚体架的三参数轨道；腕部安装不变，底盘动作不变。包含未认证物理安装候选，作为视觉敏感性研究保留；没有按成败删点。',
            '- 训练64格点与额外32个Sobol点只共享参数范围，不共享完全相同的采样分布。因此训练目录与额外目录之间的均值差，不能直接解释为“见过/没见过”造成的差异。',
            '- 先前几何核查中，外部相机相对参考安装的位移约0.11–0.72m，中位约0.36m，双相机基线0.7m保持不变；这不是仅原位转动或图像缩放。轨道围绕规范光轴前方1m的参考点，不是依据任务目标语义自动瞄准，也不保证目标信息可见。',
            '', '## 选择器学习流程','',
            '使用实际评分时保存的224×224 RGB。冻结ImageNet ResNet18；左右外部池化特征取平均，腕部特征单独编码。候选局部特征、已知任务ID及可选的该点Accel均值/标准差输入岭回归。',
            'PCA（外部8维、腕部4维）、标准化和岭回归均在seen数据拟合；按seen布局leave-one-layout-out选择alpha∈{1,10,100}，然后重拟合全部seen。目标为初态内部中心化的视角成功率差。heldout布局/风格不参与预处理、拟合或选超参。',
            '部署评分只依赖已查询点；无规范图、规范标记、候选ID或待查询图像。相机目录用于图导航，非连续未知空间搜索。已知任务ID意味着不声称未见任务语言泛化。',
            '', '## 每页解释与证据来源','']
    for page,(kind,index) in enumerate(order,1):
        if kind=='new':
            n=brief.notes[index];lines += [f'### 第{page}页：{n["title"]}','',n['takeaway'],'',n['speaker_note'],'']
        else:
            lines += [f'### 第{page}页：保留原LIBERO报告第{index+1}页','',
                      '原冻结数字与图表保留，仅统一页码和页脚。完整旧页解释见归档说明：'+f'[{previous.name}]({previous / "vla_view_research_progress_zh.md"})。','']
    lines += ['## 主要文件','',f'- [全矩阵审计]({root / "audit.json"})',f'- [统计与逐初态诊断]({root / "summary.json"})',
              f'- [同预算搜索结果]({root / "search-summary.json"})',f'- [逐任务表]({root / "per-task.csv"})',
              f'- [每候选任务均值与Accel]({root / "per-view.csv"})',f'- [每初态各方法成绩]({root / "per-initial-methods.csv"})',
              f'- [完整矩阵图]({root / "figures/01_full_matrix.png"})',f'- [查询预算曲线]({root / "figures/02_query_budget.png"})',
              f'- [逐任务方法热图]({root / "figures/03_task_methods.png"})',f'- [实测空间切片：不插值]({root / "figures/04_measured_slices.png"})',
              f'- [原PDF、说明和发布记录]({previous})',f'- [本版PDF与逐页预览]({output})','',
              '## 后续工作顺序（建议，未启动新评测）','',
              '1. 保留并报告P0的正负结果，不因没有复现LIBERO改善而改任务、换初态或筛掉困难候选。本轮完整性验收通过，不需要以修复名义全部重跑。',
              '2. 固定少量方法与预算：参考安装、不调整、Accel、RGB+Accel及匹配随机。若要形成方法结论，优先对这些预先固定的比较补独立执行噪声，而非97候选全部加倍；确认样本量根据当前配对方差及关心的最小效应规划。不能按新结果不断换赢家。',
              '3. 对多视角训练的跨环境差异，先审查训练覆盖密度与保留的中间检查点，再决定是否做同预算第二seed或增加训练预算的诊断；8250步跑完与充分收敛是不同判断。不能直接把当前差异归因于训练不够。',
              '4. 主动相机验证另立小规模、物理可达协议：执行策略适配、相机动作/观测更新、无作弊查询及固定不动/随机/方法对照先通过，再扩展任务内动态观察。P0的左右安装参数干预不是已经可用的真实头部动作空间。',
              '5. 可见性、深度/遮挡机制如进入方法，须补明确任务对象与可获取标签；现有RGB缓存不能冒充语义可见性真值。新框架与已有数据、权重、运行版本分开登记。','']
    return '\n'.join(lines)


def publish_report(output,public):
    output,public=Path(output),Path(public);record=read(output/'provenance.json')
    pdf=output/public.name;notes=output/public.with_suffix('.md').name
    if sha(pdf)!=record['pdf_sha256'] or sha(notes)!=record['notes_sha256']:raise ValueError('Review artifact changed')
    for source,target in [(pdf,public),(notes,public.with_suffix('.md'))]:
        temporary=target.with_name(target.name+'.building');shutil.copyfile(source,temporary);os.replace(temporary,target)
    record['status']='PUBLISHED_AFTER_VISUAL_REVIEW'
    write(output/'provenance.json',record)
    publication=read(Path(record['previous_publication'])/'publication.json')
    publication.update(scope='Unified LIBERO/RoboCasa P0 results; old report preserved',
         current=dict(archive=str(output),pdf_sha256=record['pdf_sha256'],notes_sha256=record['notes_sha256'],
                      new_closed_loops=0,newly_analyzed_closed_loops=310400),previous_publication=record['previous_publication'],
         provenance=str(output/'provenance.json'))
    write(public.parent/'publication.json',publication)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--publish',action='store_true')
    a=p.parse_args();build(a.config,publish=a.publish)
