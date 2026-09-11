#!/usr/bin/env python3
"""Ten-slide, figure-led research briefing. CPU-only; no experiment control.

Build with the project Python + matplotlib/Pillow and PyMuPDF. The historical
reports and evidence JSONs are read-only. Only explicitly named report outputs
and a QA directory are written. Numbers come from audited evidence tables or
the archived state-method CSV, not from images or simulated placeholders.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
DOCS=ROOT/"docs/dsol_paper1"
SRC=DOCS/"report_sources_20260907"
CSV=Path("/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/final-analysis/state-method-metrics.csv")
INK="#17304A"; MUTED="#627184"; BLUE="#285FAD"; TEAL="#198879"
RED="#B8515F"; AMBER="#B9822D"; GRAY="#99A4AF"; LINE="#DFE6EE"
LIGHT="#EFF4FA"; WHITE="#FFFFFF"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Brief:
    def __init__(self):
        for weight in ("Regular","Bold"):
            font_manager.fontManager.addfont(f"/usr/share/fonts/opentype/noto/NotoSansCJK-{weight}.ttc")
        plt.rcParams.update({"font.family":"Noto Sans CJK JP","pdf.fonttype":3,
                             "axes.unicode_minus":False,"font.size":15,
                             "mathtext.fontset":"dejavusans"})
        self.legacy=json.loads((SRC/"legacy_evidence.json").read_text())
        self.new_evidence=json.loads((SRC/"new_evidence.json").read_text())
        self.tables={t["id"].split("_")[0]:t for t in self.legacy["tables"]}
        with CSV.open(newline="") as f:
            self.state_rows=[r for r in csv.DictReader(f) if r["checkpoint_seed"]=="41"]
        self.notes=[]; self.qa=[]; self.figure_inputs=set()

    def text(self,x,y,s,size=17,color=INK,bold=False,ha="left",va="top"):
        t=self.fig.text(x,y,s,fontsize=size,color=color,fontweight="bold" if bold else "normal",
                        ha=ha,va=va,linespacing=1.35)
        self.artists.append(t)
        return t

    def box(self,x,y,w,h,face=LIGHT,edge=LINE,dashed=False):
        p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.007,rounding_size=0.012",
                         facecolor=face,edgecolor=edge,linewidth=1.3,
                         linestyle="--" if dashed else "-",transform=self.canvas.transAxes)
        self.canvas.add_patch(p)
        return p

    def arrow(self,a,b,color=BLUE,lw=2,style="-|>"):
        self.canvas.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=17,
                              linewidth=lw,color=color,transform=self.canvas.transAxes))

    def new(self,title,stage,subtitle,takeaway,source,note):
        self.fig=plt.figure(figsize=(13.333,7.5),facecolor=WHITE)
        self.canvas=self.fig.add_axes([0,0,1,1]); self.canvas.set_axis_off()
        self.artists=[]
        self.canvas.add_patch(Rectangle((.045,.941),.039,.009,color=BLUE,transform=self.canvas.transAxes))
        self.text(.096,.956,stage,11,MUTED)
        self.text(.956,.956,f"{len(self.notes)+1:02d} / 10",11,MUTED,ha="right")
        self.text(.045,.895,title,27,INK,True)
        if subtitle: self.text(.048,.823,subtitle,12,MUTED)
        if takeaway:
            self.canvas.add_patch(Rectangle((.046,.137),.005,.065,color=TEAL,transform=self.canvas.transAxes))
            self.text(.067,.198,takeaway,19,INK,True)
        self.text(.048,.087,source,8.8,MUTED)
        self.text(.048,.030,"VLA VIEW RESEARCH  ·  2026-09-07  ·  研究进展",8,MUTED)
        self.notes.append({"page":len(self.notes)+1,"title":title,"takeaway":takeaway,
                           "speaker_note":note,"source":source})

    def axes(self,rect,xgrid=False,ygrid=True):
        ax=self.fig.add_axes(rect)
        ax.spines[["top","right"]].set_visible(False)
        for side in ("bottom","left"): ax.spines[side].set_color(LINE)
        ax.set_axisbelow(True)
        if ygrid: ax.grid(axis="y",color=LINE,lw=.8)
        if xgrid: ax.grid(axis="x",color=LINE,lw=.8)
        ax.tick_params(length=0,labelcolor=INK,labelsize=13)
        return ax

    def photo(self,path,rect):
        p=DOCS/"figures"/path; self.figure_inputs.add(p)
        ax=self.fig.add_axes(rect); ax.imshow(Image.open(p)); ax.set_axis_off()
        return ax

    def camera(self,x,y,scale=.06,color=BLUE):
        self.canvas.add_patch(Rectangle((x,y),scale,.65*scale,facecolor=WHITE,
                                        edgecolor=color,lw=2,transform=self.canvas.transAxes))
        self.canvas.add_patch(Polygon([[x+scale,y+.08*scale],[x+1.38*scale,y-.02*scale],
                                        [x+1.38*scale,y+.68*scale],[x+scale,y+.57*scale]],
                                       closed=True,facecolor=LIGHT,edgecolor=color,lw=1.5,
                                       transform=self.canvas.transAxes))

    def save(self,pdf):
        self.fig.canvas.draw(); ren=self.fig.canvas.get_renderer(); problems=[]
        for t in self.artists:
            b=t.get_window_extent(ren).transformed(self.fig.transFigure.inverted())
            if b.x0<.01 or b.x1>.99 or b.y0<.008 or b.y1>.99:
                problems.append({"text":t.get_text(),"bbox":list(b.bounds)})
        self.qa.append({"page":len(self.notes),"issues":problems})
        pdf.savefig(self.fig,facecolor=WHITE); plt.close(self.fig)

    def page1(self,pdf):
        self.new("多视角训练后的剩余视角价值","研究问题",
                 "How to Use Camera Views in VLAs  ·  训练覆盖、视角选择与主动观测",
                 "研究问题：训练覆盖如何改变视角选择与观测获取的收益？",
                 "策略：AlphaBrain Pi0.5-LIBERO；外部＋腕部 RGB 与语言。静态位姿固定，图像随闭环更新。\n已完成实验主要研究静态外部视角；主动观测的净收益尚待验证。",
                 "研究对象是训练处理之后的条件观察价值：在给定任务、状态和输入权限下，改变观察条件是否提高闭环成功率，以及该增益是否能够事前预测。静态视角替换和实际相机移动是不同干预；后者还涉及时间、可达性和潜在物理状态变化。当前材料区分已完成的静态证据与拟开展的主动观测验证。")
        for x,label,key in ((.055,"同一状态 · 规范视角","canonical"),(.377,"候选外部视角","strong_info")):
            self.text(x,.742,label,17,BLUE,True)
            self.photo(f"m1_view_roles/{key}.png",[x,.447,.282,.242])
        self.arrow((.344,.56),(.366,.56))
        self.text(.685,.678,"是否存在闭环增益？",21,BLUE,True)
        self.text(.685,.59,"能否提前预测？",21,INK,True)
        self.text(.685,.502,"能否补偿获取成本？",21,TEAL,True)
        self.text(.059,.359,"训练视角覆盖",18,MUTED)
        self.arrow((.283,.339),(.366,.339),MUTED)
        self.text(.389,.359,"剩余视角价值",18,BLUE,True)
        self.arrow((.701,.339),(.765,.339),MUTED)
        self.text(.785,.359,"主动观测收益",18,TEAL,True)
        self.save(pdf)

    def page2(self,pdf):
        self.new("外部多视角后训练改善被动视角鲁棒性","01 · 训练覆盖与鲁棒性",
                 "七种训练配置 · Broad32 开发模型 · 训练 seed=41 · 每格 24 个配对状态",
                 "多视角覆盖改善未见位姿表现；配对与一致性未证实额外增益。",
                 "L-T02。2000次更新，每步32个模型样本；重复/配对配置的每步源记录数减半。FM为流匹配；格内为成功率（%）。",
                 "被动视角鲁棒性指不在执行中主动取景时，对外部相机位姿变化的耐受能力；不泛指背景、光照等全部视觉扰动。多视角指后训练数据覆盖多种外部位姿，不是同时增加多个外部输入。七种配置用于区分视角覆盖、样本重复、跨视角配对和一致性目标。相同物理状态上的配对评测支持本开发设置内的覆盖收益，但只有一个训练随机种子及24个状态，不构成跨模型或任务分布的普遍规律。配对FM相对同状态重复、以及一致性相对配对FM，在未见位姿上的配对差值区间均跨零，因此是未证实额外收益，而不是证明这些方法无效或有害。每步模型样本数相同不等于各配置的独立源记录数相同；数据多样性与腕部输入依赖仍是待检验解释，不能写成已确定原因。LIBERO-Plus与RoVi-Aug提供视角多样化训练的相关证据，不是本地配置的直接复现。CVC在训练与评测均屏蔽腕部的协议中报告一致性收益，但没有腕部有无与一致性有无的交互对照，不能把我们与其结果差异归因于腕部。")
        vals=np.array([[22,8,6],[20,8,4],[22,6,4],[22,21,22],[23,20,20],[23,19,19],[22,20,20]])/24*100
        labels=["规范训练","规范精确重复","图像增强","宽视角 · 实用非配对","宽视角 · 同状态重复","宽视角 · 配对 FM","宽视角 · 配对＋一致性"]
        ax=self.fig.add_axes([.31,.282,.58,.464])
        cmap=LinearSegmentedColormap.from_list("support",["#F6F4EE","#BBDBD6",TEAL])
        im=ax.imshow(vals,vmin=0,vmax=100,cmap=cmap,aspect="auto")
        ax.set_yticks(range(7),labels,fontsize=15)
        ax.set_xticks(range(3),["规范视角","范围内未见位姿","范围外位姿"],fontsize=15)
        ax.xaxis.tick_top(); ax.tick_params(length=0,pad=10)
        for spine in ax.spines.values(): spine.set_visible(False)
        for i in range(7):
            for j in range(3): ax.text(j,i,f"{vals[i,j]:.1f}",ha="center",va="center",fontsize=20,
                                      color=WHITE if vals[i,j]>75 else INK,fontweight="bold")
        ax.axhline(2.5,color=WHITE,lw=7)
        ax.add_patch(Rectangle((-.5,2.5),3,1,fill=False,edgecolor=BLUE,lw=2.5))
        cb=self.fig.colorbar(im,cax=self.fig.add_axes([.92,.31,.014,.38])); cb.set_ticks([0,50,100]); cb.ax.tick_params(labelsize=11,length=0); cb.outline.set_visible(False)
        self.text(.072,.241,"相关研究：LIBERO-Plus / RoVi-Aug（视角覆盖）；CVC（一致性，训练与评测均屏蔽腕部）",10.5,MUTED)
        self.save(pdf)

    def page3(self,pdf):
        self.new("宽视角后训练的鲁棒性与原始任务性能权衡","01 · 训练覆盖与鲁棒性",
                 "Broad64 正式模型（M-B）· 两种后训练配置各 3 seed；灰色为 Pi0.5-LIBERO 冻结参考",
                 "宽视角后训练提高 Camera 成功率，同时降低 Original 成功率。",
                 "L-T04。两基准均40任务；柱高为按episode汇总、再跨seed平均的成功率。相对参考同时增加后训练与覆盖。\nCamera任务等权增益为 +5.15pp，95% CI [0.99,9.62]；该统计量不同于柱高差。",
                 "Camera 是 LIBERO-Plus 的相机扰动子集，Original 是原始初态任务基准，不是同一评测总体。实用非配对后训练在Camera提升而在Original下降。由于相对冻结参考同时增加后训练和视角覆盖，本正式比较不能单独识别覆盖的因果作用。配对一致性配置的差异同样属于训练方案级结果，而不是独立损失消融。")
        colors=[GRAY,TEAL,AMBER]
        for j,(bench,label) in enumerate((("camera","LIBERO-Plus · Camera"),("original","Original LIBERO"))):
            vals=[r[2] for r in self.tables["T04"]["rows"] if r[0]==bench]
            ax=self.axes([.075+j*.47,.305,.38,.415])
            b=ax.bar(range(3),vals,width=.65,color=colors)
            ax.set_ylim(0,108); ax.set_yticks([0,25,50,75,100]); ax.set_xticks(range(3),["冻结参考","宽视角","配对＋一致性"],fontsize=14)
            if j==0: ax.set_ylabel("成功率（%）",fontsize=14)
            ax.set_title(label,fontsize=19,fontweight="bold",pad=18,color=INK)
            for bb,v in zip(b,vals): ax.text(bb.get_x()+bb.get_width()/2,v+2,f"{v:.2f}",ha="center",fontsize=18,fontweight="bold",color=INK)
            self.text(.075+j*.47,.254,"每模型 1599 次闭环" if j==0 else "每模型 2000 次闭环",11,MUTED)
        self.save(pdf)

    def page4(self,pdf):
        self.new("信息视角呈现局部收益，机制解释仍需对照","02 · 剩余视角价值",
                 "M1：“信息视角”指目标实体像素可见性较高的候选，不等同于真实任务信息量；中间状态恢复评测",
                 "信息视角在特定状态集上表现更好，但不能单独归因于可见性。",
                 "L-T08/T10。每幅图左外部、右腕部。A：Broad64开发模型，3任务/21状态/6演示；B：另一2任务/21状态/6演示。\nB比较三种训练覆盖，均非前页M-B。A按状态等权；B为演示等权差及95%CI。两状态集不可相减；零差值不证明等效。",
                 "信息视角在这里是按任务实体像素可见性定义的操作性名称，不是已经成立的任务信息量度。构造状态集A中，该视角为15/21成功，匹配对照为11/21成功。该状态等权差为19.05pp；按来源演示等权的原分析为13.33pp，二者统计权重不同。因而可以保留特定状态集内的正向观察，不能扩大为按像素占比选择视角的一般有效性。另一自然状态集B在两类位姿均纳入训练后，当前Info−Control差值为零；这提供训练熟悉度的竞争解释，但没有单独识别因果机制，也不证明两类视角普遍等效。第7页的新来源选择复验进一步区分局部视角对照与可迁移的选择规则。")
        for i,(key,label,col) in enumerate((("canonical","规范",GRAY),("strong_info","信息视角（高可见性）",TEAL),("matched_control","位移匹配对照",AMBER),("blind","Blind",RED))):
            x=.055+i*.236
            self.text(x,.782,label,16,col,True)
            self.photo(f"m1_view_roles/{key}.png",[x,.544,.217,.192])
        self.text(.064,.514,"A · 构造状态集：可见性对照",15,INK,True)
        ax=self.axes([.15,.282,.29,.173],True,False)
        ax.barh([1,0],[71.4286,52.381],height=.55,color=[TEAL,AMBER])
        ax.set_yticks([1,0],["可见性增加","匹配对照"],fontsize=13)
        ax.set_xlim(0,100); ax.set_xticks([0,50,100]); ax.set_xlabel("成功率（%）",fontsize=12)
        for y,v in ((1,71.4286),(0,52.381)): ax.text(v+2,y,f"{v:.1f}",va="center",fontsize=14,fontweight="bold")
        self.text(.562,.514,"B · 自然状态集：训练覆盖对照",15,INK,True)
        ax=self.axes([.716,.282,.22,.173],True,False)
        vals=[-6.9444,26.3889,0]; lows=[-29.1667,9.7222,0]; highs=[12.5,43.0556,0]
        ax.axvline(0,color=GRAY,lw=1)
        for y,v,lo,hi,c in zip([2,1,0],vals,lows,highs,[GRAY,TEAL,BLUE]):
            ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt="o",markersize=7,color=c,capsize=3,lw=2)
        ax.set_yticks([2,1,0],["Broad32","＋可见性高位姿","＋两类位姿"],fontsize=11.8)
        ax.set_xlim(-35,50); ax.set_xticks([-30,0,30]); ax.set_ylim(-.5,2.5)
        ax.set_xlabel("可见性高 − 匹配对照（pp）",fontsize=11)
        self.save(pdf)

    def page5(self,pdf):
        self.new("Accel 排序受初始噪声影响，混合选择仍待验证","03 · 视角选择的可预测性",
                 "可见性：目标实体像素比例；Accel：归一化去噪速度变化分数，作为不确定性代理，低分优先",
                 "可见性选择未在新来源复现；Accel 混合规则的正点值尚不确定。",
                 "L-T11/T12/T16。均为Broad64开发模型；点/线为相对各自规范基线的增益/95%CI。Accel仅作公式级视角排序迁移。\n左：48→36状态、24→18演示来源，各5推理噪声；右：97候选评分、96状态/24来源，六噪声仅用于排序集成。",
                 "可见性门控的原批次小正值未在新来源复现。Accel由动作去噪速度变化的归一化分数构成，本地将其迁移为低分优先的候选视角排序规则。对Broad64实用非配对开发模型，六组初始Flow噪声之间的候选排名平均Spearman为0.412，仅2/96状态在六组噪声下选中同一个Top1，支持排名及最小值视角对初噪敏感。排名变化本身不能证明行为变差，仍须闭环评测。Accel Top10内按可见性选择为83/96成功，对规范基线81/96仅增加2.08pp，95%CI[−3.13,7.29]，属于正点估计而不是已确认提升。六噪声用于排序集成，不是每个闭环条件重复六次。这里不是原工作完整失败检测流程的复现，负结果不能反驳原失败检测器。可见性复验与Accel评测使用不同状态集，效应只能相对各自基线解释。")
        self.text(.072,.753,"可见性：独立来源复验",18,INK,True)
        ax=self.axes([.193,.326,.286,.35],True,False); ax.axvline(0,color=GRAY,lw=1.2)
        rows=self.tables["T16"]["rows"][:2]
        for y,r,c in zip([1,0],rows,[BLUE,RED]):
            v=r[1]; lo,hi=r[2]
            ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt="o",color=c,capsize=6,markersize=11,lw=3)
            ax.text(v,y+.25,f"{v:+.2f}",ha="center",fontsize=19,color=c,fontweight="bold")
        ax.set_yticks([1,0],["原 24 来源","新 18 来源"],fontsize=14); ax.set_ylim(-.5,1.55); ax.set_xlim(-23,10)
        ax.set_xlabel("成功率变化（pp）",fontsize=14)
        self.text(.554,.753,"Accel：闭环选择评测",18,INK,True)
        ax=self.axes([.696,.326,.252,.35],True,False); ax.axvline(0,color=GRAY,lw=1.2)
        vals=[0,-1.0416667,2.0833333]; lows=[-6.25,-7.2916667,-3.125]; highs=[6.25,5.2083333,7.2916667]
        for y,v,lo,hi,c in zip([2,1,0],vals,lows,highs,[GRAY,BLUE,TEAL]):
            ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt="o",color=c,capsize=5,markersize=9,lw=2.5)
        ax.set_yticks([2,1,0],["单噪声","六噪声集成","Top10＋可见性"],fontsize=13); ax.set_ylim(-.55,2.6); ax.set_xlim(-10,10)
        ax.set_xlabel("成功率变化（pp）",fontsize=14)
        self.text(.554,.710,"六噪声 Top1 完全一致：2/96 状态",11.5,MUTED)
        self.text(.554,.252,"混合规则：+2.08pp，95% CI [−3.13, 7.29]",11.5,MUTED)
        self.save(pdf)

    def page6(self,pdf):
        self.new("候选视角呈现局部收益，整体增益尚待确认","02 · 剩余视角价值",
                 "v2 · Broad64 M-B（训练seed41）· 16示范来源 / 8已知任务 · 起点换外部位姿，随后固定",
                 "部分状态的实测成功率高于规范视角；整体增益未通过预设确认标准。",
                 "N05–N08。O/P/Q噪声相互独立；阶段内跨视角配对初始噪声与10步去噪网格。搜索强制选择非规范候选。\n原预设标准：增益≥5pp且95%CI下界>0。区间按任务内来源重采样，不涵盖新任务或训练seed不确定性。",
                 "O对每状态完整评估97候选，P使用独立噪声筛选并冻结候选，Q以新噪声确认。Q中选中候选相对规范位姿为7个状态正、6负、3平，支持局部观察差异；不是7个状态各自通过显著性检验，也不能据此声称普遍存在较大的稳定上限。整体增益为3.91pp，区间下界为零，未通过预设确认标准。任务固定规则只在开发状态拟定；每状态离线搜索则使用该待评状态的O/P闭环成败标签，二者信息权限不同。每状态候选强制为非规范位姿，没有可靠的保持选项。Q估计该有限搜索程序选中项的期望，而非整个候选域的真实最大值。置信区间在固定任务内重采来源，不能覆盖所有泛化不确定性。")
        for x,a,b in ((.071,"O · 全 97 候选","32 次噪声重复"),(.38,"P · 筛选并冻结","9 候选 · 32 新噪声"),(.69,"Q · 独立确认","冻结方法 · 64 新噪声")):
            self.box(x,.667,.225,.095,face=LIGHT,edge=LINE)
            self.text(x+.014,.741,a,15,BLUE,True); self.text(x+.014,.701,b,11,MUTED)
        self.arrow((.304,.711),(.364,.711),MUTED)
        self.arrow((.615,.711),(.675,.711),MUTED)
        ax=self.axes([.092,.291,.52,.31])
        vals=[59.08203125,60.44921875,62.98828125]
        ax.bar(range(3),vals,color=[GRAY,BLUE,TEAL],width=.65)
        for i,v in enumerate(vals): ax.text(i,v+3,f"{v:.2f}",ha="center",fontsize=21,fontweight="bold")
        ax.set_ylim(0,100); ax.set_yticks([0,25,50,75,100]); ax.set_xticks(range(3),["规范视角","每任务固定","每状态离线搜索"],fontsize=13)
        ax.set_ylabel("成功率（%）",fontsize=14)
        self.text(.69,.577,"+3.91 pp",32,TEAL,True)
        self.text(.691,.482,"相对规范视角",17,INK)
        self.text(.691,.428,"95% CI  [0.00, 7.81]",16,MUTED)
        self.text(.691,.345,"区间包含零增益\n未通过预设标准",17,INK,True)
        self.text(.09,.243,"任务固定：开发集拟定；状态搜索：使用该待评状态的 O/P 闭环标签。",11,MUTED)
        self.save(pdf)

    def page7(self,pdf):
        self.new("尚未证实状态选择优于任务固定基线","03 · 视角选择的可预测性",
                 "同一v2测试集：左为离线搜索对照，右为两类岭回归选择器；两面板的参考基线不同",
                 "局部收益尚未转化为稳定选择增益；不能据此认定视角价值不可学习。",
                 "N08–N10 / v2 CSV。左为事后补充：+2.54pp，95% CI [−0.68,5.76]；右为原Q确认结果。\n“候选图可查询”额外使用候选图像，不能视为只根据起始图像预测未见视角；两个选择器均未通过预设标准。",
                 "来源顺序由状态键固定，不按观测收益筛选。搜索相对任务固定为5正6负5平；相对规范视角的7正6负3平是另一个比较。均值+2.54pp的事后区间包含零，尚不足以证明任务内部的状态选择优势。右侧选择器只用允许特征，不使用测试闭环标签，但候选图版本拥有额外图像权限。两个岭回归基线的失败不构成所有方法不可学习的证明。")
        grouped={}
        for r in self.state_rows: grouped.setdefault(r["pair_key"],{})[r["method"]]=float(r["success"])
        ordered=sorted(grouped)
        delta=np.array([100*(grouped[k]["statewise_P_top1"]-grouped[k]["task_fixed"]) for k in ordered])
        assert len(delta)==16 and np.isclose(delta.mean(),2.5390625)
        assert (int((delta>0).sum()),int((delta<0).sum()),int((delta==0).sum()))==(5,6,5)
        self.text(.064,.749,"逐来源：离线搜索 − 任务固定",16,INK,True)
        ax=self.axes([.096,.327,.427,.35])
        colors=[TEAL if v>0 else RED if v<0 else GRAY for v in delta]
        ax.bar(range(16),delta,color=colors,width=.74)
        ax.axhline(0,color=GRAY,lw=1)
        for i,v in enumerate(delta):
            if v==0: ax.plot(i,0,"_",color=GRAY,markersize=8,mew=2)
        ax.set_xlim(-.7,15.7); ax.set_ylim(-13,46); ax.set_yticks([-10,0,20,40]); ax.set_xticks([0,7,15],["来源 1","8","16"])
        ax.set_ylabel("增益（pp）",fontsize=14)
        ax.text(.025,.95,"5 正 / 6 负 / 5 平",transform=ax.transAxes,va="top",fontsize=12,color=MUTED)
        self.text(.098,.282,"平均 +2.54pp  [−0.68, 5.76]",15,MUTED)
        self.text(.59,.749,"选择器：相对规范视角",17,INK,True)
        ax=self.axes([.734,.355,.208,.29],True,False); ax.axvline(0,color=GRAY,lw=1)
        for y,v,lo,hi,c in ((1,-.9765625,-1.953125,0,RED),(0,.48828125,-2.63671875,3.61328125,BLUE)):
            ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt="o",color=c,capsize=6,lw=3,markersize=10)
        ax.set_yticks([1,0],["几何 / 上下文","候选图可查询"],fontsize=13); ax.set_ylim(-.6,1.7); ax.set_xlim(-4,5); ax.set_xticks([-4,0,4]); ax.set_xlabel("增益（pp）",fontsize=14)
        self.text(.601,.284,"两类岭回归未显示可靠增益",14,MUTED)
        self.save(pdf)

    def page8(self,pdf):
        self.new("训练覆盖如何改变额外观察需求？","04 · 研究假设与验证设计",
                 "相关研究分别考察训练覆盖、已有观测的表征利用，以及主动观测获取",
                 "研究目标：识别观察收益的适用条件，并验证其样本外预测能力。",
                 "代表研究：RoboNVS / Moving Eye；ReconVLA / ActiveVLA；LIME / SaPaVe / TAVIS。名称链接主来源。\n多视角训练、主动观测及普通价值回归本身均已有研究；本研究的条件关系尚待实证确认。",
                 "现有研究已分析多视角训练、当前证据的重表达及主动相机运动，包括匹配训练预算和条件性主动收益。本研究拟检验训练覆盖是否改变额外观察的增量价值，以及这种变化是否由预先定义的任务证据条件解释。该定位不是首次性声明；潜在贡献取决于可重复的训练交互、独立条件验证和可实际实现的预测收益。")
        headers=["训练视角覆盖","已有观测表征利用","主动观测获取"]
        refs=["RoboNVS / Moving Eye","ReconVLA / ActiveVLA","LIME / SaPaVe / TAVIS"]
        for i in range(3):
            x=.064+i*.308; self.box(x,.469,.269,.275,face=WHITE,edge=LINE)
            self.text(x+.016,.723,headers[i],18,INK,True)
            if i==0:
                for k in range(3):
                    self.canvas.add_patch(Rectangle((x+.035+k*.046,.558+k*.012),.084,.072,facecolor=[LIGHT,"#DDE8F7","#C5DAD8"][k],edgecolor=BLUE,lw=1.1,transform=self.canvas.transAxes))
            elif i==1:
                self.canvas.add_patch(Rectangle((x+.033,.552),.09,.075,facecolor=LIGHT,edgecolor=GRAY,transform=self.canvas.transAxes))
                self.canvas.add_patch(Rectangle((x+.055,.57),.039,.03,fill=False,edgecolor=TEAL,lw=2,transform=self.canvas.transAxes))
                self.arrow((x+.133,.59),(x+.181,.59),TEAL)
                self.canvas.add_patch(Rectangle((x+.191,.547),.056,.091,facecolor="#D9ECE7",edgecolor=TEAL,transform=self.canvas.transAxes))
            else:
                self.camera(x+.025,.568,.037,GRAY); self.camera(x+.183,.568,.037,TEAL)
                self.arrow((x+.094,.588),(x+.171,.588),TEAL)
            self.text(x+.016,.511,refs[i],11.5,MUTED)
        self.arrow((.202,.451),(.42,.367),BLUE)
        self.arrow((.51,.451),(.52,.376),BLUE)
        self.arrow((.817,.451),(.628,.367),BLUE)
        self.box(.273,.27,.456,.117,face=LIGHT,edge=BLUE,dashed=True)
        self.text(.501,.364,"待检验：训练覆盖与任务证据\n如何影响额外观察收益？",18,BLUE,True,ha="center")
        self.save(pdf)

    def page9(self,pdf):
        self.new("匹配训练对照下的观察价值验证","04 · 研究假设与验证设计",
                 "拟议对照：同初始权重、数据记录、2000次更新及学习率计划；训练seed=41，仅改变外部训练视角",
                 "训练交互＝宽视角模型的观察增益 − 规范训练模型的观察增益。",
                 "截点：2026-09-07 07:50 UTC；规范训练已结束，产物验收与匹配评测待完成。旧v2数值仅为历史参照。\n各模型相对自身预设基线估计增益；静态与主动分别计算，不将两列相减解释移动成本。新结论需独立来源确认。",
                 "目标估计量是两种训练处理的观察增益之差，而不只是比较不同模型的绝对成功率。静态实验须匹配候选域、搜索预算、参考规则与噪声；主动实验须匹配可行动作域、输入权限及全局执行预算。静态配置和主动获取是不同干预，不能直接相减当作运动成本。证据条件在观察结果前定义。历史Broad分数不替代新协议的独立确认，规范模型最终权重身份验收及主动执行契约仍待完成。")
        xs=[.33,.544,.758]; ys=[.55,.395]
        for x,label in zip(xs,["固定参考","静态搜索","实际获取"]): self.text(x+.09,.745,label,18,INK,True,ha="center")
        for y,label,sub in zip(ys,["规范视角训练","宽视角训练"],["2000步训练完成","Broad64 模型"]):
            self.text(.068,y+.073,label,18,INK,True); self.text(.068,y+.018,sub,12,MUTED)
        labels=[["待测","待测","待测"],["59.08%","62.99%","待测"]]
        for r in range(2):
            for c in range(3):
                known=(r==1 and c<2)
                self.box(xs[c],ys[r]-.02,.18,.124,face="#E6F2EE" if known else WHITE,edge=TEAL if known else GRAY,dashed=not known)
                self.text(xs[c]+.09,ys[r]+.069,labels[r][c],25,TEAL if known else GRAY,True,ha="center")
                if known: self.text(xs[c]+.09,ys[r]+.013,"历史 v2",10,MUTED,ha="center")
        self.text(.07,.303,"证据条件",14,MUTED,True)
        for x,label in ((.258,"相同证据 · 不同投影"),(.51,"任务证据揭示"),(.737,"无新增证据对照")):
            self.box(x,.267,.2,.058,face=LIGHT,edge=LINE); self.text(x+.1,.311,label,13,INK,ha="center")
        self.save(pdf)

    def page10(self,pdf):
        self.new("基于预期收益的观测获取策略：方法原型","04 · 研究假设与验证设计",
                 "待验证原型：由允许输入预测动作级观察增益；选择与阈值仅在开发阶段拟定",
                 "条件规律与可部署收益尚待匹配实验及独立来源验证。",
                 "首轮范围：仿真运动学外部相机；基线为直接继续、开发集固定查询与匹配等待。不是实际头部或腕部验证。\n选择器不读取未获取的候选图；运动与推理计入统一预算，规划失败与超时保留在全部尝试的分母中。优势回归不是新算法。",
                 "选择器输入包括当前外部及腕部图像、语言、协议允许的状态和标定几何，以及候选动作参数与成本。开发标签为同状态、配对噪声、同总执行预算下，获取动作相对直接继续的平均成功率差。原型选择预测增益最高的可行动作，增益未达到开发期冻结阈值则保持。除直接继续外，还须比较开发集固定查询、随机或几何查询及同门控等待控制。训练交互、证据条件与部署收益均需在未参与开发的来源和完整任务上确认；普通优势回归只是方法基线。")
        self.box(.063,.504,.205,.19,face=LIGHT)
        self.text(.165,.663,"选择器输入",19,INK,True,ha="center")
        self.text(.165,.605,"外部 / 腕部图像＋语言\n允许的状态与几何\n候选动作及成本",12.8,MUTED,ha="center")
        self.arrow((.279,.604),(.335,.604))
        self.box(.35,.504,.249,.19,face="#E5F0FA",edge=BLUE,dashed=True)
        self.text(.474,.663,"预测配对观察增益",18,BLUE,True,ha="center")
        self.text(.474,.605,"监督：取景−继续的成功率差\n同状态、同噪声、同总预算\n方法基线：优势回归",12.5,MUTED,ha="center")
        self.arrow((.61,.628),(.694,.671),TEAL)
        self.arrow((.61,.576),(.694,.518),AMBER)
        self.box(.71,.624,.219,.096,face="#E6F2EE",edge=TEAL)
        self.text(.82,.708,"选择预测增益最高的\n可行动作",15,TEAL,True,ha="center")
        self.box(.71,.461,.219,.096,face="#FBF3E4",edge=AMBER)
        self.text(.82,.532,"增益未达阈值：保持",16,AMBER,True,ha="center")
        self.text(.069,.363,"训练覆盖效应",19,INK,True)
        self.text(.378,.363,"任务证据条件",19,INK,True)
        self.text(.7,.363,"独立来源部署收益",19,INK,True)
        self.text(.07,.306,"匹配训练对照",13,MUTED)
        self.text(.38,.306,"预定义证据干预",13,MUTED)
        self.text(.701,.306,"成功率、时间与查询成本",13,MUTED)
        self.save(pdf)

    def build(self,output,qa_dir):
        import pymupdf
        output=output.resolve(); qa_dir=qa_dir.resolve(); qa_dir.mkdir(parents=True,exist_ok=True)
        protected=[DOCS/"view_revalidation_stage1_integrated_v5_20260826_zh.pdf",DOCS/"view_value_new_experiments_zh.pdf",DOCS/"vla_view_research_unified_20260907_zh.pdf"]
        if output in protected: raise ValueError("Historical reports are protected; choose a new focus filename")
        old_hashes={str(p):sha(p) for p in protected}
        draft=qa_dir/"draft.pdf"
        with PdfPages(draft) as pdf:
            for i in (1,2,3,4,6,7,5,8,9,10): getattr(self,f"page{i}")(pdf)
        doc=pymupdf.open(draft)
        doc.set_toc([[1,n["title"],n["page"]] for n in self.notes])
        doc.set_metadata({"title":"多视角训练后的剩余视角价值｜VLA研究进展","author":"VLA View Research","subject":"训练覆盖、静态视角选择与主动观测验证"})
        links={"RoboNVS":"2603.26757","Moving Eye":"2607.02322","ReconVLA":"2508.10333","ActiveVLA":"2601.08325","LIME":"2607.02417","SaPaVe":"2603.12193","TAVIS":"2605.07943","Accel":"2607.27933","CVC":"2608.06965","LIBERO-Plus":"2510.13626","RoVi-Aug":"https://rovi-aug.github.io/"}
        for name,key in links.items():
            for page in doc:
                for r in page.search_for(name): page.insert_link({"kind":pymupdf.LINK_URI,"from":r,"uri":key if key.startswith("https://") else "https://arxiv.org/abs/"+key})
        output.parent.mkdir(parents=True,exist_ok=True); doc.save(output,garbage=4,deflate=True); doc.close()
        doc=pymupdf.open(output)
        contact=Image.new("RGB",(1600,5*470),"#DDE5ED"); draw=ImageDraw.Draw(contact)
        for i,p in enumerate(doc):
            png=qa_dir/f"page-{i+1:02d}.png"; p.get_pixmap(matrix=pymupdf.Matrix(1.35,1.35)).save(png)
            im=Image.open(png).convert("RGB"); im.thumbnail((780,439))
            x=(i%2)*800+10; y=(i//2)*470+25; contact.paste(im,(x,y)); draw.text((x,y-19),f"Page {i+1}",fill=INK)
        contact.save(qa_dir/"contact.jpg",quality=93)
        receipt={"pdf":str(output),"pdf_sha256":sha(output),"pages":len(doc),"bookmarks":len(doc.get_toc()),
                 "layout_issues":self.qa,"characters_per_page":[len(p.get_text()) for p in doc],
                 "history_hashes":old_hashes,"source_hashes":{str(p):sha(p) for p in [SRC/"legacy_evidence.json",SRC/"new_evidence.json",CSV,Path(__file__),*sorted(self.figure_inputs)]},
                 "source_editing_note":"Figures are redrawn from archived numbers. Historical PDF files and image assets are unchanged. No experimental run was launched.",
                 "training_snapshot_note":"2026-09-07T07:50:02Z read-only: 2000 updates complete, final files present; no final weight-content hash or new eval in this report task."}
        doc.close()
        assert old_hashes=={str(p):sha(p) for p in protected}
        (qa_dir/"qa.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
        (SRC/"focus_brief_build_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
        notes=["# VLA视角利用研究｜图表说明","","本说明记录各图的实验设置、统计解释与论证边界。已完成结果与拟议实验分别陈述。",""]
        for n in self.notes:
            notes.extend([f"## {n['page']:02d} · {n['title']}","",n["takeaway"],"",n["speaker_note"],"","依据："+n["source"],""])
        output.with_suffix(".md").write_text("\n".join(notes)+"\n")
        print(json.dumps({"pdf":str(output),"pages":len(self.notes),"issues":sum(len(x["issues"]) for x in self.qa),"qa_dir":str(qa_dir)},ensure_ascii=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,default=DOCS/"vla_view_research_focus_20260907_zh.pdf")
    ap.add_argument("--qa-dir",type=Path,required=True)
    a=ap.parse_args(); Brief().build(a.output,a.qa_dir)


if __name__=="__main__": main()
