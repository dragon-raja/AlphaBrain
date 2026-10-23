"""Append the search progress briefing to the current full PDF, using records only.

The first 33 pages are retained; only their page counters change. Rebuilding
replaces this appendix rather than appending it twice. No historical PDF copies,
model imports, inference, training, or W&B publication are performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle,FancyArrowPatch
import numpy as np
import pymupdf as fitz
from PIL import Image,ImageDraw
from .continuous_training_tracking import update_rows

INK='#17304A';MUTED='#627184';BLUE='#285FAD';TEAL='#198879'
RED='#B8515F';AMBER='#B9822D';GRAY='#99A4AF';LINE='#DFE6EE';LIGHT='#EDF3FA'
MODELS=['canonical','broad','native_reference']
LABELS=['规范后训练','多视角后训练','原生参考']
COLORS=[BLUE,TEAL,AMBER]
METHODS=['stay','local_near_best_accel','continuous_ppo']
METHOD_NAMES=['不调整','局部Accel','连续PPO']
BASE_PAGES=33;TOTAL=39


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Evidence:
    def __init__(self,root):self.root=Path(root);self.sources={}
    def read(self,path):
        path=Path(path).resolve();self.sources[str(path)]=sha(path)
        return json.loads(path.read_text())
    def load(self):
        r=self.root;p=r/'runs/continuous-ppo-model-conditions-20261022'
        fixed=self.read(r/'current/paired-audit/summary.json')
        common=self.read(r/'current/paired-audit/common-horizon.json')
        old=self.read(r/'current/continuous-evaluation/results/evaluation/status.json')
        roots=[r/'runs/continuous-ppo-four-worker-20261022/results/training',
               p/'results/broad/training',p/'results/native_reference/training']
        training={}
        for model,root in zip(MODELS,roots):
            status=self.read(root/'status.json');rows=update_rows(root)
            assert status['status']=='COMPLETE' and status['episodes']==512 and len(rows)==16
            for pattern in ('updates/*.json','episodes/*.json'):
                for path in root.glob(pattern):self.sources[str(path.resolve())]=sha(path)
            training[model]=dict(status=status,rows=rows)
        fleet=r/'runs/continuous-evaluation-fleet-preparation-20261023'
        campaign=self.read(fleet/'protocol/release-68f2cd8/campaign.json')
        assert campaign['budget']['task_episodes']==672
        assert not list((fleet/'results').glob('*/blocks/*/complete.json')),'Update pending-evaluation statements first'
        assert old['tasks']==192 and old['summary']['complete_pairs']==64
        return dict(fixed=fixed,common=common,old=old,training=training,campaign=campaign)


class Slide:
    def __init__(self,page,title,stage,subtitle,takeaway,foot):
        self.page=page;self.title=title;self.texts=[];self.tables=[]
        self.fig=plt.figure(figsize=(13.333,7.5),facecolor='white')
        self.canvas=self.fig.add_axes([0,0,1,1]);self.canvas.axis('off')
        self.canvas.add_patch(Rectangle((.045,.941),.039,.009,color=BLUE))
        self.text(.096,.956,stage,10.5,MUTED)
        self.text(.045,.895,title,25,INK,True)
        self.text(.048,.82,subtitle,11,MUTED)
        self.canvas.add_patch(Rectangle((.046,.13),.005,.065,color=TEAL))
        self.text(.067,.19,takeaway,16,INK,True)
        self.text(.048,.098,foot,8.2,MUTED)
        self.text(34/959.976,.028,'VLA 研究进展 · 2026-10-23 · 搜索方法与连续位姿验证',7,MUTED)
        self.text(925.5/959.976,.032,f'{page:02d} / {TOTAL}',8,MUTED,ha='right')
    def text(self,x,y,text,size=12,color=INK,bold=False,**kw):
        t=self.fig.text(x,y,text,fontsize=size,color=color,weight='bold' if bold else 'normal',va='top',linespacing=1.35,**kw)
        self.texts.append(t);return t
    def ax(self,rect):
        ax=self.fig.add_axes(rect);ax.spines[['top','right']].set_visible(False)
        for side in ('left','bottom'):ax.spines[side].set_color(LINE)
        ax.tick_params(length=0,labelsize=9,labelcolor=INK);ax.grid(axis='y',color=LINE,lw=.7);ax.set_axisbelow(True)
        return ax
    def table(self,rect,rows,headers,widths,size=10):
        ax=self.fig.add_axes(rect);ax.axis('off')
        t=ax.table(cellText=rows,colLabels=headers,colWidths=widths,loc='center',cellLoc='center')
        t.auto_set_font_size(False);t.set_fontsize(size);t.scale(1,1.8)
        for (row,col),cell in t.get_celld().items():
            cell.set_edgecolor(LINE);cell.set_facecolor(LIGHT if row==0 else 'white');cell.get_text().set_color(INK)
        self.tables.append(t);return t
    def box(self,x,y,w,h,title,body='',color=BLUE):
        self.canvas.add_patch(Rectangle((x,y),w,h,facecolor=LIGHT,edgecolor=LINE))
        self.text(x+.012,y+h-.018,title,13,color,True)
        if body:self.text(x+.012,y+h-.065,body,10.5)
    def arrow(self,a,b,color=BLUE):
        self.canvas.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color=color,lw=1.5))
    def finish(self,pdf):
        self.fig.canvas.draw();ren=self.fig.canvas.get_renderer();issues=[]
        for text in self.texts:
            b=text.get_window_extent(ren).transformed(self.fig.transFigure.inverted())
            if b.x0<.01 or b.x1>.99 or b.y0<.008 or b.y1>.99:issues.append(dict(text=text.get_text(),bounds=list(b.bounds)))
        for tab in self.tables:
            for cell in tab.get_celld().values():
                tb=cell.get_text().get_window_extent(ren);cb=cell.get_window_extent(ren)
                if tb.width>cb.width-3 or tb.height>cb.height-2:issues.append(dict(cell=cell.get_text().get_text()))
        if issues:raise ValueError(dict(page=self.page,layout=issues))
        pdf.savefig(self.fig);plt.close(self.fig)
        return dict(page=self.page,title=self.title,layout_issues=[])


def fixed_slide(data,pdf):
    summary=data['fixed'];aggregate={}
    for cohort in ('A','B'):
        for method in ('stay','local_accel','ppo'):
            rows=[r for r in summary['table'] if r['cohort']==cohort and r['method']==method]
            n=sum(r['n'] for r in rows)
            aggregate[cohort,method]=dict(n=n,success=sum(r['successes'] for r in rows)/n,
                queries=sum(r['queries']*r['n'] for r in rows)/n,
                harm=sum(r['harm'] for r in rows)/n,rescue=sum(r['rescue'] for r in rows)/n)
    s=Slide(34,'固定97候选：搜索收益已在真实任务中复核','从评分分析转向有限预算搜索 · 已完成',
        '最终仍按已访问最低Accel选位；PPO学习访问顺序与停止，查询上限8，不读取未访问视角的图像。',
        '固定候选内存在搜索收益；候选表外的连续位姿能力仍需重新验证。',
        'A：8已见任务、16留出初态×32起点×8执行种子；B：8搜索器未见任务、16初态×8起点×8种子。时限300步。\n本图为新真实执行，非旧标签回放；任务已被分析过，不是盲测。伤害＝不调整成功而换位失败，占全部配对执行。')
    ax=s.ax([.08,.34,.41,.39]);x=np.arange(2)
    for j,(method,label,col) in enumerate(zip(('stay','local_accel','ppo'),('不调整','局部Accel','PPO'),(GRAY,BLUE,TEAL))):
        vals=[aggregate[c,method]['success']*100 for c in ('A','B')]
        bars=ax.bar(x+(j-1)*.23,vals,.21,label=label,color=col)
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=10,color=INK)
    ax.set_xticks(x,['A：已见任务','B：未见任务']);ax.set_ylim(0,85);ax.set_ylabel('真实任务成功率（%）',fontsize=11)
    ax.legend(frameon=False,ncol=3,fontsize=9,loc='upper left')
    rows=[]
    for c in ('A','B'):
        for m,l in [('local_accel','局部Accel'),('ppo','PPO')]:
            v=aggregate[c,m];rows.append([c,l,f"{v['queries']:.2f}",f"{100*v['rescue']:.1f}%",f"{100*v['harm']:.1f}%"])
    s.table([.54,.49,.40,.25],rows,['组别','方法','查询数','挽救','伤害'],[.12,.28,.2,.2,.2])
    s.text(.55,.45,'保留反例与执行差异',12,RED,True)
    s.text(.55,.405,'B双壶上灶：三方法均为0%。\nB汤罐入篮：PPO 88.3%，低于局部89.8%。\nA与旧回放对齐时限后，仍有327/12,288\n条成败不同；搜索路径一致不等于执行逐条一致。',10.5)
    s.text(.08,.265,'A每方法4,096次；B每方法1,024次。不同起点与执行种子不是同等数量的独立任务。',10,MUTED)
    return s.finish(pdf)


def interface_slide(pdf):
    s=Slide(35,'连续PPO：直接输出相机角度增量，不再查97点表','连续空间方法 · 已实现',
        '操作模型和SigLIP冻结；只训练搜索actor与独立critic。动态局部候选保留为非学习局部Accel基线。',
        '学习的是“下一次看哪里、何时停止”；最终选位仍使用原Accel。',
        '训练仅在固定物理状态下移动虚拟相机、查询模型，不执行机械臂任务、不读成功标签。每查询batch=1、两条固定配对噪声。\n固定光心、yaw±30°／pitch±20°、单次每轴≤10°；这是外部相机代理，尚非真实头部运动学。整体含RL训练，不称training-free。')
    s.box(.06,.58,.24,.17,'真实连续渲染','外部RGB＋腕部RGB＋任务\n相机位姿可以落在旧97点之外')
    s.box(.36,.58,.26,.17,'冻结π0.5：一次查询','复用SigLIP特征\n提取Accel均值与波动')
    s.box(.68,.58,.26,.17,'搜索actor：历史attention','已访问记录＋角度／限位／预算\n输出STOP或连续Δyaw／Δpitch')
    s.arrow((.30,.665),(.36,.665));s.arrow((.62,.665),(.68,.665))
    s.arrow((.81,.58),(.81,.52));s.arrow((.81,.52),(.18,.52));s.arrow((.18,.52),(.18,.58))
    s.text(.365,.512,'继续搜索：执行相机增量，再查询',10.5,BLUE)
    s.box(.06,.29,.29,.15,'critic：仅训练使用','独立历史编码＋价值头\n不读完整评分表或任务成功标签',TEAL)
    s.box(.40,.29,.25,.15,'STOP／预算结束','回到已观察最低Accel位置\n必要返回复核也占查询额度',TEAL)
    s.box(.70,.29,.24,.15,'任务评测阶段','冻结操作模型执行任务\n独立统计成功、挽救与伤害',TEAL)
    s.arrow((.65,.365),(.70,.365),TEAL)
    s.text(.065,.245,'训练奖励＝最佳Accel改善 / 冻结尺度 − 0.01×新增查询数；每回合最多8次，含起始与返回复核。',10.5)
    return s.finish(pdf)


def first_continuous_slide(data,pdf):
    scopes=data['old']['summary']['scopes'];allrows=scopes['all']
    taskkeys=['goal_cream_cheese_bowl','goal_wine_rack',
        'pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate',
        'pick_up_the_alphabet_soup_and_place_it_in_the_basket']
    s=Slide(36,'连续首版：任务收益明显，停止效率仍有问题','首版连续PPO · 192条真实配对任务已完成',
        '4任务×2初态×2起始朝向×4执行种子；每方法64次。此页是旧首版actor，不是后续三组修正版成绩。',
        '保留首版收益；先修正STOP训练项和初态采样，再做同预算匹配复测。',
        '奶酪／酒瓶300步，黑碗220步，汤罐280步；同操作模型、相同起点与评分／执行噪声。伤害和挽救为相对不调整的配对条数。\n四任务、单个RL训练种子的开发证据，不是完整benchmark或独立确认；两个修复同时实施，不能分别归因其效果。')
    ax=s.ax([.08,.41,.46,.32]);x=np.arange(4)
    for j,(m,label,col) in enumerate(zip(METHODS,METHOD_NAMES,[GRAY,BLUE,TEAL])):
        vals=[scopes[t][m]['success_rate']*100 for t in taskkeys]
        bars=ax.bar(x+(j-1)*.23,vals,.21,label=label,color=col)
        ax.bar_label(bars,fmt='%.0f',fontsize=9,padding=2)
    ax.set_xticks(x,['奶酪入碗','酒瓶入架','黑碗放盘','汤罐入篮']);ax.set_ylim(0,115);ax.set_yticks([0,25,50,75,100])
    ax.set_ylabel('成功率（%）；每任务每方法16次',fontsize=10)
    ax.legend(frameon=False,fontsize=9,ncol=3,loc='upper left')
    rows=[[name,f"{allrows[m]['success']}/64",f"{100*allrows[m]['success_rate']:.1f}%",f"{allrows[m]['mean_queries']:.2f}",f"{allrows[m]['harm']} / {allrows[m]['rescue']}"] for m,name in zip(METHODS,METHOD_NAMES)]
    s.table([.58,.48,.37,.25],rows,['方法','成功数','成功率','查询','伤害/挽救'],[.25,.19,.19,.15,.22],9)
    s.text(.59,.435,'局部搜索不保证净收益',12,RED,True)
    s.text(.59,.39,'局部Accel：12次挽救被12次伤害抵消。\n连续PPO仍有2次伤害，平均查询7.75/8。',10.5)
    s.text(.08,.31,'明确问题 → 同预算修正',12,BLUE,True)
    s.text(.08,.267,'旧探索项受角度单位影响STOP梯度，1,586份决策输入未主动STOP；每批仅单初态。\n修正探索项；改为每批16初态各2回合。仍512回合／16更新，从相同未训练初始化重训。',10.5)
    return s.finish(pdf)


def curves_slide(data,pdf):
    s=Slide(37,'修正版训练：三种操作模型出现不同搜索行为','512回合／16次更新 · 全部完成，actor已冻结',
        '每点为当批32回合均值；每批包含相同16初态。显示原始批次曲线，不做平滑或挑选最佳检查点。',
        '规范组继续探索；多视角与原生组较早停止。是否提高成功率仍待任务评测。',
        '奖励＝Accel改善／固定尺度−查询成本，不是成功率。STOP比例是随机训练回合的结束原因，不等于确定性部署STOP比例。\n每条件仅一个RL训练种子、16次更新；不能据此认定收敛，亦不能用跨模型Accel绝对值排列操作能力。')
    panels=[('search/mean_total_reward_including_initial','回合奖励（含起始查询成本）',1),
        ('search/mean_accel_improvement','起始Accel − 已访最低Accel',1),
        ('search/mean_queries','实际评分查询数 / 回合（上限8）',1),
        ('search/stop_fraction','以主动STOP结束的训练回合（%）',100)]
    rects=[[.085,.535,.38,.20],[.57,.535,.38,.20],[.085,.285,.38,.17],[.57,.285,.38,.17]]
    for index,((key,title,scale),rect) in enumerate(zip(panels,rects)):
        ax=s.ax(rect)
        for model,label,col in zip(MODELS,LABELS,COLORS):
            rows=data['training'][model]['rows'];ax.plot([r['updates'] for r in rows],[r[key]*scale for r in rows],marker='o',ms=3,lw=1.6,color=col,label=label)
        ax.set_title(title,fontsize=11,loc='left',color=INK);ax.set_xlim(.7,16.3);ax.set_xticks([1,4,8,12,16])
        if index<2:ax.axhline(0,color=GRAY,lw=.6,ls='--')
        if index==2:ax.set_ylim(0,8.3)
        if index==3:ax.set_ylim(0,108)
        if index>=2:ax.set_xlabel('PPO更新次数（每次32回合）',fontsize=9)
        if index==0:ax.legend(frameon=False,fontsize=8.5,ncol=3,loc='upper left')
    return s.finish(pdf)


def models_slide(data,pdf):
    s=Slide(38,'三组训练完成；修正版的任务收益尚待验证','模型对照 · 训练结果与任务结果分开报告',
        '分别训练搜索器，操作模型均冻结。目的是验证适用性，不是同一个actor零样本跨模型迁移。',
        '训练曲线支持搜索行为不同；尚不能判断哪组能稳定救回任务、减少换位伤害。',
        '规范／多视角模型均为项目2,000步后训练；原生参考使用原始转换权重及原生预处理／解码，内部32维、输出7维。\n三组各保留最后更新；各自依赖栈与解码有效但不完全相同，不是仅后训练因素的严格因果实验。原项目零步解码失配方案已排除。')
    rows=[]
    for model,label in zip(MODELS,LABELS):
        status=data['training'][model]['status'];tail=data['training'][model]['rows'][-4:]
        rows.append([label,f"{status['queries']:,}",f"{status['queries']/512:.2f}",
            f"{sum(x['search/mean_queries'] for x in tail)/4:.2f}",f"{status['elapsed_seconds']/60:.1f}分钟",'待评测'])
    s.table([.06,.51,.89,.24],rows,['操作模型条件','训练总查询','全程查询/回合','末4批查询/回合','训练耗时','新真实成绩'],[.2,.15,.17,.19,.14,.15],10.5)
    s.box(.06,.28,.28,.17,'规范后训练：收益信号上升','末4批奖励均值 +0.179\n查询增加到5.26次；尚无收敛证据',BLUE)
    s.box(.365,.28,.28,.17,'多视角后训练：净奖励近零','末4批奖励均值 +0.004\n查询约2.38次；任务收益尚不明',TEAL)
    s.box(.67,.28,.28,.17,'原生参考：倾向少查询','末4批奖励均值 −0.012\n查询约1.23次；不能当成任务失败',AMBER)
    s.text(.07,.235,'耗时仅描述运行：原生组使用8个驻留worker，另两组为4个批后退出worker；模型和查询量也不同。',10,MUTED)
    return s.finish(pdf)


def next_slide(pdf):
    s=Slide(39,'下一步：冻结actor，先完成三模型匹配任务评测','待执行方案 · 先迁移center，再恢复准入',
        '每个操作模型分别比较不调整、局部Accel、连续PPO，并补规范视角对照；不追加训练、不扩对照。',
        '先完成控制中心迁移与真实准入，再以成功率、查询成本和换位伤害判断方法。',
        '每模型：8初态×2起始朝向×3方法×4种子＝192任务；规范视角8初态×4种子＝32任务。三模型共672任务、最多38,592条去噪轨迹。\n旧97点成绩不能替代新版基线。只读资源检查和CPU验证已通过，真实首块／八GPU容量准入未做；失败即停，不自动恢复旧队列。')
    s.table([.06,.50,.57,.25],[['奶酪入碗','A 2/3','已见任务、留出初态','300'],
        ['酒瓶入架','A 10/11','已见任务、留出初态','300'],
        ['黑碗放盘','B 0/1','搜索器未见任务','220'],
        ['汤罐入篮','B 4/5','搜索器未见任务','280']],['任务','初态索引','与训练的关系','操作上限/步'],[.23,.16,.39,.22],10)
    s.text(.67,.735,'先验收，再汇总',13,BLUE,True)
    s.text(.67,.68,'① 首块真实准入：48任务\n② 每机8 GPU容量：240任务\n③ 其余配对评测：384任务',11)
    s.text(.67,.55,'三阶段包含在672条内，不额外加量。\n同步报告规范视角能力、挽救与伤害；\n缺失标签保留缺失，不填成失败。',10.5)
    s.box(.06,.285,.27,.14,'center：CPU控制中心','计划、代码、预算、派发与进度',BLUE)
    s.box(.385,.285,.28,.14,'Fresh／GNU／Video','三台计算节点；各评测一个模型',TEAL)
    s.box(.72,.285,.23,.14,'共享 /share','模型、数据、结果与备份',BLUE)
    s.arrow((.33,.355),(.385,.355));s.arrow((.665,.355),(.72,.355))
    s.text(.065,.245,'center接管尚未完成；重启后重新核验环境、可信保活与GPU身份。当前不启动评测。',10,MUTED)
    return s.finish(pdf)


def build(project_root,pdf_path,output,font_dir):
    target=Path(pdf_path).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for weight in ('Regular','Bold'):font_manager.fontManager.addfont(str(Path(font_dir)/f'NotoSansCJK-{weight}.ttc'))
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','pdf.fonttype':3,'axes.unicode_minus':False,'font.size':11})
    evidence=Evidence(project_root);data=evidence.load();before=sha(target)
    with fitz.open(target) as old:
        if len(old) not in (BASE_PAGES,TOTAL):raise ValueError('Unexpected full-report page count')
        original=[hashlib.sha256(old[i].get_pixmap(clip=fitz.Rect(0,0,959.976,510)).samples).hexdigest() for i in range(BASE_PAGES)]
    with tempfile.TemporaryDirectory(prefix='.search-appendix-',dir=target.parent) as temporary:
        tmp=Path(temporary)
        with PdfPages(tmp/'appendix.pdf') as pdf:
            pages=[fixed_slide(data,pdf),interface_slide(pdf),first_continuous_slide(data,pdf),
                   curves_slide(data,pdf),models_slide(data,pdf),next_slide(pdf)]
        with fitz.open(target) as old,fitz.open(tmp/'appendix.pdf') as tail:
            result=fitz.open();result.insert_pdf(old,from_page=0,to_page=BASE_PAGES-1);result.insert_pdf(tail)
            for index in range(BASE_PAGES):
                page=result[index];page.add_redact_annot(fitz.Rect(897,519,940,537),fill=(1,1,1))
                page.apply_redactions(images=0,graphics=0)
                page.insert_textbox(fitz.Rect(880,522.4,925.5,536),f'{index+1:02d} / {TOTAL}',fontname='helv',fontsize=8,color=(.384,.443,.518),align=2)
            result.set_metadata(dict(title='VLA研究进展：完整报告与连续PPO进展',author='DSOL Paper 1',subject='Existing evidence through 2026-10-23; no new experiments'))
            result.save(tmp/'full.pdf',garbage=4,deflate=True);result.close()
        with fitz.open(tmp/'full.pdf') as result:
            assert len(result)==TOTAL
            retained=[hashlib.sha256(result[i].get_pixmap(clip=fitz.Rect(0,0,959.976,510)).samples).hexdigest() for i in range(BASE_PAGES)]
            if retained!=original:raise ValueError('Historical body changed')
            contact=Image.new('RGB',(1440,1215),'#dfe6ee');draw=ImageDraw.Draw(contact)
            for j in range(6):
                pix=result[BASE_PAGES+j].get_pixmap(matrix=fitz.Matrix(.75,.75))
                im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
                contact.paste(im,((j%2)*720,(j//2)*405))
            contact.save(output/'contact-sheet.png')
            extracted='\n'.join(result[i].get_text() for i in range(BASE_PAGES,TOTAL))
            (output/'appended-text.txt').write_text(extracted)
        if sha(target)!=before:raise ValueError('Current report changed during build')
        os.replace(tmp/'full.pdf',target)
    receipt=dict(status='BUILT_LAYOUT_CHECKED_REQUIRES_VISUAL_REVIEW',target=str(target),before_sha256=before,
        after_sha256=sha(target),retained_body_pages=33,body_pixels_identical=True,total_pages=TOTAL,appended=pages,
        historical_pdf_copy_created=False,new_model_calls=0,new_training_updates=0,new_task_episodes=0,sources=evidence.sources)
    (output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    # A compact numerical input file makes plots reviewable without live dashboards.
    (output/'plot-data.json').write_text(json.dumps(dict(training=data['training'],old_continuous=data['old']['summary'],
        fixed_table=data['fixed']['table']),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','target','after_sha256','total_pages','body_pixels_identical')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project-root','pdf','output','font-dir'):p.add_argument('--'+name,required=True)
    a=p.parse_args();build(a.project_root,a.pdf,a.output,a.font_dir)
