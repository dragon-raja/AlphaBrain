"""Maintain the unified final five pages, preserving the historical first18."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.plotting.window_comparison import curves, task_heatmap
from .robocasa_diagnostic_update import DiagnosticBrief, publish, fitz, PdfPages
from .robocasa_results import BLUE, TEAL, GRAY, AMBER, INK, MUTED, TASK_NAMES
from .confirmation_update import number, pp

NAMES = {'libero_canonical':'LIBERO · 规范训练', 'libero_broad':'LIBERO · 多视角训练',
         'plus_canonical':'Plus官方候选 · 规范训练', 'plus_broad':'Plus官方候选 · 多视角训练',
         'robocasa_official':'RoboCasa · π0.5 官方', 'robocasa_xr1':'RoboCasa · XR-1 官方'}
SHORT_TASKS = {'goal_cream_cheese_bowl':'奶酪入碗', 'goal_top_drawer_bowl':'开抽屉放碗',
               'goal_wine_rack':'酒瓶上架', 'libero10_bowl_bottom_drawer':'碗入底抽屉',
               'libero10_book_caddy':'书入收纳盒', 'libero10_mug_microwave':'杯入微波炉',
               'object_cream_cheese_basket':'奶酪入篮', 'spatial_drawer_bowl_plate':'抽屉碗至盘'}


class WindowBrief(DiagnosticBrief):
    def values(self, key, mode='include', scope='all'):
        return self.summary['panels'][key]['methods'][mode]['scopes'][scope]['mean']

    def table(self, rect, rows, headings, widths, size=8.5):
        ax=self.fig.add_axes(rect); ax.axis('off')
        t=ax.table(cellText=rows, colLabels=headings, colWidths=widths, loc='center', cellLoc='center')
        t.auto_set_font_size(False); t.set_fontsize(size); t.scale(1, 1.45)
        for (r,_),cell in t.get_celld().items():
            cell.set_edgecolor('#DCE3EB')
            if r==0:cell.set_facecolor('#EDF3FA')

    def libero_plus(self, pdf):
        self.new('自建97候选有窗口收益；Plus官方候选的趋势不同',
            '五窗口统一分析 · 2026-10-13',
            '同8任务／32初态；先候选试执行、再复位选定初始视角；不是执行途中换相机',
            '自建97规范模型：W1→W3提升15.92pp；不能概括为“长窗口无效”。',
            '上排：自建97候选，原32留出扰动均值为基线，每格32执行噪声。下排：官方Plus相机候选，每格4执行噪声。\n全部采用8评分噪声，任务等权；曲线为全部32初态，不是新任务泛化。候选目录与执行预算不同，不作环境间因果比较。',
            'H表示前H次重规划评分的算术均值，包含初态；LIBERO与Plus评分对应0/5/10/15/20动作步。所有候选按独立评分噪声得到得分，完整任务成败不参与选位。W1严格复现原评分。Plus只是官方相机扰动子集，并非完整LIBERO-Plus基准。规范参考在LIBERO用32执行重复，在Plus用相同初态的前4重复，因此规范值不同不是换了权重。')
        for j,key in enumerate(['libero_canonical','libero_broad','plus_canonical','plus_broad']):
            ax=self.fig.add_axes([.075+(j%2)*.47,.535 if j<2 else .272,.39,.19])
            curves(ax,self.values(key),self.values(key,'exclude'),NAMES[key])
            if j<2:ax.set_xlabel('')
            v=self.values(key)
            ax.text(.02,.02,f'不调整 {number(v[0]*100)}%  /  规范 {number(v[1]*100)}%',transform=ax.transAxes,fontsize=7.5)
        self.text(.075,.795,'蓝实线：允许规范位　绿虚线：排除规范位　黄虚线：扰动不调整　灰点线：规范参考',8.5,INK)
        self.save(pdf)

    def robocasa(self, pdf):
        self.new('RoboCasa 独立确认：长窗口未形成稳定的跨模型恢复',
            '新初态证据 · 保留原独立确认口径',
            '每模型80新初态，10已知任务；97候选×8评分噪声；冻结选位后8套独立执行噪声',
            '预定窗口3未通过整体确认；heldout 的局部改善不能替代 seen 与整体结果。',
            '允许规范位；扰动基线＝原32个额外候选均值。π0.5每25动作步重规划，XR-1每16步，H不代表相同执行时长。\nseen/heldout为布局与风格组，不是新任务。排除规范位：π各窗口覆盖79/80/79/80/78；XR为80/80/79/79/79初态，未给不完整总体打分。',
            '排除规范位需重新按96候选评分选位，不等于把选择规范位的初态删除。若新选中的视角未在冻结执行清单中，则缺少独立闭环标签，不能补0、沿用规范标签或只算已覆盖子集。π0.5和XR-1原主窗口3的95%与族校正区间均保留在统计说明。窗口4/5的较好点估计是描述性，不替换原主假设。')
        for col,model in enumerate(('official','xr1')):
            key='robocasa_'+model;ax=self.fig.add_axes([.075+col*.47,.495,.39,.255])
            exclude=self.summary['panels'][key]['methods']['exclude']['complete_window_means']
            ex=[*self.values(key)[:2],*[np.nan if v is None else v for v in exclude]]
            curves(ax,self.values(key),ex,title=NAMES[key],limits=(25,80));ax.legend(fontsize=7.5,loc='upper left',ncol=2)
            orig=self.summary['robocasa_original']['models'][model]['populations']['all']['methods']['window3']['gain_vs_stay']
            self.text(.075+col*.47,.437,f'W3恢复 {pp(orig["mean"])} pp；95%区间 [{pp(orig["ci95"][0])}, {pp(orig["ci95"][1])}]',9.5,INK)
        rows=[]
        for model,label in [('official','π0.5'),('xr1','XR-1')]:
            for scope,sl in [('pretrain','seen'),('target','heldout')]:
                v=self.values('robocasa_'+model,scope=scope)
                rows.append([label+' · '+sl,*[number(v[i]*100) for i in [1,0,2,3,4,5,6]]])
        self.table([.08,.255,.85,.14],rows,['场景组（40初态）','规范%','不调整%','W1%','W2%','W3%','W4%','W5%'],[.25]+[.107]*7,8.4)
        self.save(pdf)

    def permissions(self,pdf):
        self.new('两种Plus设置：同一扰动基线，不同的候选终点',
            '统一比较 · 相同初态与前4套执行噪声',
            '实线：Plus官方候选；虚线：自建97绝对终点。两者均从原初态重新执行完整任务',
            'Plus→97复用LIBERO候选终点，不是另一份独立多窗口验证。',
            '下表统一前4次执行噪声：不调整基线仍明显不同，来自相机扰动分布，而不是换模型或换物理初态。\n自建97是绝对终点集合；未执行从扰动起点移动相机的过程，也不是为每个扰动起点新建97个相对候选。',
            'Plus官方扰动集合每任务20–49个定义，8任务共341个；LIBERO不调整基线为自建32个留出视角。均任务等权，任务内初态/相机等权。规范模型LIBERO32重复基线22.17%，取前4次22.95%，Plus4次47.23%；差异主要来自扰动集合，不意味着完整Plus更简单。Plus→97的评分、选择、执行终点与LIBERO匹配4次完全相同，只替换扰动参考。上图允许规范位，图下保留自建97排除规范位的五窗口结果；完整初态分组在配套说明保留。')
        for col,model in enumerate(('canonical','broad')):
            ax=self.fig.add_axes([.075+col*.47,.49,.39,.25]);native=self.values('plus_'+model)
            curves(ax,native,title='规范训练' if model=='canonical' else '多视角训练',limits=(35,102))
            bridge=self.summary['bridge'][model]
            ax.plot(range(1,6),np.array(bridge['include']['mean'][2:])*100,'D--',color=TEAL,label='自建97（允许规范位）',ms=3)
            handles,labels=ax.get_legend_handles_labels()
            labels=['官方候选（允许规范位）' if x=='允许规范位' else x for x in labels]
            ax.legend(handles,labels,fontsize=7,loc='lower right',ncol=2)
            self.text(.075+col*.47,.437,'自建97排除规范位：'+ ' / '.join(number(v*100,1) for v in bridge['exclude']['mean'][2:])+' %',9,INK)
        rows=[]
        for model,label in [('canonical','规范训练'),('broad','多视角训练')]:
            b=self.transitions['baseline_matching'][model]
            rows.append([label,number(b['custom32']*100),number(b['custom4']*100),number(b['plus4']*100)])
        self.table([.08,.255,.85,.13],rows,
                   ['不调整基线','自建留出32视角\n32次执行','同32视角\n前4次执行','Plus官方扰动\n4次执行'],[.22,.26,.26,.26],9)
        self.save(pdf)

    def tasks(self,pdf):
        self.new('任务收益分布：不能用一个最佳窗口概括所有任务',
            '逐任务分析 · 全部任务保留',
            '每格＝该窗口选位成功率 − 相应扰动不调整成功率；单位pp；允许规范位',
            '自建97的正结果保留；此页Plus专指官方候选，非Plus→97。',
            '每个环境内部统一色条，环境之间独立范围；不是跨环境同难度比较。完整W1–5及排除规范位数值见配套说明。\nRoboCasa开发高能力／视角敏感组：新π0.5仅4初态、规范53.13%；XR-1为32初态、规范87.11%，W3仍未恢复。',
            '不能按新结果筛掉负任务或重新定义高能力组。热图展示任务效应异质性，不意味着已识别因果机制。规范能力、扰动损失、模型后训练、相机配置和重规划步距等均不同。LIBERO/Plus每任务4初态，Plus每候选4执行重复，逐任务结果尤其粗。')
        for col,(family,models) in enumerate([('libero',('canonical','broad')),('plus',('canonical','broad')),('robocasa',('official','xr1'))]):
            parts=[self.summary['panels'][family+'_'+m]['methods']['include']['scopes']['all']['per_task'] for m in models]
            names=sorted(parts[0]);a=np.array([[100*(parts[j][t][k+1]-parts[j][t][0]) for j in range(2) for k in (1,3,5)] for t in names])
            ax=self.fig.add_axes([.12+col*.315,.315,.205,.405]);limit=max(5,float(abs(a).max()))
            labels=[SHORT_TASKS.get(t,TASK_NAMES.get(t,t)) for t in names]
            # Long raw task IDs are never allowed to spill into neighboring panels.
            labels=[s if len(s)<=12 else s[:11]+'…' for s in labels]
            cols=['规1','规3','规5','多1','多3','多5'] if family!='robocasa' else ['π1','π3','π5','X1','X3','X5']
            title={'libero':'LIBERO · 自建97','plus':'Plus · 官方候选','robocasa':'RoboCasa · 自建97'}[family]
            im=task_heatmap(ax,a,labels,cols,limit);ax.set_title(title,fontsize=12,pad=13)
            cb=self.fig.colorbar(im,cax=self.fig.add_axes([.12+col*.315,.267,.205,.011]),orientation='horizontal');cb.ax.tick_params(labelsize=7)
        self.save(pdf)

    def mechanism(self,pdf):
        self.new('窗口能纠错，也会稳定地选错：需要区分两种更新',
            '逐初态配对审计 · 不增加评测、不拟合学习器',
            '同初态、候选目录和执行噪声配对；表中改善／退化指实测成功率变化，不是逐初态显著性判定',
            '多窗口有条件收益；评分噪声一致性不能单独保证换位有效。',
            '允许规范位。LIBERO每格32次执行，RoboCasa每格8次；执行预算不同，不用初态计数直接比较谁更稳定。\n评分时刻：LIBERO/Plus 0/5/10/15/20，π0.5 0/25/50/75/100，XR-1 0/16/32/48/64动作步。',
            'LIBERO规范W1→W3的10个改善初态分布在6类任务，前后16执行噪声均同向改善；7例转向规范位，但独立排除规范位仍有+12.01pp。W3→W5净下降约81.5%来自上层抽屉放碗。两个大幅退化初态的8/8评分噪声均支持差候选，评分分半赢家也一致；XR-1也有同类反例。因此降低评分随机性不是充分解法。计算原因是新增阶段评分覆盖旧优势；卡住、遮挡、动作进度等物理原因未被唯一识别。不能从这些结果推出长窗口普遍无效或统一在W3停止。')
        rows=[]
        for key,label in [('libero97_canonical','LIBERO97规范'),('libero97_broad','LIBERO97多视角'),
                          ('robocasa_official','RoboCasa π0.5'),('robocasa_xr1','RoboCasa XR-1')]:
            values=[]
            for pair in ('W1_W3','W3_W5'):
                r=self.transitions['panels'][key]['include'][pair]['all'];c=r['counts']
                values.extend([f"{c['improved']} / {c['degraded']} / {c['equal']}",pp(r['delta'])])
            rows.append([label,*values])
        self.table([.075,.565,.85,.18],rows,
                   ['条件','W1→W3\n改善/退化/相等','变化pp','W3→W5\n改善/退化/相等','变化pp'],[.26,.24,.13,.24,.13],9)
        self.text(.075,.50,'LIBERO规范：约82%的W3→W5净下降来自“上层抽屉放碗”。',11,INK,True)
        self.table([.075,.345,.85,.11],[['LIBERO初态5','29/32 → 3/32','8 / 8'],
                   ['LIBERO初态6','28/32 → 1/32','8 / 8'],['XR-1台面物体入柜实例','8/8 → 2/8','8 / 8']],
                   ['W3→W5的错误更新','完整任务成功次数','支持差候选的评分噪声'],[.42,.28,.30],9)
        self.text(.075,.29,'新增窗口改变的是分支状态证据，不只是同一初态的重复测量。',10.5,INK)
        self.save(pdf)


def notes_text(s):
    lines=['# 五窗口统一分析（2026-10-13）','',
           '原报告前18页正文及图表保留；19–23页统一更新。上一版PDF完整归档。',
           '', '所有数值为完整任务闭环成功率。H是累计前H次重规划评分，不是任务途中改相机。',
           'LIBERO/Plus为原8任务×4初态的开发性分析；RoboCasa为两模型各80个新初态的独立确认。不能称为统一训练配方下的跨benchmark因果实验。',
           '', '## 统计口径', '', s['uncertainty'],
           'LIBERO使用32执行重复；Plus官方及自建97终点桥接用前4重复；RoboCasa使用8独立执行重复。候选评分均8噪声。',
           '扰动基线分别为LIBERO原32留出候选、Plus每任务官方相机候选均值、RoboCasa原32额外候选均值；先初态、再任务等权，不按候选数给任务加权。',
           '规范视角是已知参考，不是学习方法。排除规范位改变可选集合，不改变规范参考或删掉选择过规范位的初态。', '']
    for key,p in s['panels'].items():
        lines+=['## '+NAMES[key],'']
        for mode,m in p['methods'].items():
            lines+=['### '+('允许规范位' if mode=='include' else '排除规范位'),'',
                    '完整选位结果覆盖：'+str(m['coverage'])+'；每个数对应W1–5覆盖初态数。', '',
                    '| 分组 | 不调整% | 规范% | W1% | W2% | W3% | W4% | W5% |',
                    '|---|---:|---:|---:|---:|---:|---:|---:|']
            for scope,r in m['scopes'].items():
                lines.append('| '+scope+' | '+' | '.join(number(v*100) for v in r['mean'])+' |')
            if 'all' not in m['scopes']:
                lines+=['','仅报告覆盖全体初态的单个窗口（%）；其余NA：'+str([None if v is None else round(v*100,4) for v in m['complete_window_means']]),
                        '全量W1–5比较缺少部分选中终点的闭环标签；不报告不完整总体，不新增实验。','']
            else:
                r=m['scopes']['all'];lines+=['','相对不调整的描述性95%区间（pp）：'+str([[round(100*v,2) for v in ci] for ci in r['gain_ci95']]),
                    'W5−W1：'+pp(r['vs_w1'][-1])+' pp；95%区间 '+str([round(100*v,2) for v in r['vs_w1_ci95'][-1]]),'',
                    '| 任务 | 不调整% | 规范% | W1% | W2% | W3% | W4% | W5% |',
                    '|---|---:|---:|---:|---:|---:|---:|---:|']
                for t,v in r['per_task'].items():lines.append('| '+t+' | '+' | '.join(number(a*100) for a in v)+' |')
        lines+=['','评分诊断（W1–5）：',str(p['signal']),'']
    lines+=['## Plus起点 → 自建97绝对终点（匹配前4执行噪声）','']
    for model,methods in s['bridge'].items():
        for mode,r in methods.items():lines.append(model+' / '+mode+'：'+str([round(v*100,4) for v in r['mean']]))
    lines+=['','## 原RoboCasa独立确认与冻结分组','',s['robocasa_original']['uncertainty']]
    for model,data in s['robocasa_original']['models'].items():
        for group,r in data['frozen_strata'].items():
            methods=r['methods'];lines.append(f'{model} / {group}，n={r["initials"]}；规范 {number(100*methods["canonical"]["success"]["mean"])}%；W3恢复 {pp(methods["window3"]["gain_vs_stay"]["mean"])} pp。')
        orig=data['populations']['all']['methods']['window3']['gain_vs_stay']
        lines.append(f'{model} 主窗口3原确认：'+str(orig))
    lines+=['','## 解释边界','',
            '直接观察：窗口改变候选排序；多数初态长窗口会换位；换位既改善也损害；评分分半稳定性不必单调增加。',
            '推理：多步评分平均的是不同分支状态，而非同一初态的重复测量；可能更能暴露错误，也可能引入与初态恢复目标无关的阶段偏好。',
            '尚未证实：错误动作确定性、遮挡、训练配方或本体状态捷径分别是首要原因。不能把描述性诊断写成因果结论。',
            'LIBERO/Plus所选探测分支在最后评分前成功比例均为0；π0.5 RoboCasa W5仅0.3125%，XR-1为0。任务提前完成不是普遍解释。',
            '结论：保留静态视角恢复正结果；多步累积不是普适改进；后续方法需在独立开发数据定规则，再以独立初态/执行噪声确认。当前不追加实验、不以事后最佳窗口替代主窗口。']
    return '\n'.join(lines)+'\n'


def build(public_dir, analysis, output_root, transition_audit):
    public_dir,analysis,output_root=map(Path,(public_dir,analysis,output_root))
    summary=read(analysis/'summary.json')
    transition_audit=Path(transition_audit)
    transitions=read(transition_audit/'summary.json')
    if summary['status']!='PASS':raise ValueError('Analysis not accepted')
    if transitions['status']!='PASS' or transitions['sources']!=summary['sources']:
        raise ValueError('Transition audit does not match synthesis sources')
    for p,h in summary['sources'].items():
        if sha(p)!=h:raise ValueError('Evidence changed: '+p)
    out=output_root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    previous=out/'previous-publication';previous.mkdir()
    for name in ('vla_view_research_progress_zh.pdf','vla_view_research_progress_zh.md','publication.json'):
        shutil.copy2(public_dir/name,previous/name)
    before={p.name:sha(p) for p in previous.iterdir()}
    brief=WindowBrief(summary,{})
    brief.transitions=transitions
    with PdfPages(out/'window-pages.pdf') as pdf:
        for method in (brief.libero_plus,brief.robocasa,brief.permissions,brief.tasks,brief.mechanism):method(pdf)
    if any(r['issues'] for r in brief.qa):raise ValueError('Figure text outside page')
    with fitz.open(previous/'vla_view_research_progress_zh.pdf') as old,fitz.open(out/'window-pages.pdf') as new:
        if len(old) not in (20,23):raise ValueError('Expected current20/23-page report')
        result=fitz.open();result.insert_pdf(old,from_page=0,to_page=17);result.insert_pdf(new)
        for i,page in enumerate(result):
            page.add_redact_annot(fitz.Rect(0,524,960,540),fill=(1,1,1));page.apply_redactions(images=0,graphics=0,text=0)
            page.insert_text((34,531),'VLA 研究进展 · 2026-10-13 · '+('历史证据保留原口径' if i<18 else 'LIBERO / Plus / RoboCasa 五窗口统一分析'),fontname='china-s',fontsize=7,color=(.39,.45,.51))
            page.insert_text((901,531),f'{i+1:02d} / 23',fontsize=8,color=(.39,.45,.51))
        result.set_toc([r for r in old.get_toc() if r[2]<=18]+[[1,n['title'],19+i] for i,n in enumerate(brief.notes)])
        result.set_metadata(dict(title='VLA研究进展｜训练覆盖、跨相机恢复与五窗口验证',author='VLA View Research'))
        target=out/'vla_view_research_progress_zh.pdf';result.save(target,garbage=4,deflate=True);result.close()
    previews=out/'previews';previews.mkdir();preservation=[]
    with fitz.open(previous/'vla_view_research_progress_zh.pdf') as old,fitz.open(target) as new:
        for i in range(18):
            clip=fitz.Rect(0,40,959.976,518)
            same=old[i].get_pixmap(clip=clip).samples==new[i].get_pixmap(clip=clip).samples
            if not same:raise ValueError('Historical figure body changed')
            preservation.append(dict(page=i+1,body_pixels_identical=True))
        for i in range(18,23):new[i].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(previews/f'page-{i+1:02d}.png')
    text=notes_text(summary)+'\n\n## 逐初态匹配审计（最新）\n\n'+str(transition_audit/'summary.json')+'\n'
    text+='自建留出32视角与Plus官方扰动不是同一分布；匹配前4执行噪声后规范基线22.95%与47.23%，多视角75.29%与77.38%。Plus→97复用LIBERO终点，不是独立多窗口验证。\n'
    text+='LIBERO规范W1→W3：10改善、0退化；W3→W5：1改善、6退化，净下降约82%来自上层抽屉放碗。两个大幅退化实例均有8/8评分噪声支持差候选。RoboCasa为双向换位相互抵消，XR-1 W3→W5平均仍+3.28pp。既不能说长窗口无效，也没有可靠通用停止规则。\n'
    text+='\n## 历史报告说明（保留原口径）\n\n'+(previous/'vla_view_research_progress_zh.md').read_text()
    (out/'vla_view_research_progress_zh.md').write_text(text)
    write(out/'preservation.json',preservation);write(out/'layout-qa.json',brief.qa)
    write(out/'speaker-notes.json',brief.notes)
    write(out/'provenance.json',dict(before=before,artifacts={n:sha(out/n) for n in ('vla_view_research_progress_zh.pdf','vla_view_research_progress_zh.md')},
          sources={str(analysis/'summary.json'):sha(analysis/'summary.json'),
                   str(transition_audit/'summary.json'):sha(transition_audit/'summary.json'),
                   str(transition_audit/'initial-transitions.json'):sha(transition_audit/'initial-transitions.json')},pages=23,
          added_closed_loop_results=0,new_experiments=0,
          scope='Clarified native Plus versus custom97; paired window-transition evidence; existing results only',
          retained_body_pages=list(range(1,19)),new_pages=list(range(19,24)),layout_pass=True,
          builder={str(Path(__file__).resolve()):sha(__file__)}))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--public-dir',required=True);p.add_argument('--analysis');p.add_argument('--output-root');p.add_argument('--publish-built');p.add_argument('--transition-audit')
    a=p.parse_args()
    if a.publish_built:publish(a.publish_built,a.public_dir)
    else:print(build(a.public_dir,a.analysis,a.output_root,a.transition_audit))
