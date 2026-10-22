"""Complete, append-only experiment-to-evidence report for saved action records."""
import argparse
import csv
import html
import json
import shutil
import re
from pathlib import Path
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read,write,sha
from AlphaBrain.research.dsol.plotting import action_structure as plots


def table(heads,rows):
    def esc(v):return html.escape(str(v))
    return '<table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in heads)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table>'


def para(text):return '<p>'+html.escape(text)+'</p>'


def figure(name,caption):
    return '<figure><img src="assets/'+name+'" alt="'+html.escape(caption,quote=True)+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'


def short(view):return view if len(view)<26 else view[:12]+'…'


def write_csv(path,rows):
    flat=[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows]
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)


def pdf_pages(pages,out,font_dir):
    import pymupdf as fitz
    css='''@font-face{font-family:cn;src:url(NotoSansCJK-Regular.ttc)}
    @font-face{font-family:cn;src:url(NotoSansCJK-Bold.ttc);font-weight:bold}
    body{font-family:cn;font-size:12pt;line-height:1.45;color:#243744}h1{font-size:26pt;color:#174c68;margin:0 0 12pt}
    h2{font-size:15pt;margin:8pt 0}p{margin:7pt 0}img{width:100%}figure{margin:8pt 0}figcaption{font-size:11pt;color:#526674}
    table{border-collapse:collapse;width:100%;font-size:10.5pt}th,td{border:.5pt solid #bdcdd5;padding:5pt;vertical-align:top}th{background:#eaf1f5}
    .small{font-size:8pt}a{color:#174c68}'''
    archive=fitz.Archive([str(out),str(font_dir)]);doc=fitz.open();layout=[]
    for i,(title,body) in enumerate(pages):
        def sized_image(match):
            from PIL import Image
            with Image.open(out/match.group(1)) as img:w,h=img.size
            width=min(1072,600*w/h)
            return '<img src="'+match.group(1)+'" style="width:'+str(width)+'pt">'
        body=re.sub(r'<img src="([^"]+)"[^>]*>',sized_image,body)
        pg=doc.new_page(width=1152,height=864)
        page_css=css if '<img ' in body else css+'body{font-size:18pt;line-height:1.5}table{font-size:14pt}th,td{padding:8pt}p{margin:12pt 0}'
        spare,scale=pg.insert_htmlbox(fitz.Rect(40,30,1112,814),'<h1>'+html.escape(title)+'</h1>'+body,css=page_css,archive=archive,scale_low=1)
        if spare<0:raise ValueError(f'PDF page {i+1} overflows: {title}')
        pg.insert_htmlbox(fitz.Rect(40,831,1112,854),f'VLA视角研究 · 2026-10-15 · 已有数据离线分析 / 新增模型调用0　　　　　　　　　{i+1} / {len(pages)}',css=css+'body{font-size:8pt}',archive=archive)
        layout.append(dict(page=i+1,title=title,spare_height=spare,scale=scale))
    doc.set_metadata(dict(title='Accel：从实验设计到动作结构证据',author='AlphaBrain research',subject='128份既有记录；完整设计、过程、正反例与边界'))
    doc.save(out/'研究过程与动作结构分析.pdf',deflate=True,garbage=4)
    preview=out/'previews';preview.mkdir()
    for i,pg in enumerate(doc):pg.get_pixmap(matrix=fitz.Matrix(1.1,1.1)).save(preview/f'{i+1:02d}.png')
    write(out/'pdf-layout.json',layout)


def build(root,output_name='report',cache=None):
    root=Path(root);cfg=read(root/'frozen-plan.json');summary=read(root/'summary.json');pairs=read(root/'transitions.json')
    groups=read(root/'groups.json');coverage=read(root/'coverage.json');index=read(root/'cases.json');evidence=read(root/'evidence.json')
    prior=Path(cfg['previous_analysis']);out=root/output_name;out.mkdir(exist_ok=False);assets=out/'assets';assets.mkdir();detail=out/'cases';detail.mkdir()
    if cache:
        for f in (Path(cache)/'assets').glob('libero_*.png'):
            shutil.copyfile(f,assets/f.name)
    plots.style(Path(cfg['font_dir'])/'NotoSansCJK-Regular.ttc')
    plots.design(assets/'design.png');plots.transitions(pairs,assets/'transitions.png')
    plots.concordance(groups,assets/'concordance.png');plots.distributions(pairs,assets/'distributions.png')
    plots.associations(evidence['associations'],assets/'associations.png')
    inherited={}
    for name in ('physical','channels','recovery'):
        source=prior/'report'/f'{name}.png';shutil.copyfile(source,assets/f'prior-{name}.png');inherited[str(source)]=sha(source)
    base=prior.parents[2]
    for source,name in [(base/'same-state-xr1-20261015T045429Z/final-report/matrices.png','same-state.png'),
                        (base/'prefix-replay-20261014/analysis/xr1-39dd9788-images.png','cabinet-observations.png'),
                        (base/'prefix-replay-20261014/analysis/libero-05-images.png','libero-observations.png')]:
        shutil.copyfile(source,assets/name);inherited[str(source)]=sha(source)
    css='''body{max-width:1280px;margin:35px auto;padding:0 22px;font:16px/1.75 system-ui;color:#223743}h1,h2{color:#174c68}section{margin:50px 0;border-top:2px solid #dfebf1;padding-top:18px}img{max-width:100%}figure{margin:18px 0}figcaption{font-size:14px;color:#526674}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #bdcdd5;padding:7px;vertical-align:top;overflow-wrap:anywhere}th{background:#eaf1f5}nav{background:#edf4f7;padding:18px}a{color:#12618b}details{margin:15px 0}input{padding:8px;width:70%}.callout{padding:15px;background:#e9f3ef}'''
    head='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>'+css+'</style>'
    all_case_rows=[]
    for n,meta in enumerate(index):
        case=read(root/'cases'/meta['file']);rr=[p for p in pairs if p['panel']==case['panel'] and p['initial']==case['initial']]
        chosen=next(p for p in rr if p['mode']=='include' and p['start']==3)
        if not chosen['changed']:chosen=next(p for p in rr if p['mode']=='include' and p['start']==1)
        filename=meta['file'].replace('.json','.png')
        if not (assets/filename).exists():plots.case_plot(case,chosen,assets/filename)
        case_rows=[[p['mode'],f"W{p['start']}→W{p['end']}",p['old_view'],p['new_view'],p['old_success'],p['new_success'],p['outcome']] for p in rr]
        body='<h1>'+plots.NAMES[case['panel']]+' / 初态'+str(case['initial'])+'</h1>'+para(case['task'])
        body+='<a href="../report.html#all-cases">返回完整报告</a>'+para('本页图展示保留规范候选的W3→W5；若不换位则展示W1→W3。全部模式见表，不按改善大小挑选。阴影为8套噪声的范围，不是置信区间。')
        body+='<img src="../assets/'+filename+'">'+table(['规范位','窗口','旧候选完整ID','新候选完整ID','旧成功率','新成功率','变化'],case_rows)
        body+=para('每条曲线、8个噪声重复、各候选动作XYZ及去噪中间能量见原始对齐JSON。预测命令不是测量的实际运动，末次预测未执行。')
        body+='<a href="../../cases/'+meta['file']+'">完整对齐JSON</a>'
        ctable=[]
        for c in case['candidates']:
            for stage in range(5):
                ctable.append([c['view'],stage+1,c['success'],*['%.6f'%np.asarray(c['features'][k])[:,stage].mean() for k in ('near_close_fraction','near_gripper_variation','near_rotation_magnitude','near_rotation_variation')],*['%.6f'%np.asarray(c['channels'][k])[:,stage].mean() for k in ('translation','rotation','gripper')]])
        body+='<details><summary>全部已选候选 × 5个阶段明细（含排除规范位的赢家）</summary>'+table(['候选','阶段','成功率','近端闭合','近端夹爪变化','近端旋转大小','近端旋转变化','平移贡献','旋转贡献','夹爪贡献'],ctable)+'</details>'
        (detail/meta['file'].replace('.json','.html')).write_text(head+body+'</html>')
        all_case_rows.append([plots.NAMES[case['panel']],case['initial'],case['task'],chosen['outcome'],meta['file'].replace('.json','.html')])
        if (n+1)%32==0:print('case plots',n+1,flush=True)
    selected=evidence['selected_cases']
    for p in selected:
        name=f"{p['panel']}-{p['initial']:02d}"
        case=read(root/'cases'/(name+'.json'));plots.denoising_plot(case,p,assets/(name+'-denoise.png'))
    selections=read(root/'selections.json')
    for filename,rows in [('all-transitions.csv',pairs),('all-selections.csv',selections),('all-coverage.csv',coverage),('within-initial-associations.csv',evidence['associations'])]:write_csv(out/filename,rows)
    pages=[];sections=[]
    def add(title,body,pdf=True):
        sections.append((title,body))
        if pdf:pages.append((title,body))
    add('今天做了什么：实验、分析和方法验证分开',
        para('前一轮两例同状态模型诊断已完成，本轮128份动作结构分析已完成。尚未完成的是新选位方法的独立有效性验证；不能把解释性分析称为新的成功率实验。')+
        table(['工作','设计与产出','调用 / 执行','当前状态'],[
        ['历史完整任务与五窗口记录','冻结成功率标签；候选先短执行评分、再独立完整执行','历史预算，非今天新增','复用；旧展示PDF保留'],
        ['恢复与工程控制','mocap按已有body恢复；原状态/渲染位姿分离；EGL上下文修复','失败与修复记录均保留','最终384帧 / 1152图逐像素一致'],
        ['两例同状态跨视角诊断','2例 × 2状态来源 × 2视角 × 3时刻 × 8噪声 = 192格','复用32；新增160诊断+12控制；动作0','完成，172/172预算'],
        ['遥测事件对齐','64分支 / 2752帧 / 320原节点；距离、开度、接触与贡献','新增调用0 / 动作0','完成，局部阶段条件不足'],
        ['本轮动作结构对齐','128归档 / 9000候选格 / 360000评分节点；固定W1/W3/W5','新增调用0 / 动作0','完成，正反例及过程全部展示'],
        ['新方法独立确认','尚未冻结一个获得支持的新选择规则','未启动','仍待研究，不宣称已经解决']]))
    add('工程过程：失败原因、修复与最终准入',
        table(['阻塞','原因 / 修复','验证边界'],[
        ['EGL启动参数','最初传入GPU UUID，但该版Robosuite要求数字设备号；改为适配器支持的编号并增加启动检查。','参数错误，不是研究假设失败'],
        ['历史状态与图像对不上','记录的qpos比渲染缓存晚2毫秒；经用户同意，分别验证原始状态身份与历史渲染位姿。','不再把两个时刻要求成同一几何；原图与模型门槛未放宽'],
        ['mocap恢复不全','依每帧已有body位姿映射槽位，支持不同顺序和移动目标，不补跑动作。','恢复已有记录，不生成新的操作轨迹'],
        ['第二场景图像破损','旧OpenGL上下文延迟释放影响新场景；先释放旧场景上下文再创建新场景。','前一轮失败停止，用户授权后新目录重跑'],
        ['最终结果','384帧、1152图逐像素一致；固定输入与重渲染模型控制一致；两例诊断完成。','工程准入通过不等于新选位指标有效']])+
        para('以上为已完成诊断的过程回顾，本轮没有重新启动这些控制或任何模型。历史失败文件、原门槛和授权变更记录均保留。'))
    add('本轮设计：从已有输入到可检查结论',figure('design.png','流程图：先冻结范围与比较对象，再计算描述量；成功标签不参与特征、权重或阈值选择。')+
        para('主问题：预测开合或旋转结构能否解释“窗口加长后选对 / 选错”的差异？主比较是同初态原W3与W5赢家，W1→W3为补充；保留和排除规范位都报告。每面板32份记录对应相同8任务，面板间初态重用，不能当128个独立任务。'))
    add('计算定义：动作变化不等于去噪变化',
        table(['量','固定计算','解释边界'],[
        ['原分数 / 通道','A=3 Σ‖vⱼ−vⱼ₋₁‖ / Σ‖vⱼ‖，只用前3个去噪输出。通道贡献=3Σ(通道差分平方和/全差分范数)/原分母；各贡献之和=A。','使用原归一化模型坐标；逐值复现旧结果'],
        ['预测闭合 / 切换','原动作先按执行器clip到[-1,1]；g>0为闭合，g=0为hold；相邻g乘积<0计直接开合切换。','不是实际关节开度；不将hold计作开合'],
        ['夹爪变化','mean |gₕ₊₁−gₕ|','区分二值开合与命令细微变化'],
        ['旋转大小 / 变化 / 反向','mean‖rₕ‖；mean‖rₕ₊₁−rₕ‖；非零向量相邻点积<0的比例。','OSC控制命令，不是物理转角；零范数反向无效'],
        ['时间范围','前5步与完整10步分别算；五个评分时刻0/5/10/15/20动作步。','只有前四个块的前5步历史执行过；第5块未执行'],
        ['新增阶段差','W3→W5取第4/5阶段，新候选特征−旧候选特征；先配对噪声，再均值。','各候选状态不同；不能宣称因果隔离']])+
        para('历史终局标签：LIBERO每候选32次独立执行；Plus每候选4次。图中成功率是有限重复的实测比例，尤其Plus变化粒度为0.25，不是无误差真值。'))
    add('两例模型诊断：实际做过的对照',figure('same-state.png','已有同状态2×2矩阵。行是原分支状态及历史，列是外部视角；每格8套固定噪声。矩阵与本轮LIBERO/Plus分析不能混为同一实验。')+
        para('控制：4个固定历史输入及8个重渲染模型控制一致；固定候选、状态、历史与噪声，不执行动作。状态项包含历史；交互说明同一视角的评分效果依赖状态/历史，不能将其直接命名为纯阶段因果效应。'))
    add('已有遥测：接触、开度、距离的中间过程',figure('prior-physical.png','完整64步短轨迹；曲线为8重复均值，色带为范围。入柜7/8掌部接触、3/8单侧手指接触、0/8同帧双指接触，不能写成已经抓住。'))
    add('已有遥测：视角反转由哪些通道构成',figure('prior-channels.png','固定采样时刻的历史好视角−历史坏视角评分差。入柜主要夹爪交互；水槽主要平移交互。连线仅帮助阅读，不表示存在中间评分。')+
        para('入柜48步夹爪贡献已升高，但未闭合、未接触；48→64反转7条中6条没有闭合执行。水槽无闭合/接触仍有状态依赖。接触开关不是充分解释。'))
    add('历史画面：保留实际观察，而不是只看数值',figure('cabinet-observations.png','已有入柜分支图像：固定展示原评分噪声0。画面用于解释可见性，不据单张图认定遮挡是因果；统计仍使用全部8套。'))
    add('覆盖与原收益：正例没有被删掉',figure('prior-recovery.png','原W1/W3/W5相对冻结扰动基线的恢复收益。四面板原收益全部保留，未拟合新规则。')+
        table(['覆盖','数量','解释'],[['归档 / 候选格 / 评分节点','128 / 9000 / 360000','候选格=模型×初态×视角；不是独立任务'],['原赢家记录 / 窗口比较 / 实际换位','768 / 512 / 261','包含/排除规范模式会重用同一初态'],['本轮候选标签','9000已知，0缺失','仅限这128份LIBERO/Plus；不代表RoboCasa缺失标签已补齐'],['逐值验证','原分数、全部原结构量、赢家一致','未调整分数公式或候选排序']]))
    add('总体结果：每个面板的改善和退化都展示',figure('transitions.png','改善/退化按历史独立完整任务成功率比较，非逐初态显著性检验；相等包括没有换位及换位后同成功率。'))
    add('区分度审计：不把局部分离当新方法',figure('concordance.png','只比较实际换位且标签为改善/退化的记录。所有相等项、未换位项在总表保留。颜色方向固定，不按结果翻转指标；不同面板方向不一致，极值可能只由1个正例产生。'))
    add('值域与重叠：展开看全部正负样本',figure('distributions.png','W3→W5，保留规范位；每个点是一个初态的候选差，含相等结果。值域重叠说明没有经验上的完全分离；不据此证明一切非线性规则均无效。'))
    mainrows=[]
    for g in groups:
        if g['start']!=3 or g['mode']!='include':continue
        f=g['features']['near_rotation_variation'];known=[x for x in f['leave_one_task_out'].values() if x is not None]
        mainrows.append([plots.NAMES[g['panel']],g['changed_outcomes'].get('improved',0),g['changed_outcomes'].get('degraded',0),f"{f['concordance']:.3f}",f"{min(known):.3f}–{max(known):.3f}" if known else '无',sum(x is None for x in f['leave_one_task_out'].values())])
    add('小样本与任务依赖：漂亮数字的边界',table(['面板','改善','退化','旋转变化排序概率','剔除一个任务后的范围','剔除后不可比较次数'],mainrows)+
        para('LIBERO规范的1.000来自1个改善与6个退化；去掉唯一改善所在任务后，正类消失，不能计算。Plus多视角也只有1个改善。因此不以高/低排序概率冻结规则。LIBERO多视角有7正7负，旋转变化约0.531且值域重叠，是更直接的反例。')+
        para('这些均为开发数据上的描述性审计，不是交叉验证成绩。未拟合阈值、未训练分类器、未搜索权重，也没有把8套评分噪声当8个独立任务扩充显著性。完整逐任务表见groups.json。'))
    add('补充关联核查：动作变化和去噪变化有关，但不等价',figure('associations.png','先在每个初态内，对全部候选的8噪声均值求相关，再展示32初态相关系数中位数。不将跨任务尺度混在一起，不把该相关性当作成功预测能力。')+
        para('这是结果后的描述性补充，不是预先冻结的决策判据。旋转命令相邻变化与旋转贡献经常正相关，但正负换位仍重叠；旋转命令大小的相关性随阶段减弱。通道所反映的生成变化与实际任务价值之间仍有缺口。'))
    add('回到原先的严重反例：LIBERO初态5',figure('libero_canonical-05.png','这一例在前期已固定关注，非本轮按结果新挑：原W3赢家29/32，原W5赢家3/32。对应完整过程见128例附录。')+
        para('旋转通道推动累计反转，预测开合没有提供区分。即使能描述旋转生成与命令的差别，也尚未形成能同时保留正例的判断规则。'))
    add('严重反例的历史观察',figure('libero-observations.png','同一原先预选的LIBERO初态5，原评分噪声0的历史图像；没有新增渲染。数值统计使用全部噪声。'))
    for p in selected:
        name=f"{p['panel']}-{p['initial']:02d}";desc='改善' if p['outcome']=='improved' else '退化'
        add(plots.NAMES[p['panel']]+f"：{desc}例（初态{p['initial']}）",figure(name+'.png','按每面板初态编号最小的W3→W5改善/退化例固定展示，不按差值大小挑例。旧候选蓝，新候选橙；阴影是噪声范围。表内完整ID与所有案例见HTML附录。')+
            para('解读顺序：先看累计评分何时换序，再看通道差如何形成总分差，最后核对预测命令；预测变化只能解释生成结构，不能代替执行效果。'))
        add(plots.NAMES[p['panel']]+f"：{desc}例的去噪中间过程",figure(name+'-denoise.png','上排第3次、下排第5次预测。只使用阴影内前两次去噪差分计算原Accel；其余曲线未用于重选或拟合。通道范数不能直接相加为总范数，精确可加贡献见前页。'))
    conclusion=[
        '保留：动作生成通道与状态/历史存在交互，可作为解释和可靠性研究线索。保留原Accel基线与原LIBERO/Plus正收益，尚不产生新总分。',
        f"暂停早期窗口的二元开合判别：{evidence['changed_transition_records']}条实际换位记录对应{evidence['candidate_role_records']}个旧/新候选角色，在新增阶段均没有预测闭合或开合切换；这里缺少事件覆盖，不能检验真正抓取阶段，更不能称通用开合机制已被否定。",
        '放弃作为通用规则：旋转更小/更平滑就应更好。正负分布重叠、方向跨面板不稳定；少数完全分离依赖仅1个改善样本。完整10步也没有给出跨面板一致的分离方向。',
        '暂不推进：全局通道降权、接触后切换、固定距离阶段开关，以及继续搜索更好看的动作阈值。它们目前缺乏同时保住正收益和减少错误换位的证据。',
        '值得继续的最小离线工作：审计已有晚期完整轨迹是否同时保存预测动作、去噪与物体反馈，先确认真实闭合/接触阶段的正反例覆盖；若没有，就明确记录证据缺口，不用早期数据外推。',
        '方法方向：下一轮若提出规则，应针对状态/历史下的证据可靠性或具体执行结果矛盾，明确可部署输入及保留正收益、报告伤害的标准；当前数据不能保证该方向有效。独立确认需要另行冻结协议，本轮不启动。']
    add('结论与下一步：哪些继续，哪些停止', ''.join(para(s) for s in conclusion))
    add('可复核性：从结论回到每个原始输入',
        para('本报告保留原始输入哈希、冻结配置、逐值复现检查、全部候选标签、全部窗口选择、全量切换明细、逐任务诊断和128个案例页面。所有结果写入新目录，原展示PDF与原报告不变。')+
        table(['文件','内容'],[['frozen-plan.json / sources.json','分析前固定口径与原始文件哈希'],['coverage.json / summary.json','覆盖、原分数复现、0新增调用'],['selections.json / transitions.json','全部768原赢家、512窗口比较，含负结果与未知逻辑'],['groups.json / evidence.json','固定特征分布、任务剔除、开合事件覆盖及描述性相关'],['cases/ 与 report/cases/','128份逐候选、逐噪声、逐阶段数组和图表'],['report/*.csv','下载后可筛选的完整表格'],['completion-receipt.json','最终测试与源文件不变核验（交付时生成）']])+
        para('报告修正：上一轮只给汇总结论的呈现不足以供研究审查。本轮把已有实验和新增离线分析明确分开，并提供设计、控制、中间过程、正反结果和结论边界；图多本身不增加证据强度。'))
    links='<nav>'+''.join(f'<a href="#s{i}">{html.escape(title)}</a> · ' for i,(title,_) in enumerate(sections))+'</nav>'
    body='<h1>Accel：从实验设计到动作结构证据</h1>'+para('2026-10-15｜本轮已有数据分析；新增模型调用0。PDF用于完整汇报，HTML保留128例与可下载全表。')+'<p><a href="研究过程与动作结构分析.pdf">下载完整PDF</a> · <a href="#all-cases">全部128例</a></p>'+links
    for i,(title,content) in enumerate(sections):body+=f'<section id="s{i}"><h2>'+html.escape(title)+'</h2>'+content+'</section>'
    body+='<section id="all-cases"><h2>全部128例：按模型、任务、初态筛选</h2><input id="filter" placeholder="输入任务、模型或初态编号"><table id="case-table"><thead><tr><th>面板</th><th>初态</th><th>任务</th><th>图中变化</th><th>过程图与完整表</th></tr></thead><tbody>'
    for row in all_case_rows:body+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row[:4])+f'<td><a href="cases/{row[4]}">打开逐例页面</a></td></tr>'
    body+='</tbody></table></section><section><h2>数据下载</h2>'+''.join(f'<p><a href="{f}">{f}</a></p>' for f in ('all-transitions.csv','all-selections.csv','all-coverage.csv','within-initial-associations.csv'))+'</section>'
    body+="<script>document.getElementById('filter').addEventListener('input',e=>{let q=e.target.value.toLowerCase();document.querySelectorAll('#case-table tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script>"
    (out/'report.html').write_text(head+body+'</html>')
    write(out/'report-pages.json',pages)
    pdf_pages(pages,out,Path(cfg['font_dir']))
    write(out/'provenance.json',dict(inputs={str(root/f):sha(root/f) for f in ('frozen-plan.json','summary.json','transitions.json','groups.json','evidence.json')},inherited_assets=inherited,source_sha256=sha(__file__),pdf_sha256=sha(out/'研究过程与动作结构分析.pdf'),pages=len(pages),case_pages=len(index)))
    print(out/'report.html',flush=True)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output-name',default='report');p.add_argument('--cache');a=p.parse_args();print(build(a.root,a.output_name,a.cache))
