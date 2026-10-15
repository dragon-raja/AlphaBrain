"""Deployment-permission-aware pages for the single current research report."""

import numpy as np
from .legacy_style import INK, MUTED, BLUE, TEAL, GRAY, AMBER

MODELS = ['canonical', 'broad']
LABELS = ['规范视角训练', '多视角训练']
SCOPE = 'heldout32'


def full_query_page(brief, pdf):
    b = brief
    b.new(
        '指标规则：单噪声、噪声平均与可见性',
        '04 · 指标规则对照，全部97候选',
        '初态 2、3：16 个评价初态；起始相机均匀取自留出目录 32 视角，随后固定执行完整任务。',
        '本轮均值未显示噪声平均的额外收益；组合规则表现依赖训练条件。',
        '单噪声：每次用1个评分噪声选点，再对8种种子的终点成绩取均值；平均噪声：先平均8评分再选1点。\n终点均复用32次完整闭环。可见性需仿真分割特权；返回规范需标定。全查询成本不相同，非等成本搜索。',
        '本页只比较指标规则，不混入学习器。每个评价初态对97个候选评分，选定视角后固定完成任务。单噪声规则对8个预存评分种子分别独立取最小Accel视角，再平均各自终点的32次闭环成功率，不挑最好的种子，也不是先平均分数。8噪声平均规则先计算每个候选的平均Accel，再取最小值。可见性取外部相机仿真指标最大；组合规则先按8噪声平均Accel取最低10个，再在其中取可见性最大。所有规则都是逐初态选择，不是一个全局固定视角。单噪声决策需97个评分样本，8噪声均值需776个；本页不是等推理成本比较。保持原位基线在留出目录32个起点均匀平均；相机获取路径未模拟。学习器与有限查询搜索集中在第十一页，旧全查询学习器结果留在已归档分析。')
    keys = ['stay', 'single_accel', 'min_accel', 'visibility', 'accel10_visibility', 'known_canonical']
    labels = ['保持扰动原位', '最低 Accel · 单噪声', '最低 Accel · 8噪声均值', '最大可见性 †', '平均 Accel 前10＋可见性 †', '已知标定 · 返回规范']
    for m, name in enumerate(MODELS):
        ax = b.axes([.28 + m * .345, .34, .26, .39], True, False)
        rows = b.metric_references['models'][name]
        for i, key in enumerate(keys):
            row = rows[key][SCOPE]['success']
            v = row['mean'] * 100
            lo, hi = np.array(row['ci95']) * 100
            ax.errorbar(v, i, xerr=[[max(0, v-lo)], [max(0, hi-v)]], fmt='o',
                        color=GRAY if key in ['stay', 'known_canonical'] else [BLUE, TEAL][m], capsize=3)
            ax.text(v, i-.19, f'{v:.2f}', fontsize=10, ha='center')
        ax.set_xlim(0, 103)
        ax.set_ylim(5.5, -.7)
        ax.set_yticks(range(6), labels if m == 0 else [], fontsize=10)
        ax.set_xlabel('完整任务成功率（%）', fontsize=10)
        ax.set_title(LABELS[m], fontsize=14)
    gains = [b.metric_references['models'][name]['min_accel'][SCOPE]['gain_vs_stay']['mean'] * 100 for name in MODELS]
    b.text(.075, .26, f'8噪声平均 Accel 相对保持原位：规范训练 {gains[0]:+.2f} pp；多视角训练 {gains[1]:+.2f} pp。', 11, INK)
    b.save(pdf)


def budget_page(brief, pdf):
    b = brief
    budgets = b.recovery['protocol']['config']['budgets']
    primary = str(b.recovery['protocol']['config']['primary_budget'])
    b.new(
        '有限查询搜索：只用已访问的候选图与评分',
        '05 · 部署输入重设后的探索性离线重放',
        '16 个评价初态 × 32 留出起点；查询预算含起点。RGB＋Accel 学习器不接收规范图或规范位姿标记。',
        '恢复收益已观察；多视角模型优于同预算随机的优势尚未确认。',
        '初态 0、1 开发；初态 2、3 评价。已查看此前测试汇总，属于探索性分析。\n随机搜索 8 次重放取均值；Accel 每个查询点用 8 个固定评分噪声；终点闭环复用原 32 次结果。',
        '相对相机位姿目录只用于导航，以平移/0.1m＋旋转/15°构造对称六近邻图。指标搜索从已访问的最好评分节点向未访问近邻展开；随机对照随机选已访问边界的一条边，再以相同评分规则选择访问过的最好终点。两者都允许保留起点，并记录回到选中终点的路径。RGB 学习器是冻结 ResNet18 特征上的岭回归，以初态内中心化的成功率差异训练；PCA、标准化、正则强度都仅使用开发初态。图像和评分是离线缓存，API 只向搜索暴露已查询点；非实际相机获取实验。分布未知、连续位姿、物理可达性及查询耗时未验证。')
    methods = [('local_accel', 'Accel 引导', BLUE, '-'), ('random_accel', '随机探查＋Accel', BLUE, '--'),
               ('local_rgb_accel', '学习器引导', TEAL, '-'), ('random_rgb_accel', '随机探查＋学习器', TEAL, '--')]
    for m, name in enumerate(MODELS):
        rows = b.recovery['models'][name]
        ax = b.axes([.09+m*.46, .35, .36, .37])
        for key, label, color, style in methods:
            values = [rows['methods'][key][str(k)][SCOPE]['success']['mean']*100 for k in budgets]
            ax.plot(budgets, values, marker='o', ms=3, color=color, linestyle=style, label=label, lw=1.7)
        ax.axhline(rows['references']['stay'][SCOPE]['success']['mean']*100, color=GRAY, lw=1, label='保持原位')
        ax.axhline(rows['references']['known_canonical'][SCOPE]['success']['mean']*100, color=AMBER, lw=1, linestyle=':', label='已知标定返回规范')
        ax.set_xticks(budgets)
        ax.set_ylim(0, 102)
        ax.set_xlabel('已查询的不同视角数', fontsize=10)
        ax.set_ylabel('完整任务成功率（%）', fontsize=10)
        ax.set_title(LABELS[m], fontsize=14)
        if m == 0:
            ax.legend(fontsize=7, ncol=2, loc='lower right', frameon=False)
    lines = []
    for m, name in enumerate(MODELS):
        rows = b.recovery['models'][name]['methods']
        a, l = [rows[key][primary][SCOPE] for key in ['local_accel', 'local_rgb_accel']]
        lines.append(f'{LABELS[m]}：{primary} 查询 Accel / 学习器 = {a["success"]["mean"]*100:.2f}% / {l["success"]["mean"]*100:.2f}%')
    b.text(.075, .275, '；'.join(lines), 10, INK)
    intervals = []
    for name in MODELS:
        row = b.recovery['models'][name]['paired_method_comparisons']['local_rgb_accel_minus_random_rgb_accel'][primary][SCOPE]
        intervals.append(f"{row['mean']*100:+.2f} [{row['ci95'][0]*100:+.2f}, {row['ci95'][1]*100:+.2f}]")
    b.text(.075, .245, '16查询，学习器引导 − 随机探查＋学习器（pp，95%区间）：' + ' / '.join(intervals) + '（规范 / 多视角训练）。', 9, MUTED)
    b.save(pdf)


def next_page(brief, pdf):
    b = brief
    b.new(
        '下一阶段：条件视角场与跨场景恢复验证',
        '两阶段计划 · 机制核验与场景扩展同步推进',
        '当前观测预测有利区域 → 到位验证 → 不再主动调整相机安装偏置。',
        '保留模板强基线，扩大独立场景；先冻结 VLA，再判断是否需要联合训练。',
        '方法核验：当前图像场的小模型尚未超过模板；指标预测与任务收益需分别验证。\nRoboCasa：旧增强仅8位置且包含腕部扰动；新主线采用匹配的规范／广域外部后训练。',
        '2026-09-16第二轮审计。LIBERO条件场仅用当前图像及已知参考坐标，不读目标候选图；初態/任务分别留出，尚未超过模板。RoboCasa旧增强使用8离散位置且同时扰动腕部，不替代新主线。复用官方Human300初始化，补做等预算规范／广域外部后训练；不重复完整基础预训练。用户确认左右相机联动、腕部安装不变。RoboCasa左右相机随底盘自然运动，安装偏置固定不等于世界位姿固定；未将底盘动作清零。现有300任务训练源窗口和400评测初态保留清单，支持范围与执行噪声需先冻结。8卡LoRA反向已跑通，OpenPI Accel插桩最终动作对照相同；这些仅为工程验收，不是算法收益。完整计划见 protocols/conditional_field_robocasa_20260916_zh.md。实际获取、连续新位姿及真机仍待验证。')
    blocks = [(.06, '阶段一 · 方法核验', '配对噪声与相对排序\n当前图像预测指标场\n模板／输入条件消融\n指标与任务收益分开'),
              (.375, '阶段二 · 多场景', '复用官方模型初始化\n规范／广域外部匹配训练\n左右联动，腕部安装不变\n布局留出与扰动恢复'),
              (.69, '训练与部署边界', '先冻结 VLA 学视角\n必要时加入任务反馈\n联合训练需单独对照\n到位真实观测再确认')]
    for x, title, body in blocks:
        b.box(x, .32, .255, .40)
        b.text(x+.015, .69, title, 18, BLUE, True)
        b.text(x+.015, .61, body, 12, MUTED)
    b.save(pdf)
