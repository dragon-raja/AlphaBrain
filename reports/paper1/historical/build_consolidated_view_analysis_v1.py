#!/usr/bin/env python3
"""One eight-page scientific brief: spatial structure, metrics, Oracle, scope.

Each build is archived in a timestamped directory. One stable PDF in docs is the
user-facing entry; an optional CPU-only watcher refreshes it after full64 passes.
"""
from __future__ import annotations

from pathlib import Path as _RepoPath
import sys as _sys
_REPOSITORY_ROOT = _RepoPath(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPOSITORY_ROOT))


import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from scripts.dsol_paper1.analysis.analyze_view_oracle_comparison_v2 import load_verified, paired_training_rows, LABELS, ORACLES
from scripts.dsol_paper1.analysis.analyze_view_metric_rules_v1 import RULES
from reports.paper1.historical.build_view_landscape_report_v1 import read_json, write_json, sha256, ranks, setup_plotting, require, PARAMS
from reports.paper1.historical.build_view_metric_comparison_brief_v1 import camera_centers

REPO=_REPOSITORY_ROOT
ROOT=Path('/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908')
OUTPUT=ROOT/'consolidated-analysis-v1'
PUBLIC=REPO/'docs/dsol_paper1/vla_view_landscape_and_metrics_20260908_zh.pdf'
TASKS={'goal_cream_cheese_bowl':'奶酪入碗','goal_top_drawer_bowl':'碗入上抽屉','goal_wine_rack':'酒瓶入架',
       'libero10_book_caddy':'书入收纳盒','libero10_bowl_bottom_drawer':'碗入下抽屉','libero10_mug_microwave':'杯入微波炉',
       'object_cream_cheese_basket':'奶酪入篮','spatial_drawer_bowl_plate':'抽屉取碗入盘'}


def csv_rows(path):
    with path.open() as stream: return list(csv.DictReader(stream))


def normalized_ranks(values):
    values=np.asarray(values)
    require(values.ndim==1 and len(values)>1,'Need at least two values to normalize ranks')
    return (ranks(values)-1)/(len(values)-1)


def empirical_hierarchy(outcomes, tasks, canonical_index=0):
    """Retrospective maxima on one matrix; NOT independently validated policies.

    Equal task weighting; equal state weighting within each task. The candidate
    set includes the canonical option at every level.
    """
    outcomes=np.asarray(outcomes, dtype=float); tasks=np.asarray(tasks)
    require(outcomes.ndim==3 and len(outcomes)==len(tasks) and len(tasks)>0,
            'Expected nonempty state x candidate x noise tensor and task labels')
    require(np.isfinite(outcomes).all() and np.isin(outcomes,[0,1]).all(), 'Invalid success cells')
    rates=outcomes.mean(2)
    groups=[np.flatnonzero(tasks==task) for task in np.unique(tasks)]
    task_rates=np.array([rates[g].mean(0) for g in groups])
    return dict(canonical=float(task_rates[:,canonical_index].mean()),
                global_fixed=float(task_rates.mean(0).max()),
                task_fixed=float(task_rates.max(1).mean()),
                statewise=float(np.mean([rates[g].max(1).mean() for g in groups])))


def read_complete(directory):
    receipt=read_json(directory/'completion.json')
    require(receipt['status']=='PASS_COMPLETE','Analysis incomplete')
    for name,digest in receipt['outputs'].items(): require(sha256(directory/name)==digest,'Analysis output changed: '+name)
    return receipt


def current_status():
    paths={'first8':ROOT/'controller-status.json','extension':ROOT/'full64-extension-v2/controller-status.json',
           'speed':ROOT/'full64-extension-v2/early-speed-gate-v1/status.json','performance':ROOT/'full64-extension-v2/performance-choice.json'}
    return {k:read_json(p) if p.exists() else None for k,p in paths.items()}


def build():
    broad=load_verified(ROOT/'broad-existing','broad')
    read_complete(ROOT/'metric-oracle-comparison-v2'); read_complete(ROOT/'metric-rules-v1')
    summary=read_json(ROOT/'metric-oracle-comparison-v2/broad_summary.json')
    metrics=read_json(ROOT/'metric-rules-v1/summary.json')
    core=read_json(ROOT/'broad-existing/summary.json')
    associations=csv_rows(ROOT/'metric-rules-v1/state_association_stability.csv')
    contrasts=read_json(ROOT/'metric-rules-v1/pairwise_rule_contrasts.json')
    q_path=Path('/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/final-analysis/analysis.json')
    q_analysis=read_json(q_path)
    require(q_analysis['status']=='PASS_Q_ANALYZED','Historical Q analysis incomplete')
    q=next(x for x in q_analysis['checkpoint_results'] if x['checkpoint_seed']==41)
    require(q['state_count']==16 and q['noise_repeats']==64 and q['noise_banks']==['Q'], 'Unexpected Q scope')
    inputs={**{str(v['path']):v['sha256'] for v in broad['input_identity'].values()},
            str(ROOT/'broad-existing/summary.json'):sha256(ROOT/'broad-existing/summary.json'),
            str(ROOT/'metric-oracle-comparison-v2/completion.json'):sha256(ROOT/'metric-oracle-comparison-v2/completion.json'),
            str(ROOT/'metric-rules-v1/completion.json'):sha256(ROOT/'metric-rules-v1/completion.json')}
    inputs[str(q_path)]=sha256(q_path)
    canonical=None; canonical_summary=None
    full=ROOT/'full64-extension-v2'
    if (full/'matched64-metrics-with-oracle/completion.json').exists():
        read_complete(full/'matched64-metrics-with-oracle')
        canonical=load_verified(full/'canonical-report-full64','canonical'); paired_training_rows(canonical,broad)
        canonical_summary=read_json(full/'matched64-metrics-with-oracle/canonical_summary.json')
        inputs.update({str(v['path']):v['sha256'] for v in canonical['input_identity'].values()})
        inputs[str(full/'matched64-metrics-with-oracle/completion.json')]=sha256(full/'matched64-metrics-with-oracle/completion.json')
    stamp=datetime.now(timezone.utc); stamp_id=stamp.strftime('%Y%m%dT%H%M%S%fZ')
    out=OUTPUT/'history'/stamp_id; out.mkdir(parents=True)
    (out/'figures').mkdir()
    status=current_status()
    setup_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.font_manager import fontManager
    from matplotlib.lines import Line2D
    fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'); fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','pdf.fonttype':3,'axes.unicode_minus':False,'font.size':10})
    states=broad['states']; ids=broad['candidate_ids']; poses=broad['poses']
    require(ids[0]=='canonical','Canonical candidate order changed')
    test_mask=np.array([s['split']=='test' for s in states])
    q_keys={s['pair_key'] for s in q['state_metrics']}
    require(q_keys=={s['pair_key'] for s in states if s['split']=='test'},'O and Q test source mismatch')
    task_ids=np.array([s['task_id'] for s in states])
    headroom={name:empirical_hierarchy(broad['success'][mask],task_ids[mask])
              for name,mask in [('all64',np.ones(len(states),dtype=bool)),('test16',test_mask)]}
    write_json(out/'empirical_headroom.json',dict(estimand='same-sample empirical maxima, not population upper bounds',
               scopes=headroom,canonical_included=True,new_gpu_calls=0))
    success=broad['success'].mean(2); accel=broad['accel'].mean(2); visibility=broad['visibility']
    geometry=np.array([[p[k] for k in PARAMS] for p in poses]); pages=[]

    def frame(fig,title,subtitle):
        fig.suptitle(title,x=.035,y=.970,ha='left',fontsize=21,fontweight='bold',color='#19354c')
        fig.text(.036,.91,subtitle,va='top',fontsize=10,color='#536577')

    def note(fig,text,y=.080): fig.text(.035,y,text,fontsize=9,color='#536577',va='top')

    def save(fig,name,pdf,conclusion):
        number=len(pages)+1
        fig.text(.035,.020,'VLA 视角空间与指标分析  |  历史 AA4 数据，渲染稳健性待复核；非主动获取验证',fontsize=8,color='#738395')
        fig.text(.97,.020,f'{number}/8',ha='right',fontsize=8,color='#738395')
        path=out/'figures'/f'{number:02d}_{name}.png'; fig.savefig(path,dpi=150); pdf.savefig(fig); plt.close(fig)
        pages.append({'page':number,'figure':str(path),'conclusion':conclusion})

    def error_points(ax,values,positions,scale=1,color='#2a6f97',offset=0,marker='o'):
        for pos,value in zip(positions,values):
            if value['mean'] is None: continue
            mean=value['mean']*scale; ci=value['ci95']
            if ci[0] is not None: ax.plot(np.array(ci)*scale,[pos+offset]*2,c=color,lw=1.3)
            ax.scatter(mean,pos+offset,c=color,s=30,marker=marker,zorder=3)

    with PdfPages(out/'consolidated_view_analysis.pdf') as pdf:
        # 1: actual catalog parameters, not a pooled world-coordinate map.
        fig=plt.figure(figsize=(16,9))
        frame(fig,'候选视角空间：同一物理状态下改变外部相机位姿',
            '已完成数据：多视角后训练模型（Broad M-B，seed41）；8 任务 × 8 来源状态 × 97 候选 × 32 闭环重复。Accel 对 8 个初噪取平均。')
        for col,(values,title,cmap) in enumerate(((success.mean(0),'闭环成功率（跨 64 状态均值）','viridis'),
                (accel.mean(0),'平均 Accel（跨状态，仅作总体趋势）','magma'),(visibility.mean(0),'可见像素占比（跨状态均值）','cividis'))):
            ax=fig.add_subplot(1,3,col+1,projection='3d')
            lo,hi=(0,1) if col==0 else (float(values.min()),float(values.max()))
            sc=ax.scatter(*geometry[1:].T,c=values[1:],cmap=cmap,vmin=lo,vmax=hi,s=28,depthshade=False)
            ax.scatter(*geometry[0],c=[values[0]],cmap=cmap,vmin=lo,vmax=hi,marker='*',s=135,edgecolor='black',depthshade=False)
            ax.set(xlabel='方位角偏移（°）',ylabel='仰角偏移（°）',zlabel='',title=title)
            ax.view_init(22,-60); ax.tick_params(labelsize=8); fig.colorbar(sc,ax=ax,shrink=.47,pad=.12).ax.tick_params(labelsize=8)
        fig.subplots_adjust(left=.015,right=.98,top=.78,bottom=.25,wspace=.10)
        note(fig,'三维坐标：方位角偏移、仰角偏移、半径倍率（纵轴）。任务是目标；状态是物体／机器人及任务进度的一个物理快照。',.17)
        note(fig,'图中坐标是相对规范相机的参数，不是各任务共用的绝对 XYZ。每次闭环外部相机固定，机器人与腕部图像继续变化。',.125)
        note(fig,'星号为规范视角；颜色不含深度明暗，不作空间插值。总体均值不能代替状态内分析。',.080)
        save(fig,'candidate_space',pdf,'候选空间需在同一状态内比较；跨状态参数均值不是统一世界位置的成功率场。')

        # 2: every historical source and candidate remains visible.
        fig,axes=plt.subplots(1,3,figsize=(16,9))
        frame(fig,'完整状态—候选矩阵：不同状态的视角价值并不相同',
            'Broad 模型；三列使用相同状态和候选顺序。每行是一个来源状态，每格是一个候选。排序图只表达相对名次，不表达原始差值大小。')
        matrices=(success,np.array([1-normalized_ranks(a) for a in accel]),np.array([normalized_ranks(v) for v in visibility]))
        ticks=[]; labels=[]
        for task in TASKS:
            where=[i for i,s in enumerate(states) if s['task_id']==task]
            ticks.append(float(np.mean(where))); labels.append(TASKS[task])
        for j,(ax,values,title) in enumerate(zip(axes,matrices,('成功率（32 次重复）','低 Accel 排名（1 为更低）','可见性排名（1 为更高）'))):
            im=ax.imshow(values,aspect='auto',vmin=0,vmax=1,cmap='viridis',interpolation='nearest')
            ax.set_title(title); ax.set_xlabel('候选索引（0 为规范视角）'); ax.set_yticks(ticks,labels if j==0 else [])
            for i in range(1,len(states)):
                if states[i]['task_id']!=states[i-1]['task_id']: ax.axhline(i-.5,c='white',lw=1.0)
                elif states[i]['split']!=states[i-1]['split']: ax.axhline(i-.5,c='white',lw=.6,ls=':')
            fig.colorbar(im,ax=ax,shrink=.68,pad=.02)
        fig.subplots_adjust(left=.12,right=.98,top=.79,bottom=.19,wspace=.19)
        note(fig,'候选为规范点＋按方位角／仰角／半径排序的 96 点。白实线分任务，白点线区分历史 development / test 来源。',.12)
        note(fig,'所有 64 状态均保留；不能把接近满分的单个快照称作“整个任务简单”，也不能凭少量示例判断全局规律。',.075)
        save(fig,'all_state_matrices',pdf,'总体相关性和个别好视角都不足以替代完整状态—候选矩阵。')

        # 3: retain the original training-support distance analysis.
        fig,axes=plt.subplots(2,2,figsize=(16,9))
        frame(fig,'几何距离与训练覆盖：状态内的平均秩相关较弱',
            'Broad 模型：左列为候选成功率，右列为平均 Accel；点值均跨 64 状态平均。ρ 为先在状态内计算，再按来源汇总。')
        distances=[np.array([p['distance_canonical_parameter_normalized'] for p in poses]),np.array([p['distance_train64_parameter_normalized'] for p in poses])]
        masks=[np.arange(97)>0,np.array([p['catalog_group']=='heldout32' for p in poses])]
        for row,(dist,mask) in enumerate(zip(distances,masks)):
            for col,values in enumerate((success.mean(0),accel.mean(0))):
                ax=axes[row,col]; support_sc=ax.scatter(dist[mask],values[mask],s=25,c=geometry[mask,2],cmap='viridis',alpha=.9,vmin=geometry[:,2].min(),vmax=geometry[:,2].max())
                key=('spearman_canonical_distance_' if row==0 else 'spearman_train_distance_')+('success' if col==0 else 'accel')+('_heldout' if row else '')
                rho=core['all'][key]['mean']
                ax.set_title(f'状态内平均秩相关 ρ = {rho:+.3f}')
                ax.set_xlabel('到规范视角的归一化参数距离（96 点）' if row==0 else '到最近后训练视角的参数距离（32 个留出视角）')
                ax.set_ylabel('成功率' if col==0 else '平均 Accel'); ax.grid(alpha=.15)
                if col==0: ax.set_ylim(0,1)
        fig.subplots_adjust(left=.075,right=.94,top=.79,bottom=.20,hspace=.66,wspace=.20)
        bar=fig.colorbar(support_sc,cax=fig.add_axes((.96,.27,.008,.44))); bar.ax.set_title('半径倍率',fontsize=8,pad=8); bar.ax.tick_params(labelsize=7)
        note(fig,'距离按方位角、仰角、半径的参数范围归一化，不是米制距离。下排只看 Broad 后训练未覆盖的 32 个候选。',.12)
        note(fig,'颜色表示半径倍率；几何相关性不证明模型“熟悉度”，也未隔离目标大小、遮挡或训练覆盖的因果作用。',.075)
        save(fig,'support_geometry',pdf,'离规范位置或后训练支持更近，不等于已经找到闭环更好的视角。')

        # 4: exact single-state XYZ mapping across three metrics.
        fig=plt.figure(figsize=(16,10))
        frame(fig,'同一状态的真实相机位置：成功率、Accel 与可见性逐点对齐',
            'Broad 模型；两行分别为不同物理状态，从缓存外参读取真实相机中心 XYZ（米）。示例按任务内来源名称固定选择，未按指标效果挑选。')
        for row,task in enumerate(('goal_cream_cheese_bowl','goal_wine_rack')):
            eligible=[(i,s) for i,s in enumerate(states) if s['task_id']==task and s['split']=='development']
            i,state=min(eligible,key=lambda p:(p[1]['source_group'],p[1]['source_state_index']))
            xyz=camera_centers(state,ids); center=xyz.mean(0); extent=np.ptp(xyz,axis=0).max()*.55
            for col,(values,title,cmap) in enumerate(((success[i],'成功率（越高越好）','viridis'),(accel[i],'平均 Accel（假设越低越好）','magma'),(visibility[i],'可见像素占比','cividis'))):
                ax=fig.add_subplot(2,3,row*3+col+1,projection='3d'); lo,hi=(0,1) if col==0 else (float(values.min()),float(values.max())+1e-12)
                sc=ax.scatter(*xyz[1:].T,c=values[1:],vmin=lo,vmax=hi,cmap=cmap,s=23,depthshade=False)
                ax.scatter(*xyz[0],c=[values[0]],vmin=lo,vmax=hi,cmap=cmap,s=110,marker='*',edgecolor='black',depthshade=False)
                ax.set(xlabel='X（米）',ylabel='Y（米）',zlabel='Z',title=f"{TASKS[task]} / {state['demo_name']} / frame {state['source_state_index']}\n{title}",
                    xlim=(center[0]-extent,center[0]+extent),ylim=(center[1]-extent,center[1]+extent),zlim=(center[2]-extent,center[2]+extent))
                ax.set_box_aspect((1,1,1)); ax.view_init(20,-60); ax.tick_params(labelsize=7)
                fig.colorbar(sc,ax=ax,shrink=.43,pad=.13).ax.tick_params(labelsize=8)
        fig.subplots_adjust(left=.02,right=.98,top=.79,bottom=.16,wspace=.08,hspace=.34)
        note(fig,'奶酪状态接近天花板；酒瓶状态具有明显视角差异。示例用于解释结构，不替代第 2 页完整矩阵。',.11)
        note(fig,'Accel／可见性色标按当前状态范围显示，不能跨行只看颜色。XYZ 未展示朝向；像素占比增大不等于信息量增加。',.070)
        save(fig,'same_state_xyz',pdf,'部分状态低 Accel 区域质量较好，但可见像素占比与闭环价值不具有简单等价关系。')

        # 5: all nine comparators, with Oracle separated as label-using references.
        model_summaries={'多视角训练':summary}
        if canonical_summary: model_summaries['规范视角训练']=canonical_summary
        fig,axes=plt.subplots(len(model_summaries),3,figsize=(16,9 if len(model_summaries)==1 else 13),squeeze=False)
        frame(fig,'指标筛选收益与 Oracle 参照：随机候选、规范视角是两条不同基线',
            '七条指标规则固定；Top10 不按成功率调参。经验 Oracle 使用成功标签，独立列为收益参照，不与可计算规则混称同类方法。')
        for row,(model_name,s) in enumerate(model_summaries.items()):
            for col,(field,scale,title) in enumerate((('success',100,'成功率（%）'),('gain_vs_uniform_pp',1,'相对均匀候选（百分点）'),('gain_vs_canonical_pp',1,'相对规范视角（百分点）'))):
                ax=axes[row,col]
                for j,rule in enumerate(RULES+ORACLES):
                    item=s['overall']['rules'][rule][field]; color='#b66a20' if rule in ORACLES else '#2a6f97'
                    error_points(ax,[item],[j],scale,color,marker='D' if rule in ORACLES else 'o')
                    if col==0: ax.text(item['mean']*100+.6,j-.12,f"{item['mean']*100:.2f}",fontsize=8)
                ax.axhline(6.5,color='#999',ls='--',lw=.7); ax.set_ylim(8.6,-.7)
                display_labels={**LABELS,ORACLES[0]:'每状态经验最佳（同 32 次选／报）',ORACLES[1]:'噪声留半复核（16 次选／16 次测）'}
                ax.set_yticks(range(9),[display_labels[r] for r in RULES+ORACLES] if col==0 else []); ax.set_xlabel(title); ax.set_title(model_name+' / 64 状态',fontsize=11)
                if col: ax.axvline(0,c='#666',lw=.8)
                else: ax.set_xlim(0,100)
                ax.grid(axis='x',alpha=.15)
        fig.subplots_adjust(left=.29,right=.975,top=.81,bottom=.23,wspace=.20,hspace=.42)
        gain=contrasts['contrasts']['visibility_refinement_within_accel_top10']['gain_pp']
        note(fig,f"Broad 的 Top10 内可见性二次筛选增量：{gain['mean']:+.2f} pp，95% 区间 [{gain['ci95'][0]:+.2f}, {gain['ci95'][1]:+.2f}]；额外收益尚未确认。",.16)
        note(fig,'经验 Oracle：每状态按 O32 平均挑一个视角，再报告同批结果，存在乐观偏差；不是每条噪声各挑一个成功视角。',.115)
        note(fig,'噪声留半复核：16 次选、另 16 次测，再交换；复用已有闭环数据，不是新算法或新增评测。横线为描述性来源 bootstrap 95% 区间。',.075)
        save(fig,'metrics_and_oracle',pdf,'低 Accel 初筛相对均匀选择有小幅收益；可见性净增益与超过规范视角的稳定收益尚未确认。')

        # 6: noise stability is separate from closed-loop quality.
        fig,axes=plt.subplots(1,3,figsize=(16,9))
        frame(fig,'指标可靠性：候选排序信号较弱，集合成员对初噪敏感',
            'Broad 模型；每状态单独计算指标与成功率的关系，再比较评分初噪之间的 Top10 集合。相关性、集合稳定性与闭环收益是不同问题。')
        keys=('rho_negative_accel_success','rho_visibility_success','top10_pairwise_jaccard')
        titles=('ρ（−Accel，成功率）','ρ（可见像素占比，成功率）','Top10 跨初噪两两 Jaccard')
        for ax,key,title in zip(axes,keys,titles):
            values=np.array([float(r[key] or 'nan') for r in associations]); values=values[np.isfinite(values)]
            ax.hist(values,bins=np.linspace(0,1,13) if key.endswith('jaccard') else np.linspace(-1,1,17),color='#4b83a5',edgecolor='white')
            mean=metrics['overall']['stability' if key.endswith('jaccard') else 'associations'][key]['mean']
            ax.axvline(mean,c='#b66a20',ls='--',lw=1.6); ax.set_title(f'{title}\n汇总均值 {mean:.3f} / {len(values)} 状态有效')
            ax.set(xlabel='指标值',ylabel='状态数量'); ax.grid(axis='y',alpha=.15)
        fig.subplots_adjust(left=.055,right=.97,top=.77,bottom=.27,wspace=.27)
        note(fig,'状态内成功率恒定时秩相关无定义，不能人为补成零。Accel 与可见性相关图各有 62 个有效状态。',.19)
        note(fig,'8 个评分初噪的 Top10 平均两两 Jaccard 约 0.288；没有状态的 8 份 Top10 完全一致。',.145)
        note(fig,'集合成员变化不自动等于集合没有价值：仍需看规则的闭环收益对照。可见性使用仿真任务实体分割，是特权像素占比代理量。',.095)
        save(fig,'metric_noise_reliability',pdf,'排序稳定性不足与部分筛选收益可以同时存在；不能把像素占比直接称作任务信息量。')

        # 7: separate empirical headroom from a finite search procedure's value.
        fig,axes=plt.subplots(1,3,figsize=(16,9))
        frame(fig,'候选空间收益与搜索兑现：75.88% 和 62.99% 回答不同问题',
            '同一 Broad M-B seed41 模型。左、中：在完整候选矩阵上事后选最佳；右：旧搜索程序选定视角后，在独立噪声上闭环评价。')
        for ax,scope,title in zip(axes[:2],('all64','test16'),('全 64 状态 / O32','其中 16 个 test 状态 / O32')):
            values=[100*headroom[scope][k] for k in ('canonical','global_fixed','task_fixed','statewise')]
            ax.bar(range(4),values,color=['#98a5b0','#7594ae','#316ba2','#198878'])
            ax.set_xticks(range(4),['规范','全局最佳\n固定','每任务\n最佳固定','每状态\n经验最佳'],fontsize=9)
            ax.set_title(title+'\n97 候选均可选，包含规范视角',fontsize=11)
            for i,v in enumerate(values): ax.text(i,v+1.2,f'{v:.2f}',ha='center',fontsize=11,fontweight='bold')
        ax=axes[2]
        values=[100*q['population'][k]['success'] for k in ('canonical','task_fixed','statewise_P_top1')]
        ax.bar(range(3),values,color=['#98a5b0','#316ba2','#198878'])
        ax.set_xticks(range(3),['规范','开发来源选定\n每任务固定','O/P 选定\n每状态视角'],fontsize=9)
        ax.set_title('同 16 个 test 状态 / 独立 Q64\n旧程序：每状态搜索强制选非规范视角',fontsize=11)
        for i,v in enumerate(values): ax.text(i,v+1.2,f'{v:.2f}',ha='center',fontsize=11,fontweight='bold')
        for ax in axes:
            ax.set_ylim(0,100); ax.set_ylabel('闭环成功率（%）'); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
        fig.subplots_adjust(left=.055,right=.975,top=.75,bottom=.31,wspace=.23)
        h=headroom['all64']
        note(fig,f"全矩阵经验收益：每状态最佳相对规范 +{100*(h['statewise']-h['canonical']):.2f} pp，相对每任务最佳固定 +{100*(h['statewise']-h['task_fixed']):.2f} pp；不是每个噪声各选一个视角。",.235)
        note(fig,'左、中重复使用同批成功标签；选择越灵活，经验最大值越容易偏高。它是经验收益空间，不是已验证的真实期望上限。',.185)
        note(fig,'右侧 +3.91 pp（95% 区间 [0.00, 7.81]）只评价旧有限搜索程序；未通过其预设门槛，不等于候选空间无稳定收益。',.135)
        note(fig,'各面板只能先与本面板基线比较。每状态指不同起始快照分别选视角，单次 rollout 内外部相机保持固定。',.085)
        save(fig,'headroom_and_independent_search',pdf,'完整矩阵显示经验收益空间；旧独立验证测量的是有限搜索程序，不能据此宣布真实 Oracle 上限很低。')

        # 8: one interpretation and one matched-training scope, no status-as-result.
        fig=plt.figure(figsize=(16,9)); ax=fig.add_axes((.045,.31,.91,.44)); ax.axis('off')
        frame(fig,'阶段结论与训练覆盖对照：区分筛选信号、稳定收益和可执行获取',
            '当前证据支持有限的候选筛选信号；论文中的训练交互与状态泛化结论，需要完整配对矩阵，不能由示例图或经验最大值代替。')
        rows=[['训练条件','多视角训练模型','规范视角训练模型'],
              ['模型／训练预算','同一初始化；seed41；2,000 步','同一初始化；seed41；2,000 步'],
              ['输入与操作条件','外部图像＋腕部图像＋语言','相同；只改变外部后训练位姿覆盖'],
              ['目标评测范围','64 来源状态 × 97 视角 × O32','相同 64 状态 × 97 视角 × O32'],
              ['完整闭环矩阵','已完成：198,656 次','已完成：198,656 次' if canonical else '评测进行中：首批 8＋补充 56 状态'],
              ['指标／Oracle 比较','七条固定规则＋两项 Oracle 参照','相同规则、候选、初噪与去噪步数'],
              ['当前可作训练效果结论','完整来源与噪声条件下的配对诊断' if canonical else '等待右侧完整矩阵；不作不等规模均值相减','来源标签保留；不是新任务或跨训练 seed 证明']]
        table=ax.table(cellText=rows,cellLoc='left',colWidths=[.20,.39,.41],loc='center'); table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1,2.4)
        for (r,c),cell in table.get_celld().items():
            cell.set_edgecolor('#d8e1e8')
            if r==0: cell.set_facecolor('#e9f0f5'); cell.set_text_props(weight='bold')
            else: cell.set_facecolor('#f7fafc' if r%2 else 'white')
        note(fig,'结论 1：在 Broad 模型中，低 Accel 相对均匀候选有部分筛选价值，但不等于已稳定超过规范视角。',.235)
        note(fig,'结论 2：在 Broad 模型中，可见像素占比辅助筛选的净增益尚未确认；初噪稳定性与任务信息性需要分别评价。',.185)
        note(fig,'结论 3：每状态经验最佳 75.88%，高于规范 64.70% 和每任务最佳固定 69.34%；稳定收益与可预测性仍须分别验证。',.135)
        note(fig,'状态均来自上述 8 个已知任务；固定外部相机的重启闭环评测不包含候选获取成本，也不能直接证明未见任务泛化。',.085)
        save(fig,'conclusions_and_matched_scope',pdf,'完整配对实验用于识别训练覆盖如何改变指标收益；当前静态证据不直接构成主动感知算法贡献。')
    require(len(pages)==8,'Expected eight-page consolidated brief')
    provenance=dict(status='PASS_CONSOLIDATED_EIGHT_PAGE_REPORT',created_at_utc=stamp.isoformat(),script=source_identity(Path(__file__)),
        inputs=inputs,canonical_full64_available=canonical is not None,pages=pages,execution_snapshot=status,new_gpu_calls=0,
        pdf_sha256=sha256(out/'consolidated_view_analysis.pdf'))
    write_json(out/'provenance.json',provenance)
    PUBLIC.parent.mkdir(parents=True,exist_ok=True)
    unified=REPO/'reports/paper1/unified.py'
    publish_unified=unified.exists() and (ROOT/'render-protocol-bridge-v1/analysis-recovery-v1.json').exists()
    if not publish_unified:
        temporary=PUBLIC.with_suffix('.building.pdf'); shutil.copyfile(out/'consolidated_view_analysis.pdf',temporary); temporary.replace(PUBLIC)
    write_json(OUTPUT/'latest.json',dict(status=provenance['status'],pdf=str(out/'consolidated_view_analysis.pdf'),
        public_report=str(REPO/'docs/dsol_paper1/vla_view_research_focus_20260907_zh.pdf') if publish_unified else str(PUBLIC),
        archive=str(out),sha256=provenance['pdf_sha256'],canonical_full64_available=canonical is not None))
    if publish_unified:
        subprocess.run([sys.executable,str(unified)],check=True)
    print('PASS_CONSOLIDATED_ANALYSIS_AND_REPORT',out,flush=True)
    return provenance


def source_identity(path): return {'path':str(path),'sha256':sha256(path)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--watch-full64',action='store_true'); args=parser.parse_args()
    if args.watch_full64:
        deadline=time.time()+7*24*3600
        while time.time()<deadline:
            target=ROOT/'full64-extension-v2/matched64-metrics-with-oracle/completion.json'
            if target.exists():
                # Execute the current on-disk builder in a fresh interpreter,
                # so later authorized report edits cannot mismatch code hashes.
                subprocess.run([sys.executable,str(Path(__file__))],check=True)
                break
            time.sleep(30)
        else: raise SystemExit('Full64 report did not complete within the seven-day watch budget')
    else: build()
