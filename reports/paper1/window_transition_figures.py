"""Two review figures from the matched audit; never publishes/replaces the PDF."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

from AlphaBrain.research.dsol.data.artifacts import read
from AlphaBrain.research.dsol.plotting.window_comparison import transition_heatmap, paired_stage_margin
from .window_synthesis import SHORT_TASKS
from .robocasa_results import TASK_NAMES


def build(audit, font):
    root=Path(audit); summary=read(root/'summary.json'); rows=read(root/'initial-transitions.json')
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
                         'axes.unicode_minus':False})
    fig,axes=plt.subplots(1,3,figsize=(16,6),layout='constrained')
    panels=[('libero97_canonical','LIBERO · 规范训练 · 32次执行'),
            ('robocasa_official','RoboCasa · π0.5官方 · 8次执行'),
            ('robocasa_xr1','RoboCasa · XR-1官方 · 8次执行')]
    for ax,(key,title) in zip(axes,panels):
        im=transition_heatmap(ax,summary['panels'][key]['include'],dict(SHORT_TASKS,**TASK_NAMES),title)
        fig.colorbar(im,ax=ax,shrink=.75,label='任务内成功率变化（百分点）')
    fig.suptitle('配对初态与执行噪声：窗口改变后的逐任务净变化\n允许规范位；每图独立色标。跨环境不是相同时长实验。',fontsize=14)
    fig.savefig(root/'task-transitions.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    cases=[('libero97_canonical',1,1,3,'LIBERO：W1→W3纠正选择'),
           ('libero97_canonical',5,3,5,'LIBERO：W3→W5破坏选择'),
           ('robocasa_xr1','39dd9788d2b8a223cfc09850',3,5,'XR-1：W3→W5破坏选择')]
    for ax,(key,initial,start,end,title) in zip(axes,cases):
        row=next(r for r in rows if r['panel']==key and r['initial']==initial and r['start']==start
                 and r['end']==end and r['mode']=='include')
        paired_stage_margin(ax,row,title)
    fig.suptitle('低Accel偏好也会稳定地选错：正值支持新候选，不等于新候选更容易成功\n三例均有8/8评分噪声支持更新；后两例的坏选择也不是评分分半不一致。',fontsize=13)
    fig.savefig(root/'paired-stage-margins.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--font',required=True)
    args=p.parse_args();build(args.audit,args.font)
