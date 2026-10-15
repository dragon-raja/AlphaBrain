"""Compact diagnostic visualization; no result selection or report mutation."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from ..data.artifacts import read


def draw(root, font):
    root=Path(root);summary=read(root/'summary.json');fixed=read(root/'fixed-support.json')
    font_manager.fontManager.addfont(font)
    plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams['axes.unicode_minus']=False
    labels={'libero97_canonical':'LIBERO · 规范训练', 'libero97_broad':'LIBERO · 多视角训练',
            'plus_native_canonical':'Plus 官方候选 · 规范训练','plus_native_broad':'Plus 官方候选 · 多视角训练',
            'robocasa_official':'RoboCasa · 官方 π0.5','robocasa_xr1':'RoboCasa · XR-1'}
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,(panel,label) in zip(axes.flat,labels.items()):
        source=fixed if panel.startswith('robocasa') else summary
        s=source['panels'][panel]['all']['chunk_translation'];v=100*np.array(s['quartile_success'])
        ax.bar(range(4),v,color=['#1c8078','#67a99e','#94aabd','#476c94'])
        ax.set_xticks(range(4),['最接近多数','次接近','次偏离','最偏离'])
        ax.set_ylim(0,100);ax.set_title(label+'\n初态内秩相关均值 = '+f"{s['rho_mean']:+.3f}",fontsize=12)
        for x,y in enumerate(v):ax.text(x,y+1,f'{y:.1f}%',ha='center',fontsize=10)
        ax.set_ylabel('完整任务成功率 (%)');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('初态跨视角动作共识：更接近多数，不等于更容易成功',fontsize=17)
    fig.text(.5,.015,'同初态、配对评分噪声；近期平移指令距离，非实际运动。LIBERO/Plus：全候选；RoboCasa：固定32扰动终点。\n先在每初态内分组，再平均；相关系数为各初态 Spearman 均值。仅描述性分析，不是新方法验证。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.09,1,.94));fig.savefig(root/'action-consensus.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--font',required=True)
    a=p.parse_args();draw(a.root,a.font)
