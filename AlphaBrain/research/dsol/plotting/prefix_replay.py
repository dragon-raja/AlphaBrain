"""Short-prefix evidence: fixed repeat0 images and all-eight-repeat curves."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from ..data.artifacts import read


def render(config_file,font):
    cfg=read(config_file);out=Path(cfg['output']);summary=read(out/'summary.json');rows=read(out/'branches.json')
    font_manager.fontManager.addfont(font);plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name();plt.rcParams['axes.unicode_minus']=False
    for case in cfg['cases']:
        ncam=2 if case['model']=='libero' else 3
        fig,axes=plt.subplots(ncam*2,5,figsize=(15,ncam*6))
        for ci,candidate in enumerate(case['candidates']):
            with np.load(Path(candidate['path'])/'00'/'telemetry.npz') as z:
                for w in range(5):
                    prefix=f"frame/{w*case['stride']:03d}/obs/"
                    if case['model']=='libero':
                        imgs=[z[prefix+'agentview_image'][::-1,::-1],z[prefix+'robot0_eye_in_hand_image'][::-1,::-1]]
                    else:imgs=[z[prefix+'video.robot0_agentview_left'],z[prefix+'video.robot0_agentview_right'],z[prefix+'video.robot0_eye_in_hand']]
                    camera_labels=['外部相机','腕部相机'] if ncam==2 else ['左外部相机','右外部相机','腕部相机']
                    for cam,img in enumerate(imgs):
                        ax=axes[ci*ncam+cam,w];ax.imshow(img);ax.set_xticks([]);ax.set_yticks([])
                        if cam==0:ax.set_title(f"第{w+1}次评分 / 动作步{w*case['stride']}")
                        if w==0:ax.set_ylabel(candidate['label']+' · '+camera_labels[cam]+f"\n旧完整成功率{100*candidate['historical_endpoint_success']:.1f}%",fontsize=10)
        fig.suptitle(case['title']+'：原噪声编号0的短前缀（未按图像结果挑选）',fontsize=15)
        fig.tight_layout(rect=(0,0,1,.97));fig.savefig(out/(case['id']+'-images.png'),dpi=135);plt.close(fig)
    fig,axes=plt.subplots(len(cfg['cases']),4,figsize=(17,3.4*len(cfg['cases'])))
    for i,case in enumerate(cfg['cases']):
        for candidate,color in zip(case['candidates'],['#176c9d','#b55a35']):
            rr=[r for r in rows if r['case']==case['id'] and r['view']==candidate['view']];a=np.array([r['flow']['score'] for r in rr]);paths=np.array([[v['eef']['path_length']*100 for v in r['stages']] for r in rr]);cmd=np.array([[v['command_translation_norm'] for v in r['stages']] for r in rr])
            for j,values in enumerate([a,np.cumsum(a,axis=1)/np.arange(1,6),paths,cmd]):
                x=np.arange(1,values.shape[1]+1);axes[i,j].plot(x,values.mean(0),'-o',color=color,label=candidate['label']);axes[i,j].fill_between(x,values.min(0),values.max(0),color=color,alpha=.12)
                axes[i,j].set_xticks(x);axes[i,j].spines[['top','right']].set_visible(False)
        axes[i,0].set_ylabel(case['title']+'\n单次评分 Accel');axes[i,1].set_ylabel('累计窗口 Accel（实际选位用）');axes[i,2].set_ylabel('实际末端路径长度 (cm)');axes[i,3].set_ylabel('预测平移指令范数')
        for ax in axes[i]:ax.set_xlabel('评分阶段 / 随后执行的动作块');ax.legend(fontsize=9)
    fig.suptitle('同一初态好坏候选：全8套原评分噪声；阴影为最小—最大值，不是置信区间',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.975));fig.savefig(out/'prefix-curves.png',dpi=145);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--font',required=True);a=p.parse_args();render(a.config,a.font)
