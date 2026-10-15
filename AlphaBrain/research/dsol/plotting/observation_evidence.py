"""Contact sheets from immutable diagnostic inputs; no image synthesis."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from ..data.artifacts import read,sha


def build(panel_path):
    panel=read(panel_path);output=Path(panel_path).parent/'contact-sheets';output.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    for benchmark,spec in panel['benchmarks'].items():
        assert sha(spec['payload'])==spec['payload_sha256']
        images=np.load(spec['payload'])['images'];records=spec['records']
        columns=[(0,0),(0,1),(1,0),(1,1),(2,0),(2,1),(0,2)] if benchmark=='robocasa' else [(0,0),(1,0),(2,0),(0,1)]
        for task in sorted({r['task'] for r in records}):
            indices=[i for i,r in enumerate(records) if r['task']==task];width=224*len(columns)
            canvas=Image.new('RGB',(width,80+len(indices)*270),'white');draw=ImageDraw.Draw(canvas)
            draw.text((8,5),f'{benchmark}: {task}',fill='black',font=font)
            draw.text((8,27),records[indices[0]]['prompt'],fill='black',font=font)
            draw.text((8,49),'Same source: canonical / broad A / broad B; wrist shared. Top/bottom: natural state change.',fill='black',font=font)
            for row,i in enumerate(indices):
                r=records[i];base=80+row*270
                for col,(view,camera) in enumerate(columns):
                    image=Image.fromarray(images[i,view,camera]).resize((224,224))
                    canvas.paste(image,(col*224,base+35))
                    camera_name=('left','right','wrist')[camera] if benchmark=='robocasa' else ('external','wrist')[camera]
                    label=f'f{r["frame"]} {r["view_ids"][view]} {camera_name}'
                    draw.text((col*224+3,base+4),label,fill='black',font=font)
            canvas.save(output/f'{benchmark}-{task}.png')
    print(output)


def response_figure(summary_path):
    """Show model-within-benchmark changes and every task, not success estimates."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    summary=read(summary_path)
    fig,axes=plt.subplots(1,3,figsize=(16,5),gridspec_kw={'width_ratios':[1.15,1,1]})
    colors=['#2966ac','#d07831']
    keys=['view_response_rms','canonical_state_response_rms','perturbed_demonstration_mse']
    for j,benchmark in enumerate(('libero','robocasa')):
        item=summary['benchmarks'][benchmark];models=item['models']
        canonical=models['canonical' if benchmark=='libero' else 'canonical_8250']
        broad=models['broad' if benchmark=='libero' else 'broad_8250']
        ratios=[broad[k]/canonical[k] for k in keys]
        bars=axes[0].bar(np.arange(3)+(j-.5)*.35,ratios,.35,label=benchmark.upper(),color=colors[j])
        axes[0].bar_label(bars,fmt='%.3f',fontsize=9,padding=3)
        errors=[]
        for model in (canonical,broad):
            errors.append(np.array(model['per_state_view_mse']).reshape(-1,2,3)[:,:,1:].mean((1,2)))
        delta=errors[1]-errors[0]
        labels=[t.replace('PickPlace','').replace('libero10_','').replace('goal_','').replace('object_','').replace('spatial_','') for t in item['tasks']]
        ax=axes[j+1];ax.barh(np.arange(len(labels)),delta,color=[colors[j] if d<=0 else '#a94d54' for d in delta])
        ax.set_yticks(np.arange(len(labels)),labels,fontsize=8);ax.invert_yaxis();ax.axvline(0,color='black',lw=.8)
        ax.set_xlabel('Broad minus canonical: demonstration MSE\nNegative = better label fit, not rollout success')
        ax.set_title(benchmark.upper()+' | each task, perturbed views')
        ax.grid(axis='x',alpha=.15)
    axes[0].axhline(1,color='black',ls='--',lw=1)
    axes[0].set_xticks(np.arange(3),['Camera-only\nresponse','Natural-state\nresponse','Perturbed-view\nlabel error'])
    axes[0].set_ylabel('Multiview / canonical model (within benchmark)')
    axes[0].set_ylim(0,1.3);axes[0].legend(frameon=False);axes[0].set_title('Reduced view response is not state collapse')
    fig.suptitle('Frozen training-source diagnostic | LIBERO: 8 pairs; RoboCasa: 10 pairs; 8 paired noises',fontsize=13)
    fig.text(.5,.015,'First 6 end-effector channels, fixed training scales. Natural states change all available inputs. No new closed-loop evaluation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.94))
    output=Path(summary_path).parent/'observation-response.png';fig.savefig(output,dpi=170);plt.close(fig)
    return output


def compact_response_figure(summary):
    """Chinese summary of the same audited ratios; no new statistical estimate."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.6), gridspec_kw={'width_ratios': [1.6, 1]})
    keys = ['view_response_rms', 'canonical_state_response_rms', 'perturbed_demonstration_mse']
    for j, (benchmark, color, label) in enumerate([
        ('libero', '#2966ac', 'LIBERO'), ('robocasa', '#d07831', 'RoboCasa')
    ]):
        models = summary['benchmarks'][benchmark]['models']
        canonical = models['canonical' if benchmark == 'libero' else 'canonical_8250']
        broad = models['broad' if benchmark == 'libero' else 'broad_8250']
        ratios = np.array([broad[k] / canonical[k] for k in keys])
        bars = axes[0].bar(np.arange(2)+(j-.5)*.3, ratios[:2], .28, color=color, label=label)
        axes[0].bar_label(bars, fmt='%.3f', padding=3, fontsize=10)
        bar = axes[1].bar(j, ratios[2], .55, color=color)
        axes[1].bar_label(bar, fmt='%.3f', padding=3, fontsize=10)
    axes[0].set_xticks([0, 1], ['只换相机：动作变化', '真实状态改变：动作变化'])
    axes[0].set_title('相机响应减小，状态响应仍保留', fontsize=11)
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set_xticks([0, 1], ['LIBERO', 'RoboCasa'])
    axes[1].set_title('扰动图上的示范动作误差', fontsize=11)
    for a in axes:
        a.axhline(1, color='#7b848c', linestyle='--', linewidth=1)
        a.set_ylim(0, 1.3)
        a.set_ylabel('多视角／规范模型', fontsize=9)
        a.tick_params(labelsize=9)
        a.grid(axis='y', alpha=.15)
    fig.subplots_adjust(left=.075, right=.99, top=.83, bottom=.20, wspace=.32)
    return fig


if __name__=='__main__':
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--panel',type=Path);group.add_argument('--summary',type=Path);args=p.parse_args()
    if args.panel:build(args.panel)
    else:print(response_figure(args.summary))
