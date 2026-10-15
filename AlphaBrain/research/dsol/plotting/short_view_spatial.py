"""Actual measured camera-parameter fields; separate explicit colorbars."""
import numpy as np


def plot(coordinates,accel,success,rows,model,split,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from .view_space import adaptive_limits,spatial_field
    for font in font_manager.findSystemFonts():
        if 'NotoSansCJK' in font:font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False})
    fig=plt.figure(figsize=(16,10));title='官方 π0.5' if model=='official' else 'XR-1'
    for n in range(6):
        ax=fig.add_subplot(2,3,n+1,projection='3d')
        values=success*100 if n==0 else accel[:,n-1]
        limits=adaptive_limits(values,minimum_span=max(float(np.ptp(values))*.05,1e-8),bounds=(0,100 if n==0 else float('inf')))
        im=spatial_field(ax,coordinates,values,limits=limits,cmap='viridis' if n==0 else 'viridis_r')
        ax.view_init(elev=23,azim=-57);ax.set_xlabel('方位角偏移（度）',fontsize=9)
        ax.set_ylabel('俯仰角偏移（度）',fontsize=9);ax.set_zlabel('半径比例',fontsize=9)
        if n==0:label='完整任务成功率（%）\n所有窗口共用同一成功率场'
        else:
            r=rows[n-1]['association'];space=r['spatial_mean_spearman'];within=r['pooled_within_initial_rank_correlation']
            label=f'窗口{n}：平均 Accel\n空间图ρ={space:+.3f}；初态内ρ={within:+.3f}'
        ax.set_title(label,fontsize=11)
        cb=fig.colorbar(im,ax=ax,shrink=.58,pad=.10);cb.ax.tick_params(labelsize=8)
    fig.suptitle(f'{title} · {split} · 20初态｜多步Accel与闭环成功率的空间对照',fontsize=16)
    fig.text(.5,.025,'96个实测扰动候选，不含规范位、不插值。坐标是初始相机组的候选参数，不是世界XYZ。\n'
        '各图独立色标：黄色=低Accel／高成功率；不同窗口的绝对大小请读各自色标。\n'
        '后续Accel来自各候选独立试执行的状态；不是同一中间状态的视角场。ρ均为排名相关，正值表示低Accel对应高成功率。',ha='center',fontsize=10)
    fig.subplots_adjust(top=.88,bottom=.15,hspace=.25,wspace=.18)
    fig.savefig(output/f'{model}_{split}_spatial.png',dpi=160);plt.close(fig)
