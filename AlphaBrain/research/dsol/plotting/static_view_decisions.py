"""Chinese static-decision comparison; exploratory offline endpoint replay."""
def plot(result, output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for font in font_manager.findSystemFonts():
        if 'NotoSansCJK' in font:
            font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False})
    names = {'official':'π0.5 官方', 'canonical':'π0.5 规范后训练',
             'broad':'π0.5 多视角后训练', 'xr1':'XR-1 官方'}
    labels = {'min_mean':'最低均值', 'min_median':'最低中位数',
              'min_mean_plus_se':'均值＋标准误', 'min_neighbor_mean':'7邻域平均',
              'seen_global_fixed':'seen全局固定', 'seen_task_fixed':'seen每任务固定',
              'calibrated_accel':'Accel校准', 'calibrated_accel_task':'任务条件Accel',
              'calibrated_pose':'仅几何校准', 'calibrated_accel_pose':'几何＋Accel',
              'calibrated_accel_pose_task':'任务条件几何＋Accel'}
    fig, axes = plt.subplots(2,2,figsize=(14,12))
    for ax,(name,data) in zip(axes.flat,result['models'].items()):
        row = data['splits']['heldout'];methods=row['methods'];keys=list(methods)
        values=np.array([methods[k]['success']*100 for k in keys])
        ax.barh(range(len(keys)),values,color=['#168576' if k.startswith('calibrated') else '#3363a2' for k in keys])
        ax.set_yticks(range(len(keys)),[labels[k] for k in keys]);ax.invert_yaxis()
        ax.axvline(row['stay']*100,color='#888888',linestyle='--',label=f'扰动不调整 {row["stay"]*100:.2f}%')
        ax.axvline(row['canonical']*100,color='#b56c27',linestyle=':',label=f'已知规范安装 {row["canonical"]*100:.2f}%')
        for i,value in enumerate(values):ax.text(value+.4,i,f'{value:.2f}',va='center',fontsize=9)
        ax.set_xlim(0,100);ax.set_title(names[name]);ax.legend(fontsize=8,loc='lower right')
        ax.set_xlabel('完整任务成功率（%）：每初态4次执行均值')
    fig.suptitle('静态规则诊断｜仅seen标签拟合，heldout 20初态评估\n全部方法并列报告，不按heldout选择赢家；已查看过的面板，不是新盲测',fontsize=13)
    fig.tight_layout(rect=(0,.025,1,.94))
    fig.text(.5,.01,'完整97候选查询／终点结果重放；校准法使用已有闭环标签与已知安装坐标，不是物理相机搜索。',ha='center',fontsize=10)
    fig.savefig(output/'static_rules_zh.png',dpi=180);plt.close(fig)
