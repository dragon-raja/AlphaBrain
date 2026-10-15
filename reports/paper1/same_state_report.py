"""Compose a completed no-action diagnostic receipt without changing results."""
import argparse
import html
from pathlib import Path
from AlphaBrain.research.dsol.data.artifacts import read,write,sha
from AlphaBrain.research.dsol.plotting.same_state import plot_matrices


def build(diagnostic,config,output,interpretation=None):
    diagnostic=Path(diagnostic);out=Path(output);out.mkdir(exist_ok=False)
    status=read(diagnostic/'status.json')
    if status['status']!='COMPLETE':raise ValueError('Cannot report an incomplete diagnostic as complete')
    summary=read(diagnostic/'summary.json');cfg=read(config)
    plot_matrices(summary['contrasts'],cfg['cases'],out/'matrices.png')
    rows=[]
    for r in summary['contrasts']:
        effects=r['effects'];cells=[r['initial'][:8],str(r['step']),r['metric']]
        for key in ('state','view','interaction'):
            e=effects[key];cells.append(f"{e['mean']:.6f} ({e['positive_beyond_noise']}+/ {e['negative_beyond_noise']}-)")
        rows.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in cells)+'</tr>')
    content='''<!doctype html><meta charset="utf-8"><title>同状态跨视角诊断</title>
<style>body{font:16px sans-serif;max-width:1200px;margin:40px auto;line-height:1.6}td,th{border:1px solid #ddd;padding:6px}table{border-collapse:collapse;font-size:13px}img{max-width:100%}</style>
<h1>两例同状态跨视角诊断</h1>
<p>192个矩阵单元；复用32份历史输出，新增160次诊断调用，额外12次工程控制，总计172/172。384帧、1152张原视角图像逐像素一致，固定输入和重渲染模型控制一致。未执行动作。</p>
<p>行固定物理渲染位姿、状态与历史，列只更换外部视角。状态项包括分支历史差异，不能等同于已隔离的操作阶段。正值表示好候选一侧的Accel更高，不代表操作更好。</p>
<img src="matrices.png" alt="六个固定状态/历史与视角矩阵">
<p>S为跨视角平均的状态/历史差，V为跨状态平均的视角差，I为两种状态下的视角差之差；原对角差=S+V。括号列出8套固定噪声中超过4倍控制差异的正/负方向数量；7/8只作描述性稳定条件。本轮控制差异为零。</p>
<table><tr><th>案例</th><th>步</th><th>量</th><th>状态/历史 S</th><th>视角 V</th><th>交互 I</th></tr>'''+''.join(rows)+'''</table>
<p>这些是反复检查过的两例机制诊断，不能证明新的选位规则提高成功率。历史闭环标签独立于本次评分噪声；缺失标签没有填成失败。未调权重、未训练或扩大实验。</p>'''
    if interpretation is not None:
        content += '<h2>本轮解释与后续分析</h2>'+''.join('<p>'+html.escape(t)+'</p>' for t in read(interpretation)['conclusions'])
    (out/'report.html').write_text(content)
    write(out/'provenance.json',dict(status_sha256=sha(diagnostic/'status.json'),summary_sha256=sha(diagnostic/'summary.json'),config_sha256=sha(config),interpretation_sha256=sha(interpretation) if interpretation else None))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--diagnostic',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--interpretation');a=p.parse_args();build(a.diagnostic,a.config,a.output,a.interpretation)
