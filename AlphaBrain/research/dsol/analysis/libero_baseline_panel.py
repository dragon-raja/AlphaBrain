"""Paired analysis of the bounded LIBERO training-start diagnostic."""
from pathlib import Path
import argparse
import numpy as np

from ..data.artifacts import read, sha, write
from ..data.libero_baseline_panel import panel_indices
from .robocasa_statistics import interval


def validate_native_noise_call(full, common):
    # The frozen evaluator passes a bounded inference seed separately from
    # the original 64-bit noise-bank seed; both refer to the same call.
    if (full['shared_noise_sha256'] != common['noise_sha256']
            or full['eval_seed'] != int(common['noise_seed']) % (2**31 - 1)
            or full['call_index'] != common['replan_index']
            or len(full['full_noise_sha256']) != 64):
        raise ValueError('Native32/shared7 noise trace mismatch')


def validate_pair(rows):
    if len(rows) != 3:
        raise ValueError('Need baseline, canonical and broad')
    for row in rows:
        if row['status'] != 'complete' or not row['explicit_flow_noise'] or not row['policy_calls']:
            raise ValueError('Incomplete or non-explicit rollout')
        if not row['success'] and row['completion_steps'] != row['max_steps']:
            raise ValueError('Unsuccessful rollout did not reach its task horizon')
        if len(row['policy_calls']) != row['inference_calls']:
            raise ValueError('Incomplete noise trace')
    for key in ('pair_key', 'selected_candidate_id', 'initialization_receipt', 'noise_bank_manifest_sha256',
                'max_steps', 'replan_steps', 'wait_steps', 'sensor_control', 'prompt', 'policy_repeat_id'):
        if any(row[key] != rows[0][key] for row in rows[1:]):
            raise ValueError('Paired protocol mismatch: '+key)
    for other in rows[1:]:
        for a, b in zip(rows[0]['policy_calls'], other['policy_calls']):
            if not a['noise_sha256'] or a['noise_sha256'] != b['noise_sha256'] or a['replan_index'] != b['replan_index']:
                raise ValueError('Flow noise hash/address mismatch')


def summarize(root):
    root = Path(root)
    plan = read(root/'plan.json')
    parent = read(root/'parent-release.json')
    if sha(root/'parent-release.json') != plan['parent_release_sha256']:
        # Parent copy is serialized, so compare content and the original bytes.
        if parent != read(Path(plan['parent'])/'release.json'):
            raise ValueError('Parent release content changed')
    if sha(Path(plan['parent'])/'release.json') != plan['parent_release_sha256']:
        raise ValueError('Parent source identity changed')
    if panel_indices(parent) != plan['indices']:
        raise ValueError('Selected panel changed')
    native = plan.get('format') == 'libero_native_reference_panel_v1'
    baseline = read(root/('baseline-receipt.json' if native else 'step-zero-receipt.json'))
    names = ['baseline', 'canonical', 'broad']
    values = np.zeros((3, 32, 5, 4), dtype=np.uint8)
    audited = []
    for flat, index in enumerate(plan['indices']):
        state, rem = divmod(flat, 20)
        view, repeat = divmod(rem, 4)
        rows = []
        for m, name in enumerate(names):
            if name == 'baseline':
                path = root/'episodes'/f'{index:06d}.json'
                weight = baseline['weights_sha256']
            else:
                ref = plan['reused'][f'{name}:{index}']
                path = Path(ref['path'])
                if sha(path) != ref['sha256']:
                    raise ValueError('Reused result changed')
                weight = plan['original_models'][name]['weights_sha256']
            row = read(path)
            if row['aa0']['index'] != index or row['aa0']['checkpoint_sha256'] != weight:
                raise ValueError('Wrong index or checkpoint')
            if not row['explicit_flow_noise'] or row['policy_repeat_id'] != repeat:
                raise ValueError('Wrong execution noise')
            if name == 'baseline' and row['aa0']['release_sha256'] != sha(root/'release.json'):
                raise ValueError('Baseline release changed')
            if native and name == 'baseline':
                trace_path = root/'noise32'/f'{index:06d}.json'
                trace = read(trace_path)
                if (trace['status'] != 'PASS' or trace['episode_sha256'] != sha(path)
                        or trace['internal_dim'] != 32 or trace['shared_dim'] != 7
                        or len(trace['calls']) != row['inference_calls']):
                    raise ValueError('Incomplete native32 noise provenance')
                for full, common in zip(trace['calls'], row['policy_calls']):
                    validate_native_noise_call(full, common)
                audited.append(dict(path=str(trace_path), sha256=sha(trace_path)))
            rows.append(row)
            values[m, state, view, repeat] = bool(row['success'])
            audited.append(dict(path=str(path), sha256=sha(path)))
        validate_pair(rows)
    tasks = np.array([s['task_id'] for s in parent['states']])
    groups = dict(canonical=[0], mild=[1, 2], strong=[3, 4], four_perturbed=[1, 2, 3, 4])
    q = values.mean(-1)
    result = dict(baseline=plan['baseline'], scope=plan['limits'], baseline_rollouts=640,
                  reused_rollouts=1280, groups={}, differences={},
                  plan_sha256=sha(root/'plan.json'),
                  uncertainty='Task/initial hierarchical bootstrap, conditional on four measured noise repeats; single training seed')
    if native:
        result.update(format=plan['format'], baseline_plot_label=plan['baseline_plot_label'],
                      plot_note=plan['plot_note'], noise_extension=plan['noise_extension'])
    for mi, name in enumerate(names):
        result['groups'][name] = {g: interval(q[mi][:, ids].mean(-1), tasks) for g, ids in groups.items()}
    for name, a, b in [('canonical_minus_start', 1, 0), ('broad_minus_start', 2, 0), ('broad_minus_canonical', 2, 1)]:
        result['differences'][name] = {g: interval((q[a][:, ids]-q[b][:, ids]).mean(-1), tasks)
                                       for g, ids in groups.items()}
    result['per_task'] = {task: {name: {g: float(q[mi][tasks == task][:, ids].mean())
                                       for g, ids in groups.items()} for mi, name in enumerate(names)}
                          for task in sorted(set(tasks))}
    result['initial_splits'] = {}
    splits = np.array([s['split'] for s in parent['states']])
    for split in sorted(set(splits)):
        result['initial_splits'][split] = {name: {g: interval(q[mi][splits == split][:, ids].mean(-1), tasks[splits == split])
                                                for g, ids in groups.items()} for mi, name in enumerate(names)}
    output = root/'analysis'
    output.mkdir(exist_ok=False)
    np.savez_compressed(output/'outcomes.npz', outcomes=values, indices=plan['indices'])
    write(output/'audit.json', dict(status='PASS', rows=audited))
    write(output/'summary.json', result)
    lines = ['# LIBERO 后训练起点小面板', '',
             '8 个任务 × 原 4 初态 × 5 个几何预选视角 × 原噪声库前 4 次。', '',
             ('**修正版：起点使用原始转换权重和原生 OpenPI 推理组件，内部32维，quantile动作解码；没有重新训练。**'
              if native else '**起点是 AlphaBrain 适配后、优化器更新前的重建模型，不是原生官方实现。**'), '',
             '四扰动平均只代表这个小面板，不代表 96 候选总体，不估计 Oracle，不检验选择器。', '',
             '| 模型 | 规范 | 轻扰动两视角 | 强扰动两视角 | 四扰动平均 |',
             '|---|---:|---:|---:|---:|']
    for name, label in zip(names, ['原生参考' if native else '后训练前起点', '规范后训练', '多视角后训练']):
        lines.append('| '+label+' | '+' | '.join(f"{result['groups'][name][g]['mean']*100:.2f}%" for g in groups)+' |')
    lines += ['', '差值置信区间与输入来源核查见 summary.json 和 audit.json。',
              '仅一个训练 seed；初态索引留出不等于任务未见，也不保证相对预训练数据未见。']
    if native:
        lines += ['', '原生32维与旧模型7维仅共享前7维执行噪声；额外25维固定生成并逐调用保存哈希。',
                  '规范/多视角后训练结果复用原严格匹配记录，保留它们训练时的mean/std解码，不套用原生解码。',
                  '该对比包含训练配方及适配实现变化，不把原生与后训练差值归因于单一训练因素。旧640次0%另存为失配诊断，不拼入此表。',
                  '', '## 分任务结果', '', '| 任务 | 原生规范 | 原生扰动 | 规范后训练规范 | 规范后训练扰动 | 多视角规范 | 多视角扰动 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for task, row in result['per_task'].items():
            values = [row[name][g] for name in names for g in ('canonical', 'four_perturbed')]
            lines.append('| '+task+' | '+' | '.join(f'{v*100:.2f}%' for v in values)+' |')
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    from ..plotting.libero_baseline_panel import plot
    plot(result, output/'checkpoint_comparison.png')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    summarize(p.parse_args().root)


if __name__ == '__main__':
    main()
