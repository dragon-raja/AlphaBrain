"""Inspect one archived state/view and its repeated live executions on CPU."""
import argparse
from pathlib import Path

import numpy as np

from AlphaBrain.research.dsol.analysis.execution_trace_audit import first_row_difference, compare_recorded_noise
from AlphaBrain.research.dsol.data.artifacts import read, sha, write


def build(config, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    root, old_root = Path(config['formal_root']), Path(config['historical_root'])
    initial, candidate = config['initial'], config['candidate']
    historic = {}
    expected_hashes = dict(read(old_root / config['record_hashes']))
    for repeat in range(8):
        index = initial * 97 * 32 + candidate * 32 + repeat
        path = old_root / config['historical_records'] / f'{index:06d}.json'
        if sha(path) != expected_hashes[str(path.relative_to(old_root))]:
            raise ValueError('Historical receipt changed')
        row = read(path)
        if row['pair_key'] != config['state_key'] or row['policy_repeat_id'] != repeat:
            raise ValueError('Historical trial identity mismatch')
        historic[repeat] = row
    statuses = sorted(list((root/'first-block').glob('*/*/status.json'))
                      + list((root/'remaining').glob('*/*/status.json')))
    references = {}; comparisons = []; receipts = []; selected_searches = []
    for status in statuses:
        for task in read(status)['tasks']:
            if task['cohort'] != 'A' or task['initial'] != initial or task['selected'] != candidate:
                continue
            if task['state_key'] != config['state_key']:
                raise ValueError('Live trial identity mismatch')
            repeat = task['repeat']; method = task['method']; start = task['start']
            path = status.parent / method / f'episode-{repeat}.npz'
            if sha(path) != task['trajectory_sha256']:
                raise ValueError('Live trajectory changed')
            with np.load(path, allow_pickle=False) as z:
                arrays = {k: z[k].copy() for k in z.files}
            if repeat not in references: references[repeat] = (str(path), arrays)
            original, reference = references[repeat]
            repeated = {k: first_row_difference(reference[k], arrays[k]) for k in reference}
            noise = compare_recorded_noise(historic[repeat]['policy_calls'], task['execution_noise_sha256'])
            comparisons.append(dict(start=start, method=method, repeat=repeat, live_success=task['success'],
                live_steps=task['steps'], old_success=historic[repeat]['success'],
                old_steps=historic[repeat]['completion_steps'], noise=noise, repeated_current=repeated,
                reference=original, path=str(path)))
            receipts.append(dict(path=str(path), sha256=sha(path)))
            selected_searches.append((method, start))
    if set(references) != set(range(8)):
        raise ValueError('Missing paired execution seeds')
    seeds = []
    for repeat in range(8):
        group = [r for r in comparisons if r['repeat'] == repeat]
        seeds.append(dict(repeat=repeat, exposures=len(group), old_success=historic[repeat]['success'],
            old_steps=historic[repeat]['completion_steps'], live_successes=sum(r['live_success'] for r in group),
            live_steps=sorted({r['live_steps'] for r in group}),
            current_arrays_identical=all(all(v['complete_equal'] for v in r['repeated_current'].values()) for r in group)))
    result = dict(status='PARTIAL_OBSERVABILITY', state_key=config['state_key'], candidate=candidate,
        candidate_id=historic[0]['selected_candidate_id'], seeds=seeds, comparisons=comparisons,
        exposures=len(comparisons), unique_current_searches=len(set(selected_searches)),
        ppo_exposures=sum(r['method']=='ppo' for r in comparisons),
        ppo_outcome_disagreements=sum(r['method']=='ppo' and r['live_success']!=r['old_success'] for r in comparisons),
        historical_first_different_action_step=None,
        reason='Old records contain 10-action block hashes and one whole-run digest, not step arrays; new execution stores only executed (at most 5 per replan) actions. Different hash supports cannot locate the first divergence.',
        initial_scoring_noise_is_execution_noise=False,
        evidence='Current replan_states are 8D policy observations, not full simulator states; sample_frames occur every 50 actions.',
        receipts=receipts, new_model_calls=0, new_simulator_steps=0)
    write(output/'audit.json', result)
    # Display completion times with failures at the same 300-step limit.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.7), layout='constrained')
    for i, row in enumerate(seeds):
        for offset, steps, success, color, label in [(-.16, row['old_steps'], row['old_success'], '#5577aa', 'Historical'),
                (.16, row['live_steps'][0], row['live_successes']==row['exposures'], '#c76b34', 'Current')]:
            ax.bar(i+offset, steps, .30, color=color, alpha=.85, label=label if i==0 else None)
            ax.text(i+offset, steps+4, 'S' if success else 'F', ha='center', fontsize=10)
    ax.set(xticks=range(8), xlabel='Paired execution seed',
           ylabel='Executed actions', ylim=(0,345),
           title=f'Initial {initial} / view {candidate}: S=success, F=timeout')
    ax.legend(loc='upper center', ncols=2)
    fig.savefig(output/'completion.png', dpi=160); plt.close(fig)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); build(read(args.config), args.output)
