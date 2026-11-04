"""Read-only interim report: settled waves and complete, three-seed task panels."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

from AlphaBrain.research.dsol.analysis.recovery_summary import paired_summary


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_results(folder):
    complete = read(folder / 'complete.json')
    if complete['status'] != 'COMPLETE' or digest(folder / 'results.json') != complete['artifacts']['results.json']:
        raise ValueError('Incomplete or changed result')
    return read(folder / 'results.json')


def task_key(row):
    return row['suite'], row['task']


def outcome_key(row):
    return row['trial_id'], row['repeat']


def check_noise(a, b):
    aa, bb = a['execution_noise_sha256'], b['execution_noise_sha256']
    if not aa or not bb or aa[:min(len(aa), len(bb))] != bb[:min(len(aa), len(bb))]:
        raise ValueError('Execution noise pairing differs')


def build(root, prepared, output, *, baseline_groups=(), require_complete=False):
    root, prepared, output = map(Path, (root, prepared, output))
    if output.exists():
        raise ValueError('New report directory required')
    sources, phases, searches = {}, defaultdict(list), {}
    def consume(folder, rows, phase):
        result = verified_results(folder)
        sources[str(folder / 'results.json')] = digest(folder / 'results.json')
        sources[str(folder / 'complete.json')] = digest(folder / 'complete.json')
        expected = {(r['trial_id'], repeat): r for r in rows for repeat in r['execution_repeats']}
        actual = {outcome_key(r): r for r in result['tasks']}
        if len(actual) != len(result['tasks']) or actual.keys() != expected.keys():
            raise ValueError('Result coverage differs from frozen rows')
        for key, row in actual.items():
            if any(row[k] != value for k, value in expected[key].items()) or type(row['success']) is not bool:
                raise ValueError('Result metadata differs')
        phases[phase].extend(result['tasks'])
        for row in result['searches']:
            if row['trial_id'] in searches:
                raise ValueError('Repeated search identity')
            searches[row['trial_id']] = row
        return len(result['tasks'])
    # Snapshot settled waves once; never include quicker unfinished-wave results.
    settlements = sorted((root / 'settlements').glob('*.json'))
    for settlement in settlements:
        phase = settlement.stem.rsplit('-', 1)[0]
        if phase not in ('baseline', 'ours'):
            continue
        receipt = read(settlement)
        if receipt['status'] != 'COMPLETE':
            raise ValueError('Unsettled wave')
        campaign_path = root / 'campaigns' / settlement.name
        campaign = read(campaign_path)
        sources[str(settlement)] = digest(settlement)
        sources[str(campaign_path)] = digest(campaign_path)
        count = 0
        for block in campaign['blocks']:
            folder = root / 'execution' / (settlement.stem + '-real') / 'results' / f"block-{block['index']:04d}"
            count += consume(folder, block['rows'], phase)
        if count != receipt['actual_episodes']:
            raise ValueError('Settlement episode count differs')
    for group in baseline_groups:
        folder = Path(group['output'])
        provenance = Path(group['source_record'])
        if (digest(provenance) != group['source_record_sha256']
                or digest(folder / 'complete.json') != group['receipt_sha256']):
            raise ValueError('Supplemental baseline provenance changed')
        if any(row['method'] not in ('stay', 'reference') for row in group['rows']):
            raise ValueError('Supplemental results must be baseline operations')
        sources[str(provenance)] = digest(provenance)
        consume(folder, group['rows'], 'baseline')
    baseline = phases['baseline']
    stay = {(r['state_key'], r['view_id'], r['repeat']): r for r in baseline if r['method'] == 'stay'}
    reference = {(r['state_key'], r['repeat']): r for r in baseline if r['method'] == 'reference'}
    if len(stay) + len(reference) != len(baseline):
        raise ValueError('Duplicate or unexpected baseline')
    planned_path = prepared / 'ours-blocks.json'
    sources[str(planned_path)] = digest(planned_path)
    planned, observed = defaultdict(set), defaultdict(set)
    expected_stay, expected_reference = set(), set()
    for block in read(planned_path):
        for row in block['rows']:
            for repeat in row['execution_repeats']:
                planned[task_key(row)].add((row['trial_id'], repeat))
                expected_stay.add((row['state_key'], row['view_id'], repeat))
                expected_reference.add((row['state_key'], repeat))
    for row in phases['ours']:
        key = outcome_key(row)
        if key in observed[task_key(row)]:
            raise ValueError('Duplicate PPO outcome')
        observed[task_key(row)].add(key)
    full_tasks = {task for task, keys in planned.items() if keys == observed[task]}
    if require_complete and (set(full_tasks) != set(planned) or set(observed) != set(planned)
                             or set(stay) != expected_stay or set(reference) != expected_reference):
        raise ValueError('Complete frozen PPO and baseline coverage required')
    paired = []
    keep = ['success', 'state_key', 'view_id', 'repeat', 'method', 'suite', 'task', 'split',
            'initial', 'trial_id', 'queries', 'angles_deg', 'selected_angles']
    for row in phases['ours']:
        if task_key(row) not in full_tasks:
            continue
        a = stay[row['state_key'], row['view_id'], row['repeat']]
        b = reference[row['state_key'], row['repeat']]
        for other in (a, b):
            check_noise(row, other)
            for k in ('state_key', 'suite', 'task', 'initial', 'split', 'max_steps', 'settle_steps', 'replan_steps'):
                if row[k] != other[k]:
                    raise ValueError('Baseline pairing metadata differs')
        if row['angles_deg'] != a['angles_deg']:
            raise ValueError('Starting view differs')
        entry = {k: row[k] for k in keep}
        entry.update(stay_success=a['success'], nominal_success=b['success'])
        paired.append(entry)
    groups = defaultdict(list)
    for row in paired:
        for name in ('all', 'split/' + row['split'], 'seed/' + row['method']):
            groups[name].append(row)
    summaries = {name: paired_summary(rows) for name, rows in groups.items()}
    costs = {}
    for name, rows in groups.items():
        unique = {r['trial_id']: searches[r['trial_id']] for r in rows}
        costs[name] = dict(searches=len(unique), mean_queries=mean(r['queries'] for r in unique.values()),
                           initial_only_fraction=mean(r['queries'] == 1 for r in unique.values()),
                           at_cap_fraction=mean(r['queries'] == r['query_cap'] for r in unique.values()),
                           stop_reasons=dict(Counter(r['reason'] for r in unique.values())))
    baseline_groups = defaultdict(list)
    for row in baseline:
        baseline_groups[row['split'] + '/' + row['method']].append(row)
        baseline_groups['all/' + row['method']].append(row)
    baselines = {}
    for name, rows in baseline_groups.items():
        by_task = defaultdict(list)
        for row in rows:
            by_task[task_key(row)].append(row['success'])
        baselines[name] = dict(episodes=len(rows), tasks=len(by_task), successes=sum(r['success'] for r in rows),
                              success_rate=mean(r['success'] for r in rows), task_macro_success=mean(map(mean, by_task.values())))
    result = dict(scope=('Complete frozen mainline panel; all prescribed PPO and baseline outcome identities covered.' if require_complete else 'Descriptive interim; frozen-order completed waves; only fully covered three-seed tasks in PPO comparisons. Not representative of all tasks; no stopping or model-selection decision.'),
                  complete_frozen_panel=require_complete,
                  source_root=str(root), settled_waves=[p.stem for p in settlements],
                  settled_outcomes={k: len(v) for k, v in phases.items()},
                  full_tasks=sorted(full_tasks), planned_tasks=len(planned),
                  excluded_partial_task_outcomes=len(phases['ours']) - len(paired),
                  paired=summaries, search_costs=costs, full_baseline=baselines,
                  baseline_reuse='Same stay/reference outcomes reused across PPO seeds for paired contrasts, not independent observations.',
                  new_model_calls=0)
    output.mkdir(parents=True)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    sources[str(Path(__file__).resolve())] = digest(Path(__file__))
    (output / 'input-hashes.json').write_text(json.dumps(sources, indent=2) + '\n')
    if paired:
        with (output / 'paired-outcomes.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'prepared', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = build(args.root, args.prepared, args.output)
    print(json.dumps(result, indent=2))
