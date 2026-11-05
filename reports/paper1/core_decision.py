"""Audit and merge U/T screening, continuation, and matched budget evidence.

Read-only CPU report; requires explicit recovery replacements, never discovers
or chooses a successful retry. No protocol, checkpoint or result is rewritten.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from AlphaBrain.research.dsol.analysis.core_decision import contrast, summarize
from AlphaBrain.research.dsol.analysis.prefix_health import training_health
from AlphaBrain.research.dsol.execution.manifest import file_digest, digest
from reports.paper1.core_selection import load_outcomes
from reports.paper1.prefix_health import Report


class Evidence:
    def __init__(self):
        self.hashes = {}
        self.phases = []

    def load(self, path):
        path = Path(path)
        data = path.read_bytes()
        stamp = hashlib.sha256(data).hexdigest()
        if str(path) in self.hashes and self.hashes[str(path)] != stamp:
            raise ValueError('Input changed during analysis')
        self.hashes[str(path)] = stamp
        return json.loads(data)

    def verify(self, path, expected):
        actual = file_digest(path)
        if actual != expected:
            raise ValueError('Artifact digest differs: ' + str(path))
        self.hashes[str(path)] = actual

    def phase(self, root, name, cell, schedule, meta):
        root = Path(root)
        label = Path(schedule).stem + f"-v{meta['version']:04d}"
        if meta['purpose'] != 'development':
            raise ValueError('Final test results cannot select the core')
        ledger = self.load(root / 'budget' / (name + '-' + label + '.json'))
        if ledger['status'] != 'COMPLETE' or ledger['settlement']['actual_episodes'] != 64:
            raise ValueError('Missing completed development ledger')
        cp = ledger['campaign']
        campaign = self.load(cp)
        if (self.hashes[cp] != ledger['campaign_sha256'] or campaign['cell'] != name
                or campaign['evaluation_version'] != meta['version']):
            raise ValueError('Campaign/checkpoint differs from settlement')
        self.verify(campaign['actor']['path'], campaign['actor']['sha256'])
        for field in ('score_noise', 'execution_noise'):
            if campaign[field] != cell[field]:
                raise ValueError('Changed scoring or operation noise manifest')
            self.verify(campaign[field], campaign['input_hashes'][campaign[field]])
        planned = self.load(schedule)
        actual_schedule = self.load(campaign['schedule'])
        if planned != actual_schedule:
            raise ValueError('Recovery changed the scientific schedule')
        folder = root / 'execution' / name / 'execution' / (label + '-real')
        completion = self.load(folder / 'completion.json')
        if completion['status'] != 'COMPLETE' or completion['dispatch_enabled'] is not False:
            raise ValueError('Phase controller not closed')
        rows = []
        for block in campaign['blocks']:
            directory = folder / 'results' / f"block-{block['index']:04d}"
            proof = self.load(directory / 'complete.json')
            result = self.load(directory / 'results.json')
            if (proof['status'] != 'COMPLETE'
                    or self.hashes[str(directory / 'results.json')] != proof['artifacts']['results.json']):
                raise ValueError('Result hash or completion differs')
            request = dict(campaign=cp, campaign_sha256=ledger['campaign_sha256'],
                           block=block, host=block['host'], stage='formal')
            if proof['request_sha256'] != digest(request):
                raise ValueError('Result request differs from frozen campaign')
            expected = {(r['trial_id'], rep): r for r in block['rows'] for rep in r['execution_repeats']}
            actual = {(r['trial_id'], r['repeat']): r for r in result['tasks']}
            searches = {r['trial_id']: r for r in result['searches']}
            if (actual.keys() != expected.keys() or len(actual) != len(result['tasks'])
                    or len(searches) != len(result['searches'])
                    or searches.keys() != {k[0] for k in expected}):
                raise ValueError('Result coverage differs')
            for key, r in actual.items():
                if any(r.get(k) != v for k, v in expected[key].items()):
                    raise ValueError('Scientific result metadata differs')
                search = searches[r['trial_id']]
                if search['queries'] != r['queries']:
                    raise ValueError('Search/operation query count differs')
                rows.append(dict(r, seed=cell['seed'],
                                 search_initial_accel=search['initial_accel'],
                                 search_best_accel=search['best_accel']))
        if len(rows) != 64:
            raise ValueError('Incomplete operation panel')
        self.phases.append(dict(root=str(root), cell=name, version=meta['version'],
                                campaign=cp, actor_sha256=campaign['actor']['sha256']))
        return rows


def build(config):
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    evidence = Evidence()
    screen, sources = load_outcomes(config['screen_roots'])
    evidence.hashes.update(sources)
    panels = {f'{v}@512': rows for v, rows in screen.items()}
    baseline = Report(config['baseline_root'], output / 'baseline-audit')
    baseline.pi_refs = baseline.load(config['pi_references'])
    replacements = config['replacements']
    used = set()
    for run in config['runs']:
        root = Path(run)
        protocol = evidence.load(root / 'protocol.json')
        for name, cell in protocol['cells'].items():
            variant = cell['core_spec']['variant']
            if cell.get('mechanism'):
                variant += '-' + cell['mechanism']['name']
            for schedule, meta in cell['evaluation_schedules'].items():
                replacement_key = f'{root.name}/{name}/{meta["version"]}'
                actual_root = replacements.get(replacement_key, root)
                if replacement_key in replacements:
                    used.add(replacement_key)
                rows = evidence.phase(actual_root, name, cell, schedule, meta)
                panels.setdefault(f'{variant}@{meta["version"]*32}', []).extend(rows)
        print('Read', root.name, flush=True)
    if used != set(replacements):
        raise ValueError('Unused recovery replacement')
    # Restore paired Stay/reference labels from pinned original raw outcomes.
    for rows in panels.values():
        if len(rows) != 192 or {r['seed'] for r in rows} != {0, 1, 2}:
            raise ValueError('All three seeds and 64 operations/seed required')
        for r in rows:
            stay, reference = baseline.pi_baseline(r), baseline.pi_baseline(r, reference=True)
            baseline.check_noise(r, stay)
            baseline.check_noise(r, reference)
            if stay['horizon'] != r['horizon'] or reference['horizon'] != r['horizon']:
                raise ValueError('Changed baseline operation horizon')
            r.update(stay_success=stay['success'], nominal_success=reference['success'])
    summaries = {k: summarize(v) for k, v in panels.items()}
    for key, rows in panels.items():
        summaries[key].update(rescues=sum(r['success'] and not r['stay_success'] for r in rows),
                              harms=sum(r['stay_success'] and not r['success'] for r in rows),
                              stay_rate=sum(r['stay_success'] for r in rows)/len(rows),
                              reference_rate=sum(r['nominal_success'] for r in rows)/len(rows))
    comparisons = {a+' minus '+b: contrast(panels[a], panels[b]) for a, b in config['comparisons']}
    curves = {}
    for name, folder in config['training'].items():
        folder = Path(folder)
        updates = [evidence.load(folder / 'updates' / f'{i:03d}.json') for i in range(1, 129)]
        episodes = [evidence.load(folder / 'episodes' / f'{i:04d}.json')['metrics'] for i in range(4096)]
        curves[name] = training_health(updates, episodes, expected_updates=128)
        print('Audited 4096 training episodes', name, flush=True)
    evidence.hashes.update(baseline.sources)
    payload = dict(status='COMPLETE_DEVELOPMENT_DECISION_EVIDENCE', summaries=summaries,
                   comparisons=comparisons, phases=evidence.phases, new_model_calls=0,
                   note='Exploratory selection panel: 3 policy seeds x 8 tasks x 2 initials x 2 starts x 2 noise repeats. Not held-out test or selection-adjusted inference.')
    for name, data in [('summary.json', payload), ('learning-curves.json', curves),
                       ('input-hashes.json', evidence.hashes), ('config.json', config)]:
        (output / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    fields = ['panel', 'seed', 'suite', 'task', 'state_key', 'view_id', 'repeat',
              'success', 'stay_success', 'nominal_success', 'queries', 'query_cap',
              'search_initial_accel', 'search_best_accel']
    with (output / 'paired-outcomes.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for panel, rows in panels.items():
            writer.writerows(dict(panel=panel, **{k: r[k] for k in fields[1:]}) for r in rows)
    from AlphaBrain.research.dsol.plotting.core_decision import figures
    figures(summaries, curves, output)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    build(json.loads(Path(args.config).read_text()))
