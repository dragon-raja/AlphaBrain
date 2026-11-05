"""Select one model-wide checkpoint from settled development phases only."""
from pathlib import Path

from AlphaBrain.research.dsol.data.artifacts import read, write
from AlphaBrain.research.dsol.data.unified_mainline import select_checkpoint
from AlphaBrain.research.dsol.execution.manifest import file_digest as sha
from .prefix_pilot import validate_campaign, request_for, validate_block


def publish_selection(root, protocol_path, family):
    root = Path(root); protocol = read(protocol_path); rows = []; hashes = {}
    target = root / 'selection' / (family + '.json')
    if target.exists():
        raise ValueError('Selection is immutable; no automatic reselection')
    for name, cell in protocol['cells'].items():
        if cell['family'] != family:
            continue
        for schedule, meta in cell['evaluation_schedules'].items():
            if meta['purpose'] != 'development':
                continue
            label = Path(schedule).stem + f"-v{meta['version']:04d}"
            ledger_path = root / 'budget' / (name + '-' + label + '.json')
            ledger = read(ledger_path)
            if ledger['status'] != 'COMPLETE' or ledger['settlement']['actual_episodes'] != 64:
                raise ValueError('Development budget has not settled completely')
            path = ledger['campaign']; campaign = validate_campaign(path)
            if (sha(path) != ledger['campaign_sha256'] or campaign['cell'] != name
                    or campaign['schedule'] != schedule or campaign['evaluation_version'] != meta['version']):
                raise ValueError('Development phase/checkpoint identity differs')
            folder = root / 'execution' / name / 'execution' / (label + '-real')
            complete = read(folder / 'completion.json')
            if complete['status'] != 'COMPLETE' or complete['dispatch_enabled'] is not False:
                raise ValueError('Development owner still active or incomplete')
            hashes[str(ledger_path)] = sha(ledger_path); hashes[path] = sha(path)
            hashes[campaign['actor']['path']] = sha(campaign['actor']['path'])
            for block in campaign['blocks']:
                directory = folder / 'results' / f"block-{block['index']:04d}"
                request = request_for(path, block['index'])
                validate_block(request, directory)
                result = read(directory / 'results.json')
                proof = read(directory / 'complete.json')
                if proof['artifacts']['results.json'] != sha(directory / 'results.json'):
                    raise ValueError('Development result bytes changed')
                planned = {(r['trial_id'], n): r for r in block['rows'] for n in (0, 1)}
                for r in result['tasks']:
                    expected = planned[(r['trial_id'], r['repeat'])]
                    if any(r.get(k) != v for k, v in expected.items()):
                        raise ValueError('Development population differs')
                    rows.append(dict(r, version=meta['version'], seed=cell['seed']))
                for f in ('complete.json', 'results.json'):
                    hashes[str(directory / f)] = sha(directory / f)
    value = select_checkpoint(rows)
    value.update(family=family, protocol_sha256=sha(protocol_path), input_hashes=hashes)
    write(target, value)
    return target, value
