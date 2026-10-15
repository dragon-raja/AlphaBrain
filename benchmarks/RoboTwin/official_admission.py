"""Finalize official resource intake; never launch training or policy inference."""
import json
from pathlib import Path


def finalize(data_run, checkpoint_run, source_report, output):
    from AlphaBrain.research.dsol.data.resource_catalog import resource_release
    from AlphaBrain.research.dsol.data.resource_import import digest, save
    from .current_data import audit_clean_task
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    releases = []
    for run in (Path(data_run), Path(checkpoint_run)):
        releases.append(resource_release(run / 'plan.json', [run]))
    source = json.loads(Path(source_report).read_text())
    if source['status'] != 'PASS':
        raise ValueError('Official source byte verification failed')
    resources = {}
    for release in releases:
        for key, value in release['resources'].items():
            if key in resources:
                raise ValueError('Duplicate official resource ID')
            resources[key] = value
    tasks = []
    for key, value in sorted(resources.items()):
        if key.startswith('official-demo-clean-'):
            result = audit_clean_task(Path(value['destination']) / 'extracted')
            save(output / 'tasks' / (key + '.json'), result)
            tasks.append({'id': key, 'status': result['status'], 'episodes': result['episodes'],
                          'frames': result['frames'], 'errors': result['errors']})
    passed = len(tasks) == 50 and all(task['status'] == 'PASS' for task in tasks)
    summary = {'schema_version': 1, 'status': 'BYTE_AND_SCHEMA_VERIFIED' if passed else 'INCOMPLETE',
               'resources': resources, 'import_releases': releases,
               'source_report': str(source_report), 'source_report_sha256': digest(source_report),
               'tasks': tasks, 'task_count': len(tasks), 'episodes': sum(x['episodes'] for x in tasks),
               'frames': sum(x['frames'] for x in tasks), 'runtime_validated': False,
               'checkpoint_training_domain_verified': False,
               'scope': 'official current clean50, full simulation assets and selected Pi05 checkpoint; not entire HF repository or policy certification',
               'randomized500_included': False, 'other_embodiment_demonstrations_included': False}
    save(output / 'release.json', summary)
    if passed:
        save(output.parent / 'official_current.json', {
            'release': str(output / 'release.json'), 'sha256': digest(output / 'release.json'),
            'status': summary['status'], 'runtime_validated': False})
    return summary
