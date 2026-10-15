"""Construct pinned official download plans; no network or IO at import."""
from pathlib import PurePosixPath


def robotwin_clean_plan(tree, revision, expected_tasks=50, limit_mib_s=64):
    if tree['revision'] != revision:
        raise ValueError('Official tree revision mismatch')
    files = [row for row in tree['files'] if row['type'] == 'file']
    clean = [row for row in files if PurePosixPath(row['path']).name == 'demo_clean.zip'
             and row['path'].startswith('dataset/')]
    tasks = [PurePosixPath(row['path']).parts[1] for row in clean]
    if len(clean) != expected_tasks or len(set(tasks)) != expected_tasks:
        raise ValueError('Official clean task coverage mismatch')
    assets = ['objects.zip', 'embodiments.zip', 'background_texture.zip']
    selected = sorted(clean, key=lambda row: row['path'])
    for name in assets:
        matches = [row for row in files if row['path'] == name]
        if len(matches) != 1:
            raise ValueError('Missing or duplicated official asset: ' + name)
        selected.extend(matches)
    resources = []
    for row in selected:
        path = PurePosixPath(row['path'])
        sha = row.get('lfs', {}).get('oid', '')
        if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Official LFS SHA256 missing: ' + str(path))
        is_data = str(path).startswith('dataset/')
        name = path.parts[1] if is_data else path.stem
        resources.append({'id': ('official-demo-clean-' if is_data else 'official-asset-') + name,
                          'kind': 'official_zip',
                          'destination': f'datasets/raw/hf-{revision[:12]}/demo_clean/{name}' if is_data
                          else f'assets/hf-{revision[:12]}/{name}',
                          'url': f'https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/{revision}/{path}?download=true',
                          'bytes': row['size'], 'sha256': sha})
    return {'schema_version': 1, 'benchmark': 'robotwin2', 'upstream_revision': revision,
            'purpose': 'official current 50-task clean release and complete simulation assets; not a formal evaluation protocol',
            'scope': {'clean_tasks': expected_tasks, 'demos_per_task_expected': 50,
                      'randomized500_included': False, 'other_embodiment_demonstrations_included': False},
            'limit_mib_s': limit_mib_s, 'resources': resources}
