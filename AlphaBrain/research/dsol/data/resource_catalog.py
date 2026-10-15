"""Read-only navigation audit, not a re-certification of scientific results."""
import json
from pathlib import Path, PurePosixPath
from .resource_import import digest


def check_pointer(path):
    """Verify pointer -> release -> resource receipt hashes (not payload bytes)."""
    pointer = json.loads(Path(path).read_text())
    release_path = Path(pointer['release'])
    if digest(release_path) != pointer['sha256']:
        raise ValueError('Release hash differs from published pointer')
    release = json.loads(release_path.read_text())
    for key, item in release['resources'].items():
        receipt = Path(item['destination']) / 'IMPORT_RECEIPT.json'
        if digest(receipt) != item['receipt_sha256']:
            raise ValueError(f'Resource receipt hash changed: {key}')
        content = json.loads(receipt.read_text())
        if content['resource']['id'] != key or content['status'] != 'BYTE_VERIFIED':
            raise ValueError(f'Resource identity mismatch: {key}')
    return {'pointer': str(path), 'release': str(release_path), 'sha256': pointer['sha256'],
            'resources': len(release['resources']), 'scope': 'metadata hash chain; no payload rehash'}


def audit(config):
    result = {'schema_version': 1, 'scope': 'paths and metadata byte identities only',
              'entries': [], 'discovered_experiments': [], 'errors': []}
    ids = [e['id'] for e in config['entries']]
    if len(set(ids)) != len(ids):
        result['errors'].append('Duplicate registry entry ID')
    retired = config.get('historical_read_mapping', {})
    owners = config.get('code_owners', {})
    for entry in config['entries']:
        row = {'id': entry['id'], 'benchmark': entry['benchmark'], 'role': entry['role'],
               'root': entry['root'], 'metadata': []}
        if not Path(entry['root']).is_absolute() or any(
                entry['root'] == old or entry['root'].startswith(old + '/') for old in retired):
            result['errors'].append(f"Invalid active root: {entry['id']}")
        if owners:
            owner = owners.get(entry['benchmark'])
            code = Path(config['repository_root']) / owner if owner else None
            if code is None or not code.is_dir():
                result['errors'].append(f"Missing code owner: {entry['id']}")
            else:
                row['code_owner'] = str(code)
        if not Path(entry['root']).is_dir():
            result['errors'].append(f"Missing root: {entry['id']}")
        for relative in entry.get('anchors', []):
            if PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts:
                result['errors'].append(f"Invalid anchor: {entry['id']}/{relative}")
                continue
            path = Path(entry['root']) / relative
            if not path.is_file():
                result['errors'].append(f"Missing anchor: {entry['id']}/{relative}")
                continue
            row['metadata'].append({'path': str(path), 'sha256': digest(path),
                                    'bytes': path.stat().st_size})
        result['entries'].append(row)
    anchors = ('release.json', 'plan.json', 'run_config.json', 'source-manifest.json',
               'preparation.json', 'launch.json')
    for root in config.get('experiment_search_roots', []):
        for path in sorted(Path(root).iterdir()):
            if not path.is_dir():
                continue
            records = [{'path': str(path / name), 'sha256': digest(path / name)}
                       for name in anchors if (path / name).is_file()]
            result['discovered_experiments'].append({
                'root': str(path), 'metadata': records,
                'admission': 'NAVIGATION_ONLY' if records else 'LEGACY_METADATA_NOT_AT_ROOT',
            })
    result['release_chains'] = []
    for pointer in config.get('release_pointers', []):
        try:
            result['release_chains'].append(check_pointer(pointer))
        except (OSError, ValueError, KeyError) as exc:
            result['errors'].append(f'Invalid release chain: {pointer}: {exc}')
    result['status'] = 'PASS' if not result['errors'] else 'INCOMPLETE'
    return result


def audit_file(path, check_mirror=True):
    config = json.loads(Path(path).read_text())
    result = audit(config)
    if config.get('project_manifest'):
        from .resource_graph import audit_graph
        manifest = Path(config['repository_root']) / config['project_manifest']
        locations = Path(config['local_resource_paths'])
        try:
            graph = audit_graph(json.loads(manifest.read_text()), json.loads(locations.read_text()))
            result['project_resources'] = graph
            result['project_manifest_sha256'] = digest(manifest)
            result['local_resource_paths_sha256'] = digest(locations)
            result['errors'].extend(graph['errors'])
        except (OSError, ValueError, KeyError) as exc:
            result['errors'].append(f'Project resource graph invalid: {exc}')
    mirror = config.get('shared_catalog')
    if check_mirror and mirror:
        if not Path(mirror).is_file() or digest(mirror) != digest(path):
            result['errors'].append('Repository catalog and shared catalog differ')
            result['status'] = 'INCOMPLETE'
    result['status'] = 'PASS' if not result['errors'] else 'INCOMPLETE'
    return result


def publish_catalog(path):
    """Publish only an admitted catalog; archive previous bytes by content hash."""
    import shutil
    path = Path(path)
    config = json.loads(path.read_text())
    result = audit_file(path, check_mirror=False)
    if result['status'] != 'PASS':
        raise ValueError(result['errors'])
    mirror = Path(config['shared_catalog'])
    if mirror.exists() and digest(mirror) != digest(path):
        archive = mirror.parent / 'catalog_history' / (digest(mirror) + '.json')
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists() and digest(archive) != digest(mirror):
            raise ValueError('Catalog history collision')
        if not archive.exists():
            shutil.copyfile(mirror, archive)
    # Same JSON bytes as the repository copy, atomic replacement.
    mirror.parent.mkdir(parents=True, exist_ok=True)
    temporary = mirror.with_name(mirror.name + '.publishing')
    if temporary.exists():
        raise FileExistsError('Unfinished catalog publication exists')
    with temporary.open('xb') as output:
        output.write(path.read_bytes())
    temporary.replace(mirror)
    return {'shared_catalog': str(mirror), 'sha256': digest(mirror), 'status': 'PUBLISHED'}


def resource_release(original_plan, attempt_roots):
    """Combine admitted resources without overwriting failed attempt evidence."""
    original_plan = Path(original_plan)
    original_sha = digest(original_plan)
    plan = json.loads(original_plan.read_text())
    expected = {r['id'] for r in plan['resources']}
    resources, attempts = {}, []
    for root in map(Path, attempt_roots):
        state = json.loads((root / 'status.json').read_text())
        if state['status'] == 'RUNNING':
            raise ValueError('Import is still running')
        actual_plan = json.loads((root / 'plan.json').read_text())
        if digest(root / 'plan.json') != state['plan_sha256']:
            raise ValueError('Attempt plan changed')
        if state['plan_sha256'] != original_sha and actual_plan.get('parent_plan_sha256') != original_sha:
            raise ValueError('Attempt has no original-plan lineage')
        if digest(root / 'importer.py') != state['importer_sha256']:
            raise ValueError('Attempt importer changed')
        attempts.append({'root': str(root), 'status': state['status'],
                         'state_sha256': digest(root / 'status.json'),
                         'plan_sha256': state['plan_sha256'], 'importer_sha256': state['importer_sha256']})
        items = {r['id']: r for r in actual_plan['resources']}
        for key, result in state['resources'].items():
            if key in resources or key not in expected:
                raise ValueError('Duplicate or unexpected resource')
            receipt_path = Path(result['destination']) / 'IMPORT_RECEIPT.json'
            if digest(receipt_path) != result['receipt_sha256']:
                raise ValueError('Resource receipt changed')
            receipt = json.loads(receipt_path.read_text())
            if receipt['resource'] != items[key] or receipt['status'] != 'BYTE_VERIFIED':
                raise ValueError('Resource identity mismatch')
            resources[key] = dict(result, total_bytes=receipt['total_bytes'])
    if set(resources) != expected:
        raise ValueError(f'Missing resources: {sorted(expected - set(resources))}')
    return {'schema_version': 1, 'benchmark': plan['benchmark'], 'status': 'BYTE_VERIFIED',
            'original_plan_sha256': original_sha, 'attempts': attempts, 'resources': resources,
            'runtime_validated': False, 'checkpoint_training_domain_verified': False,
            'verification_scope': 'Import receipts and plan/source identities; not a new full payload rehash'}
