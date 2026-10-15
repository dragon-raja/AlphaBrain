"""Version-pinned public HF snapshots; downloaded Python is never imported.

Uses the existing resource-import staging and publication contract. Ordinary
files are checked against Git blob identities, LFS files against content SHA256.
No implicit authentication or model execution is performed.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import quote
from urllib.request import urlopen


def validate_metadata(item, metadata):
    revision = item['revision']
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('A full immutable revision is required')
    if metadata.get('sha') != revision or metadata.get('id') != item['repo_id']:
        raise ValueError('Snapshot identity mismatch')
    files = metadata['siblings']
    seen = set()
    for entry in files:
        name = entry['rfilename']
        path = PurePosixPath(name)
        if (not path.parts or path.is_absolute() or '..' in path.parts
                or '\\' in name or str(path) != name or name in seen
                or name == 'IMPORT_RECEIPT.json'):
            raise ValueError('Unsafe or duplicate snapshot path')
        seen.add(name)
        expected = entry.get('lfs', {}).get('sha256', entry.get('blobId', ''))
        length = 64 if 'lfs' in entry else 40
        if not re.fullmatch('[0-9a-f]{%d}' % length, expected) or entry['size'] < 0:
            raise ValueError('Missing file identity')
    if not files or sum(e['size'] for e in files) > item['max_bytes']:
        raise ValueError('Empty snapshot or size exceeds approved bound')
    return files


def verify_file(path, entry):
    path = Path(path)
    size = path.stat().st_size
    if size != entry['size']:
        raise ValueError('Snapshot file size mismatch: ' + entry['rfilename'])
    content = hashlib.sha256()
    blob = hashlib.sha1(('blob %d\0' % size).encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            content.update(block)
            blob.update(block)
    expected = entry.get('lfs', {}).get('sha256', entry.get('blobId'))
    actual = content.hexdigest() if 'lfs' in entry else blob.hexdigest()
    if actual != expected:
        raise ValueError('Snapshot content identity mismatch: ' + entry['rfilename'])
    return dict(path=entry['rfilename'], bytes=size, sha256=content.hexdigest(),
                upstream_identity=expected, upstream_hash='sha256' if 'lfs' in entry else 'git_blob_sha1')


def acquire_snapshot(item, staging, progress, limit_mib_s):
    from .resource_import import save, target

    repo = item['repo_id']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
        raise ValueError('Expected a public owner/repository ID')
    if not re.fullmatch(r'[0-9a-f]{40}', item['revision']):
        raise ValueError('A full immutable revision is required')
    url = 'https://huggingface.co/api/models/' + repo + '/revision/' + item['revision'] + '?blobs=true'
    with urlopen(url, timeout=45) as response:
        metadata = json.load(response)
    entries = validate_metadata(item, metadata)
    staging.mkdir(parents=True, exist_ok=False)
    save(staging / 'UPSTREAM_METADATA.json', metadata)
    records = []
    # Fetch inspectable configuration/code first, then large weights.
    for entry in sorted(entries, key=lambda e: (e['size'], e['rfilename'])):
        destination = target(staging, entry['rfilename'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_url = 'https://huggingface.co/' + repo + '/resolve/' + item['revision'] + '/' + quote(entry['rfilename'], safe='/')
        result = subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
            '--retry', '3', '--connect-timeout', '20', '--max-time', '3600',
            '--speed-time', '120', '--speed-limit', '1024', '--limit-rate', f'{limit_mib_s}M',
            '--output', str(destination), file_url], capture_output=True)
        if result.returncode:
            # Do not leak signed redirect URLs or proxy credentials in receipts.
            raise RuntimeError('Public download failed (curl %d): %s' % (result.returncode, entry['rfilename']))
        records.append(verify_file(destination, entry))
        progress(len(records), sum(r['bytes'] for r in records))
    return records
