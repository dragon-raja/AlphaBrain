"""Explicit, checksum-verified resource admission; no work happens at import time.

An import is not a simulator or scientific acceptance. Failed staging directories
are preserved. Existing final destinations are never overwritten.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tarfile
import time
import zipfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def target(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Destination must be a nonempty relative path')
    path = root / str(relative)
    if path.resolve() != path.absolute():
        raise ValueError('Symlinks are not allowed in destination paths')
    return path


def inventory(root, exclude=()):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Expected a real source directory')
    result = {}
    for path in sorted(root.rglob('*')):
        if any(part in exclude for part in path.relative_to(root).parts):
            continue
        if path.is_symlink():
            raise ValueError(f'Symlink in source: {path}')
        if path.is_dir():
            continue
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f'Non-regular source: {path}')
        result[str(path.relative_to(root))] = [info.st_size, info.st_mtime_ns]
    if not result:
        raise ValueError('Empty source')
    return result


def copy_tree(source, staging, progress, limit_mib_s, exclude=()):
    source = Path(source)
    before = inventory(source, exclude)
    staging.mkdir(parents=True, exist_ok=False)
    records = []
    for name, (size, _) in before.items():
        src, dst = source / name, staging / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        copied, started = 0, time.monotonic()
        with src.open('rb') as reader, dst.open('xb') as writer:
            for block in iter(lambda: reader.read(4 * 1024 * 1024), b''):
                writer.write(block)
                h.update(block)
                copied += len(block)
                delay = copied / (limit_mib_s * 1024**2) - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
        if copied != size or digest(dst) != h.hexdigest():
            raise ValueError(f'Copy verification failed: {name}')
        records.append({'path': name, 'bytes': size, 'sha256': h.hexdigest()})
        progress(len(records), sum(r['bytes'] for r in records))
    if inventory(source, exclude) != before:
        raise ValueError('Source inventory changed during copy')
    return records


def inspect_zip(archive):
    with zipfile.ZipFile(archive) as z:
        seen = set()
        for info in z.infolist():
            path = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if (path.is_absolute() or '..' in path.parts or '\\' in info.filename
                    or not path.parts or str(path) in seen
                    or stat.S_ISLNK(mode)
                    or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR))):
                raise ValueError(f'Unsafe ZIP entry: {info.filename}')
            seen.add(str(path))
        return z.infolist()


def extract_zip(archive, destination):
    entries = inspect_zip(archive)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    with zipfile.ZipFile(archive) as z:
        for entry in entries:
            dst = target(destination, entry.filename)
            if entry.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(entry) as reader, dst.open('xb') as writer:
                shutil.copyfileobj(reader, writer, 4 * 1024 * 1024)
            if dst.stat().st_size != entry.file_size:
                raise ValueError('Extracted size mismatch')
            records.append({'path': entry.filename, 'bytes': entry.file_size, 'sha256': digest(dst)})
    return records


def extract_tar(archive, destination):
    """Extract only regular files/directories; never follow archive links."""
    destination = Path(destination)
    with tarfile.open(archive, 'r:*') as stream:
        entries = stream.getmembers()
        seen = set()
        for entry in entries:
            path = PurePosixPath(entry.name)
            if (path.is_absolute() or '..' in path.parts or '\\' in entry.name
                    or not path.parts or str(path) in seen
                    or not (entry.isfile() or entry.isdir())):
                raise ValueError(f'Unsafe TAR entry: {entry.name}')
            seen.add(str(path))
        destination.mkdir(parents=True, exist_ok=False)
        records = []
        for entry in entries:
            dst = target(destination, entry.name)
            if entry.isdir():
                dst.mkdir(parents=True, exist_ok=True)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(entry) as reader, dst.open('xb') as writer:
                shutil.copyfileobj(reader, writer, 4 * 1024 * 1024)
            if dst.stat().st_size != entry.size:
                raise ValueError('Extracted TAR size mismatch')
            records.append({'path': entry.name, 'bytes': entry.size, 'sha256': digest(dst)})
    return records


def acquire(item, root, progress, limit_mib_s):
    final = target(root, item['destination'])
    staging = final.with_name(final.name + '.partial')
    if final.exists() or staging.exists():
        raise FileExistsError(f'Refusing to overwrite existing resource: {final}')
    final.parent.mkdir(parents=True, exist_ok=True)
    if item['kind'] == 'copy_tree':
        records = copy_tree(item['source'], staging, progress, limit_mib_s, item.get('exclude', ()))
    elif item['kind'] == 'official_hf_snapshot':
        from .hf_snapshot import acquire_snapshot
        records = acquire_snapshot(item, staging, progress, limit_mib_s)
    elif item['kind'] in ('official_zip', 'official_tar'):
        if not item['url'].startswith('https://') or len(item['sha256']) != 64:
            raise ValueError('Pinned HTTPS URL and SHA256 required')
        staging.mkdir()
        archive_name = 'original.zip' if item['kind'] == 'official_zip' else 'original.tar.gz'
        archive = staging / archive_name
        subprocess.run(['curl', '--fail', '--location', '--retry', '3',
                        '--connect-timeout', '30', '--speed-time', '120', '--speed-limit', '1024',
                        '--limit-rate', f'{limit_mib_s}M', '--output', str(archive), item['url']], check=True)
        if archive.stat().st_size != item['bytes'] or digest(archive) != item['sha256']:
            raise ValueError('Official archive size/SHA256 mismatch')
        records = [{'path': archive_name, 'bytes': item['bytes'], 'sha256': item['sha256']}]
        extractor = extract_zip if item['kind'] == 'official_zip' else extract_tar
        records += [dict(r, path='extracted/' + r['path'])
                    for r in extractor(archive, staging / 'extracted')]
    else:
        raise ValueError('Unsupported resource kind')
    save(staging / 'IMPORT_RECEIPT.json', {
        'schema_version': 1, 'resource': item, 'files': records,
        'file_count': len(records), 'total_bytes': sum(r['bytes'] for r in records),
        'status': 'BYTE_VERIFIED', 'runtime_validated': False,
        'training_domain_verified': False, 'completed_unix': time.time(),
    })
    staging.rename(final)
    return {'destination': str(final), 'receipt_sha256': digest(final / 'IMPORT_RECEIPT.json'),
            'status': 'BYTE_VERIFIED', 'file_count': len(records)}


def run(plan_path, root, record):
    """Run a fresh import; record FAILED on any error, never resume implicitly."""
    plan_path, root, record = Path(plan_path), Path(root), Path(record)
    if not root.is_absolute() or root.resolve() != root or not record.is_absolute():
        raise ValueError('Explicit real absolute roots required')
    plan = json.loads(plan_path.read_text())
    destinations = [target(root, i['destination']) for i in plan['resources']]
    if len(set(destinations)) != len(destinations) or any(
            a in b.parents for a in destinations for b in destinations if a != b):
        raise ValueError('Duplicate or nested resources')
    if len({i['id'] for i in plan['resources']}) != len(plan['resources']):
        raise ValueError('Duplicate resource IDs')
    limit = plan.get('limit_mib_s', 32)
    if limit <= 0:
        raise ValueError('Positive transfer limit required')
    record.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(plan_path, record / 'plan.json')
    shutil.copyfile(__file__, record / 'importer.py')
    state = {'status': 'RUNNING', 'pid': os.getpid(), 'started_unix': time.time(),
             'plan_sha256': digest(plan_path), 'importer_sha256': digest(__file__), 'resources': {}}

    def progress(files, size):
        state.update(verified_files=files, verified_bytes=size, heartbeat_unix=time.time())
        save(record / 'status.json', state)

    try:
        for item in plan['resources']:
            state.update(current=item['id'], verified_files=0, verified_bytes=0)
            progress(0, 0)
            state['resources'][item['id']] = acquire(item, root, progress, limit)
        state.update(status='BYTE_VERIFIED', current=None, completed_unix=time.time())
    except BaseException as exc:
        state.update(status='FAILED', error=repr(exc), failed_unix=time.time())
        raise
    finally:
        save(record / 'status.json', state)
