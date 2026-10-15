"""Versioned, non-destructive workspace backup with content receipts (CPU only).

Only explicitly configured roots are copied. Symlinks are preserved, not followed;
system dependencies and credentials are not a bootable machine image.
"""
import argparse
from datetime import datetime, timezone
import fnmatch
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import time

from .artifacts import read, write


def backup(config_file):
    cfg = read(config_file); destination = Path(cfg['destination']).resolve()
    sources = [Path(p).absolute() for p in cfg['sources']]
    if (not destination.is_absolute() or str(destination) in ('/', '/share', '/workspace', '/root')
            or any(str(p) in ('/', '/workspace', '/root', '/share') or not p.exists() for p in sources)
            or any(destination == p or p in destination.parents for p in sources)):
        raise ValueError('Explicit existing non-root sources and a separate destination required')
    if len(set(sources)) != len(sources) or any(a in b.parents for a in sources for b in sources if a != b):
        raise ValueError('Backup sources must not overlap')
    destination.mkdir(parents=True, exist_ok=True); destination.chmod(0o700)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    manifest_path = destination/'manifest.json'
    previous = read(manifest_path)['files'] if manifest_path.exists() else {}
    records = dict(previous); changed = skipped = copied_bytes = 0; unstable = []; excluded = []
    started = time.monotonic(); last_status = 0
    state = dict(status='RUNNING', host=cfg['host'], sources=[str(p) for p in sources],
                 started_utc=stamp, filesystem_scope='Explicit project/runtime roots, not full operating system or credentials')
    def status(final=False):
        nonlocal last_status
        state.update(changed_files=changed, unchanged_files=skipped, copied_bytes=copied_bytes,
                     elapsed_seconds=time.monotonic()-started, unstable_files=unstable,
                     excluded_count=len(excluded), checked_utc=datetime.now(timezone.utc).isoformat())
        write(destination/'status.json', state); last_status=time.monotonic()
        if final:
            write(destination/'receipts'/f'{stamp}.json', dict(state, excluded=excluded))
    status()
    def paths():
        for source in sources:
            if source.is_file() or source.is_symlink():
                yield source
                continue
            for current, dirs, files in os.walk(source, followlinks=False):
                for name in list(dirs):
                    path=Path(current)/name
                    if path.is_symlink():
                        dirs.remove(name); yield path
                for name in files: yield Path(current)/name
    try:
        for source in paths():
            key=str(source); relative=source.relative_to(source.anchor)
            if any(fnmatch.fnmatch(key, pattern) or fnmatch.fnmatch(source.name, pattern)
                   for pattern in cfg.get('exclude_patterns', [])):
                excluded.append(key); continue
            target=destination/'files'/relative
            try:before=source.lstat()
            except FileNotFoundError:
                unstable.append(key);continue
            kind='symlink' if source.is_symlink() else 'file'
            identity=dict(bytes=before.st_size, mtime_ns=before.st_mtime_ns, mode=before.st_mode, kind=kind)
            if kind=='symlink':identity['link_target']=os.readlink(source)
            old=previous.get(key)
            if old and all(old.get(k)==v for k,v in identity.items()) and (target.exists() or target.is_symlink()):
                if cfg.get('verify_unchanged',False):
                    if kind=='symlink':
                        digest=hashlib.sha256(os.readlink(target).encode()).hexdigest()
                    else:
                        check=hashlib.sha256()
                        with target.open('rb') as stream:
                            while data:=stream.read(4*1024*1024):check.update(data)
                        digest=check.hexdigest()
                    if digest!=old['sha256']:raise IOError('Previously backed-up content changed: '+key)
                skipped+=1
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                temp=None
                try:
                    if kind=='symlink':
                        temp=target.with_name(target.name+'.'+stamp+'.tmp')
                        temp.symlink_to(identity['link_target'])
                        digest=hashlib.sha256(identity['link_target'].encode()).hexdigest()
                    else:
                        digest=hashlib.sha256()
                        with source.open('rb') as src, tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as dst:
                            temp=Path(dst.name)
                            while data:=src.read(4*1024*1024):
                                dst.write(data); digest.update(data); copied_bytes+=len(data)
                                rate=cfg.get('max_mib_per_second',32)*1024*1024
                                delay=copied_bytes/rate-(time.monotonic()-started)
                                if delay>0:time.sleep(min(delay,1))
                            dst.flush();os.fsync(dst.fileno())
                        after=source.lstat()
                        if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
                            unstable.append(key);continue
                        shutil.copystat(source,temp);digest=digest.hexdigest()
                        # Verify destination bytes independently before publishing the file.
                        check=hashlib.sha256()
                        with temp.open('rb') as stream:
                            while data:=stream.read(4*1024*1024):check.update(data)
                        if check.hexdigest()!=digest:raise IOError('Backup checksum mismatch: '+key)
                    if target.exists() or target.is_symlink():
                        version=destination/'versions'/stamp/relative;version.parent.mkdir(parents=True,exist_ok=True)
                        target.rename(version)
                    temp.replace(target);temp=None
                    records[key]=dict(identity,sha256=digest,backup=str(target));changed+=1
                except FileNotFoundError:
                    unstable.append(key)
                finally:
                    if temp is not None and (temp.exists() or temp.is_symlink()):temp.unlink()
            if time.monotonic()-last_status>10:status()
        write(manifest_path,dict(host=cfg['host'],files=records,sources=[str(p) for p in sources],
              no_deletion=True,credentials_excluded=True,scope='File backup; final quiescent pass required for running jobs'))
        state['status']='PASS' if not unstable else 'NEEDS_INCREMENTAL_PASS'
        status(final=True)
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc));status(final=True);raise
    return state


def checkpoint_loop(config_file, interval_seconds, until_utc):
    """Bounded protection of evolving local queue ledgers; no job control."""
    if interval_seconds < 60:
        raise ValueError('Checkpoint interval must be at least one minute')
    deadline=datetime.fromisoformat(until_utc)
    if deadline.tzinfo is None:raise ValueError('Explicit timezone required')
    while datetime.now(timezone.utc)<deadline:
        backup(config_file)
        remaining=(deadline-datetime.now(timezone.utc)).total_seconds()
        # Small sleep slices keep signal handling responsive.
        end=time.monotonic()+min(interval_seconds,max(0,remaining))
        while time.monotonic()<end:time.sleep(min(30,end-time.monotonic()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--repeat-seconds',type=float)
    parser.add_argument('--until-utc')
    args=parser.parse_args()
    if not args.execute:parser.error('Explicit --execute required for copying')
    if args.repeat_seconds is not None:
        if not args.until_utc:parser.error('Bounded checkpoint loop requires --until-utc')
        checkpoint_loop(args.config,args.repeat_seconds,args.until_utc)
    else:
        backup(args.config)
