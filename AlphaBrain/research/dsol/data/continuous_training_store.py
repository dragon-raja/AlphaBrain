"""Durable continuous-PPO state. Only clean boundaries are explicitly resumable."""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_file(path, writer):
    path = Path(path); temporary = path.with_name(path.name + '.tmp')
    with temporary.open('xb') as stream:
        writer(stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def write_json(path, value):
    atomic_file(path, lambda f: f.write((json.dumps(value, indent=2, allow_nan=False)+'\n').encode()))


def pack(value):
    if isinstance(value, np.ndarray): return torch.from_numpy(value.copy())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k:pack(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)): return type(value)(pack(v) for v in value)
    return value


def numpy_tree(value):
    if isinstance(value, torch.Tensor): return value.cpu().numpy().copy()
    if isinstance(value, dict): return {k:numpy_tree(v) for k,v in value.items()}
    if isinstance(value, list): return [numpy_tree(v) for v in value]
    return value


def rng_state():
    state = np.random.get_state()
    return dict(torch=torch.get_rng_state(), python=random.getstate(),
                numpy=[state[0], state[1].tolist(), *state[2:]])


def restore_rng(state):
    torch.set_rng_state(state['torch']); random.setstate(state['python'])
    n = state['numpy']; np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))


def fork_complete_stage(source, destination, binding, *, previous_schedule, schedule,
                        source_files, expected_update):
    """Copy a hash-bound clean prefix into a separately authorized longer stage.

    Optimizer, RNG, reward statistics and episode bytes are preserved. The old
    output is never rebound or resumed. An incomplete destination is deliberately
    left for audit; this function cannot retry or overwrite it.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists() or source.resolve() == destination.resolve():
        raise ValueError('New, separate training-stage output required')
    # NFS maps an exclusive flock to a write lock and requires a writable fd.
    # r+ opens the existing lock without creating, truncating or writing it.
    with (source/'writer.lock').open('r+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError: raise ValueError('Source trainer is still active') from None
        if any((source/n).exists() for n in ('inflight.json', 'failure.json')) or any(source.rglob('*.tmp')):
            raise ValueError('Ambiguous source stage cannot be extended')
        old = json.loads((source/'binding.json').read_text())
        if (set(old) != set(binding)
                or any(old[k] != binding[k] for k in old if k not in ('config', 'config_sha256', 'schedule_sha256', 'retained_checkpoint_updates'))
                or not set(old.get('retained_checkpoint_updates',[])).issubset(binding.get('retained_checkpoint_updates',[]))
                or set(old['config']) != set(binding['config'])
                or any(old['config'][k] != binding['config'][k] for k in old['config'] if k != 'updates')
                or type(expected_update) is not int or expected_update != old['config']['updates']
                or binding['config']['updates'] <= expected_update
                or len(previous_schedule) != expected_update*old['config']['episodes_per_update']
                or schedule[:len(previous_schedule)] != previous_schedule
                or len(schedule) != binding['config']['updates']*binding['config']['episodes_per_update']
                or old['schedule_sha256'] != fingerprint(previous_schedule)
                or binding['schedule_sha256'] != fingerprint(schedule)):
            raise ValueError('Only an exact schedule-prefix extension is allowed')
        names = {'binding.json', 'latest.json'}
        for folder in ('episodes', 'checkpoints', 'updates'):
            names.update(str(p.relative_to(source)) for p in (source/folder).iterdir() if p.is_file())
        if set(source_files) != names:
            raise ValueError('Incomplete source-stage file manifest')
        for name, stamp in source_files.items():
            p = source/name
            if p.is_symlink() or not p.resolve().is_relative_to(source.resolve()) or digest(p) != stamp:
                raise ValueError('Source-stage evidence changed')
        latest = json.loads((source/'latest.json').read_text())
        if latest['update'] != expected_update or source_files.get(latest['path']) != latest['sha256']:
            raise ValueError('Wrong complete source checkpoint')
        state = torch.load(source/latest['path'], map_location='cpu', weights_only=True)
        receipts = sorted((source/'episodes').glob('*.json'))
        if (state['update'] != expected_update or state['cursor'] != len(previous_schedule)
                or len(receipts) != len(previous_schedule)
                or len(list((source/'episodes').glob('*.pt'))) != len(receipts)):
            raise ValueError('Source is not a complete episode boundary')
        for index, path in enumerate(receipts):
            receipt = json.loads(path.read_text())
            if (receipt['index'] != index or receipt['metrics']['initial_identity'] != previous_schedule[index]
                    or source_files.get(receipt['path']) != receipt['sha256']):
                raise ValueError('Source episode prefix differs')
        destination.mkdir(parents=True, exist_ok=False)
        for name in sorted(names-{'binding.json'}):
            target = destination/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source/name, target)
            if digest(target) != source_files[name]: raise ValueError('Stage-copy hash mismatch')
        write_json(destination/'binding.json', binding)
        write_json(destination/'stage-origin.json', dict(source=str(source), update=expected_update,
            source_files=source_files, checkpoint=latest, optimizer_rng_reward_preserved=True))


class ContinuousTrainingStore:
    """New outputs only; per-episode receipts plus initial/latest-two checkpoints.

    Inputs to torch.load are this store's hash-checked tensor-only records.
    Failed or interrupted operations are not automatically rolled back/replayed.
    """
    def __init__(self, root, binding, *, resume=False):
        retained=binding.get('retained_checkpoint_updates',[])
        if (not isinstance(retained,list) or any(type(v) is not int or v<0 for v in retained)
                or len(set(retained))!=len(retained)):
            raise ValueError('Explicit unique checkpoint retention updates required')
        self.root = Path(root)
        if not resume:
            self.root.mkdir(parents=True, exist_ok=False)
            for name in ('episodes', 'checkpoints', 'updates'): (self.root/name).mkdir()
            write_json(self.root/'binding.json', binding)
        else:
            if json.loads((self.root/'binding.json').read_text()) != binding:
                raise ValueError('Changed protocol, schedule, source or backend identity')
            self.assert_clean()
        self.binding = binding
        self._lock = (self.root/'writer.lock').open('a')
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock.close()
            raise ValueError('Another trainer owns this output') from None
        try:self.assert_clean()
        except BaseException:
            self.close();raise

    def close(self):
        if not self._lock.closed:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()

    def assert_clean(self):
        if (any((self.root/n).exists() for n in ('inflight.json', 'failure.json'))
                or any(self.root.rglob('*.tmp'))):
            raise ValueError('Failed/ambiguous operation: stop; no automatic replay')

    def begin(self, kind, index):
        if self._lock.closed:raise ValueError('Closed trainer cannot write')
        self.assert_clean()
        write_json(self.root/'inflight.json', dict(kind=kind, index=index))

    def finish(self):
        (self.root/'inflight.json').unlink()
        fd=os.open(self.root,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)

    def fail(self, error):
        if not (self.root/'failure.json').exists():
            write_json(self.root/'failure.json', dict(status='FAILED_STOPPED', error=repr(error), automatic_retry=False))

    def _save(self, path, payload):
        if path.exists(): raise FileExistsError(path)
        atomic_file(path, lambda f: torch.save(pack(payload), f))
        return dict(path=str(path.relative_to(self.root)), sha256=digest(path))

    def load(self, receipt):
        path = self.root/receipt['path']
        if not path.resolve().is_relative_to(self.root.resolve()) or digest(path) != receipt['sha256']:
            raise ValueError('Checkpoint/episode identity mismatch')
        return torch.load(path, map_location='cpu', weights_only=True)

    def save_episode(self, index, episode, metrics):
        receipt = self._save(self.root/'episodes'/f'{index:04d}.pt', dict(episode=episode, rng=rng_state()))
        write_json(self.root/'episodes'/f'{index:04d}.json', dict(index=index, **receipt, metrics=metrics))

    def episodes(self):
        records = [json.loads(p.read_text()) for p in sorted((self.root/'episodes').glob('*.json'))]
        if [r['index'] for r in records] != list(range(len(records))):
            raise ValueError('Missing or duplicate episode cursor')
        # A data file without its commit receipt means an interrupted transaction.
        if len(list((self.root/'episodes').glob('*.pt'))) != len(records):
            raise ValueError('Uncommitted episode data')
        return records

    def save_checkpoint(self, update, payload, metrics=None):
        receipt = self._save(self.root/'checkpoints'/f'update-{update:03d}.pt', dict(**payload, rng=rng_state()))
        if metrics is not None: write_json(self.root/'updates'/f'{update:03d}.json', metrics)
        write_json(self.root/'latest.json', dict(update=update, **receipt))
        # Retain the initial state and two newest update states, not 512 copies.
        removed=[]
        for p in sorted((self.root/'checkpoints').glob('update-*.pt')):
            number=int(p.stem.split('-')[-1])
            if number != 0 and number < update-1 and number not in self.binding.get('retained_checkpoint_updates',[]):
                removed.append(dict(path=str(p.relative_to(self.root)), sha256=digest(p), bytes=p.stat().st_size));p.unlink()
        if removed: write_json(self.root/'updates'/f'{update:03d}-retention.json', dict(reason='superseded resumable optimizer state; episode evidence retained',removed=removed))
