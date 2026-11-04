"""Direct-channel learner receipts, independent of a mutable NFS latest pointer."""
import json
import os
import subprocess
import time
from pathlib import Path
from .manifest import file_digest
from .queue import _durable_write


def execute_update(command, log_path, root, *, run, version, final_version=256):
    """Run one CPU learner and accept its exact next-version stdout once.

    An exclusive per-version intent is durable before the command starts. Any
    command/receipt failure retains it, so even a new log cannot replay an update.
    Recovery must reconcile the writer's immutable version separately.
    """
    if (type(version) is not int or type(final_version) is not int
            or not 1 <= version <= final_version <= 256):
        raise ValueError('Explicit expected output version required')
    if (Path(root)/run/f'{version:04d}.json').exists():
        raise ValueError('Version already accepted; optimizer must not repeat')
    directory=Path(root)/run
    directory.mkdir(parents=True,exist_ok=True)
    intent=directory/f'{version:04d}.attempt.json'
    with intent.open('x') as stream:
        json.dump(dict(run=run,version=version,log=str(log_path),status='ATTEMPTED'),stream)
        stream.flush();os.fsync(stream.fileno())
    descriptor=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    with Path(log_path).open('xb') as log:
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=log)
        log.write(process.stdout)
        log.flush()
        process.check_returncode()
    value = accept(process.stdout, root, run=run, version=version,final_version=final_version)
    _durable_write(Path(log_path).with_suffix('.publication.json'), value)
    return value


def accept(text, root, *, run, version, final_version=256, timeout=60, clock=time.monotonic, sleep=time.sleep, hasher=file_digest):
    """Accept one completed CPU update; waiting never reruns the optimizer.

    The caller supplies stdout from the authenticated, successful learner SSH
    process. Only the unique versioned policy file is read over the filesystem.
    No shared mutable publication pointer authorizes a subsequent collection.
    """
    value=json.loads(text)
    if (type(final_version) is not int or not 1<=final_version<=256 or not 0<=version<=final_version
            or value['run']!=run or value['version']!=version or value['episodes']!=version*32
            or value['operation_model_updates']!=0 or value['optimizer_device']!='cpu'
            or value['status']!=('COMPLETE' if version==final_version else 'PAUSED_AT_UPDATE_BOUNDARY')
            or Path(value['policy']).name!=f'policy-{version:04d}.pt'):
        raise ValueError('Learner returned a stale or mixed publication')
    deadline=clock()+timeout
    while True:
        try:actual=hasher(value['policy'])
        except FileNotFoundError:actual=None
        if actual==value['policy_sha256']:break
        if clock()>=deadline:
            raise TimeoutError('Published policy not visible with its direct-channel hash; update is not replayed')
        sleep(.25)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    target=root/run/f'{version:04d}.json';target.parent.mkdir(exist_ok=True)
    if target.exists():
        if json.loads(target.read_text())!=value:raise ValueError('Conflicting publication for an existing version')
    else:_durable_write(target,value)
    return value


def published(root, run, version):
    value=json.loads((Path(root)/run/f'{version:04d}.json').read_text())
    if value['version']!=version or value['run']!=run:
        raise ValueError('Publication cache identity differs')
    if file_digest(value['policy'])!=value['policy_sha256']:
        raise ValueError('Versioned policy changed')
    return value
