"""Immutable versioned requests for queue-owned resident workers.

This is transport only: it does not start processes, allocate GPU resources,
retry work, alter budgets or decide which scientific records to collect.
"""
from pathlib import Path
import time

from .manifest import file_digest
from .queue import _durable_write
from ..data.artifacts import read


class ResidentMailbox:
    def __init__(self, root):self.root=Path(root)

    def publish(self, version, request):
        if type(version) is not int or version<0:raise ValueError('Invalid resident version')
        if (self.root/'stop.json').exists():raise ValueError('Resident mailbox closed')
        path=self.root/f'{version:04d}.request.json'
        if path.exists():raise ValueError('Request already published; no automatic replay')
        if version and not (self.root/f'{version-1:04d}.complete.json').exists():
            raise ValueError('Previous resident request not complete')
        _durable_write(path,dict(version=version,request=str(request),sha256=file_digest(request)))
        return path

    def next(self, version, *, timeout, alive=lambda:True):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            path=self.root/f'{version:04d}.request.json'
            if path.exists():
                value=read(path)
                if value['version']!=version or file_digest(value['request'])!=value['sha256']:
                    raise ValueError('Resident request version/hash changed')
                if (self.root/f'{version:04d}.complete.json').exists():
                    raise ValueError('Completed resident work cannot be replayed')
                return value
            if (self.root/'stop.json').exists():
                if read(self.root/'stop.json')['completed_versions']!=version:
                    raise ValueError('Resident stop version mismatch')
                return None
            if not alive():raise RuntimeError('Resident coordinator no longer alive')
            time.sleep(.1)
        raise TimeoutError('Resident request wait exceeded bound')

    def complete(self, version, request_digest, receipt):
        request=read(self.root/f'{version:04d}.request.json')
        if request['sha256']!=request_digest or file_digest(request['request'])!=request_digest:
            raise ValueError('Resident request changed before completion')
        path=self.root/f'{version:04d}.complete.json'
        if path.exists():raise ValueError('Duplicate resident completion')
        _durable_write(path,dict(version=version,request_sha256=request_digest,
                                 receipt=str(receipt),receipt_sha256=file_digest(receipt)))

    def result(self, version):
        path=self.root/f'{version:04d}.complete.json'
        if not path.exists():return None
        value=read(path);request=read(self.root/f'{version:04d}.request.json')
        if (value['version']!=version or value['request_sha256']!=request['sha256']
                or file_digest(request['request'])!=request['sha256']
                or file_digest(value['receipt'])!=value['receipt_sha256']):
            raise ValueError('Resident completed result changed')
        return value

    def close(self, completed_versions):
        if (self.root/'stop.json').exists():raise ValueError('Resident stop already issued')
        if completed_versions and self.result(completed_versions-1) is None:
            raise ValueError('Cannot stop before last submitted request completes')
        _durable_write(self.root/'stop.json',dict(completed_versions=completed_versions))
