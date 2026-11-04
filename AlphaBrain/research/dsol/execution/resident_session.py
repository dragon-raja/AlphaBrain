"""Local custody of an idle/running resident slot, separate from task receipts."""
from copy import deepcopy
from pathlib import Path

from .manifest import digest
from .queue import _durable_write


def session_pin(launch):
    return dict(owner=launch['owner'], epoch=launch['epoch'], slot=launch['slot'],
                binding_sha256=launch['binding_sha256'], permit_sha256=digest(launch['permit']))


class ResidentSession:
    def __init__(self, root):
        self.root = Path(root)
        self.pin = None
        self.row = None
        self.path = None

    def check(self, launch):
        pin = session_pin(launch)
        if self.pin is not None and pin != self.pin:
            raise RuntimeError('Resident slot/owner/epoch/permit changed')
        if self.pin is None:
            self.pin = deepcopy(pin)
            self.path = self.root/'residents'/(digest(launch['slot'])+'.json')

    def update(self, task_row, status):
        if status not in ('RUNNING', 'IDLE', 'UNKNOWN'):
            raise ValueError('Invalid resident state')
        self.check(task_row['launch'])
        self.row = deepcopy(task_row)
        self.row.update(status=status, resident=True)
        _durable_write(self.path, self.row)

    def close(self):
        if self.row is not None:
            self.row['status'] = 'CLOSED' if self.row['status'] == 'IDLE' else 'UNKNOWN'
            _durable_write(self.path, self.row)
