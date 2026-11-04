"""Single-process immutable runtime leases; no model imports or launch policy.

The authority owns permissions. This object only retains an already authorized
resource across sequential blocks. Any failed lease poisons the whole session.
"""
from contextlib import contextmanager
import threading


class ResidentRuntimeCache:
    def __init__(self):
        self.runtime = None
        self.identity = None
        self.failed = False
        self.loads = 0
        self.reuses = 0
        self._lock = threading.Lock()

    @contextmanager
    def lease(self, identity, factory, bind):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Concurrent resident runtime lease')
        try:
            if self.failed:
                raise RuntimeError('Failed resident session requires explicit recovery')
            if not isinstance(identity, str) or not identity:
                raise ValueError('Immutable runtime identity required')
            reused = self.runtime is not None
            if reused and identity != self.identity:
                raise RuntimeError('Resident runtime identity changed')
            if not reused:
                self.runtime = factory()
                self.identity = identity
                self.loads += 1
            else:
                self.reuses += 1
            bind(self.runtime)
            yield self.runtime, reused
        except BaseException:
            self.failed = True
            raise
        finally:
            self._lock.release()
