"""CPU-only local listener probe, before any model ownership is granted."""
from pathlib import Path
import os
import tempfile
from multiprocessing.connection import Listener


def probe_unix_listener(directory=None):
    """Exercise the actual OS path limit; do not infer it from a mock queue."""
    with tempfile.TemporaryDirectory(prefix='dsol-search-sim-',dir=directory) as temporary:
        address=str(Path(temporary)/'sim.sock')
        try:
            with Listener(address,family='AF_UNIX'):
                return dict(status='PASS',path_bytes=len(os.fsencode(address)),
                            temporary_root=str(Path(temporary).parent))
        except OSError as error:
            raise ValueError('Simulator IPC unavailable; configure a short owned TMPDIR') from error
