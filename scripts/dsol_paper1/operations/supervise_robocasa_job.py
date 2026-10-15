"""Record a single explicitly requested subprocess, including abnormal exits.

No retries, protocol changes, GPU imports, or unrelated process management.
Launch this lightweight supervisor detached with an explicit command after --.
"""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

from AlphaBrain.research.dsol.data.artifacts import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not args.execute or not command:
        parser.error('Explicit --execute and command required')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    with (args.output / 'stdout.log').open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        def forward(signum, _frame):
            if child.poll() is None:
                child.send_signal(signum)
        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGHUP, forward)
        while True:
            record = dict(supervisor_pid=os.getpid(), child_pid=child.pid, command=command,
                          started_unix=started, heartbeat_unix=time.time(), status='RUNNING')
            write(args.output / 'status.json', record)
            try:
                code = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                pass
        record.update(status='COMPLETE' if code == 0 else 'ERROR',
                      returncode=code, finished_unix=time.time())
        write(args.output / 'status.json', record)
        raise SystemExit(code if code >= 0 else 128 - code)


if __name__ == '__main__':
    main()
