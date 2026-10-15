"""Explicit bounded-scope CPU intake watcher; no retries, GPU or training."""
import argparse
import json
from pathlib import Path
import time
import traceback
from AlphaBrain.research.dsol.data.resource_import import save, digest
from benchmarks.RoboTwin.official_admission import finalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-run', 'checkpoint-run', 'source-report', 'output', 'record'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(vars(args)))
        return
    record = Path(args.record)
    record.mkdir(parents=True, exist_ok=False)
    state = {'status': 'WAITING_FOR_IMPORTS', 'started_unix': time.time(), 'arguments': vars(args),
             'entrypoint_sha256': digest(__file__)}
    save(record / 'launch.json', state)
    try:
        while True:
            imports = [json.loads((Path(run) / 'status.json').read_text())
                       for run in (args.data_run, args.checkpoint_run)]
            if any(item['status'] == 'FAILED' for item in imports):
                raise RuntimeError('Import failed; no automatic retry or partial publication')
            state.update(heartbeat_unix=time.time(),
                         imports=[{'status': item['status'], 'completed_resources': len(item['resources']),
                                   'current': item.get('current')} for item in imports])
            save(record / 'status.json', state)
            if all(item['status'] == 'BYTE_VERIFIED' for item in imports):
                break
            time.sleep(30)
        state['status'] = 'AUDITING_ALL_50_TASKS'
        save(record / 'status.json', state)
        result = finalize(args.data_run, args.checkpoint_run, args.source_report, args.output)
        state.update(status='COMPLETE' if result['status'] == 'BYTE_AND_SCHEMA_VERIFIED' else 'FAILED',
                     release=str(Path(args.output) / 'release.json'), completed_unix=time.time())
    except BaseException:
        state.update(status='FAILED', traceback=traceback.format_exc())
        raise
    finally:
        save(record / 'status.json', state)


if __name__ == '__main__':
    main()
