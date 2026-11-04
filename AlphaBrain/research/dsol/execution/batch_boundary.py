"""Read a bound operator pause request only at a completed optimizer boundary.

This grants no execution permission and never cancels or retries work. Callers
must finish and publish the entire current batch before invoking this function.
Requests belong on the controller's local filesystem, alongside its ledger.
"""
import json
from pathlib import Path


def pause_requested(path, *, experiment_id, config_sha256, completed_version):
    path = Path(path)
    if not path.exists():
        return False
    request = json.loads(path.read_text())
    required = {'schema', 'experiment_id', 'config_sha256', 'after_version'}
    if (set(request) != required
            or request['schema'] != 'PAUSE_AT_COMPLETE_BATCH_V1'
            or request['experiment_id'] != experiment_id
            or request['config_sha256'] != config_sha256):
        raise ValueError('Pause request is not bound to this execution')
    target = request['after_version']
    if type(target) is not int or target < 0:
        raise ValueError('Nonnegative complete batch version required')
    if type(completed_version) is not int or completed_version < 0:
        raise ValueError('Invalid completed batch version')
    return completed_version >= target
