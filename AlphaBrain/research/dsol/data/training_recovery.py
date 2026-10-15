"""CPU-only provenance and replay gates for interrupted matched training."""

import math
from pathlib import Path

from .artifacts import read, sha


def recovery_source(run, step, manifest, smoke, source, initialization):
    run = Path(run)
    config = read(run / 'run_config.json')
    for key, path in [('manifest_sha256', manifest), ('smoke_sha256', smoke),
                      ('external_train_sha256', source)]:
        if config[key] != sha(path):
            raise ValueError('Recovery provenance mismatch: ' + key)
    if (config['shuffle'] is not False or config['seed'] != 41
            or config['global_batch'] != 8 or config['fsdp_devices'] != 8
            or Path(config['checkpoint']).resolve() != Path(initialization).resolve()
            or not 0 < step < config['total_steps']):
        raise ValueError('Recovery protocol mismatch')
    candidates = list((run / 'checkpoints').glob('*/*/' + str(step)))
    if len(candidates) != 1:
        raise ValueError('Exactly one explicit recovery checkpoint required')
    checkpoint = candidates[0]
    metadata = read(checkpoint / '_CHECKPOINT_METADATA')
    if not metadata.get('commit_timestamp_nsecs'):
        raise ValueError('Checkpoint is not committed')
    for name in ('params/manifest.ocdbt', 'train_state/manifest.ocdbt', 'assets/norm_stats.json'):
        if not (checkpoint / name).is_file():
            raise ValueError('Incomplete recovery checkpoint: ' + name)
    return dict(run=str(run.resolve()), step=step, checkpoint=str(checkpoint.resolve()),
                run_config_sha256=sha(run / 'run_config.json'),
                checkpoint_metadata_sha256=sha(checkpoint / '_CHECKPOINT_METADATA'))


def compare_replayed_metrics(actual, expected, *, rtol=1e-5, atol=1e-7, numerical_diagnostic=False):
    """Stop, rather than silently continue, when original-step replay disagrees."""
    if actual['step'] != expected['step']:
        raise ValueError('Replay optimizer step mismatch')
    limits = dict(loss=.01, grad_norm=.02, param_norm=1e-6) if numerical_diagnostic else {
        key: rtol for key in ('loss', 'grad_norm', 'param_norm')}
    for key in limits:
        if (not math.isfinite(actual[key]) or not math.isfinite(expected[key])
                or not math.isclose(actual[key], expected[key], rel_tol=limits[key], abs_tol=atol)):
            raise ValueError(f'Recovery replay differs: {key}: actual={actual[key]!r}, expected={expected[key]!r}')
    return dict(step=actual['step'], status='PASS', relative_limits=limits, atol=atol,
                interpretation='Numerical sanity check, not bitwise or statistical equivalence',
                differences={k: actual[k] - expected[k] for k in ('loss', 'grad_norm', 'param_norm')})
