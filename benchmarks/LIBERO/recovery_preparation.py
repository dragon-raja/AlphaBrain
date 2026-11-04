"""CPU preparation shared with the real recovery validator; no model loading."""
from copy import deepcopy
from pathlib import Path

from AlphaBrain.research.dsol.data.artifacts import read
from AlphaBrain.research.dsol.data import native_recovery
from AlphaBrain.research.dsol.execution.manifest import file_digest
from AlphaBrain.research.dsol.execution.queue import _durable_write
from .search_runtime import selected_checkpoint


def bind_model_assets(campaign):
    """Add missing model identities from the already bound simulator config.

    Never bless today's arbitrary bytes: expected hashes come from the admitted
    model declaration. Existing declarations may not be silently overwritten.
    """
    c = deepcopy(campaign)
    hashes = c['input_hashes']
    paired = c['paired_config']
    if hashes.get(paired) != file_digest(paired):
        raise ValueError('Unbound or changed paired config')
    sim_path = read(paired)['cohorts']['A']['sim_config']
    if hashes.get(sim_path) != file_digest(sim_path):
        raise ValueError('Unbound or changed simulator config')
    model = selected_checkpoint(read(sim_path), allow_native=True)
    expected = {
        str(Path(model['path']) / 'model.safetensors'): model['weights_sha256'],
        str(Path(model['path']) / 'assets/physical-intelligence/libero/norm_stats.json'):
            model['normalization_sha256'],
    }
    for path, value in expected.items():
        if path in hashes and hashes[path] != value:
            raise ValueError('Conflicting model asset identity: ' + path)
        if file_digest(path) != value:
            raise ValueError('Model asset differs from admitted identity: ' + path)
        hashes[path] = value
    return c


def prepare_campaigns(common, planned, output, *, prefix, wave_size=288,
                      expected_limits, validator=None):
    """Prepare unique waves, validating actual files before publishing allocation.

    The caller supplies stage/actor bindings and explicit output namespace.
    A failed preparation remains evidence and cannot be retried in place.
    """
    if validator is None:
        from .controlled_recovery import validate_campaign
        validator = validate_campaign
    if not prefix or Path(prefix).name != prefix or prefix in ('.', '..'):
        raise ValueError('Explicit safe phase prefix required')
    if native_recovery.limits(planned) != expected_limits:
        raise ValueError('Recovery budget differs from approved allocation')
    common = bind_model_assets(common)
    blocks = []
    for split in ('validation', 'seen_new_initials', 'unseen_test'):
        blocks.extend(native_recovery.blocks([r for r in planned if r['split'] == split]))
    if sum(len(b['rows']) for b in blocks) != len(planned):
        raise ValueError('Unknown recovery split')
    phases = []
    for number, chunk in enumerate(native_recovery.waves(blocks, size=wave_size)):
        phase = f'{prefix}-{number:03d}'
        directory = Path(output) / phase
        directory.mkdir(parents=True, exist_ok=False)
        chunk = [dict(b, index=i, host=('fresh', 'gnu', 'video')[i % 3])
                 for i, b in enumerate(chunk)]
        rows = [r for b in chunk for r in b['rows']]
        methods = {r['method'] for r in rows}
        c = dict(common, blocks=chunk, limits=native_recovery.limits(rows),
                 actors={k: v for k, v in common['actors'].items() if k in methods})
        path = directory / 'campaign.json'
        _durable_write(path, c)
        validator(path)
        phases.append(dict(phase=phase, campaign=str(path), sha256=file_digest(path),
                           limits=c['limits'], searches=len(rows)))
    return dict(status='FROZEN_OUTCOME_INDEPENDENT', phases=phases,
                limits=expected_limits, automatic_retry=False, wave_size=wave_size,
                scientific_rows_and_order_unchanged=True)
