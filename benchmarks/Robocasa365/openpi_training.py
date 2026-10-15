"""Matched-data configuration adapter and real-batch identity checks for OpenPI."""

import dataclasses
import hashlib
from pathlib import Path

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha


def matched_config(manifest_path, checkpoint, *, batch_size=8, fsdp_devices=8):
    from openpi.shared import normalize
    from openpi.training import config, weight_loaders

    path, checkpoint = Path(manifest_path), Path(checkpoint)
    manifest = read(path)
    release = read(path.parent / 'release.json')
    if (manifest['status'] != 'DATA_READY_NOT_TRAINED'
            or release['manifests'][manifest['arm']]['sha256'] != sha(path)):
        raise ValueError('Manifest is not the verified data release')
    for key in ('source_window_ledger', 'view_exposure_manifest'):
        if sha(manifest[key]) != manifest[key+'_sha256']:
            raise ValueError('Released source/exposure manifest changed')
    training = manifest['training']
    if (batch_size != training['global_batch'] or fsdp_devices != training['fsdp_devices']
            or checkpoint.resolve() != Path(manifest['initialization']).resolve()
            or manifest['training_shuffle'] is not False or manifest['dataset_count'] != 300):
        raise ValueError('Requested training differs from matched protocol')
    base = config.get_config('pi05_pretrain_human300_all_lora_fast25')
    if base.model.action_horizon != 50 or base.model.train_image_augmentation:
        raise ValueError('Unexpected action horizon/image augmentation')
    data = dataclasses.replace(base.data, data_dirs=manifest['data_dirs'],
        source_window_ledger=manifest['source_window_ledger'], view_organization=manifest['view_organization'],
        view_exposure_manifest=manifest['view_exposure_manifest'], ordinary_image_augmentation=None,
        base_config=dataclasses.replace(base.data.base_config or config.DataConfig(),
            norm_stats=normalize.load(checkpoint / 'assets')))
    cfg = dataclasses.replace(base, data=data, batch_size=batch_size, fsdp_devices=fsdp_devices,
        seed=training['seed'], wandb_enabled=False, num_workers=4,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(checkpoint / 'params')))
    return cfg, manifest


def audit_batch(observation, actions, arm):
    """Check four exact source-window pairs after the real input transforms."""
    values = dict(state=np.asarray(observation.state), actions=np.asarray(actions),
                  prompt=np.asarray(observation.tokenized_prompt),
                  prompt_mask=np.asarray(observation.tokenized_prompt_mask))
    if len(values['actions']) != 8 or values['actions'].shape[1:] != (50, 32):
        raise ValueError('Unexpected real action batch')
    for key, array in values.items():
        if not np.isfinite(array).all() or not np.array_equal(array[::2], array[1::2]):
            raise ValueError('Source-window pair mismatch: ' + key)
    images = {key: np.asarray(value) for key, value in observation.images.items()}
    if set(images) != {'base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb'}:
        raise ValueError('Expected left exterior, wrist and right exterior inputs')
    pair_equal = {key: bool(np.array_equal(array[::2], array[1::2])) for key, array in images.items()}
    if not pair_equal['left_wrist_0_rgb']:
        raise ValueError('Wrist inputs differ within source-window pairs')
    if arm == 'canonical_matched' and not all(pair_equal.values()):
        raise ValueError('Canonical exposures must repeat exactly')
    if arm == 'broad_exterior_matched' and all(pair_equal.values()):
        raise ValueError('First broad batch contains no exterior view change')
    for array in images.values():
        if array.shape != (8, 224, 224, 3) or not np.isfinite(array).all():
            raise ValueError('Unexpected image shape or values')
    return dict(status='PASS', arm=arm, within_pair_source_exact=True, image_pair_equal=pair_equal,
        non_image_hashes={key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                          for key, value in values.items()},
        wrist_hash=hashlib.sha256(np.ascontiguousarray(images['left_wrist_0_rgb']).tobytes()).hexdigest())
