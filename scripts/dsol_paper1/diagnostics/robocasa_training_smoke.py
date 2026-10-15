"""Opt-in, bounded OpenPI/JAX training-memory probe; no dataset or checkpoint writes.

This runtime adapter imports an explicitly supplied external training module.
Synthetic or explicitly released real batches test optimizer/backward memory;
neither path writes a model checkpoint or claims task performance.
"""

import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--openpi-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--fsdp-devices', type=int, default=8)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--data-manifest', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute or not 1 <= args.steps <= 3:
        parser.error('Explicit --execute and 1..3 bounded optimizer steps required')
    if args.output.exists():
        raise FileExistsError(args.output)
    import jax
    import numpy as np
    from openpi.training import config, sharding, weight_loaders
    source = args.openpi_root / 'scripts/train.py'
    spec = importlib.util.spec_from_file_location('dsol_external_openpi_train_probe', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = dataclasses.replace(config.get_config('pi05_pretrain_human300_all_lora_fast25'),
        batch_size=args.batch_size, fsdp_devices=args.fsdp_devices, wandb_enabled=False,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(args.checkpoint / 'params')))
    manifest = None
    if args.data_manifest:
        from benchmarks.Robocasa365.openpi_training import matched_config, audit_batch
        cfg, manifest = matched_config(args.data_manifest, args.checkpoint,
            batch_size=args.batch_size, fsdp_devices=args.fsdp_devices)
    if args.batch_size % jax.device_count():
        raise ValueError('Global batch must divide visible GPU count')
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    rng, init_rng = jax.random.split(jax.random.key(41))
    loader_iter, batch_audits = None, []
    if manifest:
        from openpi.training import data_loader
        print('Loading released real data with shuffle=False', flush=True)
        loader = data_loader.create_data_loader(cfg, sharding=data_sharding, shuffle=False, num_batches=args.steps)
        loader_iter = iter(loader)
        batch = next(loader_iter)
        obs, act = batch
        batch_audits.append(audit_batch(obs, act, manifest['arm']))
        print(json.dumps(batch_audits[-1]), flush=True)
    else:
        obs, act = cfg.model.fake_obs(args.batch_size), cfg.model.fake_act(args.batch_size)
        batch = jax.device_put((obs, act), data_sharding)
    started = time.monotonic()
    state, state_sharding = module.init_train_state(cfg, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    init_seconds = time.monotonic() - started
    step = jax.jit(functools.partial(module.train_step, cfg),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated), donate_argnums=(1,))
    records = []
    for i in range(args.steps):
        if i and loader_iter is not None:
            batch = next(loader_iter)
            batch_audits.append(audit_batch(*batch, manifest['arm']))
        start = time.monotonic()
        with sharding.set_mesh(mesh):
            state, info = step(rng, state, batch)
        jax.block_until_ready(info)
        record = dict(step=i + 1, seconds=time.monotonic() - start,
                      **{k: float(np.asarray(v)) for k, v in info.items()})
        records.append(record)
        print(json.dumps(record), flush=True)
    payload = dict(status='PASS' if all(np.isfinite(r['loss']) and np.isfinite(r['grad_norm']) for r in records) else 'FAIL',
        scope=('Released real batches: exact paired input checks and optimizer updates; no task quality claim'
               if manifest else 'Synthetic batch: optimizer/memory only; no real data or task quality claim'),
        data_manifest=str(args.data_manifest) if manifest else None, batch_audits=batch_audits,
        data_manifest_sha256=hashlib.sha256(args.data_manifest.read_bytes()).hexdigest() if manifest else None,
        checkpoint=str(args.checkpoint), config=cfg.name, global_batch=args.batch_size,
        per_device_batch=args.batch_size // jax.device_count(), fsdp_devices=args.fsdp_devices,
        images=3, image_resolution=[224, 224], action_shape=list(act.shape), token_length=cfg.model.max_token_len,
        init_seconds=init_seconds, optimizer_steps=records,
        device_memory=[dict(device=str(d), stats=d.memory_stats()) for d in jax.devices()],
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        checkpoint_norm_sha256=hashlib.sha256((args.checkpoint / 'assets/norm_stats.json').read_bytes()).hexdigest(),
        runtime_environment={k: os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'LD_PRELOAD',
            'NCCL_P2P_DISABLE', 'NCCL_IB_DISABLE', 'NCCL_CUMEM_HOST_ENABLE',
            'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_PYTHON_CLIENT_MEM_FRACTION']},
        checkpoints_written=0, existing_weights_modified=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write('\n')
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
