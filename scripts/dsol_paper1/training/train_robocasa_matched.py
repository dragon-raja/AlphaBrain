"""Explicit matched OpenPI training after the same-arm real-data smoke gate.

Delegates model initialization/update and checkpoint encoding to OpenPI. Uses
the frozen ledger order, preserves other jobs, and only writes a new run root.
"""

import argparse
import dataclasses
import functools
import importlib.util
import json
import signal
import shutil
from pathlib import Path
import time

from AlphaBrain.research.dsol.data.artifacts import read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--openpi-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-manifest', type=Path, required=True)
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-run', type=Path, help='Read-only source run; output must still be new')
    parser.add_argument('--resume-step', type=int, help='Committed optimizer step, with original metrics for 3 replay steps')
    parser.add_argument('--indexed-resume', action='store_true', help='Seek the explicitly deterministic ledger sampler; no random augmentation allowed')
    parser.add_argument('--replay-only', action='store_true', help='Only three diagnostic recovery updates; save no weights and never continue past the replay window')
    parser.add_argument('--numerical-replay', action='store_true', help='Explicit numerical gate: loss1%, grad_norm2%, param_norm1e-6; strict replay remains default')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    if args.output.exists():
        raise FileExistsError('Fresh training root required; never overwrite a run')
    if bool(args.resume_run) != (args.resume_step is not None):
        parser.error('--resume-run and --resume-step must be provided together')
    if args.indexed_resume and not args.resume_run:
        parser.error('--indexed-resume requires recovery')
    if args.replay_only and not args.resume_run:
        parser.error('--replay-only requires recovery')
    if args.numerical_replay and not args.resume_run:
        parser.error('--numerical-replay requires recovery')
    smoke = read(args.smoke)
    if (smoke.get('status') != 'PASS' or smoke.get('data_manifest_sha256') != sha(args.data_manifest)
            or smoke.get('global_batch') != 8 or smoke.get('fsdp_devices') != 8
            or len(smoke.get('optimizer_steps', [])) < 2 or not smoke.get('batch_audits')):
        raise ValueError('Same-arm real-data FSDP8 smoke is required')
    import jax
    import numpy as np
    from openpi.training import checkpoints, data_loader, sharding
    from benchmarks.Robocasa365.openpi_training import matched_config, audit_batch

    cfg, manifest = matched_config(args.data_manifest, args.checkpoint)
    total = manifest['training']['optimizer_updates_per_arm']
    if total * cfg.batch_size != manifest['exposures'] or jax.device_count() != 8:
        raise ValueError('Training exposure count/device mismatch')
    cfg = dataclasses.replace(cfg, exp_name=manifest['arm'], num_train_steps=total,
                              checkpoint_base_dir=str(args.output / 'checkpoints'))
    source = args.openpi_root / 'scripts/train.py'
    if sha(source) != smoke['source_sha256'] or sha(args.checkpoint / 'assets/norm_stats.json') != smoke['checkpoint_norm_sha256']:
        raise ValueError('Training implementation/normalization changed after smoke')
    spec = importlib.util.spec_from_file_location('dsol_openpi_matched_train', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recovery = None
    expected_replay = {}
    if args.resume_run:
        from AlphaBrain.research.dsol.data.training_recovery import recovery_source
        recovery = recovery_source(args.resume_run, args.resume_step, args.data_manifest,
                                   args.smoke, source, args.checkpoint)
        for line in (args.resume_run / 'metrics.jsonl').read_text().splitlines():
            row = json.loads(line)
            if args.resume_step < row['step'] <= args.resume_step + 3:
                expected_replay[row['step']] = row
        if len(expected_replay) != 3:
            raise ValueError('Three original post-checkpoint steps required for recovery replay gate')
    args.output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[3]
    source_paths = [Path(__file__).resolve(), source.resolve(),
        repo / 'benchmarks/Robocasa365/openpi_training.py',
        repo / 'AlphaBrain/research/dsol/data/training_recovery.py',
        args.openpi_root / 'src/openpi/training/checkpoints.py',
        args.openpi_root / 'src/openpi/training/data_loader.py',
        args.openpi_root / 'src/openpi/groot_utils/groot_openpi_ledger_dataset.py']
    archive = args.output / 'source_snapshot'
    archive.mkdir()
    sources = {}
    for ordinal, path in enumerate(source_paths):
        target = archive / f'{ordinal:02d}_{path.name}'
        shutil.copy2(path, target)
        sources[str(path)] = dict(path=str(target), sha256=sha(target))
    write(args.output / 'source_snapshot.json', sources)
    write(args.output / 'run_config.json', dict(manifest_sha256=sha(args.data_manifest), manifest=manifest,
        smoke_sha256=sha(args.smoke), external_train_sha256=sha(source), entrypoint_sha256=sha(__file__),
        checkpoint=str(args.checkpoint), shuffle=False, seed=cfg.seed, global_batch=cfg.batch_size,
        fsdp_devices=cfg.fsdp_devices, total_steps=total, keep_period=2000,
        recovery=recovery, indexed_resume=args.indexed_resume, numerical_replay=args.numerical_replay,
        replay_only=args.replay_only, no_other_processes_modified=True))
    manager = None
    completed = 0
    started = time.monotonic()
    def interrupted(signum, _frame):
        raise KeyboardInterrupt('Received signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        mesh = sharding.make_mesh(cfg.fsdp_devices)
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        loader = data_loader.create_data_loader(cfg, sharding=data_sharding, shuffle=False, num_batches=total)
        batches = iter(loader)
        batch = next(batches)
        first_audit = audit_batch(*batch, manifest['arm'])
        if first_audit != smoke['batch_audits'][0]:
            raise ValueError('First actual training batch differs from same-arm smoke')
        write(args.output / 'first_batch_audit.json', first_audit)
        rng, init_rng = jax.random.split(jax.random.key(cfg.seed))
        state, state_sharding = module.init_train_state(cfg, init_rng, mesh, resume=bool(recovery))
        if recovery:
            import orbax.checkpoint as ocp
            state_shape = state
            with ocp.CheckpointManager(str(Path(recovery['checkpoint']).parent),
                    item_handlers={'assets': checkpoints.CallbackHandler(),
                                   'train_state': ocp.PyTreeCheckpointHandler(),
                                   'params': ocp.PyTreeCheckpointHandler()},
                    options=ocp.CheckpointManagerOptions(read_only=True, create=False)) as previous:
                state = checkpoints.restore_state(previous, state, loader, step=args.resume_step)
            # Record the abstract initialization/restoration signature. Do not
            # coerce weak typing: use the upstream restored state unchanged.
            scalar_types = dict(expected_weak=bool(getattr(state_shape.step, 'weak_type', False)),
                                restored_weak=bool(getattr(state.step, 'weak_type', False)),
                                expected_dtype=str(state_shape.step.dtype), restored_dtype=str(state.step.dtype))
            write(args.output / 'restored_scalar_signature.json', scalar_types)
            completed = int(np.asarray(state.step))
            if completed != args.resume_step:
                raise ValueError('Restored optimizer step mismatch')
            # Replay the original loader, including worker RNG consumption. No
            # optimizer updates occur while advancing to the next exact batch.
            if args.indexed_resume:
                if (not cfg.data.source_window_ledger or cfg.data.ordinary_image_augmentation is not None
                        or cfg.model.train_image_augmentation):
                    raise ValueError('Indexed recovery requires the deterministic no-augmentation ledger')
                torch_loader = loader._data_loader.torch_loader
                torch_loader.batch_sampler.sampler = range(completed * cfg.batch_size, len(torch_loader.dataset))
                batches = iter(loader)
                batch = next(batches)
            else:
                for skipped in range(completed):
                    batch = next(batches)
                    if (skipped + 1) % 100 == 0:
                        write(args.output / 'status.json', dict(status='RESTORING_DATA_CURSOR',
                            completed=completed, total=total, batches_advanced=skipped + 1))
            write(args.output / 'recovery.json', dict(source=recovery, status='REPLAY_PENDING',
                optimizer_restored=True, rng='seed41_split_then_fold_in_restored_step',
                consumed_batches=completed, next_optimizer_step=completed + 1,
                cursor_method='indexed_deterministic_ledger' if args.indexed_resume else 'sequential_replay',
                next_batch_audit=audit_batch(*batch, manifest['arm'])))
        jax.block_until_ready(state)
        update = jax.jit(functools.partial(module.train_step, cfg),
            in_shardings=(replicated, state_sharding, data_sharding),
            out_shardings=(state_sharding, replicated), donate_argnums=(1,))
        manager, _ = checkpoints.initialize_checkpoint_dir(cfg.checkpoint_dir, keep_period=2000,
                                                           overwrite=False, resume=False)
        with (args.output / 'metrics.jsonl').open('x') as log:
            start_step = completed
            replay_results = []
            stop_step = start_step + 3 if args.replay_only else total
            for index in range(start_step, stop_step):
                step_start = time.monotonic()
                if index > start_step:
                    batch = next(batches)
                with sharding.set_mesh(mesh):
                    state, info = update(rng, state, batch)
                jax.block_until_ready(info)
                record = {key: float(np.asarray(value)) for key, value in info.items()}
                if not all(np.isfinite(value) for value in record.values()):
                    raise FloatingPointError('Nonfinite training metrics')
                completed = int(np.asarray(state.step))
                if completed != index + 1:
                    raise ValueError('Optimizer step identity mismatch')
                record.update(step=completed, seconds=time.monotonic()-step_start)
                if completed in expected_replay:
                    from AlphaBrain.research.dsol.data.training_recovery import compare_replayed_metrics
                    write(args.output / f'replay-step-{completed}.json',
                          dict(actual=record, expected=expected_replay[completed]))
                    try:
                        replay_results.append(compare_replayed_metrics(record, expected_replay[completed],
                                              numerical_diagnostic=args.numerical_replay))
                    except ValueError as exc:
                        if not args.replay_only:
                            raise
                        replay_results.append(dict(step=completed, status='STRICT_MISMATCH', error=str(exc)))
                    write(args.output / 'recovery.json', dict(source=recovery,
                        status=('DIAGNOSTIC_ONLY' if args.replay_only else
                                'PASS' if len(replay_results) == 3 else 'REPLAY_PENDING'),
                        optimizer_restored=True, data_cursor_restored=True, replay=replay_results))
                log.write(json.dumps(record)+'\n')
                log.flush()
                if completed == 1 or completed % 100 == 0 or completed == total:
                    log.flush()
                    write(args.output / 'status.json', dict(status='RUNNING', completed=completed, total=total,
                        elapsed_seconds=time.monotonic()-started, latest=record))
                    print(json.dumps(record), flush=True)
                if not args.replay_only and (completed % 2000 == 0 or completed == total):
                    checkpoints.save_state(manager, state, loader, completed)
        manager.wait_until_finished()
        if args.replay_only:
            write(args.output / 'status.json', dict(status='DIAGNOSTIC_COMPLETE_NOT_RESUMED',
                  completed=completed, diagnostic_updates=3, checkpoints_written=0))
            return
        write(args.output / 'status.json', dict(status='COMPLETE', completed=completed, total=total,
            elapsed_seconds=time.monotonic()-started, checkpoint=str(cfg.checkpoint_dir / str(completed))))
    except BaseException as exc:
        write(args.output / 'status.json', dict(status='ERROR', completed=completed, total=total,
            elapsed_seconds=time.monotonic()-started, error=f'{type(exc).__name__}: {exc}'))
        raise
    finally:
        if manager is not None:
            manager.close()


if __name__ == '__main__':
    main()
