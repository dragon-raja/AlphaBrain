"""Bounded baseline runner using the original frozen LIBERO evaluation runtime.

This file is copied into the new experiment. All policy / simulator imports
resolve to the parent frozen checkout; historical source is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    os.replace(tmp, path)


def frozen_core(plan):
    parent = Path(plan['parent'])
    if sha(parent/'release.json') != plan['parent_release_sha256']:
        raise ValueError('Parent release changed')
    sys.path.insert(0, str(parent/'repo'))
    sys.path.insert(0, str(parent/'repo/scripts/dsol_paper1'))
    from scripts.dsol_paper1 import standard_initial_aa0_v1 as standard
    return standard.core


def step_zero(root, plan):
    """Reconstruct training initialization, without an optimizer or data loader."""
    core = frozen_core(plan)
    original = read(root/'parent-release.json')
    # Verify frozen model/bridge implementation bytes before initialization.
    for name, digest in original['source_hashes'].items():
        if '/AlphaBrain/' in name and sha(name) != digest:
            raise ValueError('Frozen framework source changed: '+name)
    from omegaconf import OmegaConf
    from accelerate.utils import set_seed
    import torch
    from safetensors.torch import save_file
    from AlphaBrain.model.framework import build_framework
    from AlphaBrain.model.modules.action_model.pi0_flow_matching_head.weight_bridge import load_pi0_weights
    torch.set_num_threads(2)
    config_path = Path(plan['training_config'])
    if sha(config_path) != plan['training_config_sha256']:
        raise ValueError('Training config changed')
    cfg = OmegaConf.load(config_path)
    source = Path(cfg.trainer.pretrained_checkpoint)/'model.safetensors'
    source_digest = sha(source)
    if source_digest != '0f8c489e37b01c72251c45f2e73595894f3933fc6297f4f1cf95fc8737db4c74':
        raise ValueError('Original training weight identity mismatch')
    if int(cfg.seed) != 41:
        raise ValueError('Wrong model construction seed')
    set_seed(41)
    model = build_framework(cfg)
    summary = load_pi0_weights(model, str(source), strict=False, verbose=True)
    if (len(summary['matched']) != 814 or len(summary['missing']) != 121
            or len(summary['adapted']) != 3 or summary['shape_mismatch']):
        raise ValueError('Bridge differs from training receipt')
    # Preserve all inherited/random-initialized tensors, including zero LoRA B.
    output = root/'step-zero-model'
    output.mkdir(exist_ok=False)
    weights = output/'model.safetensors'
    save_file({k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}, str(weights))
    shutil.copy2(config_path, output/'framework_config.yaml')
    shutil.copytree(config_path.parent/'vlm_pretrained', output/'vlm_pretrained')
    receipt = dict(status='RECONSTRUCTED_NOT_NATIVE_OFFICIAL', optimizer_updates=0,
                   source=str(source), source_sha256=source_digest, seed=41,
                   config_sha256=sha(config_path), bridge=summary,
                   weights_sha256=sha(weights), path=str(output))
    write(root/'step-zero-receipt.json', receipt)
    release = dict(original)
    release['models'] = {'baseline': dict(path=str(output), weights_sha256=receipt['weights_sha256'])}
    release['baseline_panel'] = dict(count=640, plan_sha256=sha(root/'plan.json'), parent=plan['parent'])
    write(root/'release.json', release)
    print(json.dumps({'status': receipt['status'], 'missing': summary['missing']}), flush=True)


def episode(args, plan):
    core = frozen_core(plan)
    core.ROOT = args.root
    native = plan.get('format') == 'libero_native_reference_panel_v1'
    noise_trace = []
    if native:
        import numpy as np
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        infer = WebsocketClientPolicy.infer
        def traced_infer(client, observation):
            result = infer(client, observation)
            seed = int(observation['_eval_seed'])
            extra_seed = int.from_bytes(hashlib.sha256(
                ('libero-native32-extra25-v1::'+str(seed)).encode()).digest()[:16], 'little')
            full = np.empty((10, 32), dtype=np.float32)
            full[:, 7:] = np.random.Generator(np.random.PCG64(extra_seed)).standard_normal((10, 25), dtype=np.float32)
            full[:, :7] = observation['_eval_noise']
            full_hash = hashlib.sha256(full.tobytes()).hexdigest()
            if result.get('full_noise_sha256') != full_hash:
                raise ValueError('Native32 full noise cannot be independently reconstructed')
            noise_trace.append(dict(call_index=len(noise_trace), eval_seed=int(observation['_eval_seed']),
                shared_noise_sha256=result['noise_sha256'], full_noise_sha256=full_hash))
            return result
        WebsocketClientPolicy.infer = traced_infer
    core.episode(argparse.Namespace(index=args.index, output=args.output, port=args.port,
                                  gpu=args.gpu, model='baseline', host='fresh'), read(args.root/'release.json'))
    if native:
        row = read(args.output)
        if len(noise_trace) != row['inference_calls']:
            raise ValueError('Incomplete native32 noise receipt')
        write(args.root/'noise32'/f'{args.index:06d}.json', dict(status='PASS', index=args.index,
              episode_sha256=sha(args.output), internal_dim=32, shared_dim=7, calls=noise_trace))


def worker(args, plan):
    core = frozen_core(plan)
    root = args.root
    gpu = plan['gpus'][args.worker]
    native = plan.get('format') == 'libero_native_reference_panel_v1'
    port = plan.get('base_port', 31951)+args.worker
    out = root/'workers'/str(args.worker)
    out.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', port)) == 0:
            raise ValueError('Port already owned')
    baseline = read(root/('baseline-receipt.json' if native else 'step-zero-receipt.json'))
    release = read(root/'release.json')
    source = Path(plan['policy_source']) if native else Path(plan['parent'])/'repo/scripts/cabi_vla/serve_alphabrain_pi05_websocket.py'
    source_hash = plan['policy_source_sha256'] if native else release['source_hashes'][str(source)]
    if sha(source) != source_hash:
        raise ValueError('Frozen policy server changed')
    if sha(Path(baseline['path'])/'model.safetensors') != baseline['weights_sha256']:
        raise ValueError('Step-zero weights changed')
    started = time.time()
    assigned = plan['indices'][args.worker::plan['workers']]
    if args.limit:
        assigned = assigned[:args.limit]
    server_log = (out/'server.log').open('a')
    command = [str(core.python()), str(source), '--checkpoint', baseline['path'],
               '--device', 'cuda:0', '--cpu-threads', '2', '--port', str(port)]
    environment = core.env(gpu=gpu)
    if native:
        parent = Path(plan['parent'])
        environment['PYTHONPATH'] = ':'.join(map(str, [plan['native_overlay'],
            parent/'runtime/policy/lib/python3.12/site-packages', parent/'runtime/openpi-src',
            parent/'runtime/openpi-client-src', plan['native_deps']]))
        environment.update(JAX_PLATFORMS='cpu', OPENPI_DATA_HOME=plan['cache'])
        command = [str(core.python()), str(source), '--parent', plan['parent'],
                   '--alpha-checkpoint', plan['alpha_checkpoint'], '--native-checkpoint', baseline['path'],
                   '--variant', 'native', '--normalization', 'quantile', '--port', str(port)]
    server = subprocess.Popen(command, env=environment, stdout=server_log, stderr=subprocess.STDOUT)
    try:
        while True:
            if server.poll() is not None:
                raise ValueError('Policy server failed')
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1', port)) == 0:
                    break
            if time.time()-started > 480:
                raise TimeoutError('Policy startup timeout')
            time.sleep(2)
        for n, index in enumerate(assigned):
            output = root/'episodes'/f'{index:06d}.json'
            if not output.exists():
                command = [str(core.python(True)), str(Path(__file__).resolve()), 'episode', '--root', str(root),
                           '--index', str(index), '--gpu', str(gpu), '--port', str(port), '--output', str(output)]
                with (out/'episodes.log').open('a') as log:
                    subprocess.run(command, env=core.sim_env(), stdout=log, stderr=subprocess.STDOUT,
                                   timeout=600, check=True)
            row = read(output)
            if row['aa0']['checkpoint_sha256'] != baseline['weights_sha256'] or row['aa0']['index'] != index:
                raise ValueError('Episode identity mismatch')
            if row['aa0']['release_sha256'] != sha(root/'release.json'):
                raise ValueError('Episode release mismatch')
            if native:
                full_noise = read(root/'noise32'/f'{index:06d}.json')
                if full_noise['status'] != 'PASS' or full_noise['episode_sha256'] != sha(output):
                    raise ValueError('Missing or inconsistent full native noise trace')
            write(out/'status.json', dict(status='RUNNING', completed=n+1, count=len(assigned),
                        gpu=gpu, started=started, heartbeat=time.time(), last_index=index))
        write(out/'status.json', dict(status='COMPLETE', completed=len(assigned), count=len(assigned),
                                     gpu=gpu, started=started, finished=time.time()))
    except BaseException as exc:
        write(out/'status.json', dict(status='ERROR', error=repr(exc), gpu=gpu, heartbeat=time.time()))
        raise
    finally:
        # Only the policy child created by this worker is terminated.
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        server_log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['step-zero', 'worker', 'episode'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--worker', type=int, default=0)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--index', type=int)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--port', type=int, default=31951)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    plan = read(a.root/'plan.json')
    if plan['count'] != 640 or sha(__file__) != plan['runner_sha256']:
        raise ValueError('Unrecognized budget or modified runner')
    if a.mode == 'step-zero':
        step_zero(a.root, plan)
    elif a.mode == 'worker':
        worker(a, plan)
    else:
        episode(a, plan)


if __name__ == '__main__':
    main()
