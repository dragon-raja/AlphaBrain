"""Opt-in, subprocess-only observers for the runtime migration gate.

The deployed evaluator and policy implementations are invoked unchanged. Probe
outputs belong to a separate diagnostic directory, never the scientific ledger.
Compatible with the frozen simulator's Python 3.8 interpreter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace


def write(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def identity(value):
    import numpy as np
    a = np.ascontiguousarray(value)
    return dict(dtype=str(a.dtype), shape=list(a.shape), sha256=hashlib.sha256(a.tobytes()).hexdigest())


def origins():
    result = {}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename and (name.startswith(('AlphaBrain', 'scripts.dsol_paper1', 'scripts.vla_shared'))
                         or name in {'standard_initial_aa0_v1', 'dualhost_aa0_v1', 'explicit_flow_noise',
                                     'evaluate_dsol_libero_hdf5_views', 'evaluate_pi05_libero_plus_views',
                                     'libero_camera_pose', 'standard_initialization_v1'}):
            p = Path(filename).resolve()
            if p.is_file():
                result[name] = dict(path=str(p), sha256=sha(p))
    return result


def serve(args):
    import numpy as np
    namespace = runpy.run_path(str(args.entry), run_name='runtime_policy_probe')
    implementation = namespace['AlphaBrainPi05Policy']

    class ObservedPolicy(implementation):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            head = self._model.flow_matching_head
            original = head.embed_suffix
            self._observed_times = []
            self._observed_noise = None

            def observed_suffix(state, x_t, timestep, *pos, **kwargs):
                if not self._observed_times:
                    self._observed_noise = identity(x_t[0].detach().cpu().numpy())
                self._observed_times.append(timestep.detach().cpu().numpy().tolist())
                return original(state, x_t, timestep, *pos, **kwargs)

            head.embed_suffix = observed_suffix
            write(args.output / 'server-identity.json', dict(
                entry=str(args.entry), entry_sha256=sha(args.entry), modules=origins(),
                metadata=self.metadata, num_inference_steps=int(head.num_inference_steps),
                observer='read-only suffix input observation; no additional policy inference'))

        def infer(self, request):
            self._observed_times = []
            self._observed_noise = None
            response = super().infer(request)
            assert len(self._observed_times) == 10, 'Unexpected number of denoising evaluations'
            assert self._observed_noise == identity(np.asarray(request['_eval_noise'], dtype=np.float32))
            response['runtime_validation'] = dict(times=self._observed_times, initial_noise=self._observed_noise)
            return response

    namespace['main'].__globals__['AlphaBrainPi05Policy'] = ObservedPolicy
    sys.argv = [str(args.entry), '--checkpoint', str(args.checkpoint), '--device', 'cuda:0',
                '--cpu-threads', '2', '--port', str(args.port)]
    namespace['main']()


def episode(args):
    import numpy as np
    namespace = runpy.run_path(str(args.entry), run_name='runtime_episode_probe')
    core = namespace['core']
    release = json.loads(args.release.read_text())
    from openpi_client import websocket_client_policy
    original = websocket_client_policy.WebsocketClientPolicy
    calls = []

    class ObservedClient:
        def __init__(self, *a, **kw):
            self.client = original(*a, **kw)

        def infer(self, request):
            response = self.client.infer(request)
            numeric = {k: identity(v) for k, v in request.items() if isinstance(v, np.ndarray)}
            metadata = {k: v for k, v in request.items() if not isinstance(v, np.ndarray)}
            if not calls:
                np.savez_compressed(args.output / 'first-input.npz', **{
                    k: v for k, v in request.items() if isinstance(v, np.ndarray)})
            probe = response['runtime_validation']
            assert probe['initial_noise'] == numeric['_eval_noise']
            assert len(probe['times']) == 10
            calls.append(dict(inputs=numeric, metadata=metadata, actions=identity(response['actions']),
                              flow=probe))
            return response

    websocket_client_policy.WebsocketClientPolicy = ObservedClient
    try:
        core.episode(SimpleNamespace(output=str(args.output / 'episode.json'), index=args.index,
                                    host='fresh', model=args.model, port=args.port, gpu=args.gpu), release)
        row = json.loads((args.output / 'episode.json').read_text())
        assert row['aa0']['counts']['calls'] == len(calls)
        write(args.output / 'calls.json', calls)
        write(args.output / 'probe.json', dict(scientific_ledger=False, status='COMPLETE',
              entry=str(args.entry), entry_sha256=sha(args.entry), modules=origins(),
              release_sha256=sha(args.release), observer_sha256=sha(__file__)))
    finally:
        websocket_client_policy.WebsocketClientPolicy = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['serve', 'episode'])
    parser.add_argument('--entry', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--index', type=int)
    parser.add_argument('--model', choices=['canonical', 'broad'])
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    assert not args.output.exists(), 'Never overwrite a probe result'
    args.output.mkdir(parents=True)
    # Fixed, explicit source roots. No walking all subdirectories for imports.
    for path in reversed([args.repo, args.entry.parent,
                          args.repo / ('scripts/vla_shared' if (args.repo / 'scripts/vla_shared').is_dir() else 'scripts/cabi_vla')]):
        sys.path.insert(0, str(path))
    (serve if args.mode == 'serve' else episode)(args)


if __name__ == '__main__':
    main()
