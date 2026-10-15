"""Explicit frozen RGB encoding of saved P0 observations; no VLA or simulator.

Uses the *actual scored input* for each model/view, including native render
differences. Image hashes deduplicate identical inputs without changing them.
"""
from pathlib import Path
import argparse
import json

import numpy as np

from ..data.artifacts import read, write, sha
from ..data.evaluation_matrix import validate_plan


def encode(config, *, device='cpu'):
    import torch
    from torchvision.models import resnet18
    cfg = read(config)
    root = Path(cfg['output'])
    plan = read(cfg['matrix_plan'])
    reference = validate_plan(plan)
    jobs = sorted(reference['jobs'], key=lambda j: (j['split'], j['task'], j['scene_seed'], j['initial_id']))
    mapping = {(w['arm'], i): w['worker_id'] for w in plan['workers'] for i in w['initials']}
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(cfg['encoder_weights'], map_location='cpu', weights_only=True))
    model.fc = torch.nn.Identity()
    torch.set_num_threads(12)
    model.eval().to(device)
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    output = root / 'features'
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(plan_sha256=sha(cfg['matrix_plan']), encoder_sha256=sha(cfg['encoder_weights']),
                    implementation_sha256=sha(__file__), device=device,
                    preprocessing='Actual 224x224 RGB; /255 and ImageNet channel normalization; frozen ResNet18 pooled512',
                    exterior='average left/right pooled embeddings; wrist separately; no unqueried canonical image')
    write(output / 'provenance.json', identity)
    for i, job in enumerate(jobs):
        target = output / (job['initial_id'] + '.npz')
        if target.exists():
            continue
        cache, keys, pending = {}, [], {}
        for arm in ('canonical', 'broad'):
            cell_root = Path(plan['source']).parent / 'workers' / mapping[(arm, job['initial_id'])] / 'cells' / job['initial_id']
            for view in plan['view_ids']:
                cell = cell_root / view
                scores = read(cell / 'scores.json')
                local_keys = []
                with np.load(cell / 'scores.npz', allow_pickle=False) as arrays:
                    for channel in ('observation/image', 'observation/right_image', 'observation/wrist_image'):
                        key = scores['input_arrays'][channel]['sha256']
                        local_keys.append(key)
                        if key not in pending:
                            pending[key] = arrays[channel].copy()
                keys.append(local_keys)
        batch_keys = list(pending)
        with torch.inference_mode():
            for offset in range(0, len(batch_keys), 64):
                part = batch_keys[offset:offset+64]
                batch = torch.from_numpy(np.stack([pending[k] for k in part])).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255
                z = model((batch-mean)/std).cpu().numpy()
                cache.update(zip(part, z))
        external = np.stack([(cache[a]+cache[b])/2 for a,b,c in keys]).reshape(2,97,512)
        wrist = np.stack([cache[c] for a,b,c in keys]).reshape(2,97,512)
        np.savez_compressed(target, exterior=external, wrist=wrist)
        write(output / (job['initial_id'] + '.json'), dict(sha256=sha(target), identity=identity))
        if (i+1) % 10 == 0:
            print(json.dumps(dict(encoded_initials=i+1, total=400)), flush=True)
            write(output / 'progress.json', dict(encoded_initials=i+1, total=400))
    write(output / 'complete.json', dict(complete=True, initials=400, identity=identity))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    encode(args.config, device=args.device)
