"""Outcome-blind, bounded LIBERO step-zero comparison against a frozen matrix."""
from copy import deepcopy
from pathlib import Path
import argparse
import shutil

from .artifacts import read, sha, write


def select_candidates(candidates):
    """Five diagnostic poses, not an estimate of the full candidate average."""
    reference = [i for i, c in enumerate(candidates) if c['selected_candidate_id'] == 'canonical']
    if len(reference) != 1:
        raise ValueError('Exactly one canonical required')
    chosen = reference[:]
    targets = [(-1/3, -1/3, 0), (1/3, 1/3, 0), (-1, -1, 1), (1, 1, 1)]
    for target in targets:
        available = [i for i, c in enumerate(candidates)
                     if c['selected_candidate_id'].startswith('broad_heldout_') and i not in chosen]
        if not available:
            raise ValueError('Four distinct heldout poses required')
        def distance(i):
            p = candidates[i]['pose']
            v = (p['azimuth_deg']/60, p['elevation_deg']/25, (p['radius_scale']-1)/.25)
            return sum((a-b)**2 for a, b in zip(v, target)), candidates[i]['selected_candidate_id']
        chosen.append(min(available, key=distance))
    return chosen


def panel_indices(release):
    states = release['states']
    if len(states) != 32 or len({s['task_id'] for s in states}) != 8:
        raise ValueError('Preserve all eight tasks and 32 official initials')
    if release['candidate_count'] != 97 or release['noise_repeats'] != 32:
        raise ValueError('Unexpected parent matrix')
    candidates = release['protocol']['state_blocks'][0]['candidates']
    if any(b['candidates'] != candidates for b in release['protocol']['state_blocks']):
        raise ValueError('Different candidate catalogs across initials')
    selected = select_candidates(candidates)
    return [si*97*32+ci*32+n for si in range(32) for ci in selected for n in range(4)]


def prepare(parent, output, source):
    parent, output, source = Path(parent), Path(output), Path(source)
    original = read(parent/'release.json')
    indices = panel_indices(original)
    reused = {}
    for model in ('canonical', 'broad'):
        for index in indices:
            si = index//3104
            state = original['states'][si]
            owners = [h for h, spec in original['hosts'].items() if si in spec['state_indices']]
            if len(owners) != 1:
                raise ValueError('Ambiguous original host')
            p = parent/'hosts'/owners[0]/f"dense-init-{state['initialization']['init_state_index']}"/model/f'{si:02d}'/f'{index:06d}.json'
            row = read(p)
            if row['aa0']['checkpoint_sha256'] != original['models'][model]['weights_sha256']:
                raise ValueError('Old checkpoint mismatch')
            if row['aa0']['index'] != index or row['pair_key'] != state['pair_key']:
                raise ValueError('Old initial/index mismatch')
            reused[f'{model}:{index}'] = dict(path=str(p), sha256=sha(p))
    output.mkdir(parents=True, exist_ok=False)
    runner = output/'source'/'baseline_panel.py'
    runner.parent.mkdir()
    shutil.copy2(source/'benchmarks/LIBERO/baseline_panel.py', runner)
    config = Path(original['models']['canonical']['path'])/'framework_config.yaml'
    plan = dict(format='libero_step_zero_panel_v1', parent=str(parent),
                parent_release_sha256=sha(parent/'release.json'), indices=indices,
                selected_candidate_indices=select_candidates(original['protocol']['state_blocks'][0]['candidates']),
                repeats=[0, 1, 2, 3], count=640, workers=4, gpus=[1, 3, 5, 7],
                runner=str(runner), runner_sha256=sha(runner),
                training_config=str(config), training_config_sha256=sha(config),
                original_models=original['models'], reused=reused,
                baseline='AlphaBrain bridge reconstructed at seed41 before optimizer updates; NOT native official implementation',
                limits='Five preselected views only; no oracle, no selector fitting, no training; no new seeds/initials',
                authorization='User explicitly approved 640 whole-task rollouts; existing jobs and keepalive untouched')
    write(output/'plan.json', plan)
    write(output/'parent-release.json', original)
    return plan


def prepare_native(legacy_root, output, source, checkpoint, diagnostic_runtime, cache):
    """Correct the same authorized 640-cell panel; never overwrite its first run."""
    legacy_root, output, source = map(Path, (legacy_root, output, source))
    checkpoint, diagnostic_runtime, cache = map(Path, (checkpoint, diagnostic_runtime, cache))
    old = read(legacy_root/'plan.json')
    parent = Path(old['parent'])
    original = read(parent/'release.json')
    indices = panel_indices(original)
    if indices != old['indices'] or sha(parent/'release.json') != old['parent_release_sha256']:
        raise ValueError('Original authorized panel changed')
    if read(diagnostic_runtime/'analysis/summary.json')['success_counts']['native-quantile-rollout'] != 8:
        raise ValueError('Native whole-task diagnostic not accepted')
    for ref in old['reused'].values():
        if sha(ref['path']) != ref['sha256']:
            raise ValueError('Existing post-training result changed')
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output/'source'
    snapshot.mkdir()
    paths = ['benchmarks/LIBERO/baseline_panel.py', 'benchmarks/LIBERO/baseline_dispatch.py',
             'benchmarks/LIBERO/initialization_diagnostic.py',
             'AlphaBrain/research/dsol/data/libero_baseline_panel.py',
             'AlphaBrain/research/dsol/analysis/libero_baseline_panel.py',
             'AlphaBrain/research/dsol/plotting/libero_baseline_panel.py']
    identities = {}
    for rel in paths:
        target = snapshot/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/rel, target)
        identities[str(target)] = sha(target)
    runner = snapshot/'benchmarks/LIBERO/baseline_panel.py'
    policy = snapshot/'benchmarks/LIBERO/initialization_diagnostic.py'
    plan = deepcopy(old)
    plan.update(format='libero_native_reference_panel_v1', source_repo=str(source),
        runner=str(runner), runner_sha256=sha(runner), policy_source=str(policy), policy_source_sha256=sha(policy),
        workers=16, gpus=list(range(8))*2, base_port=32300, legacy_root=str(legacy_root),
        alpha_checkpoint=str(legacy_root/'step-zero-model'), native_checkpoint=str(checkpoint),
        native_overlay=str(diagnostic_runtime/'native312-overlay'), native_deps=str(diagnostic_runtime/'native-deps'),
        cache=str(cache), baseline='Native OpenPI PyTorch reference, original converted pi05_libero weights, quantile decoding, internal action_dim32',
        baseline_plot_label='Native OpenPI reference',
        plot_note='Five preselected poses. Native32 shares first7 flow-noise coordinates with the adapted7 policies; no training.',
        limits='Same 32 initials, five preselected views, four repeats. No full-space average, Oracle or selector claim. '
               'Native32 and adapted7 are not identical architectures; only their first7 noise coordinates are paired.',
        noise_extension=dict(algorithm='Preserve frozen first7; extra_seed=uint128le(sha256(libero-native32-extra25-v1::eval_seed)[:16]); PCG64(extra_seed).standard_normal((10,25),dtype=float32)',
                             shared_dimensions=7, native_dimensions=32, full_hashes='noise32/{index:06d}.json'),
        authorization='User requested correction and re-evaluation of the same 640-cell panel; no training or expansion',
        source_snapshot_hashes=identities)
    receipt = dict(status='NATIVE_REFERENCE_NO_TRAINING', path=str(checkpoint),
                   weights_sha256=sha(checkpoint/'model.safetensors'),
                   norm_stats_sha256=sha(checkpoint/'assets/physical-intelligence/libero/norm_stats.json'),
                   diagnostic_summary_sha256=sha(diagnostic_runtime/'analysis/summary.json'))
    release = deepcopy(original)
    release['models'] = {'baseline': dict(path=str(checkpoint), weights_sha256=receipt['weights_sha256'])}
    release['baseline_correction'] = dict(kind=plan['format'], count=640, noise_extension=plan['noise_extension'],
                                         policy_source_sha256=plan['policy_source_sha256'])
    write(output/'plan.json', plan)
    write(output/'parent-release.json', original)
    write(output/'baseline-receipt.json', receipt)
    write(output/'release.json', release)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    a = p.parse_args()
    plan = prepare(a.parent, a.output, a.source)
    print({'count': plan['count'], 'selected_candidate_indices': plan['selected_candidate_indices']})


if __name__ == '__main__':
    main()
