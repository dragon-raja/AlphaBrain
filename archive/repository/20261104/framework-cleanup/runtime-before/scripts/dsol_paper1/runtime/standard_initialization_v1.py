"""Official LIBERO initial conditions, shared by rendering and rollout."""

from pathlib import Path as _RepoPath
import sys as _sys
_REPOSITORY_ROOT = _RepoPath(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPOSITORY_ROOT))

from pathlib import Path
import numpy as np
from scripts.dsol_paper1.runtime.render_bridge_common_v1 import array_identity, read, sha

KIND = 'official_libero_initial_state_v1'


def load_initial(spec):
    init = spec['initialization']
    assert init['kind'] == KIND
    assert sha(init['array_path']) == init['array_sha256']
    assert sha(spec['bddl_file']) == spec['bddl_sha256']
    state = np.load(init['array_path'], allow_pickle=False)
    assert array_identity(state) == init['array_identity']
    return state


def initialize(env, spec, physical):
    """No demonstration XML/state; official reset, set initial, 10 no-op steps."""
    from scripts.dsol_paper1.runtime.evaluate_dsol_libero_hdf5_views import DUMMY_ACTION
    from evaluate_pi05_libero_plus_views import physics_state_sha256
    env.seed(int(spec['environment_seed']))
    env.reset()
    observation = env.set_init_state(physical)
    if env.check_success():
        raise ValueError('Official initial state already satisfies task: inspect, do not replace silently')
    raw, _ = physics_state_sha256(env)
    for _ in range(int(spec['initialization']['settle_steps'])):
        observation, _, done, _ = env.step(DUMMY_ACTION)
        if done:
            raise ValueError('Task completed during initialization; inspect protocol')
    settled, size = physics_state_sha256(env)
    return observation, dict(kind=KIND, official_file=spec['initialization']['official_file'],
                            official_file_sha256=spec['initialization']['official_file_sha256'],
                            init_state_index=spec['initialization']['init_state_index'],
                            raw_physics_sha256=raw, settled_physics_sha256=settled,
                            settled_physics_size=size, settle_steps=10,
                            demonstration_state_used=False, demonstration_xml_used=False)


def verify_policy_input(spec, example):
    asset = read(spec['initial_asset_record'])
    expected = next(v for v in asset['input_identities']
                    if v['candidate_id'] == spec['selected_candidate_id'])
    for key, identity in expected['inputs'].items():
        if array_identity(example[key]) != identity:
            raise ValueError('Rendered metric/rollout input mismatch: ' + key)
    return sha(spec['initial_asset_record'])
