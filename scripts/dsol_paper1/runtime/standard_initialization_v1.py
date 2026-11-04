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


from benchmarks.LIBERO.simulation_inputs import load_initial


from benchmarks.LIBERO.simulation_inputs import initialize


from benchmarks.LIBERO.simulation_inputs import verify_policy_input
