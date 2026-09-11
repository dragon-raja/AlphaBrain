"""Freeze an execution manifest without editing the scientific release."""

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[4]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))

from pathlib import Path
import shutil
import time
from scripts.dsol_paper1.dynamic_initial_scheduler_v1 import ROOT, EXT, FROZEN_ENTRY, core
def main():
    r=core.release()
    core.require(not EXT.exists(),'Extension already prepared')
    EXT.mkdir(parents=True)
    for name, source in {
        'dynamic_initial_scheduler_v1.py': _layout_root / 'scripts/dsol_paper1/dynamic_initial_scheduler_v1.py',
        'transition_initial_dynamic_v1.py': Path(__file__).parent / 'transition_initial_dynamic_v1.py',
    }.items():
        shutil.copy2(source,EXT/name)
    core.write(EXT/'manifest.json',dict(schema='execution_only_dynamic_scheduler_v1',utc=time.time(),
        scientific_release_sha256=core.sha(ROOT/'release.json'),scheduler_sha256=core.sha(EXT/'dynamic_initial_scheduler_v1.py'),
        transition_sha256=core.sha(EXT/'transition_initial_dynamic_v1.py'),frozen_episode_entry=str(FROZEN_ENTRY),
        frozen_episode_entry_sha256=core.sha(FROZEN_ENTRY),scientific_budget=r['total_episodes'],
        change='static per-worker slices -> host-local shared FIFO queue',workers=32,policy_servers=16,
        host_initial_partition_unchanged=True,checkpoint_noise_views_termination_unchanged=True,
        completed_records_immutable=True,keepalive_untouched=True,inflight_drained_before_switch=True,
        migration_gate='192 new episodes per host, exact comparison to static full traces, repeated and cross-host',
        new_episode_code=False,require_gate_before_remaining_dense=True))
    print(EXT)


if __name__ == "__main__":
    main()
