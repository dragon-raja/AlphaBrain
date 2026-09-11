"""Freeze an execution manifest without editing the scientific release."""
from pathlib import Path
import shutil
import time
from dynamic_initial_scheduler_v1 import ROOT,EXT,FROZEN_ENTRY,core
r=core.release()
core.require(not EXT.exists(),'Extension already prepared')
EXT.mkdir(parents=True)
for name in ['dynamic_initial_scheduler_v1.py','transition_initial_dynamic_v1.py']:
    shutil.copy2(Path(__file__).parent/name,EXT/name)
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
