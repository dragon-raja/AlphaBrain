"""Archive safely aborted transition, update only transition implementation."""
from pathlib import Path
import shutil
import time
from dynamic_initial_scheduler_v1 import EXT,core
m=core.read(EXT/'manifest.json')
core.require('transition_revision' not in m,'Already amended')
for h in ['fresh','gnu']:
    p=EXT/h/'transition.json';r=core.read(p)
    core.require(r['status'] in ['DRAINING_INFLIGHT_EPISODES','TRANSITION_ABORTED_OLD_DISPATCH_RESUMED'],'Unexpected prior transition state')
    status=core.read(EXT.parent.parent/'hosts'/h/'status.json')
    core.require(status['pid']==r['old_controller']['pid'] and status['phase'].startswith('dense-'),'Original run no longer active')
    p.rename(p.with_name('transition-attempt1-aborted.json'))
    (EXT/f'{h}-transition.log').rename(EXT/f'{h}-transition-attempt1.log')
shutil.copy2(EXT/'manifest.json',EXT/'manifest-transition-attempt1.json')
shutil.copy2(Path(__file__).parent/'transition_initial_dynamic_v1.py',EXT/'transition_initial_dynamic_v1.py')
m.update(transition_revision=2,transition_sha256=core.sha(EXT/'transition_initial_dynamic_v1.py'),
         transition_amended_utc=time.time(),reason='Dispatch pause did not persist; use resumable owned-controller restart instead',
         inflight_drained_before_switch=False,incomplete_episode_policy='rerun only missing atomic outputs with original frozen identity and noise')
core.write(EXT/'manifest.json',m)
print('RETRY_PREPARED')
