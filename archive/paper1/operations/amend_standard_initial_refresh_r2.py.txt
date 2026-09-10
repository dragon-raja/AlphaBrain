"""Explicit pre-data amendment; archive failed gate/assets, preserve initial/noise."""
from pathlib import Path
import shutil
import time
from render_bridge_common_v1 import read,sha
from dualhost_aa0_v1 import write

root=Path('/share/longjunyu/alphabrain/experiments/dsol-standard-initial-aa0-v1-20260908')
r=read(root/'release.json');before=sha(root/'release.json')
assert 'amendment' not in r
for host in ['fresh','gnu']:
    status=read(root/'hosts'/host/'status.json')
    assert status['phase']=='FAILED_REQUIRES_REVIEW'
    assert not list((root/'hosts'/host).glob('gate-*/*/*/*.json'))
    assert not list((root/'hosts'/host).glob('dense-*/*/*/*.json'))
archive=root/'attempt-1-canonical-cached-observation'
archive.mkdir()
shutil.copy2(root/'release.json',archive/'release.json')
for name in ['hosts','assets','fresh-controller.log','gnu-controller.log']:
    (root/name).rename(archive/name)
source=Path(__file__).parent/'evaluate_dsol_libero_hdf5_views.py'
dest=root/'repo/scripts/dsol_paper1/evaluate_dsol_libero_hdf5_views.py'
shutil.copy2(source,dest)
r['source_hashes'][str(dest)]=sha(dest)
r['amendment']=dict(id='r2-canonical-observation-refresh',utc=time.time(),previous_release_sha256=before,
    valid_closed_loop_records_before=0,archived_attempt=str(archive),
    reason='Canonical returned final no-op cached image; explicitly refresh before first policy observation, matching posed branch',
    initialization_noise_candidates_models_unchanged=True,scientific_budget_unchanged=True)
write(root/'release.json',r)
write(root/'amendment-r2.json',dict(**r['amendment'],release_sha256=sha(root/'release.json')))
print('AMENDED_PRE_DATA',sha(root/'release.json'))
