"""CPU-only admission checks; no simulator or GPU is imported."""
import copy

import numpy as np
import pytest

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.evaluation_admission import resolve_initial, load_candidate
from AlphaBrain.research.dsol.data.render_reference import CAMERAS, LIMITS, save_reference


def released_initial(tmp_path):
    snapshot = tmp_path / 'snapshot'
    write(snapshot / 'manifest.json', {'snapshot_numeric_sha256': 'physical'})
    job = dict(path=str(snapshot), initial_id='initial',
               snapshot_manifest_sha256=sha(snapshot / 'manifest.json'))
    ref = tmp_path / 'reference/reference.json'
    save_reference(job, {k: np.zeros((256, 256, 3), np.uint8) for k in CAMERAS},
                   ref.parent, provenance={'kind': 'test'})
    row = {k: dict(max_abs=0, mean_abs=0., fraction_gt1=0., fraction_gt2=0.,
                   **{'pass': True}) for k in CAMERAS}
    validation = tmp_path / 'validation.json'
    write(validation, dict(status='PASS', initial_id='initial', reference_sha256=sha(ref),
        pixel_gate=LIMITS, repeats=4, numeric_start='physical', numeric_end='physical',
        checks=[copy.deepcopy(row) for _ in range(4)]))
    candidate = tmp_path / 'candidates'
    write(candidate / 'summary.json', {'status': 'PASS'})
    plan, release = tmp_path / 'plan.json', tmp_path / 'release.json'
    write(plan, dict(jobs=[job], pixel_gate=LIMITS))
    write(release, dict(complete=True, eligible_for_formal_release=True, plan_sha256=sha(plan),
        total=1, counts={'PASS': 1}, records=[dict(initial_id='initial', status='PASS',
        reference_file=str(ref), reference_sha256=sha(ref), validation_file=str(validation),
        validation_sha256=sha(validation), candidate_root=str(candidate),
        candidate_summary_sha256=sha(candidate / 'summary.json'))]))
    return plan, release


def test_admit_released_initial(tmp_path):
    plan, release = released_initial(tmp_path)
    admitted = resolve_initial(plan, release, 'initial')
    assert set(admitted['images']) == set(CAMERAS)
    assert admitted['release_sha256'] == sha(release)
    with pytest.raises(ValueError):
        resolve_initial(plan, release, 'not-in-population')


@pytest.mark.parametrize('change', ['incomplete', 'population', 'plan', 'candidate', 'validation'])
def test_reject_changed_or_incomplete_release(tmp_path, change):
    plan, release = released_initial(tmp_path)
    data = read(release)
    if change == 'incomplete':
        data['complete'] = False
    elif change == 'population':
        data['records'][0]['initial_id'] = 'another'
    elif change == 'plan':
        write(plan, dict(read(plan), extra='changed'))
    elif change == 'candidate':
        write(tmp_path / 'candidates/summary.json', {'status': 'changed'})
    else:
        validation = tmp_path / 'validation.json'
        receipt = read(validation)
        receipt['checks'][0][CAMERAS[0]]['max_abs'] = 3
        write(validation, receipt)
        data['records'][0]['validation_sha256'] = sha(validation)
    write(release, data)
    with pytest.raises(ValueError):
        resolve_initial(plan, release, 'initial')


def test_candidate_geometry_and_images_are_checked(tmp_path):
    from PIL import Image
    plan, release = released_initial(tmp_path)
    root = tmp_path / 'candidates'
    for name in ('left.png', 'right.png', 'wrist.png'):
        Image.fromarray(np.zeros((256, 256, 3), np.uint8)).save(root / name)
    np.savez(root / 'geometry.npz', view_ids=np.array(['view']),
             camera_positions=np.zeros((1, 3, 3)), camera_quaternions=np.zeros((1, 3, 4)))
    names = ['geometry.npz', 'left.png', 'right.png', 'wrist.png']
    write(root / 'summary.json', dict(status='PASS', initial_id='initial',
        snapshot_manifest_sha256=read(plan)['jobs'][0]['snapshot_manifest_sha256'],
        views=[dict(view_id='view', images=names[1:])],
        files={name: dict(bytes=(root/name).stat().st_size, sha256=sha(root/name)) for name in names}))
    data = read(release)
    data['records'][0]['candidate_summary_sha256'] = sha(root / 'summary.json')
    write(release, data)
    admission = resolve_initial(plan, release, 'initial')
    candidate = load_candidate(admission, 'view')
    assert candidate['positions'].shape == (3, 3)
    assert len(candidate['images']) == 3
    with pytest.raises(ValueError):
        load_candidate(admission, 'absent')
    (root / 'left.png').write_bytes(b'changed')
    with pytest.raises(ValueError):
        load_candidate(admission, 'view')
