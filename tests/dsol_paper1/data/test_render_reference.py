import copy
import numpy as np
import pytest

from AlphaBrain.research.dsol.data.artifacts import sha, write
from AlphaBrain.research.dsol.data.render_reference import CAMERAS, LIMITS, save_reference, load_reference, assert_validation


def fixture(tmp_path):
    snapshot = tmp_path / 'snapshot'
    write(snapshot / 'manifest.json', {'snapshot_numeric_sha256': 'physical'})
    job = dict(path=str(snapshot), initial_id='initial', snapshot_manifest_sha256=sha(snapshot / 'manifest.json'))
    images = {k: np.zeros((256, 256, 3), dtype=np.uint8) for k in CAMERAS}
    save_reference(job, images, tmp_path / 'reference', provenance={'kind': 'test'})
    return job, tmp_path / 'reference/reference.json'


def test_reference_roundtrip_and_integrity(tmp_path):
    job, ref = fixture(tmp_path)
    record, images = load_reference(job, ref)
    assert set(images) == set(CAMERAS)
    assert record['status'] == 'FROZEN_PENDING_INDEPENDENT_VALIDATION'
    with pytest.raises(ValueError):
        load_reference(dict(job, initial_id='wrong'), ref)
    (ref.parent / 'rgb.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError):
        load_reference(job, ref)


def test_validation_checks_numbers_not_only_pass_label(tmp_path):
    job, ref = fixture(tmp_path)
    row = {k: dict(max_abs=0, mean_abs=0., fraction_gt1=0., fraction_gt2=0., **{'pass': True}) for k in CAMERAS}
    receipt = dict(status='PASS', initial_id=job['initial_id'], reference_sha256=sha(ref), pixel_gate=LIMITS,
                   repeats=4, numeric_start='physical', numeric_end='physical', checks=[copy.deepcopy(row) for _ in range(4)])
    assert_validation(job, ref, receipt)
    receipt['checks'][0][CAMERAS[0]]['max_abs'] = 3
    with pytest.raises(ValueError):
        assert_validation(job, ref, receipt)


def test_reference_cannot_be_overwritten(tmp_path):
    job, ref = fixture(tmp_path)
    with pytest.raises(FileExistsError):
        save_reference(job, {}, ref.parent, provenance={})
