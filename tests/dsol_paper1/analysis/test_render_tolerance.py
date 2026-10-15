import pytest
import numpy as np

from AlphaBrain.research.dsol.analysis.render_tolerance import calibration_population, propose_mean_gate, confirmation_population, validate_fixed_gate
from AlphaBrain.research.dsol.analysis.render_calibration_results import pixel_difference


def check(mean, maximum=1):
    return dict(mean_abs=mean, max_abs=maximum, fraction_gt1=0., fraction_gt2=0.)


def test_validation_cannot_increase_proposed_threshold():
    proposal = propose_mean_gate([check(.003)], [check(.006)])
    assert proposal['proposed_mean_abs'] == .005
    assert proposal['status'] == 'HOLD'


def test_hard_gates_not_relaxed():
    proposal = propose_mean_gate([check(.003)], [check(.003, 3)])
    assert proposal['status'] == 'HOLD'
    assert proposal['validation_hard_failures'] == 1


def test_bounded_menu_and_empty_groups():
    assert propose_mean_gate([check(.003)], [check(.004)])['status'] == 'VALIDATED_PROPOSAL_NOT_APPLIED'
    assert propose_mean_gate([check(.02)], [check(.02)])['proposed_mean_abs'] is None
    with pytest.raises(ValueError):
        propose_mean_gate([], [check(0)])


def test_only_seen_and_independent_layouts():
    rows = [dict(task='t', split='pretrain', scene_seed=i, layout_id=20+i) for i in range(2)]
    rows.append(dict(task='t', split='target', scene_seed=0, layout_id=1))
    assert len(calibration_population(rows)) == 2
    rows[1]['layout_id'] = 20
    with pytest.raises(ValueError):
        calibration_population(rows)


def test_difference_uses_signed_arithmetic_and_validates_uint8():
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    b = a.copy(); b[0, 0, 0] = 255
    result = pixel_difference(a, b)
    assert result['max_abs'] == 255
    assert result['mean_abs'] == 255 / 12
    assert result['fraction_gt2'] == 1 / 12
    with pytest.raises(ValueError):
        pixel_difference(a, b.astype(float))


def test_confirmation_metadata_selection_excludes_development_layouts():
    rows = [dict(task='t', split='pretrain', scene_seed=i, layout_id=20+i) for i in range(5)]
    selected = confirmation_population(rows, {20, 21, 22})
    assert selected[0]['scene_seed'] == 3
    assert selected[0]['role'] == 'confirmation'


def test_fixed_proposal_cannot_adapt_to_confirmation_failure():
    result = validate_fixed_gate([check(.006)], .005)
    assert result['status'] == 'HOLD'
    assert result['proposed_mean_abs'] == .005
def test_fraction_proposal_does_not_raise_gate_on_validation_failure():
    from AlphaBrain.research.dsol.analysis.render_tolerance import propose_fraction_gate
    base = dict(max_abs=2, mean_abs=.001, fraction_gt1=.00005, fraction_gt2=0.)
    result = propose_fraction_gate([base], [dict(base, fraction_gt1=.0002)])
    assert result['status'] == 'HOLD'
    assert result['proposed_fraction_gt1'] == .0001

