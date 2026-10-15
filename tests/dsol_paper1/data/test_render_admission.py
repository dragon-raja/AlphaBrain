import pytest
from AlphaBrain.research.dsol.data.artifacts import write, sha
from AlphaBrain.research.dsol.data.render_admission import confirmed_pixel_gate


def test_only_confirmed_complete_proposals_can_be_loaded(tmp_path):
    plan = tmp_path / 'plan.json'
    write(plan, dict(mode='confirmation', fixed_mean_abs=.005, initials=[dict(role='confirmation', task='t')]))
    write(tmp_path / 'status.json', dict(complete=True, records=[
        {'job': f'confirmation/t/process-{i}', 'returncode': 0} for i in range(2)]))
    result = dict(source_directory=str(tmp_path), plan_sha256=sha(plan), proposal=dict(
        status='VALIDATED_FIXED_PROPOSAL_NOT_APPLIED', proposed_mean_abs=.005,
        validation_hard_failures=0, validation_max_mean_abs=.003,
        unchanged_limits=dict(max_abs=2, fraction_gt1=.0001, fraction_gt2=0.)))
    target = tmp_path / 'result.json'; write(target, result)
    assert confirmed_pixel_gate(target)['mean_abs'] == .005
    result['proposal']['status'] = 'HOLD'; write(target, result)
    with pytest.raises(ValueError):
        confirmed_pixel_gate(target)


def test_cannot_relax_hard_bounds(tmp_path):
    result = dict(proposal=dict(status='VALIDATED_FIXED_PROPOSAL_NOT_APPLIED', proposed_mean_abs=.005,
                  unchanged_limits=dict(max_abs=20, fraction_gt1=.0001, fraction_gt2=0.)))
    target = tmp_path / 'result.json'; write(target, result)
    with pytest.raises(ValueError, match='Hard limits'):
        confirmed_pixel_gate(target)
