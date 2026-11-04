import numpy as np
import json
import pytest
from reports.paper1.target_spatial_recovery import audit_pairing
from reports.paper1.spatial_selection_frontier import build
from reports.paper1.model_spatial_mechanisms import matched_rows
from AlphaBrain.research.dsol.execution.manifest import file_digest


@pytest.mark.parametrize('fault', [None, 'geometry', 'mean', 'hash'])
def test_three_model_match_recovers_true_members_and_rejects_bad_identity(tmp_path, fault):
    fastwam=[]; native=[]; refs={'views': {}}
    for state in range(16):
        for view in range(27):
            key=f'state-{state}'; task=f'task-{state//2}'; angles=[view, 0]
            row=dict(state_key=key, task=task, view_id=view, angles_deg=angles,
                     p3=[1., 3.], success=[True, False])
            fastwam.append(row)
            native.append(dict(state_key=key, task=task, view_id=view, yaw_deg=view,
                               pitch_deg=0, accel=2., success=[True, False]))
            directory=tmp_path/f'{state}-{view}';directory.mkdir()
            score=directory/'scores.json'
            score.write_text(json.dumps([dict(state_key=key, view_id=view, member=i,
                                              score=1.+2*i) for i in range(4)]))
            refs['views'][f'{key}::{view}']=dict(query=str(directory/'query.npz'),
                artifacts={str(score):file_digest(score)})
    groot=[dict(r) for r in fastwam]
    if fault=='geometry': groot[-1]['angles_deg']=[100,0]
    if fault=='mean': native[-1]['accel']=5.
    if fault=='hash': score.write_text('[]')
    if fault:
        with pytest.raises(ValueError): matched_rows(fastwam, groot, native, refs, {})
    else:
        consumed={}; rows=matched_rows(fastwam, groot, native, refs, consumed)
        assert len(rows)==len(consumed)==432
        assert rows[0]['p3']==[1.,3.]  # Never duplicate the archived mean as members.


def test_subset_report_excludes_reference_and_keeps_opposite_score_rankings(tmp_path):
    rows = [dict(state_key='state', task='task', view_id=i, angles_deg=[i, 0],
                 p3=[float(i)] * 2, p9=[float(27-i)] * 2,
                 success=[i in (0, 26)] * 2) for i in range(27)]
    source = tmp_path/'rows.json'; source.write_text(json.dumps(rows))
    result = build(source, tmp_path/'report')
    values = {(x['score'], x['subset_size']): x['task_mean_success']
              for x in result['frontier']}
    assert values['p3', 1] == pytest.approx(1 / 26)
    assert values['p9', 1] == pytest.approx(1 / 26)
    assert values['p3', 26] == 0
    assert values['p9', 26] == 1
    assert result['new_model_calls'] == 0
    assert not result['outcome_labels_used_for_selection']


@pytest.mark.parametrize('fault', [None, 'score_noise', 'execution_noise', 'actor', 'pose'])
def test_actual_noise_and_nonactor_endpoint_audit(tmp_path, fault):
    points = {}
    for view in range(27):
        path = tmp_path/f'{view}.npz'
        camera = np.eye(4); camera[0, 3] = view
        noise = np.zeros((1, 32, 7))
        if fault == 'score_noise' and view == 26: noise += 1
        np.savez(path, robot_state=np.zeros(8), camera_to_world_opencv=camera,
                 full_noise_0=noise, full_noise_1=noise + 2)
        result = dict(method='actor:should-not-appear' if fault == 'actor' and view == 26 else 'stay',
                      angles_deg=[view, 0],
                      selected_angles=[-1 if fault == 'pose' and view == 26 else view, 0],
                      execution_noise_sha256=['different' if fault == 'execution_noise' and view == 26 else 'same'])
        points['state', view] = dict(score_query=str(path), angles_deg=[view, 0],
                                    executions={'0': dict(result=result)})
    if fault:
        with pytest.raises(ValueError): audit_pairing(points)
    else:
        assert audit_pairing(points)['status'] == 'PASS'
