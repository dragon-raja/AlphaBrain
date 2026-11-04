import json
from pathlib import Path
import numpy as np
import pytest
from reports.paper1.target_spatial_panel import build
from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores


@pytest.mark.parametrize('duplicate', [False, True])
def test_complete_merge_and_duplicate_rejection(tmp_path, monkeypatch, duplicate):
    from benchmarks.LIBERO import controlled_target_panel
    from AlphaBrain.research.dsol.plotting import spatial_score_validity
    monkeypatch.setattr(controlled_target_panel, 'validate_block', lambda *args: {})
    monkeypatch.setattr(spatial_score_validity, 'figures', lambda *args: None)
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    trace = np.broadcast_to(np.arange(1., 11.)[:, None, None], (10, 32, 7)).copy()
    score = compute_accel_scores(trace[None], prefixes=[3])[3]['score'][0]
    query = tmp_path/'query.npz'
    np.savez(query, trace_0=trace, trace_1=trace, score_0=score, score_1=score)
    points = [dict(state_key=f'panel::suite::task-{i//2}::{i}', view_id=v,
                   angles_deg=[float(v), 0.], score_query=str(query),
                   executions={str(j): dict(result=dict(success=v<7)) for j in (0, 1)})
              for i in range(16) for v in range(27)]
    last = points[-1]
    if not duplicate: last['score_query'] = None; last['executions'] = {}
    reuse = tmp_path/'reuse.json'; write(reuse, dict(points=points, artifacts={}))
    write(tmp_path/'campaign.json', dict(reuse_manifest=str(reuse)))
    write(tmp_path/'completion.json', dict(status='COMPLETE'))
    block = tmp_path/'block'
    write(block/'results.json', dict(scores=[dict(state_key=last['state_key'], view_id=26, query_path=str(query))],
        tasks=[dict(state_key=last['state_key'], view_id=26, repeat=j, success=False) for j in (0, 1)]))
    write(tmp_path/'execution/panel-real/binding.json', dict(tasks={'block':dict(request={}, output=str(block))}))
    if duplicate:
        with pytest.raises(ValueError, match='Duplicate scoring'): build(tmp_path, tmp_path/'report')
    else:
        result = build(tmp_path, tmp_path/'report')
        assert result['positions'] == 432
        assert all(a['tasks'] == 8 and a['states'] == 16 for a in result['aggregate'])
        assert len((tmp_path/'report/spatial.csv').read_text().splitlines()) == 433
