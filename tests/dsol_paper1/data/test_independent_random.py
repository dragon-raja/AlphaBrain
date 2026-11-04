import json
import pytest
from AlphaBrain.research.dsol.data import independent_random as module
from AlphaBrain.research.dsol.execution.manifest import file_digest


def test_full_population_preserves_pairing_and_has_independent_paths():
    original=[dict(method='B-seed-0',state_key=f'state-{i//8}',view_id=i%8,
                   execution_noise='paired',angles_deg=[-17.5,9.5]) for i in range(3200)]
    result=module.rows(original)
    assert len(result)==9600
    assert len({r['acquisition_seed'] for r in result})==9600
    for source,triple in zip(original,zip(*[iter(result)]*3)):
        for seed,row in enumerate(triple):
            assert row['execution_noise']==source['execution_noise']
            assert row['angles_deg']==source['angles_deg']
            assert row['method']==f'random_acquisition-seed-{seed}'
    reordered=module.rows(list(reversed(original)))
    key=lambda r:(r['state_key'],r['view_id'],r['method'])
    assert sorted(result,key=key)==sorted(reordered,key=key)
    with pytest.raises(ValueError):module.rows(original[:-1])
    with pytest.raises(ValueError):module.rows(original[:-1]+original[:1])


def test_protocol_binds_acquisition_rule_and_disallows_actors(tmp_path):
    base=tmp_path/'baseline.json';base.write_text('{}')
    path=tmp_path/'rule.json'
    p=dict(schema=module.PROTOCOL_SCHEMA,seeds=[0,1,2],root_seed=20261030,
        proposal='uniform_legal_increment',stop='query_cap',selection='minimum_visited',
        query_cap=8,selection_members=[0,1],validation_members=[2,3],execution_repeats=[0,1],
        diagnostics='selected_and_start_not_used_for_selection',training_updates=0,
        baseline_campaign_sha256=file_digest(base))
    path.write_text(json.dumps(p))
    c=dict(independent_random_protocol=str(path),baseline_campaign=str(base),actors={},
           input_hashes={str(path):file_digest(path)})
    assert module.validate_contract(c)==p
    c['actors']={'foreign':{}}
    with pytest.raises(ValueError):module.validate_contract(c)
    c['actors']={};p['selection']='random_from_local';path.write_text(json.dumps(p))
    with pytest.raises(ValueError):module.validate_contract(c)
    c['input_hashes'][str(path)]=file_digest(path)
    with pytest.raises(ValueError):module.validate_contract(c)
