import hashlib
import json
import numpy as np
import pytest
from benchmarks.Robocasa_tabletop.snapshot_io import save_snapshot, load_snapshot


def test_roundtrip_rng_controller_arrays_and_nested_key_types(tmp_path):
    state={'rng':np.random.get_state(),'python':(None,{3:np.arange(6).reshape(2,3)}),
           'limits':np.array([-np.inf,np.inf]),'warmstart':np.arange(3,dtype=np.float64),
           'native_torque_sentinel':np.array(None)}
    p=tmp_path/'state.json';contract={'layout':0,'controller':'absolute'}
    sha=save_snapshot(p,state,task_contract=contract)
    value=load_snapshot(p,expected_sha256=sha,task_contract=contract)
    assert type(value['rng']) is tuple and type(value['python']) is tuple
    assert value['native_torque_sentinel'].shape==() and value['native_torque_sentinel'].item() is None
    np.testing.assert_array_equal(value['rng'][1],state['rng'][1])
    np.testing.assert_array_equal(value['limits'],state['limits'])
    np.testing.assert_array_equal(value['python'][1][3],state['python'][1][3])
    with pytest.raises(FileExistsError):save_snapshot(p,state,task_contract=contract)
    with pytest.raises(ValueError,match='contract'):
        load_snapshot(p,expected_sha256=sha,task_contract={'layout':1})
    p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError,match='digest'):load_snapshot(p,expected_sha256=sha,task_contract=contract)


def test_replay_never_loads_object_arrays(tmp_path):
    with pytest.raises(ValueError):save_snapshot(tmp_path/'invalid',np.array([object()],dtype=object),task_contract={})
    p=tmp_path/'malformed.json'
    p.write_text(json.dumps(dict(schema='GR1_REPLAY_DATA_V1',task_contract={},
        state=dict(type='array',dtype='O',shape=[1],data='AAAAAAAAAAA='))))
    with pytest.raises(ValueError,match='numeric'):
        load_snapshot(p,expected_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),task_contract={})
