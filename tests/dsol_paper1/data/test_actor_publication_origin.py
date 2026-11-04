"""Same shapes must not authorize a different model, preprocessing or run."""
from copy import deepcopy
import pytest
from AlphaBrain.research.dsol.data.native_training import publication_origin
from AlphaBrain.research.dsol.selectors.search_policy_factory import validate_publication


def fixture(tmp_path):
    checkpoint=tmp_path/'initial';checkpoint.write_bytes(b'synthetic CPU origin')
    suites=('libero_spatial','libero_object','libero_goal','libero_10')
    p=dict(seeds=[0,1,2],family='synthetic',model_hashes={s:'a'*64 for s in suites},
        runtime_contract_hashes={s:'b'*64 for s in suites},native_policy=dict(producer_sha256='c'*64),
        initial_checkpoint=str(checkpoint),query_cap=8,query_cost=.01)
    origin=publication_origin(p,'d'*64,0)
    class Receiver:
        feature_contract_sha256=origin['feature_contract_sha256']
        loaded=False
        def load_state_dict(self,state_dict,strict):self.loaded=True
    payload=dict(version=1,feature_contract_sha256=origin['feature_contract_sha256'],
        training_protocol_sha256='d'*64,origin=deepcopy(origin),state_dict={})
    return Receiver(),payload,origin


@pytest.mark.parametrize('field',['family','model_hashes','runtime_contract_hashes','producer_sha256',
    'training_protocol_sha256','initial_checkpoint_sha256','seed','scoring','missing_origin'])
def test_same_shape_foreign_source_rejected_before_tensor_load(tmp_path,field):
    receiver,payload,origin=fixture(tmp_path)
    if field=='missing_origin':payload.pop('origin')
    elif field in ('model_hashes','runtime_contract_hashes'):payload['origin'][field]['libero_spatial']='f'*64
    elif field=='seed':payload['origin'][field]=1
    elif field=='scoring':payload['origin'][field]['prefix']=9
    else:payload['origin'][field]='f'*64
    with pytest.raises(ValueError,match='origin'):validate_publication(receiver,payload,version=1,expected_origin=origin)
    assert not receiver.loaded


def test_explicit_context_required_and_correct_origin_loads(tmp_path):
    receiver,payload,origin=fixture(tmp_path)
    with pytest.raises(TypeError):validate_publication(receiver,payload,version=1)
    with pytest.raises(ValueError):validate_publication(receiver,payload,version=1,expected_origin={})
    assert not receiver.loaded
    validate_publication(receiver,payload,version=1,expected_origin=origin)
    assert receiver.loaded
