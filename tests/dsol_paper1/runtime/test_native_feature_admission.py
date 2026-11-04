"""Engineering comparison kernel with synthetic calls; never GPU admission."""
import numpy as np
import random
import pytest
torch=pytest.importorskip('torch')
from benchmarks.LIBERO.native_feature_admission import Triplet
from benchmarks.LIBERO.local_search_rollout import input_identity
from AlphaBrain.research.dsol.data.native_features import NativeFeatures,to_wire,layout_spec
from AlphaBrain.research.dsol.data.native_observation import NativeStream


@pytest.mark.parametrize('fault',[None,'actions','full_actions','trace','rng','numpy_rng','python_rng','stale','producer'])
def test_triplet_accounts_every_forward_and_rejects_semantic_changes(tmp_path,fault):
    class Calls:
        n=0
        def infer(self,example,noise,identity,capture=True,*,feature_capture=True,native_request=None):
            self.n+=1;draw=torch.rand(1).item();actions=np.full((10,7),draw)
            value=dict(actions=actions,full_actions=np.full((32,7),draw),trace=np.zeros((4,10,7)),score=.2)
            if not capture:return value
            if feature_capture:
                packet=NativeFeatures(layout_spec((NativeStream('head',3,'per_view',('head',),('movable',)),)),
                    'stale' if fault=='stale' else native_request['query_id'],native_request['input_sha256'],
                    native_request['history_sha256'],'e'*64,(np.ones(12,np.float32),),('f' if fault=='producer' else 'd')*64)
                value['native_features']=to_wire(packet)
                if fault=='actions':value['actions'][0,0]+=1
                if fault=='full_actions':value['full_actions'][31,0]+=1
                if fault=='trace':value['trace'][0,0,0]+=1
                if fault=='rng':torch.rand(1)
                if fault=='numpy_rng':np.random.random()
                if fault=='python_rng':random.random()
            return value
    example=dict(image=[np.zeros((2,2))],state=np.zeros(7));calls=Calls();check=Triplet(calls,'e'*64,producer_sha256='d'*64,evidence=tmp_path/'pairs')
    if fault:
        with pytest.raises(ValueError):check.infer(example,np.zeros((10,7)),dict(step=0))
    else:
        check.infer(example,np.zeros((10,7)),dict(step=0))
        assert (tmp_path/'pairs/00001/values.npz').exists()
        with np.load(tmp_path/'pairs/00001/values.npz') as values:
            np.testing.assert_array_equal(values['raw_full_actions'],values['native_full_actions'])
    assert calls.n==3
