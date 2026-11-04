from copy import deepcopy
import numpy as np
import pytest
from AlphaBrain.research.dsol.data.native_observation import NativeStream
from AlphaBrain.research.dsol.data.native_features import NativeFeatures,layout_spec,from_wire,to_wire


def packet():
    streams=(NativeStream('head',3,'per_view',('head',),('movable',)),NativeStream('wrist',5,'per_view',('wrist',),('context',)))
    return NativeFeatures(layout_spec(streams),'q','a'*64,'b'*64,'c'*64,(np.arange(12,dtype=np.float32),None),'d'*64)


def test_real_missing_context_mask_and_copy_survive_wire():
    p=packet();wire=to_wire(p);restored=from_wire(wire)
    assert restored.packed().shape==(32,) and restored.values[1] is None
    assert wire['mask']==[True,False] and not restored.packed()[12:].any()
    wire['layout']['streams'][0]['name']='changed';wire['values'][0][:]=99
    assert restored.layout_sha256==p.layout_sha256 and restored.values[0][0]==0


@pytest.mark.parametrize('fault',['fake_missing','no_movable','wrong_shape','nan','dtype','layout'])
def test_false_or_incompatible_observations_rejected(fault):
    wire=to_wire(packet())
    if fault=='fake_missing':wire['values'][1]=np.zeros(20,np.float32)
    if fault=='no_movable':wire['mask'][0]=False;wire['values'][0]=None
    if fault=='wrong_shape':wire['values'][0]=np.zeros(8,np.float32)
    if fault=='nan':wire['values'][0][0]=np.nan
    if fault=='dtype':wire['values'][0]=wire['values'][0].astype(np.float64)
    if fault=='layout':wire['layout']['streams'][0]['channels']=8
    with pytest.raises(ValueError):from_wire(wire)
