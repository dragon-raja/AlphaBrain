"""Processor-layout tests; no Torch, checkpoint, network, renderer or GPU."""
import numpy as np
import pytest
from AlphaBrain.research.dsol.data.vlm_image_tokens import split_image_tokens


def example():
    ids=np.array([[0,7,99,99,99,99,8,99,99,99,99,99,99,9]])
    hidden=np.arange(14*3,dtype=np.float32).reshape(1,14,3)
    return dict(hidden=hidden,input_ids=ids,image_grid_thw=np.array([[1,4,4],[1,4,6]]),
        camera_ids=('external','wrist'),image_token_id=99,spatial_merge_size=2,
        attention_mask=np.array([[0,*([1]*13)]]))


def test_two_real_cameras_keep_processor_order_and_distinct_grids():
    x=example();out=split_image_tokens(**x)
    assert [v['camera_id'] for v in out]==['external','wrist']
    assert [v['grid'] for v in out]==[(1,2,2),(1,2,3)]
    np.testing.assert_array_equal(out[0]['tokens'],x['hidden'][:,2:6])
    np.testing.assert_array_equal(out[1]['tokens'],x['hidden'][:,7:13])
    out[0]['tokens'][0,0,0]=-100
    assert x['hidden'][0,2,0]!=-100


def test_single_ego_camera_needs_no_wrist_substitute():
    x=example();x.update(input_ids=x['input_ids'][:,:7],hidden=x['hidden'][:,:7],
        attention_mask=x['attention_mask'][:,:7],image_grid_thw=x['image_grid_thw'][:1],camera_ids=('ego',))
    out=split_image_tokens(**x)
    assert len(out)==1 and out[0]['camera_id']=='ego'


@pytest.mark.parametrize('fault',['temporal','merge','count','duplicate','padding','noncontiguous','nonfinite'])
def test_wrong_or_ambiguous_provenance_fails_closed(fault):
    x=example()
    if fault=='temporal':x['image_grid_thw'][0,0]=2
    elif fault=='merge':x['image_grid_thw'][0,1]=5
    elif fault=='count':x['input_ids'][0,2]=5
    elif fault=='duplicate':x['camera_ids']=('ego','ego')
    elif fault=='padding':x['attention_mask'][0,3]=0
    elif fault=='noncontiguous':x['input_ids'][0,3]=5;x['input_ids'][0,6]=99
    elif fault=='nonfinite':x['hidden'][0,3,1]=np.nan
    with pytest.raises(ValueError):split_image_tokens(**x)
