"""CPU fake tower verifies feature order/pooling; no operation model forward."""
from types import SimpleNamespace
import pytest
torch=pytest.importorskip('torch')
from AlphaBrain.model.pi05_features import StandaloneSiglipFeatures,SiglipFeatureCapture


class Tower(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.));self.calls=0

    def forward(self,pixel_values):
        self.calls+=1
        tokens=torch.arange(256,dtype=torch.float32).reshape(1,256,1).expand(1,256,1152)
        return SimpleNamespace(last_hidden_state=tokens+pixel_values.mean()*self.weight)


def test_bridge_matches_existing_capture_with_no_operation_forward():
    torch.set_num_threads(1)
    pixels=dict(external=torch.zeros(1,3,224,224),wrist=torch.ones(1,3,224,224))
    tower=Tower();tap=SiglipFeatureCapture(tower,patch_grid=(16,16),view_names=('external','wrist'))
    with tap:
        for v in pixels.values():tower(pixel_values=v)
    expected=tap.features();rng=torch.get_rng_state()
    result=StandaloneSiglipFeatures(tower)(pixels)
    assert tower.calls==4 and not tower.training and not tower.weight.requires_grad
    assert torch.equal(torch.get_rng_state(),rng)
    for k in pixels:assert torch.equal(result[k],expected[k].reshape(1,4608))
    assert not tower._forward_hooks


@pytest.mark.parametrize('fault',['missing_wrist','range','shape','dtype'])
def test_bridge_refuses_missing_or_implicitly_preprocessed_views(fault):
    pixels=dict(external=torch.zeros(1,3,224,224),wrist=torch.ones(1,3,224,224))
    if fault=='missing_wrist':pixels.pop('wrist')
    if fault=='range':pixels['external'].fill_(255)
    if fault=='shape':pixels['external']=torch.zeros(1,3,256,256)
    if fault=='dtype':pixels['external']=pixels['external'].to(torch.uint8)
    tower=Tower()
    with pytest.raises(ValueError):StandaloneSiglipFeatures(tower)(pixels)
    assert tower.calls==0


def test_selective_loader_materializes_nonpersistent_position_ids(tmp_path):
    transformers=pytest.importorskip('transformers')
    safetensors=pytest.importorskip('safetensors.torch')
    import hashlib
    from AlphaBrain.model.pi05_features import load_standalone_siglip
    config=dict(hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,image_size=224,patch_size=14,vision_use_head=False)
    source=transformers.SiglipVisionModel(transformers.SiglipVisionConfig(**config))
    prefix='paligemma_with_expert.paligemma.model.vision_tower.'
    state={prefix+k:v for k,v in source.state_dict().items()}
    state['unused_language_or_action_parameter']=torch.ones(3)
    checkpoint=tmp_path/'test.safetensors';safetensors.save_file(state,checkpoint)
    stamp=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    loaded=load_standalone_siglip(checkpoint,expected_sha256=stamp,vision_config=config,device='cpu')
    assert not any(t.is_meta for t in loaded.tower.buffers())
    assert torch.equal(loaded.tower.vision_model.embeddings.position_ids,torch.arange(256).expand(1,-1))
    assert not any(p.requires_grad for p in loaded.tower.parameters())
    for key,value in loaded.tower.state_dict().items():
        # Frozen native runtime rounds the checkpoint, then restores the
        # embeddings' compute dtype. This is not a precision-improvement test.
        expected=source.state_dict()[key].bfloat16()
        if key.startswith('vision_model.embeddings.'):expected=expected.float()
        assert torch.equal(value,expected)
    with pytest.raises(ValueError,match='checkpoint changed'):
        load_standalone_siglip(checkpoint,expected_sha256='0'*64,vision_config=config,device='cpu')
