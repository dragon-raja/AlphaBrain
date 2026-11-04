import pytest
torch=pytest.importorskip('torch')
from AlphaBrain.model.native_features import NativeVisionCapture


def test_eager_extractor_after_graph_replay_is_fresh_without_extra_forward():
    class Vision:
        def __init__(self):self.calls=0;self.buffer=torch.zeros(2,8,3)
        def extract_vision_feature(self,pixels):
            self.calls+=1;self.buffer.copy_(pixels);return self.buffer
    m=Vision();original=m.extract_vision_feature
    with NativeVisionCapture(m,shape=(2,8,3)) as tap:
        output=m.extract_vision_feature(torch.ones(2,8,3))
        assert output is m.buffer
    first=tap.features();m.extract_vision_feature(torch.full((2,8,3),2.))
    assert torch.equal(first,torch.ones(2,8,3)) and m.calls==2
    assert m.extract_vision_feature==original
    with pytest.raises(RuntimeError):
        with tap:pass
    broken=NativeVisionCapture(m,shape=(2,8,3))
    with pytest.raises(RuntimeError):
        with broken:raise RuntimeError('unknown forward result')
    with pytest.raises(ValueError):broken.features()
