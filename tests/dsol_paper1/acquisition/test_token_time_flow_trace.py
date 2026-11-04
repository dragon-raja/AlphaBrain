"""Eager floating-time observation mechanics; no VLA weights or GPU admission."""
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from AlphaBrain.research.dsol.acquisition.flow_trace import TokenTimeFlowTrace
from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores


class Encoder(torch.nn.Module):
    def forward(self,x,t,e):return x+t.unsqueeze(-1)


class Decoder(torch.nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,x,e):
        self.calls+=1
        return torch.cat((torch.ones_like(x[:,:1]),x*.2+1),dim=1)


def sample(encoder,decoder,*,grid=(0.,.25,.5,.75),failure=None):
    x=torch.randn(1,16,64)
    for i,t in enumerate(grid):
        times=torch.full((1,16),t)
        if failure=='rtc':times[:,0]=1.
        v=decoder(encoder(x,times,None),None)[:,-16:]
        dt=([*grid,1.][i+1]-t)
        x=x+dt*v
        if failure=='exception':raise RuntimeError('injected')
    return x


@pytest.mark.parametrize('grid',[(0.,.25,.5,.75),(0.,.05,.4,.9)])
def test_observation_preserves_full_actions_rng_calls_and_native_steps(grid):
    encoder,decoder=Encoder(),Decoder()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23);expected=sample(encoder,decoder,grid=grid);rng=torch.get_rng_state()
        torch.manual_seed(23)
        trace=TokenTimeFlowTrace(timesteps=grid,terminal_time=1.,horizon=16,dimensions=64)
        with trace.capture(encoder,decoder):actual=sample(encoder,decoder,grid=grid)
        assert torch.equal(actual,expected) and torch.equal(torch.get_rng_state(),rng)
    assert decoder.calls==8
    values=trace.arrays();assert values['velocities'].shape==(4,1,16,64)
    for i,dt in enumerate(values['deltas'][:-1]):
        np.testing.assert_array_equal(values['states'][i+1],values['states'][i]+dt*values['velocities'][i])
    # Explicit semantic action mask/window: padded dimensions must not enter Accel.
    velocity=values['velocities'][:,0,:8,:7][None]
    metrics=compute_accel_scores(velocity,prefixes=(3,4))
    assert not metrics[3]['degenerate'][0] and np.isfinite(metrics[4]['score'][0])
    with pytest.raises(ValueError):compute_accel_scores(velocity,prefixes=(9,))
    assert not encoder._forward_pre_hooks and not decoder._forward_hooks
    assert not hasattr(encoder,trace._marker)


@pytest.mark.parametrize('failure',['rtc','exception','incomplete','shape','grid'])
def test_bad_or_partial_traces_fail_closed_and_remove_hooks(failure):
    encoder,decoder=Encoder(),Decoder()
    trace=TokenTimeFlowTrace(timesteps=(0.,.2,.5,.75) if failure=='grid' else (0.,.25,.5,.75),
        terminal_time=1.,horizon=15 if failure=='shape' else 16,dimensions=64)
    with pytest.raises((ValueError,RuntimeError)):
        with trace.capture(encoder,decoder):sample(encoder,decoder,
            grid=(0.,.25) if failure=='incomplete' else (0.,.25,.5,.75),failure=failure)
    assert not encoder._forward_pre_hooks and not decoder._forward_hooks
    assert not hasattr(encoder,trace._marker) and not hasattr(decoder,trace._marker)
    with pytest.raises(ValueError):trace.arrays()


def test_nested_or_reused_capture_rejected():
    encoder,decoder=Encoder(),Decoder()
    a=TokenTimeFlowTrace(timesteps=(0.,.25,.5,.75),terminal_time=1.,horizon=16,dimensions=64)
    b=TokenTimeFlowTrace(timesteps=(0.,.25,.5,.75),terminal_time=1.,horizon=16,dimensions=64)
    with a.capture(encoder,decoder):
        with pytest.raises(ValueError):
            with b.capture(encoder,decoder):pass
        sample(encoder,decoder)
    with pytest.raises(ValueError):
        with a.capture(encoder,decoder):pass
    assert not hasattr(encoder,a._marker) and not encoder._forward_pre_hooks


@pytest.mark.parametrize('grid,end',[((),1.),((0.,),1.),((0.,.5,.5),1.),((0.,float('nan')),1.),((0.,.5),.4),((False,.5),1.)])
def test_invalid_native_grid_rejected(grid,end):
    with pytest.raises(ValueError):TokenTimeFlowTrace(timesteps=grid,terminal_time=end,horizon=16,dimensions=64)
