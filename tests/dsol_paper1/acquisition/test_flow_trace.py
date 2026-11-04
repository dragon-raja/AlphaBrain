"""Actual Torch hook mechanics on a CPU mock Euler sampler, no VLA/physics."""
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from AlphaBrain.research.dsol.acquisition.flow_trace import FlowTrace,SchedulerTrace


class Encoder(torch.nn.Module):
    def forward(self,x,t,e):return x


class Decoder(torch.nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,x,e):
        self.calls+=1
        return torch.cat((torch.zeros_like(x[:,:1]),x*.2+1),dim=1)


def sample(encoder,decoder,steps=4,fail=False):
    x=torch.randn(1,40,132)
    for i in range(steps):
        encoded=encoder(x,torch.full((1,),int(i/steps*1000)),None)
        velocity=decoder(encoded,None)[:,-40:]
        x=x+velocity/steps
        if fail:raise RuntimeError('injected')
    return x


def test_native_actions_rng_and_call_count_unchanged():
    encoder,decoder=Encoder(),Decoder()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3);expected=sample(encoder,decoder);expected_rng=torch.get_rng_state()
        torch.manual_seed(3)
        trace=FlowTrace(steps=4,horizon=40,dimensions=132,time_buckets=1000)
        with trace.capture(encoder,decoder):actual=sample(encoder,decoder)
        assert torch.equal(actual,expected) and torch.equal(torch.get_rng_state(),expected_rng)
    assert decoder.calls==8
    a=trace.arrays();assert a['velocities'].shape==(4,1,40,132)
    np.testing.assert_array_equal(a['states'][0],a['initial_noise'])
    for i in range(3):np.testing.assert_array_equal(a['states'][i+1],a['states'][i]+a['velocities'][i]/4)
    assert not encoder._forward_pre_hooks and not decoder._forward_hooks
    with pytest.raises(ValueError):
        with trace.capture(encoder,decoder):pass


@pytest.mark.parametrize('failure',['exception','incomplete','shape','time'])
def test_failures_remove_hooks(failure):
    encoder,decoder=Encoder(),Decoder()
    trace=FlowTrace(steps=4,horizon=39 if failure=='shape' else 40,dimensions=132,time_buckets=999 if failure=='time' else 1000)
    with pytest.raises((RuntimeError,ValueError)):
        with trace.capture(encoder,decoder):sample(encoder,decoder,steps=2 if failure=='incomplete' else 4,fail=failure=='exception')
    assert not encoder._forward_pre_hooks and not decoder._forward_hooks
    with pytest.raises(ValueError):trace.arrays()


class Scheduler:
    def build_inference_schedule(self):
        return torch.tensor([1000.,800.,300.]),torch.tensor([-.2,-.5,-.3])

    @staticmethod
    def step(model_output,delta,sample):return sample+model_output*delta


def native_sample(scheduler,fail=None):
    x=torch.randn(1,32,7);times,deltas=scheduler.build_inference_schedule()
    for i,(t,dt) in enumerate(zip(times,deltas)):
        v=x*.2+t/1000
        if fail=='nan':v.fill_(float('nan'))
        x=scheduler.step(v,dt+.1 if fail=='grid' else dt,x)
        if fail=='exception':raise RuntimeError('injected native failure')
        if fail=='incomplete':break
    return x


def test_native_nonuniform_scheduler_observer_preserves_actions_rng_and_methods():
    scheduler=Scheduler();original=dict(scheduler.__dict__)
    torch.manual_seed(19);expected=native_sample(scheduler);rng=torch.get_rng_state()
    torch.manual_seed(19);trace=SchedulerTrace(steps=3,horizon=32,dimensions=7)
    with trace.capture(scheduler):actual=native_sample(scheduler)
    assert torch.equal(actual,expected) and torch.equal(torch.get_rng_state(),rng)
    assert scheduler.__dict__==original
    arrays=trace.arrays()
    np.testing.assert_array_equal(arrays['timesteps'],[1000,800,300])
    for i in range(2):
        np.testing.assert_array_equal(arrays['states'][i+1],
            arrays['states'][i]+arrays['velocities'][i]*arrays['deltas'][i])


@pytest.mark.parametrize('fault',['exception','incomplete','grid','nan','shape'])
def test_scheduler_observer_fails_closed_and_restores_methods(fault):
    scheduler=Scheduler();original=dict(scheduler.__dict__)
    trace=SchedulerTrace(steps=3,horizon=31 if fault=='shape' else 32,dimensions=7)
    with pytest.raises((ValueError,RuntimeError)):
        with trace.capture(scheduler):native_sample(scheduler,fault)
    assert scheduler.__dict__==original
    with pytest.raises(ValueError):trace.arrays()
