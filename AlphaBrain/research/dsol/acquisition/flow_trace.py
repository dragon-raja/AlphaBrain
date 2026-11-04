"""Observe a native Euler flow sampler without extra model calls or action edits.

The caller owns RNG isolation, inference admission, action masks and score
definition. This recorder is not an inference adapter or a dispatch permit.
"""
from contextlib import contextmanager
import numpy as np


class FlowTrace:
    def __init__(self, *, steps, horizon, dimensions, time_buckets):
        values=(steps,horizon,dimensions,time_buckets)
        if any(type(v) is not int or v<=0 for v in values):
            raise ValueError('Positive integer sampler dimensions required')
        self.steps,self.horizon,self.dimensions,self.time_buckets=values
        self.states=[];self.velocities=[];self.timesteps=[]
        self.active=False;self.used=False;self.complete=False

    def _copy(self, tensor):
        # No in-place changes; clone on device, transfer only after sampling.
        return tensor.detach().clone()

    def _encoder(self, module, args):
        if len(args)<2:raise ValueError('Encoder must expose actions and time')
        state,time=args[:2]
        if len(state.shape)!=3 or tuple(state.shape[1:])!=(self.horizon,self.dimensions):
            raise ValueError('Native state shape changed')
        if len(self.states)>=self.steps:raise ValueError('Unexpected extra flow step')
        self.states.append(self._copy(state));self.timesteps.append(self._copy(time))

    def _decoder(self, module, args, output):
        if len(output.shape)!=3 or output.shape[1]<self.horizon or output.shape[2]!=self.dimensions:
            raise ValueError('Native velocity shape changed')
        if len(self.velocities)+1!=len(self.states):raise ValueError('Unexpected decoder order')
        self.velocities.append(self._copy(output[:,-self.horizon:]))

    def _validate_times(self):
        for i,t in enumerate(self.timesteps):
            value=t.detach().cpu().numpy()
            if value.shape!=(self.states[i].shape[0],) or not np.all(value==int(i/self.steps*self.time_buckets)):
                raise ValueError('Native time grid changed')

    @contextmanager
    def capture(self, encoder, decoder):
        if self.used:raise ValueError('A trace records exactly one sampler call')
        self.used=True;self.active=True;handles=[]
        try:
            handles.append(encoder.register_forward_pre_hook(self._encoder))
            handles.append(decoder.register_forward_hook(self._decoder))
            yield self
            if len(self.states)!=self.steps or len(self.velocities)!=self.steps:
                raise ValueError('Incomplete native flow trace')
            self._validate_times()
            self.complete=True
        finally:
            for handle in handles:handle.remove()
            self.active=False

    def arrays(self):
        if not self.complete or self.active:raise ValueError('Complete trace required')
        array=lambda t:t.detach().float().cpu().numpy().copy()
        return dict(states=np.stack([array(t) for t in self.states]),
                    velocities=np.stack([array(t) for t in self.velocities]),
                    times=np.arange(self.steps,dtype=np.float64)/self.steps,
                    initial_noise=array(self.states[0]))


class TokenTimeFlowTrace(FlowTrace):
    """Observe an eager Euler head with floating per-action-token times.

    The caller binds the exact native schedule, actual action window/mask,
    serialized head and *unguided* sampler. RTC/guidance and compiled graph
    replay require separate adapters; decoder output may differ from the
    integrated velocity there. This does not grant inference permission or
    make a new model numerically admitted.
    """
    _marker='_dsol_token_time_trace'

    def __init__(self, *, timesteps, terminal_time, horizon, dimensions):
        if (not isinstance(timesteps,(tuple,list)) or len(timesteps)<2
                or any(isinstance(v,(bool,str)) for v in (*timesteps,terminal_time))):
            raise ValueError('Explicit real native time grid required')
        grid=np.asarray([*timesteps,terminal_time],dtype=np.float64)
        if not np.isfinite(grid).all() or not np.all(np.diff(grid)>0):
            raise ValueError('Finite increasing native Euler grid required')
        self.native_times=grid[:-1].copy();self.terminal_time=float(grid[-1])
        super().__init__(steps=len(timesteps),horizon=horizon,dimensions=dimensions,time_buckets=1)

    def _validate_times(self):
        for i,t in enumerate(self.timesteps):
            expected=t.new_full((self.states[i].shape[0],self.horizon),float(self.native_times[i]))
            if tuple(t.shape)!=tuple(expected.shape) or not bool((t==expected).all()):
                raise ValueError('Native per-token time grid changed or RTC prefix present')
        if any(not np.isfinite(t.detach().float().cpu().numpy()).all()
               for t in (*self.states,*self.velocities)):
            raise ValueError('Nonfinite native Euler trace')

    @contextmanager
    def capture(self, encoder, decoder):
        if encoder is decoder or any(hasattr(m,self._marker) for m in (encoder,decoder)):
            raise ValueError('Exclusive encoder and decoder observation required')
        for module in (encoder,decoder):setattr(module,self._marker,True)
        try:
            with super().capture(encoder,decoder):yield self
        finally:
            for module in (encoder,decoder):delattr(module,self._marker)

    def arrays(self):
        values=super().arrays()
        values['times']=self.native_times.copy()
        values['deltas']=np.diff(np.append(self.native_times,self.terminal_time))
        return values


class SchedulerTrace:
    """Read-only observer for a serialized native Euler scheduler invocation.

    Unlike FlowTrace, this records the sampler's actual, possibly nonuniform
    timesteps/deltas. It does not resample velocities or choose an Accel metric.
    No extra denoiser call or RNG operation is made. Not thread-safe: callers
    must serialize access to the scheduler, including uninstrumented inference.
    """
    def __init__(self, *, steps, horizon, dimensions):
        if any(type(x) is not int or x < 1 for x in (steps,horizon,dimensions)):
            raise ValueError('Positive native sampler dimensions required')
        self.steps,self.shape=steps,(1,horizon,dimensions)
        self.states=[];self.velocities=[];self.deltas=[]
        self.schedule=None;self.used=False;self.complete=False;self.active=False

    @staticmethod
    def _array(t):
        return t.detach().float().cpu().numpy().copy()

    @contextmanager
    def capture(self, scheduler):
        if self.used or getattr(scheduler,'_dsol_trace_active',False):
            raise ValueError('A native scheduler trace requires one exclusive call')
        names=('build_inference_schedule','step')
        previous={k:scheduler.__dict__[k] for k in names if k in scheduler.__dict__}
        build,step=scheduler.build_inference_schedule,scheduler.step
        self.used=True;self.active=True;scheduler._dsol_trace_active=True
        def observed_build(*args,**kwargs):
            if self.schedule is not None:raise ValueError('Repeated native schedule')
            result=build(*args,**kwargs)
            if len(result)!=2 or any(tuple(t.shape)!=(self.steps,) for t in result):
                raise ValueError('Native schedule shape changed')
            self.schedule=tuple(t.detach().clone() for t in result)
            return result
        def observed_step(model_output,delta,sample):
            if (self.schedule is None or len(self.states)>=self.steps
                    or tuple(sample.shape)!=self.shape or tuple(model_output.shape)!=self.shape
                    or tuple(delta.shape)!=()):
                raise ValueError('Native Euler step contract changed')
            self.states.append(sample.detach().clone())
            self.velocities.append(model_output.detach().clone())
            self.deltas.append(delta.detach().clone())
            return step(model_output,delta,sample)
        scheduler.build_inference_schedule=observed_build;scheduler.step=observed_step
        try:
            yield self
            if self.schedule is None or len(self.states)!=self.steps:
                raise ValueError('Incomplete native scheduler trace')
            times,expected=map(self._array,self.schedule)
            deltas=np.stack([self._array(x) for x in self.deltas])
            values=[*self.states,*self.velocities,*self.schedule]
            if not all(np.isfinite(self._array(x)).all() for x in values):
                raise ValueError('Nonfinite native trace')
            if not np.array_equal(deltas,expected):raise ValueError('Native step/grid mismatch')
            if not ((np.all(np.diff(times)<0) and np.all(deltas<0))
                    or (np.all(np.diff(times)>0) and np.all(deltas>0))):
                raise ValueError('Native Euler grid must be strictly monotone')
            self.complete=True
        finally:
            for name in names:
                if name in previous:setattr(scheduler,name,previous[name])
                else:delattr(scheduler,name)
            del scheduler._dsol_trace_active
            self.active=False

    def arrays(self):
        if not self.complete or self.active:raise ValueError('Complete native trace required')
        return dict(states=np.stack([self._array(x) for x in self.states]),
                    velocities=np.stack([self._array(x) for x in self.velocities]),
                    timesteps=self._array(self.schedule[0]),
                    deltas=self._array(self.schedule[1]),initial_noise=self._array(self.states[0]))
