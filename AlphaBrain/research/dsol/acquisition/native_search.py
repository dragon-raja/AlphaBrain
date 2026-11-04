"""Native streams on the shared continuous query/motion/reward lifecycle.

No dummy wrist, alternative PPO, movement constraints or query refund path.
"""
import numpy as np
from .continuous_search import ContinuousSearch
from .local_view_search import LocalViewSearch
from .search_env import unit_vector
from ..data.native_features import NativeQueryResult,layout_spec,read_layout
from ..data.native_observation import validate_streams
from ..execution.manifest import digest


class _NativeFeatureSession:
    def __init__(self,*args,streams,producer_sha256,feature_layout=None,**kwargs):
        self.streams=tuple(streams);validate_streams(self.streams)
        self.producer_sha256=producer_sha256
        layout=layout_spec(self.streams) if feature_layout is None else feature_layout
        if read_layout(layout)!=self.streams:raise ValueError('Explicit layout streams differ from policy')
        self.layout_sha256=digest(layout)
        if 'wrist' in kwargs or 'visual_dim' in kwargs:raise ValueError('Native layout owns the feature dimensions')
        super().__init__(*args,visual_dim=sum(4*s.channels for s in self.streams),**kwargs)

    def _copy_value(self,value,visual):
        if not isinstance(value,NativeQueryResult):raise ValueError('Native acquired features required; legacy bridge rejected')
        packet=value.features;packet.validate()
        if packet.layout_sha256!=self.layout_sha256 or packet.producer_sha256!=self.producer_sha256 or not np.array_equal(packet.packed(),visual):
            raise ValueError('Native policy/measurement layout or values differ')
        # The source owns its arrays; the session owns an isolated measurement.
        from ..data.native_features import from_wire,to_wire
        packet=from_wire(to_wire(packet))
        return NativeQueryResult(visual.copy(),float(value.accel_mean),float(value.accel_std),value.score_samples,packet)

    def _verify_return(self,expected,actual):
        super()._verify_return(expected,actual)
        a,b=expected.features,actual.features
        if (a.layout_sha256,a.input_sha256,a.history_sha256,a.model_sha256)!=(b.layout_sha256,b.input_sha256,b.history_sha256,b.model_sha256):
            raise ValueError('Native return provenance differs')
        if [v is not None for v in a.values]!=[v is not None for v in b.values]:
            raise ValueError('Native camera availability changed on return')

    def observation(self):
        obs=super().observation();obs.pop('visual')
        n=len(obs['geometry']);mask=np.zeros((n,len(self.streams)),dtype=bool)
        for i,stream in enumerate(self.streams):
            values=np.zeros((n,4*stream.channels),np.float32)
            for j,(_,value) in enumerate(self.records):
                feature=value.features.values[i]
                if feature is not None:values[j]=unit_vector(feature);mask[j,i]=True
            obs[f'feature_{i}']=values
        obs['feature_mask']=mask
        return obs


class NativeContinuousSearch(_NativeFeatureSession,ContinuousSearch):
    """Native acquired streams with unchanged continuous STOP/motion/reward."""


class NativeLocalSearch(_NativeFeatureSession,LocalViewSearch):
    """Same frozen local candidate/search rule, using real target observations."""


def padded_native_observation(obs,*,history_size=7):
    n=len(obs['geometry'])
    if not 1<=n<=history_size:raise ValueError('Native complete-history capacity exceeded')
    row_keys={'geometry','scores','flags','feature_mask'}|{k for k in obs if k.startswith('feature_') and k!='feature_mask'}
    result={k:np.array(v,copy=True) for k,v in obs.items() if k not in row_keys|{'action_mask'}}
    for key in row_keys:
        array=np.asarray(obs[key])
        if len(array)!=n:raise ValueError('Native feature history rows differ')
        out=np.zeros((history_size,*array.shape[1:]),array.dtype);out[:n]=array;result[key]=out
    result['action_mask']=np.zeros(history_size+1,np.int8);result['action_mask'][:n+1]=obs['action_mask']
    return result


def native_gym(streams,*,producer_sha256,feature_layout=None):
    """Import optional Gym only when explicitly constructing a collector."""
    layout=layout_spec(tuple(streams)) if feature_layout is None else feature_layout
    if read_layout(layout)!=tuple(streams):raise ValueError('Explicit gym layout streams differ')
    import gymnasium as gym
    from .continuous_search_gym import ContinuousSearchGym
    class NativeSearchGym(ContinuousSearchGym):
        def __init__(self):
            self.core=None;self.expected_layout=digest(layout)
            spaces=dict(geometry=gym.spaces.Box(-np.inf,np.inf,(7,12),np.float32),
                scores=gym.spaces.Box(-np.inf,np.inf,(7,2),np.float32),flags=gym.spaces.Box(-np.inf,np.inf,(7,4),np.float32),
                feature_mask=gym.spaces.MultiBinary((7,len(streams))),
                joint_state=gym.spaces.Box(-1,1,(6,),np.float32),delta_low=gym.spaces.Box(-10,10,(2,),np.float64),
                delta_high=gym.spaces.Box(-10,10,(2,),np.float64),action_mask=gym.spaces.MultiBinary(8),
                **{'global':gym.spaces.Box(-np.inf,np.inf,(6,),np.float32)})
            spaces.update({f'feature_{i}':gym.spaces.Box(-np.inf,np.inf,(7,4*s.channels),np.float32) for i,s in enumerate(streams)})
            self.observation_space=gym.spaces.Dict(spaces)
            self.action_space=gym.spaces.Dict(stop=gym.spaces.Discrete(2),delta_deg=gym.spaces.Box(-10,10,(2,),np.float64))

        def _validate_session(self,core):
            if (not isinstance(core,NativeContinuousSearch) or core.status!='NEW'
                    or core.config.query_budget!=8 or core.layout_sha256!=self.expected_layout or core.producer_sha256!=producer_sha256):
                raise ValueError('Fresh native session and matching policy contract required')

        def _observation(self,obs):return padded_native_observation(obs)
    return NativeSearchGym()
