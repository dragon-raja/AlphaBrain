"""Synthetic renderer/runtime for exercising real collectors; no simulator/VLA."""
from contextlib import contextmanager
from pathlib import Path
import json
import numpy as np
from AlphaBrain.research.dsol.data.native_features import NativeFeatures,to_wire,read_layout
from benchmarks.LIBERO.local_search_rollout import input_identity
from AlphaBrain.research.dsol.execution.manifest import digest


@contextmanager
def renderer(path,config,initial,output):
    pose=np.eye(4)
    def receipt():return dict(camera_to_world=pose.copy(),physical_state_id='synthetic-fixed',settled=True)
    def request(message):
        nonlocal pose
        if message['kind']=='continuous_info':return dict(reference=np.eye(4),receipt=receipt())
        if message['kind']=='move_pose':pose=np.array(message['camera_to_world'],copy=True);return dict(receipt=receipt())
        if message['kind'] not in ('query_pose','begin_execution'):raise AssertionError('No hidden simulation action permitted')
        return dict(receipt=receipt(),example=dict(image=[pose.copy(),np.ones((2,2))],state=np.zeros(7),
            camera_intrinsics=np.eye(3),camera_to_world_opencv=pose.copy(),instruction='synthetic',lang='synthetic'))
    yield request


class Runtime:
    def __init__(self,layout,limit,*,warmup=False,output=None):
        self.layout=layout;self.calls=0;self.limit=limit;self.native_needs_warmup=warmup;self.fail=False;self.stale=False
        self.output=Path(output) if output is not None else None
    def infer(self,example,noise,identity,capture=True,*,native_request=None,warmup=False):
        self.calls+=1
        if self.output is not None:
            with (self.output/'calls.jsonl').open('a') as stream:stream.write(json.dumps(dict(index=self.calls,**identity))+'\n')
        if self.fail:raise RuntimeError('Unknown synthetic call outcome')
        if warmup:self.native_needs_warmup=False;return dict(actions=np.zeros((10,7)))
        if not capture:return dict(actions=np.zeros((10,7)))
        assert capture and native_request is not None
        streams=read_layout(self.layout);pose=example['camera_to_world_opencv']
        packet=NativeFeatures(self.layout,'wrong' if self.stale else native_request['query_id'],input_identity(example),
            native_request['history_sha256'],'e'*64,tuple(np.arange(4*s.channels,dtype=np.float32)+pose[0,0] for s in streams),digest(dict(layout=self.layout)))
        return dict(actions=np.zeros((10,7)),trace=np.zeros((4,10,7)),score=1.+abs(pose[0,1]),native_features=to_wire(packet))
