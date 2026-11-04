"""Real fresh render -> budgeted native infer -> provenance checked features.

Shares motion/receipts with the legacy source. No feature tower or extra score
calibration, and no reuse until pixel/state/pose identity has been checked.
"""
import json
import numpy as np
from .local_view_runtime import RenderedViewSource
from .local_search_rollout import input_identity
from .search_domain import exact_input
from AlphaBrain.research.dsol.data.native_features import NativeQueryResult,from_wire
from AlphaBrain.research.dsol.acquisition.local_view_search import ViewMeasurement
from AlphaBrain.research.dsol.execution.manifest import digest


class NativeRenderedViewSource(RenderedViewSource):
    def __init__(self,*args,expected_model_sha256,expected_producer_sha256,**kwargs):
        super().__init__(*args,**kwargs);self.expected_model_sha256=expected_model_sha256;self.expected_producer_sha256=expected_producer_sha256

    def query(self):
        if self.failed:raise ValueError('Native source previously failed; no automatic retry')
        self.queries+=1
        try:
            reply=self.request(dict(kind='query_pose'));receipt=self._receipt(reply['receipt']);example=reply['example']
            key=tuple(receipt.camera_to_world.reshape(-1));sha=input_identity(example)
            if key in self.references:exact_input(self.references[key][0],example)
            wrapped=getattr(self.runtime,'runtime',self.runtime)
            if getattr(wrapped,'native_needs_warmup',False):
                self.runtime.infer(example,self.noises[0],dict(query=0,member=0,purpose='native_graph_warmup'),capture=False,warmup=True)
            values=[];packets=[]
            history=digest(dict(physical_state_id=self.physical,operation_history=getattr(wrapped,'search_history_rule','frozen_single_current_frame')))
            for m,noise in enumerate(self.noises):
                request=dict(query_id=f'{digest(str(self.output.resolve()))}:{self.queries}:{m}',input_sha256=sha,history_sha256=history)
                value=self.runtime.infer(example,noise,dict(case=self.output.name,query=self.queries,member=m),capture=True,native_request=request)
                packet=from_wire(value['native_features'])
                if (packet.query_id!=request['query_id'] or packet.input_sha256!=sha or packet.history_sha256!=history
                        or packet.model_sha256!=self.expected_model_sha256 or packet.producer_sha256!=self.expected_producer_sha256):
                    raise ValueError('Stale or wrong native query features')
                values.append(value);packets.append(packet)
            a,b=packets
            if (a.layout_sha256!=b.layout_sha256 or a.model_sha256!=b.model_sha256
                    or [v is not None for v in a.values]!=[v is not None for v in b.values]
                    or any(not np.array_equal(x,y) for x,y in zip(a.values,b.values))):
                raise ValueError('Current-frame native features unexpectedly depend on scoring noise')
            if key in self.references:
                for old,new in zip(self.references[key][1],values):
                    for name in ('actions','trace','score'):
                        if not np.array_equal(old[name],new[name]):raise ValueError('Native return prediction differs')
                    previous=from_wire(old['native_features'])
                    if (previous.layout_sha256!=a.layout_sha256 or previous.model_sha256!=a.model_sha256
                            or not np.array_equal(previous.packed(),a.packed())):
                        raise ValueError('Native return feature provenance differs')
            else:self.references[key]=(example,values)
            saved=dict(robot_state=example['state'],camera_intrinsics=example['camera_intrinsics'],camera_to_world_opencv=example['camera_to_world_opencv'])
            for i,image in enumerate(example['image']):saved[f'image_{i}']=image
            for m,value in enumerate(values):
                for name in ('actions','trace','score','full_actions','timesteps','deltas'):
                    if name in value:saved[f'{name}_{m}']=value[name]
                saved[f'native_metadata_{m}']=json.dumps(packets[m].metadata(),sort_keys=True)
                for i,array in enumerate(packets[m].values):
                    if array is not None:saved[f'native_feature_{m}_{i}']=array
            np.savez_compressed(self.output/f'query-{self.queries:02d}.npz',**saved)
            self.last_example,self.last_values=example,values
            scores=np.asarray([v['score'] for v in values])
            return ViewMeasurement(receipt,NativeQueryResult(a.packed(),float(scores.mean()),float(scores.std()),2,a))
        except BaseException:
            self.failed=True;raise
