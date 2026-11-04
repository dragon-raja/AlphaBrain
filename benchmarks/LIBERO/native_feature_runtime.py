"""Bounded same-forward target features; model-specific taps end at this layer."""
import math
from copy import deepcopy
from AlphaBrain.research.dsol.data.native_features import read_layout,NativeFeatures,to_wire
from AlphaBrain.research.dsol.execution.manifest import digest


class NativeRuntimeFeatures:
    def __init__(self,spec,*,family,model_sha256):
        if (set(spec)!={'schema','family','tap','shape','grid','layout'}
                or spec['schema']!='LIBERO_NATIVE_FEATURE_TAP_V1' or spec['family']!=family):
            raise ValueError('Explicit family-specific native feature tap required')
        self.spec=deepcopy(spec);self.streams=read_layout(spec['layout']);self.model_sha256=model_sha256
        self.shape=tuple(spec['shape']);self.grid=tuple(spec['grid'])
        if (len(self.shape)!=3 or any(type(n) is not int or n<1 for n in self.shape)
                or len(self.grid)!=3 or self.grid[0]!=1 or min(self.grid[1:])<2
                or any(type(n) is not int or n<1 for n in self.grid) or math.prod(self.grid)!=self.shape[1]):
            raise ValueError('Declared single-frame native token grid required')
        if family=='fastwam':
            if (spec['tap']!='video_expert.prepare.first_frame' or self.shape[0]!=1 or len(self.streams)!=1
                    or self.streams[0].fusion_scope!='joint'):
                raise ValueError('FastWAM current-frame prepare tokens remain joint')
        elif family=='groot_n15':
            if (spec['tap']!='eagle.extract_vision_feature' or self.shape[0]!=len(self.streams)
                    or any(s.fusion_scope!='per_view' for s in self.streams)):
                raise ValueError('Eagle native vision retains actual view order')
        else:raise ValueError('Unregistered native model family')
        if any(s.channels!=self.shape[-1] for s in self.streams):raise ValueError('Native channel dimensions differ')
        # This LIBERO processor admits exactly these two real sensors. GR1 is
        # a separate processor contract, never a head duplicated into a wrist.
        cameras=tuple(k for s in self.streams for k in s.camera_ids)
        roles=tuple(k for s in self.streams for k in s.camera_roles)
        if cameras!=('external','wrist') or roles!=('movable','context'):
            raise ValueError('LIBERO native processor camera order differs')

    def validate_request(self,example,request):
        from .local_search_rollout import input_identity
        if (not isinstance(request,dict) or set(request)!={'query_id','input_sha256','history_sha256'}
                or not isinstance(request['query_id'],str) or not request['query_id'].strip()
                or request['input_sha256']!=input_identity(example)):
            raise ValueError('Fresh native query/input provenance required')
        for key in ('input_sha256','history_sha256'):
            if not isinstance(request[key],str) or len(request[key])!=64 or any(c not in '0123456789abcdef' for c in request[key]):
                raise ValueError('Native history/input hash required')

    def call(self,model,infer,*,example,request):
        self.validate_request(example,request)
        from AlphaBrain.model.native_features import NativeVisionCapture,pool_spatial_grid
        if self.spec['family']=='fastwam':
            from AlphaBrain.model.fastwam_features import FastWAMFeatureCapture
            with FastWAMFeatureCapture(model.video_expert,query_id=request['query_id'],token_shape=self.shape) as tap:
                result=infer()
            tokens=tap.features(query_id=request['query_id'])
        else:
            with NativeVisionCapture(model.vlm_backbone.vlm,shape=self.shape) as tap:result=infer()
            tokens=tap.features()
        values=tuple(pool_spatial_grid(tokens[i:i+1],self.grid).cpu().numpy().reshape(-1).copy()
                     for i in range(len(self.streams)))
        packet=NativeFeatures(self.spec['layout'],request['query_id'],request['input_sha256'],
            request['history_sha256'],self.model_sha256,values,digest(self.spec))
        return result,to_wire(packet)
