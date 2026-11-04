"""Opt-in native-feature history for the existing continuous PPO objective.

Settled real image grids and explicitly grouped learned-query sequences are
supported through distinct hashed producer/layout contracts. Joint features stay joint. This module never loads an operation model or changes legacy pi05
weights/observations. Runtime capture-on/off admission is still required.
"""
import math
import torch
from torch import nn
from AlphaBrain.model.native_features import pool_spatial_grid
from ..data.native_observation import validate_feature_metadata,NativeStream,validate_streams
from .continuous_search_policy import ContinuousActorCritic


def capture_fastwam_query(video_expert, infer, spec, observation, *, grid):
    """Candidate adapter entry: one existing forward, then validated pooling.

    The caller owns dispatch/budget and must supply a serialized eager forward.
    This function is opt-in; no production runtime calls it by default.
    """
    from AlphaBrain.model.fastwam_features import FastWAMFeatureCapture
    observation.validate()
    if spec.source_kind!='target_native' or spec.fusion_scope!='joint':
        raise ValueError('FastWAM capture is a target-native joint representation')
    # Reject all metadata/grid faults before spending a model call.
    validate_feature_metadata(spec,observation,query_id=observation.query_id,
        history_sha256=observation.history_sha256,observed_shape=spec.shape,observed_dtype=spec.dtype)
    if (len(spec.shape)!=3 or spec.shape[0]!=1 or not isinstance(grid,tuple)
            or len(grid)!=3 or any(type(n) is not int or n<1 for n in grid)
            or grid[0]!=1 or min(grid[1:])<2 or math.prod(grid)!=spec.shape[1]):
        raise ValueError('Explicit single-frame spatial token grid required')
    with FastWAMFeatureCapture(video_expert,query_id=observation.query_id,token_shape=spec.shape) as tap:
        result=infer()
    pooled=pool_native_grid(tap.features(query_id=observation.query_id),spec,observation,
        query_id=observation.query_id,history_sha256=observation.history_sha256,grid=grid)
    return result,pooled


def pool_native_grid(tokens, spec, observation, *, query_id, history_sha256, grid):
    """Detached 2x2 pooling over an explicitly declared THW token order.

    Grid order must be established by the model adapter, never inferred from
    token count. No temporal or generated tokens, camera splitting or padding.
    """
    dtype=str(tokens.dtype).removeprefix('torch.')
    validate_feature_metadata(spec,observation,query_id=query_id,history_sha256=history_sha256,
                              observed_shape=tuple(tokens.shape),observed_dtype=dtype)
    if (not isinstance(grid,tuple) or len(grid)!=3 or grid[0]!=1
            or any(type(n) is not int or n<1 for n in grid) or min(grid[1:])<2
            or tokens.ndim!=3 or tokens.shape[0]!=1 or math.prod(grid)!=tokens.shape[1]
            or not torch.isfinite(tokens).all()):
        raise ValueError('Explicit finite single-frame spatial token grid required')
    return pool_spatial_grid(tokens,grid)


class NativeHistory(nn.Module):
    def __init__(self,streams,width=128,heads=4,layers=2):
        super().__init__();self.streams=tuple(streams)
        if not streams or width<1 or heads<1 or width%heads or layers<1:
            raise ValueError('Nonempty streams and valid attention dimensions required')
        validate_streams(self.streams)
        self.projections=nn.ModuleList([nn.Sequential(nn.Linear(4*s.channels,width),nn.LayerNorm(width),nn.GELU()) for s in streams])
        self.identities=nn.Parameter(torch.zeros(len(streams),width))
        self.fusion=nn.Linear(width,1)
        self.record=nn.Linear(width+12+2+4,width)
        self.context=nn.Sequential(nn.Linear(12,width),nn.Tanh())
        self.history=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,heads,width*2,dropout=0.,
            activation='gelu',batch_first=True),layers,enable_nested_tensor=False)

    def forward(self,obs):
        keys={'geometry','scores','flags','global','action_mask','joint_state','delta_low','delta_high','feature_mask'}
        keys|={f'feature_{i}' for i in range(len(self.streams))}
        if set(obs)!=keys:raise ValueError('Native observation schema; no fake wrist or privileged fields')
        geom=obs['geometry']
        if geom.ndim!=3 or geom.shape[-1]!=12:raise ValueError('Geometry shape')
        b,n=geom.shape[:2];mask=obs['feature_mask']
        shapes={'scores':(b,n,2),'flags':(b,n,4),'global':(b,6),'joint_state':(b,6),
                'action_mask':(b,n+1),'delta_low':(b,2),'delta_high':(b,2),'feature_mask':(b,n,len(self.streams))}
        shapes.update({f'feature_{i}':(b,n,4*s.channels) for i,s in enumerate(self.streams)})
        if any(tuple(obs[k].shape)!=v for k,v in shapes.items()):raise ValueError('Native observation shape')
        if mask.dtype!=torch.bool or any(not torch.isfinite(v).all() for v in obs.values()):
            raise ValueError('Finite inputs and boolean availability required')
        visited=obs['flags'][...,0]>.5
        movable=torch.tensor(['movable' in s.camera_roles for s in self.streams],device=mask.device)
        if (not visited.any(1).all() or (mask&~visited[...,None]).any()
                or not ((mask&movable).any(-1)|~visited).all()):
            raise ValueError('Each acquired observation needs a real movable view')
        projected=[]
        for i,proj in enumerate(self.projections):
            # Only operation features are detached; search-side projections learn.
            x=obs[f'feature_{i}'].detach();v=x.new_zeros(b,n,self.record.out_features)
            v[mask[...,i]]=proj(x[mask[...,i]])+self.identities[i]
            projected.append(v)
        features=torch.stack(projected,dim=2)
        logits=self.fusion(features).squeeze(-1).masked_fill(~mask,-1e9)
        weights=torch.softmax(logits,dim=-1)*mask
        fused=(features*weights[...,None]).sum(2)
        score=torch.where(visited[...,None],obs['scores'],torch.zeros_like(obs['scores']))
        records=self.record(torch.cat([fused,geom,score,obs['flags']],-1))
        records=records+self.context(torch.cat([obs['global'],obs['joint_state']],-1))[:,None]
        memory=self.history(records,src_key_padding_mask=~visited)
        return (memory*visited[...,None]).sum(1)/visited.sum(1,keepdim=True)


class NativeContinuousActorCritic(ContinuousActorCritic):
    """Same STOP/motion distribution and PPO loss, separate native histories."""
    def __init__(self,streams,width=128,heads=4,layers=2):
        nn.Module.__init__(self)
        self.actor_history=NativeHistory(streams,width,heads,layers)
        self.critic_history=NativeHistory(streams,width,heads,layers)
        self.stop_head=nn.Linear(width,1);self.mean_head=nn.Linear(width,2)
        self.log_std=nn.Parameter(torch.full((2,),-.5))
        self.value_head=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
        nn.init.zeros_(self.stop_head.weight);nn.init.zeros_(self.stop_head.bias)
        nn.init.zeros_(self.mean_head.weight);nn.init.zeros_(self.mean_head.bias)

    def load_legacy_encoder(self,*args,**kwargs):
        raise ValueError('Native representation requires an explicit weight mapping; no implicit legacy load')
