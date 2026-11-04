"""Portable native-feature wire contract, independent of Torch and simulators.

The adapter must establish raw-token provenance and pooling equivalence. This
contract validates already-pooled acquired features, never invents camera views.
"""
from dataclasses import asdict,dataclass
from copy import deepcopy
import numpy as np
from .native_observation import NativeStream,validate_streams
from .search_replay import QueryResult
from ..execution.manifest import digest

SCHEMA='NATIVE_SEARCH_FEATURE_LAYOUT_V1'


def layout_spec(streams):
    validate_streams(streams)
    return dict(schema=SCHEMA,pooling='explicit_spatial_2x2',streams=[asdict(s) for s in streams])


def read_layout(value):
    if (set(value)!={'schema','pooling','streams'} or
            (value['schema'],value['pooling']) not in
            ((SCHEMA,'explicit_spatial_2x2'),('NATIVE_SEQUENCE_FEATURE_LAYOUT_V1','explicit_ordered_query_groups_4'))):
        raise ValueError('Explicit native feature layout required')
    streams=tuple(NativeStream(**dict(s,camera_ids=tuple(s['camera_ids']),camera_roles=tuple(s['camera_roles']))) for s in value['streams'])
    validate_streams(streams)
    return streams


@dataclass(frozen=True)
class NativeFeatures:
    layout: dict
    query_id: str
    input_sha256: str
    history_sha256: str
    model_sha256: str
    values: tuple
    producer_sha256: str
    # None denotes a genuinely absent stream, never a fabricated zero camera.

    def validate(self):
        streams=read_layout(self.layout)
        if not isinstance(self.query_id,str) or not self.query_id.strip():raise ValueError('Feature query identity required')
        for value in (self.input_sha256,self.history_sha256,self.model_sha256,self.producer_sha256):
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Feature input/history/model identity required')
        if not isinstance(self.values,tuple) or len(self.values)!=len(streams):raise ValueError('Feature streams differ')
        movable=False
        for stream,value in zip(streams,self.values):
            if value is None:continue
            array=np.asarray(value)
            if array.shape!=(4*stream.channels,) or array.dtype!=np.float32 or not np.isfinite(array).all():
                raise ValueError('Finite float32 pooled native stream required')
            movable|='movable' in stream.camera_roles
        if not movable:raise ValueError('Acquired movable camera features required')
        return streams

    def packed(self):
        streams=self.validate()
        return np.concatenate([np.zeros(4*s.channels,np.float32) if v is None else np.array(v,copy=True)
                               for s,v in zip(streams,self.values)])

    @property
    def layout_sha256(self):return digest(self.layout)

    def metadata(self):
        self.validate()
        return dict(schema='NATIVE_SEARCH_FEATURE_PACKET_V1',layout=deepcopy(self.layout),layout_sha256=self.layout_sha256,
            query_id=self.query_id,input_sha256=self.input_sha256,history_sha256=self.history_sha256,
            model_sha256=self.model_sha256,producer_sha256=self.producer_sha256,mask=[v is not None for v in self.values])


@dataclass(frozen=True)
class NativeQueryResult(QueryResult):
    features: NativeFeatures


def from_wire(value):
    """Runtime-local wire packet; explicit masks survive disk/replay boundaries."""
    required={'schema','layout','layout_sha256','query_id','input_sha256','history_sha256','model_sha256','producer_sha256','mask','values'}
    if set(value)!=required or value['schema']!='NATIVE_SEARCH_FEATURE_PACKET_V1':raise ValueError('Native feature wire schema differs')
    streams=read_layout(value['layout']);mask=value['mask']
    if (len(mask)!=len(streams) or any(type(v) is not bool for v in mask)
            or len(value['values'])!=len(streams) or digest(value['layout'])!=value['layout_sha256']):
        raise ValueError('Native feature layout/mask differs')
    arrays=[]
    for present,array in zip(mask,value['values']):
        if not present and array is not None:raise ValueError('Missing stream cannot contain an image substitute')
        arrays.append(None if not present else np.array(array,copy=True))
    packet=NativeFeatures(deepcopy(value['layout']),value['query_id'],value['input_sha256'],value['history_sha256'],value['model_sha256'],tuple(arrays),value['producer_sha256'])
    packet.validate();return packet


def to_wire(packet):
    return dict(packet.metadata(),values=[None if v is None else np.array(v,copy=True) for v in packet.values])
