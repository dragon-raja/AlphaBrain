"""Native RLDX joint cognition for search; real per-action history for execution.

The 64 learned queries are four explicitly ordered groups, never an 8x8 image.
At a static acquisition pose the original reset padding fills the four frames.
Operation calls require begin_operation and one observation per executed action.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read
from AlphaBrain.research.dsol.execution.manifest import digest
from AlphaBrain.research.dsol.data.native_features import NativeFeatures, to_wire
from AlphaBrain.research.dsol.data.query_features import pool_four_query_groups
from .rldx_runtime import RLDXRuntime, LiberoHistory, validate_contract


def noise_seed(noise):
    x = np.asarray(noise, dtype='<f4')
    if x.shape != (10, 7) or not np.isfinite(x).all(): raise ValueError('Explicit paired scientific noise required')
    return int.from_bytes(hashlib.sha256(b'RLDX_SEARCH_NATIVE_NOISE_V1\0'+x.tobytes()).digest()[:8], 'little') % 2**63


def feature_spec():
    return dict(schema='RLDX_RESET_COGNITION_FEATURE_V1', family='rldx',
        tap='action_head_condition.backbone_features', shape=[1, 64, 4096],
        history='native_reset_padded_four_frames', video_deltas=[-6,-4,-2,0],
        excluded_state_feature='state_features (not supplied to the search policy)',
        layout=dict(schema='NATIVE_SEQUENCE_FEATURE_LAYOUT_V1', pooling='explicit_ordered_query_groups_4',
            streams=[dict(name='joint_cognition', channels=4096, fusion_scope='joint',
                camera_ids=['external','wrist'], camera_roles=['movable','context'])]))


def visual_feature_spec():
    """Current-frame native spatial tokens before any language-layer fusion."""
    return dict(schema='RLDX_RESET_VISUAL_FEATURE_V1',family='rldx',
        tap='backbone.qwen_model.model.get_image_features',
        history='native_reset_padded_four_frames',video_deltas=[-6,-4,-2,0],
        frame_order='frame_major_camera_minor',selected_frame=3,frames=4,merge_size=2,
        native_camera_keys=['front_view','left_wrist_view'],
        temporal_receptive_field='native_motion_module_preserved',
        layout=dict(schema='NATIVE_SEARCH_FEATURE_LAYOUT_V1',pooling='explicit_spatial_2x2',
            streams=[dict(name=c,channels=4096,fusion_scope='per_view',camera_ids=[c],camera_roles=[role])
                     for c,role in [('external','movable'),('wrist','context')]]))


class RLDXSearchRuntime:
    search_history_rule = 'native_reset_padded_four_frames'

    def __init__(self, config, output, limit):
        from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
        reject_production('benchmarks/LIBERO/rldx_search_runtime.py:RLDXSearchRuntime.__init__')
        if config['native_features'] not in (feature_spec(),visual_feature_spec()): raise ValueError('Unreviewed feature/history mapping')
        self.spec = config['native_features']; self.output=Path(output); self.limit=limit; self.calls=0; self.failed=False
        m=config['models'][config['operation_model']]
        c=validate_contract(m['runtime_contract'], m['runtime_contract_sha256'])
        if m['weights_sha256'] != digest(c['checkpoint_shards']): raise ValueError('Joint checkpoint identity differs')
        self.model_sha256=m['weights_sha256']; self.history=None
        child=self.output/'native-rldx'; child.mkdir(exist_ok=False)
        self.native=RLDXRuntime(m['runtime_contract'], m['runtime_contract_sha256'], child, limit)

    def begin_operation(self, example):
        if self.history is not None: raise ValueError('Previous operation was not closed')
        self.history=LiberoHistory(example)

    def observe_action(self, example):
        if self.history is None: raise ValueError('No active per-action history')
        self.history.append(example)

    def end_operation(self): self.history=None

    def infer(self, example, noise, identity, *, capture=False, native_request=None):
        from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
        from .local_search_rollout import input_identity
        reject_production('benchmarks/LIBERO/rldx_search_runtime.py:RLDXSearchRuntime.infer')
        if self.failed or self.calls>=self.limit: raise ValueError('Unknown outcome or exhausted RLDX search budget')
        if capture:
            if self.history is not None: raise ValueError('Acquisition cannot use a manipulation history')
            if (not isinstance(native_request,dict) or set(native_request)!= {'query_id','input_sha256','history_sha256'}
                    or native_request['input_sha256']!=input_identity(example)):
                raise ValueError('Fresh sensor and history provenance required')
            history=LiberoHistory(example)
        else:
            if native_request is not None or self.history is None: raise ValueError('Operation needs explicit native temporal history')
            history=self.history
        if self.native.output != self.output/'native-rldx':
            child=self.output/'native-rldx'; child.mkdir(exist_ok=False)
            self.native.bind_task(child,self.limit)
        from .controlled_rldx_operation import input_stamp
        stamp=input_stamp(history)
        self.calls+=1; self.failed=True; seed=noise_seed(noise)
        visual=self.spec==visual_feature_spec()
        value=self.native.infer(history,seed,identity,instrument=True,features=capture and not visual,
                                **({'visual_features':True} if capture and visual else {}))
        result=dict(actions=value['actions'], normalized_actions=value['normalized_actions'],
            trace=value['velocities'][:,0,:8,:7].copy(), seed=seed, scores=value['scores'],native_input_sha256=stamp)
        if capture:
            features=(tuple(value['visual_features'][camera].reshape(-1) for camera in ('external','wrist')) if visual else
                      (pool_four_query_groups(value['backbone_features'],queries=64,channels=4096),))
            packet=NativeFeatures(self.spec['layout'],native_request['query_id'],native_request['input_sha256'],
                native_request['history_sha256'],self.model_sha256,features,digest(self.spec))
            result.update(native_features=to_wire(packet),score=result['scores']['3'])
        with (self.output/'calls.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(index=self.calls,**identity,seed=str(seed)))+'\n');stream.flush()
        self.failed=False
        return result


def rollout_rldx_episode(request, infer, example, bank, state_key, repeat, identity, *, runtime, record):
    """Preserve all eight individual sensor updates inside each 16-action chunk."""
    from .local_search_rollout import input_identity
    runtime.begin_operation(example)
    steps=0; success=False; chunks=[]; actions=[]; inputs=[]; noises=[]; seeds=[]; physics=[]; offsets=[];native_inputs=[];normalized=[]
    try:
        while steps<720 and not success:
            noise=bank.get(state_key,repeat,len(chunks)); address=dict(identity,state_key=state_key,repeat=repeat,replan=len(chunks))
            stamp=input_identity(example); record(dict(kind='inference_attempt',identity=address,input_sha256=stamp,noise_sha256=noise['noise_sha256']))
            value=infer(example,noise['noise'],address,capture=False); chunk=np.asarray(value['actions'])
            if chunk.shape!=(16,7) or not np.isfinite(chunk).all(): raise ValueError('Native 16x7 decoded actions required')
            native_inputs.append(value['native_input_sha256']);normalized.append(value['normalized_actions']);
            offsets.append(steps); chunks.append(chunk.copy());inputs.append(stamp);noises.append(noise['noise_sha256']);seeds.append(value['seed'])
            for action in chunk[:8]:
                reply=request(dict(kind='step',actions=action[None]))
                if reply['executed']!=1: raise ValueError('Exactly one real sensor-history update required')
                example=reply['example']; runtime.observe_action(example);steps+=1;actions.append(action.copy());physics.append(reply['physics']);success=reply['success']
                if success:break
            record(dict(kind='executed',identity=address,steps=steps,success=bool(success)))
    finally:runtime.end_operation()
    return dict(success=bool(success),steps=steps,replans=len(chunks),horizon=720,repeat=repeat,state_key=state_key,
        input_sha256=inputs,execution_noise_sha256=noises,final_input_sha256=input_identity(example)),dict(
        native_input_sha256=np.asarray(native_inputs),normalized_actions=np.asarray(normalized),
        actions=np.asarray(actions),full_decoded_action_chunks=np.asarray(chunks),replan_action_offsets=np.asarray(offsets),
        native_seeds=np.asarray(seeds,dtype=np.int64),physics=np.asarray(physics))
