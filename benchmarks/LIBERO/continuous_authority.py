"""Bound LIBERO block agent: live grants, per-task locks and fixed executor.

Imports only CPU primitives until a verified real GO. CPU contract mode cannot
enter the scientific executor. Old fleet/worker entrypoints remain closed.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr, nullcontext
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

from AlphaBrain.research.dsol.execution.authority_wire import Wire, identity, require_local
from AlphaBrain.research.dsol.execution.manifest import digest, file_digest, task_key
from AlphaBrain.research.dsol.execution.queue import _durable_write


def source_identity():
    from AlphaBrain.research.dsol.execution.source_release import source_tree_identity
    root = Path(__file__).resolve().parents[2]
    return source_tree_identity(root, ('AlphaBrain/common', 'AlphaBrain/model', 'AlphaBrain/research/dsol', 'benchmarks/LIBERO'))


def load_binding(path):
    binding = json.loads(Path(path).read_text())
    if binding.get('schema') != 'LIBERO_LIVE_BINDING_V1' or binding.get('mode') not in ('libero','cpu_contract'):
        raise ValueError('Reviewed model binding required')
    if type(binding.get('resident_workers', False)) is not bool:
        raise ValueError('Explicit resident binding required')
    if (binding.get('resident_workers') and binding['mode']=='libero'
            and binding.get('scientific_adapter') not in ('controlled_rldx_operation_v1','controlled_rldx_v1','native_view_panel_v1','controlled_training_v1','controlled_recovery_v1','controlled_admission_v1','controlled_groot_admission_v1','controlled_target_evaluation_v1','controlled_target_recovery_v1','controlled_target_panel_v1')):
        raise ValueError('Resident model adapter is not registered')
    if type(binding.get('resident_simulator',False)) is not bool:
        raise ValueError('Explicit resident simulator flag required')
    if binding.get('resident_simulator') and not binding.get('resident_workers'):
        raise ValueError('Simulator residency requires a resident authority worker')
    return binding


def node_for(binding):
    from AlphaBrain.research.dsol.execution.model_agent import node_for as shared
    return shared(binding)


def verify_node(binding, *, sources=True):
    from AlphaBrain.research.dsol.execution.model_agent import verify_node as shared
    return shared(binding,sources=sources)


def prepare(root,binding):
    from AlphaBrain.research.dsol.execution.model_agent import prepare as shared
    return shared(root,binding)


def validate_scientific_spec(binding, key):
    """CPU-only identity/budget proof, repeated before every real GO."""
    if binding.get('scientific_adapter')=='controlled_rldx_operation_v1':
        from .controlled_rldx_operation import validate_spec
        return validate_spec(binding,key)
    if binding.get('scientific_adapter')=='controlled_rldx_v1':
        from .controlled_rldx import validate_spec
        return validate_spec(binding,key)
    if binding.get('scientific_adapter')=='controlled_groot_admission_v1':
        from .controlled_groot_admission import validate_spec
        return validate_spec(binding,key)
    if binding.get('scientific_adapter')=='controlled_target_panel_v1':
        from .controlled_target_panel import validate_spec
        return validate_spec(binding,key)
    if binding.get('scientific_adapter')=='controlled_target_recovery_v1':
        from .controlled_target_recovery import validate_spec
        return validate_spec(binding,key)
    if binding.get('scientific_adapter')=='controlled_target_evaluation_v1':
        from .controlled_target_evaluation import validate_spec
        return validate_spec(binding,key)
    if binding.get('scientific_adapter')=='controlled_admission_v1':
        from .controlled_admission import validate_spec
        return validate_spec(binding,key)
    if binding.get('scientific_adapter')=='controlled_recovery_v1':
        from .controlled_recovery import validate_spec
        return validate_spec(binding,key)
    if binding.get('scientific_adapter')=='controlled_training_v1':
        from .controlled_training import validate_spec
        return validate_spec(binding,key)
    if binding.get('scientific_adapter')=='native_view_panel_v1':
        from .native_view_panel import validate_spec
        return validate_spec(binding,key)
    if binding.get('scientific_adapter') is not None:
        raise ValueError('Unregistered scientific adapter')
    from .continuous_evaluation_fleet import request_for
    spec=binding['tasks'][key];req=spec['request'];block=req['block']
    if req != request_for(binding['campaign'],req['model'],block['index']):
        raise ValueError('Frozen scientific request changed')
    addresses=[dict(model=req['model'],task=r['task'],initial=r['state_key'],view=f"{r['start']}:{r['method']}",
                    stream='execution',repeat=s,noise=f'paired-seed-{s}')
               for r in block['rows'] for s in r['execution_repeats']]
    expected=dict(episodes=block['tasks'],score=block['limits'].get('score',0),paired_execution=0,nominal_execution=0)
    expected['paired_execution' if block['kind']=='paired' else 'nominal_execution']=block['limits']['execution']
    if (spec['limits']!=expected or spec['addresses']!=addresses or spec['host']!=req['host']
            or spec['stage']!=req['stage']
            or key!=task_key(spec['identity_experiment_id'],req['campaign_sha256'],addresses)):
        raise ValueError('Frozen scientific addresses or per-block budget differ')
    return expected


def validate_launch(binding, launch, *, root=None):
    if binding['mode']=='cpu_contract':
        from AlphaBrain.research.dsol.execution.model_agent import validate_cpu_launch
        return validate_cpu_launch(binding,launch,root=root)
    host,node = verify_node(binding)
    real = binding['mode'] == 'libero'
    if (launch.get('operation') != ('LIBERO_BLOCK' if real else 'CPU_BLOCK_CONTRACT')
            or launch.get('binding_sha256') != digest(binding)
            or launch.get('resident_workers', False) is not binding.get('resident_workers', False)):
        raise ValueError('Worker mode/binding differs from live grant')
    task = launch['task']; key = launch['key']; spec = binding['tasks'][key]
    expected = dict(key=key, addresses=spec['addresses'], output=spec['output'],
                    identity_experiment_id=spec['identity_experiment_id'], eligible_slots=spec['eligible_slots'],
                    payload=dict(binding_sha256=digest(binding),request=spec['request']))
    if (task != expected or digest(task) != launch['task_sha256'] or spec['host'] != host
            or launch['slot'] not in spec['eligible_slots'] or launch['limits'] != spec['limits']):
        raise ValueError('Unbound block, host, slot, output or budget')
    permit=launch['permit'];now=time.time()
    times=[permit.get(k) for k in ('issued_at','dispatch_until','finish_until')]
    if any(type(t) not in (int,float) or not math.isfinite(t) for t in times):raise ValueError('Invalid grant clock')
    if (permit.get('authorization') != ('EXPLICIT_MODEL_STAGE_V1' if real else 'CPU_SIMULATION_ONLY')
            or permit.get('owner') != launch['owner'] or permit.get('epoch') != launch['epoch']
            or type(launch['epoch']) is not int or launch['epoch']<1 or launch['attempt']!=1
            or permit.get('stage') != spec['stage'] or not times[0] <= now < times[1] < times[2]
            or (real and permit.get('binding_sha256') != digest(binding))):
        raise RuntimeError('Current stage grant missing/expired')
    if Path(task['output']).exists():
        raise RuntimeError('Existing output is never overwritten or automatically replayed')
    if real:
        validate_scientific_spec(binding,key)
        from AlphaBrain.research.dsol.execution import worker_gpu
        if root is None:raise RuntimeError('Bound worker store required for GPU ownership')
        result=worker_gpu.preflight(root,binding,launch,node)
        gpu=next((r for r in result['rows'] if str(r['index'])==launch['slot']['gpu_id']),None)
        # Only the current bound sibling may share this GPU; unknown occupants
        # and a loading sibling's outstanding peak reservation still block it.
        if gpu is None or not gpu['allowed']:
            reason = result.get('error') or (gpu or {}).get('category', 'Missing GPU telemetry')
            raise RuntimeError('Selected GPU/keepalive is not admitted: '+str(reason))
    return spec,node


def execute_block(binding, launch, guard, *, runtime_cache=None, simulator_pool=None):
    """Only the already-frozen paired/nominal loops run; no old coordinator."""
    if binding['mode'] != 'libero':raise RuntimeError('CPU contract cannot import a model executor')
    from AlphaBrain.research.dsol.execution.dispatch_policy import _live_model_scope
    panel=binding.get('scientific_adapter')=='native_view_panel_v1'
    collection=binding.get('scientific_adapter')=='controlled_training_v1'
    recovery=binding.get('scientific_adapter')=='controlled_recovery_v1'
    target_recovery=binding.get('scientific_adapter')=='controlled_target_recovery_v1'
    target_panel=binding.get('scientific_adapter')=='controlled_target_panel_v1'
    target=target_panel or target_recovery or binding.get('scientific_adapter')=='controlled_target_evaluation_v1'
    groot_admission=binding.get('scientific_adapter')=='controlled_groot_admission_v1'
    rldx_operation=binding.get('scientific_adapter')=='controlled_rldx_operation_v1'
    rldx_admission=binding.get('scientific_adapter')=='controlled_rldx_v1'
    admission=rldx_operation or rldx_admission or groot_admission or binding.get('scientific_adapter')=='controlled_admission_v1'
    if rldx_operation:from .controlled_rldx_operation import validate_block
    elif rldx_admission:from .controlled_rldx import validate_block
    elif groot_admission:from .controlled_groot_admission import validate_block
    elif target_panel:from .controlled_target_panel import validate_block
    elif target_recovery:from .controlled_target_recovery import validate_block
    elif target:from .controlled_target_evaluation import validate_block
    elif admission:from .controlled_admission import validate_block
    elif recovery:from .controlled_recovery import validate_block
    elif collection:from .controlled_training import validate_block
    elif panel:from .native_view_panel import validate_block
    else:from .continuous_evaluation_fleet import validate_block
    spec=binding['tasks'][launch['key']];req=spec['request'];output=Path(spec['output'])
    guard()
    os.environ['CUDA_VISIBLE_DEVICES']=launch['slot']['gpu_id']
    os.environ['WANDB_MODE']='disabled'
    with _live_model_scope(launch,guard):
        # stdout belongs to the wire; scientific diagnostics must not corrupt it.
        with redirect_stdout(sys.stderr):
            if rldx_operation:
                from .controlled_rldx_operation import run
                run(req,output,guard=guard,runtime_cache=runtime_cache,simulator_pool=simulator_pool)
            elif rldx_admission:
                from .controlled_rldx import run
                run(req,output,guard=guard,runtime_cache=runtime_cache,simulator_pool=simulator_pool)
            elif groot_admission:
                from .controlled_groot_admission import run
                run(req,output,guard=guard,runtime_cache=runtime_cache,simulator_pool=simulator_pool)
            elif target_panel:
                from .controlled_target_panel import run
                run(req,output,guard=guard,runtime_cache=runtime_cache,simulator_pool=simulator_pool)
            elif target_recovery:
                from .controlled_target_recovery import run
                run(req,output,guard=guard,runtime_cache=runtime_cache,simulator_pool=simulator_pool)
            elif target:
                from .controlled_target_evaluation import run
                run(req,output,guard=guard,runtime_cache=runtime_cache,simulator_pool=simulator_pool)
            elif admission:
                from .controlled_admission import run
                run(req,output,guard=guard,runtime_cache=runtime_cache,simulator_pool=simulator_pool)
            elif recovery:
                from .controlled_recovery import run
                run(req,output,guard=guard,runtime_cache=runtime_cache,simulator_pool=simulator_pool)
            elif collection:
                from .controlled_training import run
                run(req,output,guard=guard,runtime_cache=runtime_cache,simulator_pool=simulator_pool)
            elif panel:
                from .native_view_panel import run
                run(req,output,guard=guard,runtime_cache=runtime_cache,simulator_pool=simulator_pool)
            else:
                if req['block']['kind']=='paired':from .continuous_evaluation import run
                else:from .continuous_nominal import run
                config=req['paired'] if req['block']['kind']=='paired' else req['nominal']
                run(config,output,execute=True,block_index=req['block']['index'],guard=guard)
    artifacts={str(p.relative_to(output)):file_digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    _durable_write(output/'complete.json',dict(status='COMPLETE',request_sha256=digest(req),artifacts=artifacts))
    proof=validate_block(req,output)
    counts=dict(episodes=req['block']['limits']['episodes'] if panel or collection or recovery or admission or target else req['block']['tasks'],score=proof['calls'].get('score',0),paired_execution=0,nominal_execution=0)
    if admission:
        counts['paired_execution']=proof['calls'].get('execution',0)
        return counts,proof['receipt_sha256']
    if collection:return counts,proof['receipt_sha256']
    counts['paired_execution' if panel or recovery or target or req['block']['kind']=='paired' else 'nominal_execution']=proof['calls']['execution']
    return counts,proof['receipt_sha256']


def verify_completed(binding, row):
    if row['status']=='COMPLETE' and binding['mode']=='libero':
        if binding.get('scientific_adapter')=='controlled_rldx_operation_v1':from .controlled_rldx_operation import validate_block
        elif binding.get('scientific_adapter')=='controlled_rldx_v1':from .controlled_rldx import validate_block
        elif binding.get('scientific_adapter')=='controlled_groot_admission_v1':from .controlled_groot_admission import validate_block
        elif binding.get('scientific_adapter')=='controlled_target_panel_v1':from .controlled_target_panel import validate_block
        elif binding.get('scientific_adapter')=='controlled_target_recovery_v1':from .controlled_target_recovery import validate_block
        elif binding.get('scientific_adapter')=='controlled_target_evaluation_v1':from .controlled_target_evaluation import validate_block
        elif binding.get('scientific_adapter')=='controlled_admission_v1':from .controlled_admission import validate_block
        elif binding.get('scientific_adapter')=='controlled_recovery_v1':from .controlled_recovery import validate_block
        elif binding.get('scientific_adapter')=='controlled_training_v1':from .controlled_training import validate_block
        elif binding.get('scientific_adapter')=='native_view_panel_v1':from .native_view_panel import validate_block
        else:from .continuous_evaluation_fleet import validate_block
        spec=binding['tasks'][row['launch']['key']]
        proof=validate_block(spec['request'],spec['output'])
        if proof['receipt_sha256']!=row['receipt']['result_sha256']:
            raise ValueError('Completed block changed')


def group_alive(owner):
    from AlphaBrain.research.dsol.execution.model_agent import group_alive as shared
    return shared(owner)


def make_simulator_pool(path):
    from .simulator_session import SimulatorSessionPool
    return SimulatorSessionPool(path)


def serve(root,binding,wire):
    from AlphaBrain.research.dsol.execution.model_agent import serve as shared
    return shared(root,binding,wire,sys.modules[__name__])


def serve_task(root,binding,wire,request,session,store,**kwargs):
    from AlphaBrain.research.dsol.execution.model_agent import serve_task as shared
    return shared(root,binding,wire,request,session,store,adapter=sys.modules[__name__],**kwargs)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--simulate',action='store_true')
    args=parser.parse_args(argv)
    wire=Wire(sys.stdin.fileno(),sys.stdout.fileno(),timeout=60)
    try:
        binding=load_binding(args.binding)
        if args.prepare:
            args.root.parent.mkdir(parents=True,exist_ok=True);prepare(args.root,binding)
        serve(args.root,binding,wire)
    except Exception as error:
        try:wire.send(dict(kind='ERROR',error=type(error).__name__+': '+str(error)))
        except Exception:pass
        return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
