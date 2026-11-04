"""Common immutable worker plans and local controller preparation.

Benchmark binding validation is explicit; this never launches a worker.
"""
from copy import deepcopy
import json
from pathlib import Path
from . import queue
from .manifest import digest, file_digest, validate
from .controller_ledger import prepare

def plans(binding, binding_path):
    result={};stamp=digest(binding)
    for stage in binding['budget']['stages']:
        specs={k:v for k,v in binding['tasks'].items() if v['stage']==stage}
        slots=[];tasks=[]
        for key,spec in specs.items():
            for slot in spec['eligible_slots']:
                if slot not in slots:slots.append(slot)
            tasks.append(dict(key=key,addresses=spec['addresses'],output=spec['output'],
                identity_experiment_id=spec['identity_experiment_id'],eligible_slots=spec['eligible_slots'],
                payload=dict(binding_sha256=stamp,request=spec['request'])))
        if not tasks:continue
        plan=dict(format='dsol_work_queue_v1',experiment_id='center-live-'+stage,
                  protocol_sha256=binding['budget']['campaign_sha256'],binding_sha256=stamp,
                  slots=slots,tasks=tasks,backend=dict(module='DISABLED_DYNAMIC_IMPORT',factory='none'),
                  artifacts=[dict(path=str(Path(binding_path).resolve()),sha256=file_digest(binding_path)),
                             dict(path=binding['campaign'],sha256=binding['budget']['campaign_sha256'])])
        validate(plan);result[stage]=plan
    return result


def prepare_controller(binding_path, root, transport_config, *, load_binding):
    binding=load_binding(binding_path);root=Path(root)
    if transport_config.get('resident_workers',False) is not binding.get('resident_workers',False):
        raise ValueError('Transport and immutable binding disagree on resident workers')
    from AlphaBrain.research.dsol.execution.authority_wire import require_local
    root.parent.mkdir(parents=True,exist_ok=True);require_local(root.parent)
    root.mkdir(exist_ok=False)
    for stage,plan in plans(binding,binding_path).items():queue.prepare(plan,root/stage)
    prepare(root/'custody',binding['budget'],simulation_only=binding['mode']=='cpu_contract',
            execution_contract='LIVE_MODEL_V1' if binding['mode']=='libero' else None)
    queue._durable_write(root/'transport.json',transport_config)
    return dict(status='PREPARED_DISPATCH_OFF',binding_sha256=digest(binding),root=str(root))

