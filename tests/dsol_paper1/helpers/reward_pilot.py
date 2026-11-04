"""Synthetic, hash-bound adaptation contract; never a live execution permit."""
from pathlib import Path
import json
from AlphaBrain.research.dsol.data.reward_pilot import SCHEMA,FIXED_SCALE
from AlphaBrain.research.dsol.execution.manifest import file_digest


def contract(root, schedule, initial):
    root=Path(root)
    def save(name,value):
        path=root/name;path.write_text(json.dumps(value));return str(path)
    rows=json.loads(Path(schedule).read_text())
    pairs=save('paired.json',[dict(seed=s,rows=[dict(r,episode=r['source_episode']) for r in rows]) for s in (0,1,2)])
    cases=save('cases.json',[])
    checkpoints=[dict(seed=s,path=str(initial),sha256=file_digest(initial),update=64) for s in (0,1,2)]
    plan=save('approved-plan.json',dict(initial_checkpoints=checkpoints,synthetic=True))
    auth=save('approval.json',dict(schema='USER_APPROVED_ONLINE_REWARD_PILOT_V1',plan=plan,
        plan_sha256=file_digest(plan),total_additional_call_cap=169216,synthetic=True))
    p=dict(schema=SCHEMA,arms=['B','G','L'],seeds=[0,1,2],updates=8,episodes_per_update=32,
        initial_update=64,initial_cursor=2048,fixed_scale=FIXED_SCALE,query_cap=8,query_cost=.01,
        floor_fraction=.1,diagnostic_calls=4,call_caps=dict(training=36864,evaluation=131328,engineering=1024,total=169216),
        authorization=auth,approved_plan=plan,paired_schedules=pairs,evaluation_cases=cases,
        schedules={str(s):str(schedule) for s in (0,1,2)},initial_checkpoints=checkpoints)
    p['input_hashes']={x:file_digest(x) for x in [auth,plan,pairs,cases,str(schedule),str(initial)]}
    return save('adaptation.json',p)
