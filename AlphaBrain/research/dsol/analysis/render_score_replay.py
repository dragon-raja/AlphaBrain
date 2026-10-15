"""Bounded input-only replay; the caller injects the resident inference adapter."""
from pathlib import Path
import time
import numpy as np
from ..data.artifacts import read,write,sha
from ..data.render_score_panel import request_from_cell
from ..data.evaluation_matrix import flow_noise


def run(plan_path,worker_id,output,predict):
    plan=read(plan_path);worker=next(w for w in plan['workers'] if w['worker_id']==worker_id)
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.time()
    status=dict(status='RUNNING',worker_id=worker_id,plan_sha256=sha(plan_path),source_sha256=sha(__file__),
        completed=0,total=len(worker['entry_indices']),no_rollouts=True,no_training=True)
    write(output/'status.json',status)
    try:
        for index in worker['entry_indices']:
            entry=plan['entries'][index];requests={};metadata={}
            for source,cell in entry['cells'].items():
                cell=Path(cell)
                if sha(cell/'scores.json')!=entry['metadata_sha256'][source] or sha(cell/'scores.npz')!=entry['arrays_sha256'][source]:
                    raise ValueError('Frozen source changed')
                metadata[source]=read(cell/'scores.json');requests[source]=request_from_cell(cell,metadata[source])
            old_arm=worker['arm'];old=metadata[old_arm]
            with np.load(Path(entry['cells'][old_arm])/'scores.npz') as z:old_actions=z['actions'].copy()
            scores=[];actions=[];gate=plan['runtime_replay_gate'];control={}
            for source in ('canonical','broad'):
                if source=='broad' and not entry['different']:
                    scores.append(scores[0].copy());actions.append(actions[0].copy());continue
                ss=[];aa=[]
                for repeat in range(8):
                    traced=predict(requests[source],flow_noise(entry['initial_id'],'score',repeat))
                    if traced['steps']!=10 or not np.array_equal(np.asarray(traced['times']),np.asarray(old['rows'][repeat]['times'],dtype=np.float32)):
                        raise ValueError('Denoising time grid changed')
                    ss.append(traced['scores']['3']['score']);aa.append(traced['actions'])
                scores.append(np.asarray(ss));actions.append(np.stack(aa))
            scores=np.array(scores);actions=np.array(actions)
            slot=0 if old_arm=='canonical' else 1
            reference=np.array([row['metrics']['3']['score'] for row in old['rows']])
            control=dict(score_max_abs=float(np.max(np.abs(scores[slot]-reference))),
                action_max_abs=float(np.max(np.abs(actions[slot]-old_actions))))
            if not np.allclose(scores[slot],reference,atol=gate['score_atol'],rtol=0) or not np.allclose(actions[slot],old_actions,atol=gate['action_atol'],rtol=gate['action_rtol']):
                write(output/'failed-control.json',dict(entry_index=index,control=control,gate=gate))
                raise ValueError('Native-input replay failed numerical gate; do not attribute runtime drift to pixels')
            # Exact repeat sentinel, same request/noise in the current process.
            if entry['view_id']=='canonical':
                repeat=predict(requests['canonical'],flow_noise(entry['initial_id'],'score',0))
                control['same_input_repeat_action_exact']=bool(np.array_equal(repeat['actions'],actions[0,0]))
                control['same_input_repeat_score_abs']=abs(float(repeat['scores']['3']['score'])-float(scores[0,0]))
                if not control['same_input_repeat_action_exact'] or control['same_input_repeat_score_abs']>1e-12:
                    raise ValueError('Same-input replay is nondeterministic in this runtime')
            np.savez_compressed(output/f'cell-{index:04d}.npz',scores=scores,actions=actions)
            write(output/f'cell-{index:04d}.json',dict(entry_index=index,control=control,arrays_sha256=sha(output/f'cell-{index:04d}.npz')))
            status.update(completed=status['completed']+1,elapsed_seconds=time.time()-started,last_entry=index)
            write(output/'status.json',status)
            if status['completed']%10==0:print(worker_id,status['completed'],'/',status['total'],round(time.time()-started,1),flush=True)
        status['status']='COMPLETE'
    except Exception as exc:
        status.update(status='ERROR',error=f'{type(exc).__name__}: {exc}');raise
    finally:write(output/'status.json',status)
