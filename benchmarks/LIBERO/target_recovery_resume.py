"""Explicit audited continuation of target evaluation, never automatic retry.

Restore deterministic simulator prefixes using durably recorded predictions.
Only missing predictions invoke the model again; their previous uncertain cost
remains charged in the parent audit. All inputs/noises must match before reuse.
"""
from pathlib import Path
import json
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read
from AlphaBrain.research.dsol.execution.manifest import digest,file_digest
from AlphaBrain.research.dsol.data.flow_noise import tensor_sha256
from .local_search_rollout import input_identity
from .continuous_evaluation import append_event


def validate_proof(c):
    path=c.get('explicit_recovery')
    if path is None:return {}
    if c['input_hashes'].get(path)!=file_digest(path):raise ValueError('Unbound recovery audit')
    proof=read(path)
    if (proof.get('status')!='AUDITED_EXPLICIT_CONTINUATION' or not proof.get('old_workers_exited')
            or proof.get('automatic_retry') is not False or proof.get('uncertain_calls_refunded') is not False):
        raise ValueError('Explicit stopped-owner recovery proof required')
    for p,h in proof['artifacts'].items():
        if file_digest(p)!=h:raise ValueError('Audited recovery evidence changed: '+p)
    return proof


class RecoveredCalls:
    """One row, exact replay records plus a separately accounted live backend."""
    def __init__(self,calls,case,output,guard):
        self.calls,self.case,self.output,self.guard=calls,Path(case),Path(output),guard
        self.counts=dict(score=0,execution=0)
        self.attempts=[json.loads(s) for s in (self.case.parent/'model-attempts.jsonl').read_text().splitlines()] if (self.case.parent/'model-attempts.jsonl').exists() else []
        self.predictions={};self.executed={}
        for repeat in (0,1):
            path=self.case/f'episode-{repeat}/events.jsonl'
            events=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
            for event in events:
                key=(repeat,event['identity']['replan'])
                if event['kind']=='prediction':
                    if key in self.predictions:raise ValueError('Duplicate durable prediction')
                    self.predictions[key]=event
                elif event['kind']=='executed':self.executed[key]=event
        self.used=set()

    @property
    def stage(self):return self.calls.stage
    @stage.setter
    def stage(self,value):self.calls.stage=value

    def infer(self,example,noise,identity,capture=True):
        self.guard();stage=self.stage;cached=None;digest=input_identity(example)
        if stage=='score':
            key=('score',identity['query'],identity['member'])
            file=self.case/f"query-{identity['query']:02d}.npz"
            attempts=[a for a in self.attempts if a['identity'].get('case')==self.case.name
                and a['identity'].get('query')==identity['query'] and a['identity'].get('member')==identity['member']
                and a['identity']['stage']=='score']
            if file.exists():
                if len(attempts)!=1 or attempts[0]['input_sha256']!=digest or attempts[0]['noise_sha256']!=tensor_sha256(noise):
                    raise ValueError('Cached score input/noise mismatch')
                suffix='_'+str(identity['member'])
                with np.load(file) as z:cached={k[:-len(suffix)]:z[k].copy() for k in z.files if k.endswith(suffix)}
                if not {'actions','trace','external','wrist','score'}.issubset(cached):raise ValueError('Incomplete cached score')
        else:
            key=('execution',identity['repeat'],identity['replan'])
            event=self.predictions.get(key[1:])
            if event:
                if event['identity']!=identity or event['input_sha256']!=digest:raise ValueError('Recovered prediction input/identity mismatch')
                attempts=[a for a in self.attempts if a['identity'].get('stage')=='execution'
                    and a['identity'].get('repeat')==identity['repeat'] and a['identity'].get('replan')==identity['replan']
                    and all(a['identity'].get(k)==identity[k] for k in ('state_key','start','method'))]
                if len(attempts)!=1 or attempts[0]['noise_sha256']!=tensor_sha256(noise):raise ValueError('Recovered prediction noise mismatch')
                actions=np.asarray(event['full_decoded_actions'],dtype=np.float32)
                if actions.shape!=(32,7) or not np.isfinite(actions).all():raise ValueError('Invalid durable action prediction')
                cached=dict(full_actions=actions,actions=actions[:10])
        if cached is None:return self.calls.infer(example,noise,identity,capture=capture)
        if key in self.used:raise ValueError('Recovery prediction consumed twice')
        self.used.add(key);self.counts[stage]+=1
        append_event(self.output/'reused-calls.jsonl',dict(stage=stage,identity=identity,input_sha256=digest,
            noise_sha256=tensor_sha256(noise),source=str(self.case),model_calls=0))
        return cached

    def verify_trajectory(self,repeat,result,arrays):
        """Every known executed prefix must retain termination and action bytes."""
        for (r,index),event in self.executed.items():
            if r!=repeat:continue
            if index>=result['replans']:raise ValueError('Recovered trajectory ended before known prefix')
            begin=int(arrays['replan_action_offsets'][index]);known=np.asarray(event['actions'],dtype=np.float32)
            actual=arrays['actions'][begin:begin+len(known)]
            if not np.array_equal(actual,known):raise ValueError('Recovered executed actions differ')
            if event['success']!=(result['success'] and index==result['replans']-1):
                raise ValueError('Recovered prefix termination differs')


def completed_episode(case,repeat,row):
    path=Path(case)/f'episode-{repeat}/result.json'
    if not path.exists():return None
    result=read(path)
    if any(result.get(k)!=v for k,v in row.items()) or result['repeat']!=repeat:
        raise ValueError('Recovered completed episode identity differs')
    with np.load(result['trajectory']) as z:
        if len(z['actions'])!=result['steps'] or len(z['full_decoded_action_chunks'])!=result['replans']:
            raise ValueError('Incomplete recovered completed trajectory')
    return result


def validate_native_baseline_proof(c):
    """Admit only the reviewed FastWAM no-search continuation scope."""
    proof=validate_proof(c)
    if not proof or c.get('phase')!='baseline' or read(c['protocol']).get('family')!='fastwam':
        raise ValueError('Explicit FastWAM baseline recovery proof required')
    for block in c['blocks']:
        for row in block['rows']:
            ref=proof.get('records',{}).get(digest(row))
            if row['method'] not in ('stay','reference') or not ref or not Path(ref['case']).is_absolute():
                raise ValueError('Unplanned native baseline recovery row')
    return proof


def cached_baseline_calls(c,rows,value,output):
    """Count proven cached calls without charging them as new model calls.

    A journal count alone is insufficient: every source must be bound in the
    stopped-owner proof and each reused operation address must be unique.
    Search-policy/feature/diagnostic recovery remains outside this contract.
    """
    path=Path(output)/'reused-calls.jsonl'
    counts=dict(score=0,execution=0)
    if not path.exists():return counts
    proof=validate_native_baseline_proof(c)
    artifacts=proof['artifacts'];cases={str(Path(proof['records'][digest(row)]['case'])):row for row in rows}
    results={(v['trial_id'],v['repeat']):v for v in value['tasks']}
    used=set();episodes=set();searches=set()

    def bound(source):
        source=Path(source)
        if artifacts.get(str(source))!=file_digest(source):
            raise ValueError('Unbound cached baseline source')
        return source

    for line in path.read_text().splitlines():
        event=json.loads(line)
        if event.get('model_calls')!=0:raise ValueError('Cached baseline call cannot consume or refund new calls')
        stage=event['stage'];source=Path(event['source'])
        if stage=='score':
            row=cases.get(str(source.parent))
            if source.name!='search.json' or row is None or event.get('count')!=0 or str(source) in searches:
                raise ValueError('Invalid or duplicate baseline search reuse')
            info=read(bound(source))
            if info['queries']!=0 or any(info.get(k)!=v for k,v in row.items()):
                raise ValueError('Cached baseline search identity differs')
            searches.add(str(source));continue
        if stage!='execution':raise ValueError('Unsupported cached baseline stage')
        if source.name=='result.json':
            case=source.parent.parent;row=cases.get(str(case))
            if row is None:raise ValueError('Unknown cached baseline episode source')
            prior=read(bound(source));repeat=prior['repeat']
            if repeat not in (0,1) or source.parent.name!=f'episode-{repeat}':
                raise ValueError('Cached baseline repeat identity differs')
            bound(prior['trajectory'])
            checked=completed_episode(case,repeat,row)
            if checked!=results.get((row['trial_id'],repeat)) or event.get('count')!=checked['replans']:
                raise ValueError('Cached baseline completed episode differs')
            attempts=[json.loads(s) for s in bound(case.parent/'model-attempts.jsonl').read_text().splitlines() if s]
            attempts=[v for v in attempts if v['identity'].get('trial_id')==row['trial_id'] and v['identity'].get('repeat')==repeat]
            events=[json.loads(s) for s in bound(case/f'episode-{repeat}/events.jsonl').read_text().splitlines() if s]
            predictions=[v for v in events if v['kind']=='prediction']
            if len(attempts)!=checked['replans'] or len(predictions)!=checked['replans']:
                raise ValueError('Cached baseline completed episode journals differ')
            with np.load(checked['trajectory'],allow_pickle=False) as z:
                for j,(attempt,prediction) in enumerate(zip(attempts,predictions)):
                    ident=dict(row,repeat=repeat,replan=j)
                    if (attempt['identity']!=dict(ident,stage='execution') or prediction['identity']!=ident
                            or prediction['input_sha256']!=attempt['input_sha256']
                            or attempt['input_sha256']!=checked['input_sha256'][j]
                            or attempt['noise_sha256']!=checked['execution_noise_sha256'][j]
                            or not np.array_equal(z['full_decoded_action_chunks'][j],np.asarray(prediction['full_decoded_actions'],dtype=np.float32))):
                        raise ValueError('Cached baseline completed episode prediction differs')
            keys={(row['trial_id'],repeat,j) for j in range(checked['replans'])}
            marker=(row['trial_id'],repeat)
            if marker in episodes or used.intersection(keys):raise ValueError('Duplicate cached baseline operation')
            episodes.add(marker);used.update(keys);counts['execution']+=checked['replans']
        else:
            row=cases.get(str(source));ident=event.get('identity',{});repeat=ident.get('repeat');replan=ident.get('replan')
            if (row is None or repeat not in (0,1) or type(replan) is not int or replan<0
                    or ident!=dict(row,repeat=repeat,replan=replan) or event.get('count',1)!=1):
                raise ValueError('Cached baseline prediction identity differs')
            result=results.get((row['trial_id'],repeat))
            if result is None or replan>=result['replans']:raise ValueError('Cached baseline prediction outside result')
            events=bound(source/f'episode-{repeat}/events.jsonl')
            matches=[json.loads(s) for s in events.read_text().splitlines() if s]
            matches=[v for v in matches if v['kind']=='prediction' and v['identity']==ident]
            attempts=[json.loads(s) for s in bound(source.parent/'model-attempts.jsonl').read_text().splitlines() if s]
            attempts=[v for v in attempts if v['identity']==dict(ident,stage='execution')]
            if (len(matches)!=1 or len(attempts)!=1 or event['input_sha256']!=matches[0]['input_sha256']
                    or event['input_sha256']!=attempts[0]['input_sha256'] or event['noise_sha256']!=attempts[0]['noise_sha256']
                    or event['input_sha256']!=result['input_sha256'][replan]
                    or event['noise_sha256']!=result['execution_noise_sha256'][replan]):
                raise ValueError('Cached baseline prediction input/noise differs')
            with np.load(result['trajectory'],allow_pickle=False) as z:
                if not np.array_equal(z['full_decoded_action_chunks'][replan],np.asarray(matches[0]['full_decoded_actions'],dtype=np.float32)):
                    raise ValueError('Cached baseline prediction actions differ')
            key=(row['trial_id'],repeat,replan)
            if key in used or (row['trial_id'],repeat) in episodes:raise ValueError('Duplicate cached baseline operation')
            used.add(key);counts['execution']+=1
    # Totals cannot detect an actual call overlapping a reused prediction.
    # Require exact, disjoint per-replan coverage before accepting any count.
    row_ids={row['trial_id']:row for row in rows}
    for line in (Path(output)/'model-attempts.jsonl').read_text().splitlines():
        attempt=json.loads(line);ident=attempt['identity']
        if ident['stage']!='execution':continue
        row=row_ids.get(ident.get('trial_id'));repeat=ident.get('repeat');j=ident.get('replan')
        result=results.get((ident.get('trial_id'),repeat))
        if (row is None or repeat not in (0,1) or type(j) is not int or result is None or not 0<=j<result['replans']
                or ident!=dict(row,repeat=repeat,replan=j,stage='execution')
                or attempt['input_sha256']!=result['input_sha256'][j]
                or attempt['noise_sha256']!=result['execution_noise_sha256'][j]):
            raise ValueError('Actual baseline continuation address/input/noise differs')
        key=(row['trial_id'],repeat,j)
        if key in used:raise ValueError('Actual baseline call duplicates a reused prediction')
        used.add(key)
    expected={(v['trial_id'],v['repeat'],j) for v in value['tasks'] for j in range(v['replans'])}
    if used!=expected:raise ValueError('Incomplete native baseline continuation coverage')
    return counts
