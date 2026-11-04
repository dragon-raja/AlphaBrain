"""Verify completed structural-pilot artifacts before selecting a task arm."""
from pathlib import Path
from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.execution.manifest import file_digest as sha
from AlphaBrain.research.dsol.analysis.core_selection import choose_structure
from benchmarks.LIBERO.controlled_campaign import phase_evidence


def load_outcomes(roots):
    outcomes={};sources={}
    def load(path):
        sources[str(path)]=sha(path);return read(path)
    for root in map(Path,roots):
        done=load(root/'completion.json')
        if done['status']!='COMPLETE' or done.get('failures') or done['dispatch_enabled'] is not False:raise ValueError('Structure screening incomplete')
        p=load(root/'protocol.json')
        if p.get('design_stage','single_factors') not in ('single_factors','combinations','task_condition','seed_replication'):raise ValueError('Wrong selection source')
        for name,cell in p['cells'].items():
            variant=cell['core_spec']['variant']+('T' if cell.get('task_condition') else '');schedules=cell['evaluation_schedules']
            if len(schedules)!=1:raise ValueError('Single predeclared development endpoint required')
            schedule,meta=next(iter(schedules.items()))
            if meta!=dict(purpose='development',version=16,method='ppo'):raise ValueError('Changed development endpoint')
            label=Path(schedule).stem+'-v0016';ledger=load(root/'budget'/(name+'-'+label+'.json'))
            if ledger['status']!='COMPLETE' or ledger['settlement']['actual_episodes']!=64:raise ValueError('Unsettled development phase')
            cp=Path(ledger['campaign']);campaign=load(cp)
            if sources[str(cp)]!=ledger['campaign_sha256'] or campaign['cell']!=name:raise ValueError('Changed campaign')
            if sha(campaign['actor']['path'])!=campaign['actor']['sha256']:raise ValueError('Changed evaluated actor')
            config=load(root/'bindings'/name/(label+'.json'));folder=phase_evidence(config,label)
            for block in campaign['blocks']:
                b=folder/'results'/f"block-{block['index']:04d}";receipt=load(b/'complete.json');result=load(b/'results.json')
                if sources[str(b/'results.json')]!=receipt['artifacts']['results.json']:raise ValueError('Changed operation outcomes')
                expected={(r['trial_id'],rep) for r in block['rows'] for rep in r['execution_repeats']}
                if {(r['trial_id'],r['repeat']) for r in result['tasks']}!=expected or len(result['tasks'])!=len(expected):raise ValueError('Incomplete block')
                searches={r['trial_id']:r for r in result['searches']}
                for r in result['tasks']:
                    search=searches[r['trial_id']]
                    if search['queries']!=r['queries']:raise ValueError('Mismatched search and outcome')
                    outcomes.setdefault(variant,[]).append(dict(r,seed=cell['seed'],search_initial_accel=search['initial_accel'],search_best_accel=search['best_accel']))
    return outcomes,sources


def build(roots,output):
    outcomes,sources=load_outcomes(roots)
    decision=choose_structure(outcomes);write(output,dict(status='PASS_COMPLETE_DEVELOPMENT_SELECTION',**decision,input_hashes=sources));return decision
