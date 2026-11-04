"""Bounded mounting diagnostic composed with the existing authority runner."""
import argparse
from pathlib import Path
import time
from AlphaBrain.research.dsol.data.artifacts import read
from AlphaBrain.research.dsol.execution.manifest import file_digest
from AlphaBrain.research.dsol.execution.queue import exclusive, _durable_write
from .controlled_campaign import run_stage


def run(config_path, *, execute=False):
    if not execute: raise ValueError('Explicit bounded experiment execution required')
    c=read(config_path); root=Path(c['output'])
    for p,h in c['input_hashes'].items():
        if file_digest(p)!=h:raise ValueError('Changed mechanism execution input')
    p=read(c['protocol']); cap=sum(p['call_caps'][k] for k in ('score','paired_execution','nominal_execution'))
    if c['total_call_cap']!=cap:raise ValueError('Global cap mismatch')
    prior_calls=0
    if c.get('prior_attempt_accounting'):
        path=c['prior_attempt_accounting']
        if file_digest(path)!=c['input_hashes'].get(path):raise ValueError('Unbound prior attempt')
        prior=read(path)
        if (prior['status']!='KNOWN_TERMINATED' or prior['live_owned_processes']
                or prior['unknown_calls']!=0 or prior['actual_episodes']!=0):
            raise ValueError('Prior attempt not reconciled')
        prior_calls=prior['actual_model_calls']
        if type(prior_calls) is not int or prior_calls<0:raise ValueError('Invalid prior consumption')
    (root/'reservations').mkdir(exist_ok=False);(root/'settlements').mkdir(exist_ok=False)
    lock=root/'controller';lock.mkdir(exist_ok=False)
    with exclusive(lock):
        calls=reserved=prior_calls
        try:
            with (root/'controller.attempt').open('x') as f:f.write('ONE_ATTEMPT_NO_REPLAY\n')
            run_stage(c,'probe-gate',c['campaigns'][0],kind='recovery',cpu=True)
            for index,campaign in enumerate(c['campaigns']):
                phase=f'probe-{index:02d}'
                budget=read(campaign)['limits'];amount=sum(budget[k] for k in ('score','paired_execution','nominal_execution'))
                # Only completed, reconciled phases release unused headroom.
                # Prior engineering consumption remains charged; an exception
                # keeps this entire phase reservation unresolved.
                reserved=calls+amount
                if reserved>cap:raise ValueError('Global reservation exhausted')
                _durable_write(root/'reservations'/(phase+'.json'),dict(cap=amount,cumulative=reserved,
                    campaign=campaign,sha256=file_digest(campaign),unknown_consumption_refundable=False))
                _durable_write(root/'current.json',dict(status='RUNNING',phase=phase,actual_model_calls=calls,
                    committed_cap=reserved,total_cap=cap,automatic_retry=False,updated=time.time()))
                result=run_stage(c,phase,campaign,kind='recovery',cpu=False)
                calls+=result['actual_model_calls']
                _durable_write(root/'settlements'/(phase+'.json'),result)
            from AlphaBrain.research.dsol.data.coordinate_validation import SCHEMA
            if p.get('schema')==SCHEMA:
                from reports.paper1.coordinate_validation import build
            else:
                from reports.paper1.coordinate_recovery import build
            build(config_path,root/'report')
            _durable_write(root/'current.json',dict(status='COMPLETE',actual_model_calls=calls,total_cap=cap,
                prior_engineering_calls=prior_calls,scientific_model_calls=calls-prior_calls,
                searches=len(p.get('rows',[None]*144)),manipulation_episodes=p['call_caps'].get('episodes',288),
                training_updates=0,dispatch_enabled=False,updated=time.time()))
        except BaseException as e:
            _durable_write(root/'current.json',dict(status='HALTED_NO_RETRY',actual_model_calls_known=calls,
                committed_cap=reserved,error=repr(e),updated=time.time()))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--execute',action='store_true');a=parser.parse_args();run(a.config,execute=a.execute)
