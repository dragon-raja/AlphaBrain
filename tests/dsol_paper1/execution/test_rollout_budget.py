import multiprocessing as mp
import pytest
from AlphaBrain.research.dsol.execution.rollout_budget import RolloutBudget


def consume(root,index):
    b=RolloutBudget(root);key=str(index)
    b.claim(key,key,'v0')
    for member in (0,1):b.attempt(key,key,'v0',1,member)
    b.complete(key,key,'v0',2,'a'*64)


def test_four_processes_share_one_global_budget(tmp_path):
    b=RolloutBudget.create(tmp_path/'budget',8)
    b.reserve([dict(key=str(i),owner=str(i),version='v0',cap=2) for i in range(4)])
    ctx=mp.get_context('spawn');workers=[ctx.Process(target=consume,args=(b.root,i)) for i in range(4)]
    for p in workers:p.start()
    for p in workers:p.join(20);assert p.exitcode==0
    s=b.inspect();assert s['spent']==8 and s['reserved']==0
    assert len(b.recoverable([str(i) for i in range(4)]))==4
    with pytest.raises(ValueError):b.reserve([dict(key='extra',owner='x',version='v0',cap=2)])


def test_failure_attempt_is_charged_and_no_silent_replay(tmp_path):
    b=RolloutBudget.create(tmp_path/'budget',16)
    b.reserve([dict(key='episode',owner='worker',version='v0',cap=16)])
    with pytest.raises(ValueError):b.claim('episode','other','v0')
    with pytest.raises(ValueError):b.claim('episode','worker','v1')
    b.claim('episode','worker','v0');assert b.attempt('episode','worker','v0',1,0)==1
    with pytest.raises(ValueError):b.attempt('episode','worker','v0',1,0)
    with pytest.raises(ValueError):b.recoverable(['episode'])
    b.fail('injected inference failure')
    with pytest.raises(ValueError):b.attempt('episode','worker','v0',1,1)
    with pytest.raises(ValueError):b.claim('episode','worker','v0')
    assert b.inspect()['spent']==1


def test_unused_reservation_released_only_on_verified_completion(tmp_path):
    b=RolloutBudget.create(tmp_path/'budget',20)
    b.reserve([dict(key='one',owner='w',version='v0',cap=16)])
    with pytest.raises(ValueError):b.reserve([dict(key='two',owner='w',version='v1',cap=16)])
    b.claim('one','w','v0')
    for m in (0,1):b.attempt('one','w','v0',1,m)
    with pytest.raises(ValueError):b.complete('one','w','v0',4,'a'*64)
    b.complete('one','w','v0',2,'a'*64)
    b.reserve([dict(key='two',owner='w',version='v1',cap=16)])
    assert b.inspect()['spent']==2 and b.inspect()['reserved']==16
