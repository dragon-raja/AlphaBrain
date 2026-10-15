from copy import deepcopy

from AlphaBrain.research.dsol.execution.cluster import audit


def snapshot(host):
    return dict(host=host,control='/queue/'+host,status='RUNNING',retired=False,
        resource_control={},gpus='inventory',
        manifest=dict(protocol_sha256='a'*64,slots=[dict(host=host,id='0')]),
        tasks=[dict(output='/results/'+host,addresses=[dict(model='m',initial=host)],
                    status='PENDING',completed=0,slot=None)])


def test_cluster_detects_idle_resources_and_duplicate_trials():
    a,b=snapshot('a'),snapshot('b')
    result=audit([a,b])
    assert result['status']=='PASS'
    assert len(result['alerts'])==2
    b['tasks'][0]['addresses']=deepcopy(a['tasks'][0]['addresses'])
    assert audit([a,b])['status']=='HOLD'


def test_draining_not_misreported_as_idle():
    a=snapshot('a');a['resource_control']=dict(drained=[dict(host='a',gpu='0')])
    assert not audit([a])['alerts']
    a['retired']=True
    assert audit([a])['status']=='HOLD'


def test_distinct_queues_cannot_reserve_the_same_gpu():
    a,b=snapshot('a'),snapshot('a')
    b['control']='/queue/other';b['tasks'][0]['output']='/results/other'
    b['tasks'][0]['addresses']=[dict(model='other', initial='other')]
    result=audit([a,b])
    assert result['status']=='HOLD'
    assert result['duplicate_ownership'][0]['kind']=='gpu_reservation'


def test_completed_queue_releases_gpu_but_keeps_result_ownership():
    a,b=snapshot('a'),snapshot('a')
    a['status']='COMPLETE';a['tasks'][0]['status']='COMPLETE'
    b['control']='/queue/next';b['tasks'][0]['output']='/results/next'
    b['tasks'][0]['addresses']=[dict(model='next',initial='next')]
    assert audit([a,b])['status']=='PASS'
    b['tasks'][0]['addresses']=deepcopy(a['tasks'][0]['addresses'])
    assert audit([a,b])['status']=='HOLD'
