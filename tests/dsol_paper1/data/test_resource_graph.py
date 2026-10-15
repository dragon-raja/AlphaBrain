from AlphaBrain.research.dsol.data.resource_graph import audit_graph


def graph():
    return {'resources':[{'id':'weight','kind':'checkpoint','backend':'openpi_jax','availability':'present'}],
            'execution_bindings':[{'id':'run','backend':'openpi_jax','resources':['weight'],'checkpoints':['weight']}]}


def test_portable_graph_needs_no_private_paths():
    assert audit_graph(graph())['status'] == 'PASS'


def test_checkpoint_backend_mismatch_is_not_silent():
    value=graph(); value['execution_bindings'][0]['backend']='alphabrain_torch'
    assert audit_graph(value)['status'] == 'INCOMPLETE'


def test_missing_and_retired_local_roots(tmp_path):
    locations={'resource_paths':{'weight':str(tmp_path/'absent')},'retired_roots':[str(tmp_path)]}
    assert len(audit_graph(graph(),locations)['errors']) == 2


def test_provenance_cycle():
    value=graph(); value['resources'][0]['parents']=['weight']
    assert 'Resource provenance cycle' in audit_graph(value)['errors']


def test_pending_resources_are_not_claimed_complete():
    value=graph(); value['resources'][0]['availability']='acquiring'
    result=audit_graph(value,{'resource_paths':{}})
    assert result['status']=='PASS'
    assert result['resources'][0]['availability']=='acquiring'
    assert result['resources'][0]['exists'] is False


def test_bound_revision_cannot_silently_change():
    value = graph()
    value['resources'][0]['revision'] = 'official-pinned'
    value['execution_bindings'][0]['required_revisions'] = {'weight': 'official-pinned'}
    assert audit_graph(value)['status'] == 'PASS'
    value['resources'][0]['revision'] = 'some-cache-version'
    assert audit_graph(value)['status'] == 'INCOMPLETE'


def test_revision_lock_must_be_an_execution_resource():
    value = graph()
    value['execution_bindings'][0]['required_revisions'] = {'unbound': 'commit'}
    assert audit_graph(value)['status'] == 'INCOMPLETE'
