"""Identity rules for approved full-length prefix comparisons, without execution."""
from .artifacts import read
from .score_prefix import validate as validate_score
from ..execution.manifest import file_digest


def validate_mainline(p):
    if (p['schema']!='PREFIX_MAINLINE_V1' or p['updates']!=64 or p['episodes_per_update']!=32
            or p['seeds']!=[0,1,2] or p['maximum_model_calls']!=5200000
            or p['query_cap']!=8 or p['query_cost']!=.01):
        raise ValueError('Approved full-prefix mainline contract required')
    expected={(family,prefix,seed) for family,prefix in [('pi05',10),('groot_n15',4),('rldx',4)] for seed in p['seeds']}
    actual=[]
    for name,c in p['cells'].items():
        validate_score(c['score_contract']);actual.append((c['family'],c['prefix'],c['seed']))
        if (name!=f"{c['family']}-p{c['prefix']}-seed-{c['seed']}"
                or c['score_contract']['family']!=c['family'] or c['score_contract']['prefix']!=c['prefix']):
            raise ValueError('Mixed prefix/model/seed identity')
    if len(actual)!=9 or set(actual)!=expected:raise ValueError('Only the nine approved full-prefix runs are permitted')


def validate_admission(c,cell,b):
    key=b['host']+'/'+b['rows'][0]['suite'];path=c['admission_proofs'][key]
    if c['input_hashes'].get(path)!=file_digest(path):raise ValueError('Unbound actual-entry admission')
    p=read(path)
    if (p['status']!='PASS_PREFIX_NUMERICAL_AND_TRAJECTORY' or p['family']!=cell['family']
            or p['source_identity']!=c['source_identity'] or p['model_hashes']!=cell['model_hashes']
            or p['admitted_host']!=b['host'] or p['admitted_suite']!=b['rows'][0]['suite']):
        raise ValueError('Wrong source, host, suite, model or stale admission')
    if cell.get('core_spec'):
        from ..execution.manifest import digest
        if (cell.get('native_policy') or {}).get('schema')=='UNIFIED_NATIVE_CONTINUOUS_POLICY_V1':
            if (p.get('native_policy_sha256')!=digest(cell['native_policy'])
                    or p.get('feature_producer_sha256')!=digest(cell['feature_producer'])):
                raise ValueError('Native feature producer/core adapter admission differs')
            if cell['family']=='rldx' and p.get('visual_capture_equal') is not True:
                raise ValueError('Pre-language capture lacks capture-free numeric admission')
        if p.get('search_core_sha256')!=digest(cell['core_spec']):
            raise ValueError('Core variant has not passed its actual-entry collector admission')
        expected_mechanism=digest(cell['mechanism']) if cell.get('mechanism') else None
        if p.get('mechanism_sha256')!=expected_mechanism:
            raise ValueError('Mechanism has not passed its actual-entry admission')
        expected_task=digest(cell['task_condition']) if cell.get('task_condition') else None
        if p.get('task_condition_sha256')!=expected_task:
            raise ValueError('Task feature input has not passed its actual-entry collector admission')


def validate_evaluation_row(c,cell,row,state):
    meta=cell['evaluation_schedules'].get(c['schedule'])
    if meta is None or c['evaluation_version']!=meta['version']:
        raise ValueError('Unplanned evaluation schedule or checkpoint')
    if meta['purpose']=='development':
        if meta['version'] not in (16,32) or state['split']!='validation':raise ValueError('Development checkpoint/split differs')
    elif meta['purpose'] in ('final','baseline'):
        expected_split={'seen_new_initials':'train','validation':'validation','unseen_test':'test_unseen_task'}.get(row.get('split'))
        if (meta['version']!=64 or row.get('split') not in ('seen_new_initials','validation','unseen_test')
                or state['split']!=expected_split):
            raise ValueError('Full final checkpoint/task split differs')
    else:raise ValueError('Unknown evaluation purpose')
    if (c.get('stay',False)!=(meta['method'] in ('stay','reference'))
            or (meta['purpose']=='baseline' and (cell['family']!='rldx' or cell['seed']!=0))
            or row['settle_steps']!=state['initialization']['settle_steps']
            or row['query_cap']!=8 or row['execution_repeats']!=[0,1]):
        raise ValueError('Changed baseline ownership, initialization or query/repeat contract')
