"""Explicit manipulation contracts; historical native execution stays 10/5."""
import re

FASTWAM_SCHEMA = 'FASTWAM_LIBERO_OPERATION_V1'
FASTWAM_HORIZONS = dict(libero_spatial=400, libero_object=400, libero_goal=400,
                       libero_10=700, libero_90=700)
RLDX_SCHEMA = 'RLDX_LIBERO_OPERATION_V1'
RLDX_HORIZONS = dict(libero_spatial=720, libero_object=720, libero_goal=720, libero_10=720)
GROOT_SCHEMA = 'GROOT_N15_LIBERO_OPERATION_V1'
GROOT_HORIZONS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)


def groot_contract(runtime_sha256):
    return dict(schema=GROOT_SCHEMA, runtime_contract_sha256=runtime_sha256,
                predicted_actions=10, executed_actions_per_replan=10,
                clip=None, action_dtype='float64', horizons=GROOT_HORIZONS.copy(),
                execution_repeat_ids=[0,1], settle_steps=10, action_ensemble=False)


def validate_groot(contract):
    stamp=contract.get('runtime_contract_sha256','')
    if (not isinstance(stamp,str) or re.fullmatch('[0-9a-f]{64}',stamp) is None
            or contract != groot_contract(stamp)):
        raise ValueError('Explicit native GR00T 10/10 operation contract required')
    return contract


def fastwam_contract(runtime_sha256):
    return dict(schema=FASTWAM_SCHEMA, runtime_contract_sha256=runtime_sha256,
                predicted_actions=32, executed_actions_per_replan=10,
                clip=None, action_dtype='float64', horizons=FASTWAM_HORIZONS.copy(),
                execution_repeat_ids=[0, 1], settle_steps=30,
                action_ensemble=False)


def validate_fastwam(contract):
    stamp=contract.get('runtime_contract_sha256','')
    if (not isinstance(stamp,str) or re.fullmatch('[0-9a-f]{64}',stamp) is None
            or contract!=fastwam_contract(stamp)):
        raise ValueError('Explicit native FastWAM 32/10 operation contract required')
    return contract



def rldx_contract(runtime_sha256):
    return dict(schema=RLDX_SCHEMA, runtime_contract_sha256=runtime_sha256,
                predicted_actions=16, executed_actions_per_replan=8,
                clip=None, action_dtype='float32', horizons=RLDX_HORIZONS.copy(),
                execution_repeat_ids=[0,1], settle_steps=10, action_ensemble=False)


def validate_rldx(contract):
    stamp=contract.get('runtime_contract_sha256','')
    if (not isinstance(stamp,str) or re.fullmatch('[0-9a-f]{64}',stamp) is None
            or contract!=rldx_contract(stamp)):
        raise ValueError('Explicit native RLDX 16/8 operation contract required')
    return contract

def simulator_action_contract(config):
    if 'operation_contract' not in config:
        return 5, 'float32'
    if config['operation_contract'].get('schema')==RLDX_SCHEMA:
        validate_rldx(config['operation_contract'])
        model=config['models'][config['operation_model']]
        if (config.get('continuous_execution')!='PAIRED_EVALUATION_ONLY'
                or config.get('raw_observation')!='LIBERO_RAW_RGB256_PROPRIO_V1'
                or model.get('implementation')!='rldx_native'
                or model['runtime_contract_sha256']!=config['operation_contract']['runtime_contract_sha256']):
            raise ValueError('RLDX raw/execution/model binding differs')
        return 8,'float32'
    is_groot=config['operation_contract'].get('schema')==GROOT_SCHEMA
    validator=validate_groot if is_groot else validate_fastwam
    validator(config['operation_contract'])
    if (config.get('continuous_execution')!='PAIRED_EVALUATION_ONLY'
            or config.get('raw_observation')!='LIBERO_RAW_RGB256_PROPRIO_V1'):
        name='GR00T' if is_groot else 'FastWAM'
        raise ValueError(name+' operation requires raw observations and execution gate')
    model=config['models'][config['operation_model']]
    implementation='groot_native' if is_groot else 'fastwam_native'
    if (model.get('implementation')!=implementation or model['runtime_contract_sha256']
            !=config['operation_contract']['runtime_contract_sha256']):
        raise ValueError('Simulator/model operation contract differs')
    return 10, 'float64'
