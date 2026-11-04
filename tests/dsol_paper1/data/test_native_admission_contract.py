"""Train-state admission may cover held-out inputs, never a changed runtime.

    All identities here are synthetic and are not production admission evidence.
"""
import json
from copy import deepcopy
import pytest
from AlphaBrain.research.dsol.data.native_training import admission_runtime_sha256,validate_admission
from AlphaBrain.research.dsol.execution.manifest import digest,file_digest


def fixture(tmp_path):
    native=dict(layout=['synthetic'],tap='synthetic')
    spec=dict(layout=native['layout'],producer_sha256=digest(native))
    sim=dict(inputs='synthetic-train',anchors='synthetic-train-noise',native_features=native,
             models={'synthetic':dict(weights_sha256='e'*64)},operation_contract=dict(steps=5))
    path=tmp_path/'sim.json';path.write_text(json.dumps(sim))
    protocol=dict(native_policy=spec,family='synthetic',native_source_identity={'synthetic_cpu_unit_test':True})
    proof=dict(status='PASS_NATIVE_FEATURE_TRAJECTORY',scope='REAL_NATIVE_MODEL',family='synthetic',
        admitted_sim_config=str(path),sim_config_sha256=file_digest(path),runtime_identity_sha256=admission_runtime_sha256(sim),
        native_policy_sha256=digest(spec),source_identity=protocol['native_source_identity'],
        actions_exact=True,trace_exact=True,rng_exact=True,fresh_features=True,return_exact=True)
    evidence=tmp_path/'proof.json';evidence.write_text(json.dumps(proof))
    campaign=dict(native_admission=str(evidence),input_hashes={str(evidence):file_digest(evidence),str(path):file_digest(path)})
    return campaign,sim,protocol,path


def test_same_runtime_accepts_separately_bound_new_population(tmp_path):
    c,sim,p,path=fixture(tmp_path)
    validate_admission(c,sim,p)
    target=dict(sim,inputs='synthetic-evaluation',anchors='synthetic-evaluation-noise')
    validate_admission(c,target,p)
    del c['input_hashes'][str(path)]
    with pytest.raises(ValueError,match='Unbound'):validate_admission(c,target,p)


@pytest.mark.parametrize('change',['weights','tap','operation','extra','original_bytes'])
def test_changed_runtime_or_original_proof_source_is_rejected(tmp_path,change):
    c,original,p,path=fixture(tmp_path);sim=deepcopy(original)
    if change=='weights':sim['models']['synthetic']['weights_sha256']='f'*64
    elif change=='tap':sim['native_features']['tap']='other'
    elif change=='operation':sim['operation_contract']['steps']=10
    elif change=='extra':sim['new_runtime_behavior']=True
    else:path.write_text('{}')
    with pytest.raises(ValueError):validate_admission(c,sim,p)
