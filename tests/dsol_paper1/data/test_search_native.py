import json
import numpy as np
import pytest

from AlphaBrain.research.dsol.data.artifacts import sha
from AlphaBrain.research.dsol.data.search_native import load_native_search


@pytest.fixture
def native_bank(tmp_path):
    source=tmp_path/"old";native=tmp_path/"native";source.mkdir();native.mkdir()
    ids=["canonical"]+[f"broad_train_{i:03d}" for i in range(64)]+[f"broad_heldout_{i:03d}" for i in range(32)]
    states=[dict(pair_key=f"{i}",task_id=f"task{i//4}",task_ordinal=i//4,
                 initialization=dict(init_state_index=i%4)) for i in range(32)]
    (source/"states.json").write_text(json.dumps(states))
    np.savez_compressed(source/"matrix.npz",camera=np.tile(np.eye(4),(32,97,1,1)),
                        candidate_ids=ids,success=np.ones((2,32,97,32),np.int8))
    (source/"provenance.json").write_text(json.dumps({'artifacts':{n:sha(source/n) for n in ('matrix.npz','states.json')}}))
    inputs=tmp_path/"inputs.json"
    inputs.write_text(json.dumps(dict(records=[dict(state=s,candidate_ids=ids) for s in states])))
    visual=np.zeros((32,97,4608),np.float32);visual[...,0]=9
    wrist=np.zeros_like(visual);wrist[...,0]=3
    scores=np.ones((32,97,2));scores[[i for i in range(32) if i%4>=2]]=1000
    np.savez_compressed(native/"siglip-scores.npz",scores=scores,external=visual,wrist=wrist)
    np.savez(native/"reference.npz",synthetic=True)
    receipt=dict(status="COMPLETE",batch_size=1,noise_count=2,attempted=dict(admission=66,export=6208),
                 asset_sha256=sha(native/"siglip-scores.npz"),reference_sha256=sha(native/"reference.npz"),
                 inputs_sha256=sha(inputs))
    (native/"features.json").write_text(json.dumps(receipt))
    return source,native,inputs


def test_native_features_and_development_only_scale(native_bank):
    data,outcomes=load_native_search(*native_bank)
    assert data.visual.shape[-1]==4608
    assert np.all(data.visual[...,0]==9) and np.all(data.wrist[...,0]==3)
    assert data.score_scales==(1.,)
    assert data.accel_members.shape==(1,32,97,2)
    assert outcomes.paired(0,0,0,1)["paired_repeats"]==32
    assert not (native_bank[0]/"embeddings.npz").exists()


def test_native_identity_hash_is_required(native_bank):
    source,native,inputs=native_bank
    inputs.write_text(inputs.read_text()+"\n")
    with pytest.raises(ValueError,match="source changed"):
        load_native_search(source,native,inputs)
