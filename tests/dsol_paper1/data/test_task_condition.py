import hashlib,json
import numpy as np
import pytest
from AlphaBrain.research.dsol.data.task_condition import validate_spec,task_vector,instruction_key
from tests.dsol_paper1.helpers.task_condition import task_spec


def test_cache_rejects_wrong_source_text_and_mutation(tmp_path):
    spec=task_spec();text='pick the cup';p=tmp_path/'cache.json'
    p.write_text(json.dumps(dict(spec=spec,instructions={instruction_key(text):dict(text=text,vector=list(range(8)))})))
    b=dict(spec=spec,cache=str(p),cache_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    a=task_vector(b,text);a[:]=999
    assert np.array_equal(task_vector(b,text),np.arange(8,dtype=np.float32))
    with pytest.raises(ValueError):task_vector(b,'pick the plate')
    with pytest.raises(ValueError):task_vector(dict(b,spec=dict(spec,family='other')),text)
    with pytest.raises(ValueError):task_vector(dict(b,cache_sha256='0'*64),text)


@pytest.mark.parametrize('change',[{'inputs':['instruction','image']},{'inputs':['task_id']},{'dimension':0},{'attention':'image_and_text'},{'pooling':'unmasked_mean'}])
def test_no_joint_or_id_features_misrepresented_as_task_only(change):
    with pytest.raises(ValueError):validate_spec(dict(task_spec(),**change))
