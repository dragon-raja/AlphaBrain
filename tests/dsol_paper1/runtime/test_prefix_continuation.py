import json,sys
from pathlib import Path
import pytest
from benchmarks.LIBERO.prefix_continuation import frozen_stage


def test_frozen_phase_intent_prevents_replay_even_if_stdout_unknown(tmp_path):
    source=tmp_path/'source';folder=source/'benchmarks/LIBERO';folder.mkdir(parents=True)
    (source/'benchmarks/__init__.py').write_text('');(folder/'__init__.py').write_text('')
    (folder/'controlled_campaign.py').write_text('def run_stage(c,phase,path,*,kind,cpu):\n assert not cpu and kind=="training"\n return dict(status="COMPLETE",actual_model_calls=0,phase=phase)\n')
    cfg=tmp_path/'config.json';cfg.write_text('{}');log=tmp_path/'phase.log'
    value=frozen_stage(source,cfg,'batch',tmp_path/'campaign.json',log)
    assert value['status']=='COMPLETE' and value['actual_model_calls']==0
    with pytest.raises(FileExistsError):frozen_stage(source,cfg,'batch',tmp_path/'campaign.json',log)
