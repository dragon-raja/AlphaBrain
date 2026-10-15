import numpy as np
from benchmarks.Robocasa365.same_state_diagnostic import history_from_record


def test_full_seven_frame_history_is_kept_for_upstream_sampling(tmp_path):
    path=tmp_path/'data.npz'
    arrays={}
    for step in range(10,17):
        arrays[f'frame/{step:03d}/obs/state.test']=np.array([step])
        arrays[f'frame/{step:03d}/obs/video.camera']=np.full((2,2,3),step,dtype='uint8')
        arrays[f'frame/{step:03d}/sim/qpos']=np.array([999])
    np.savez(path,**arrays)
    with np.load(path) as z:history=history_from_record(z,16)
    assert len(history)==7
    assert [int(history[i]['state.test'][0]) for i in (0,2,4,6)]==[10,12,14,16]
    assert all(len(h)==2 for h in history)
