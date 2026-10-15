import numpy as np
from AlphaBrain.research.dsol.analysis.prefix_replay import path_metrics


def test_path_not_net_motion():
    m=path_metrics(np.array([[0,0,0],[1,0,0],[0,0,0]]))
    assert m['path_length']==2 and m['displacement']==0
