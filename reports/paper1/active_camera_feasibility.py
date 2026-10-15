"""Render the explicit CPU feasibility audit, without changing the main PDF."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from AlphaBrain.research.dsol.data.artifacts import read,write,sha
from AlphaBrain.research.dsol.plotting.active_camera_feasibility import audit_figure


def build(config):
    out=Path(read(config)['output']);audit=out/'audit.json';provenance=read(out/'provenance.json')
    if sha(audit)!=provenance['outputs']['audit.json']:raise ValueError('Audit changed')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False})
    fig=audit_figure(read(audit));image=out/'feasibility.png';fig.savefig(image,dpi=160,bbox_inches='tight');plt.close(fig)
    from AlphaBrain.research.dsol.plotting import active_camera_feasibility
    write(out/'figure-provenance.json',dict(input_sha256=sha(audit),image_sha256=sha(image),sources={p:sha(p) for p in (__file__,active_camera_feasibility.__file__)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);build(p.parse_args().config)
