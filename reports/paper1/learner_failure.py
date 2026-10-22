"""Present existing attribution artifacts without fitting or inference."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from matplotlib import font_manager
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from AlphaBrain.research.dsol.data.artifacts import read,write,sha
from AlphaBrain.research.dsol.plotting.learner_failure import label_stability,decomposition,cross_scores,example_images


def run(config):
    cfg=read(config);root=Path(cfg['output']);out=root/'figures';out.mkdir(exist_ok=False)
    font_manager.fontManager.addfont(cfg['font'])
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=cfg['font']).get_name(),'axes.unicode_minus':False,'font.size':10})
    v2=read(root/'v2-labels.json');lib=read(root/'recovery-attribution.json');robo=read(root/'robocasa/attribution.json')
    label_stability(v2['rows'],out/'01-label-stability.png')
    decomposition([('LIBERO：规范训练模型',lib['rows'],'canonical'),('LIBERO：多视角训练模型',lib['rows'],'broad'),
                   ('RoboCasa：规范训练 π0.5',robo['rows'],'canonical'),('RoboCasa：多视角训练 π0.5',robo['rows'],'broad')],out/'02-search-ranking.png')
    cross_scores([('LIBERO / 规范训练',lib['cross_scoring'],'canonical'),('LIBERO / 多视角训练',lib['cross_scoring'],'broad'),
                  ('RoboCasa / 规范训练',robo['cross_scoring'],'canonical'),('RoboCasa / 多视角训练',robo['cross_scoring'],'broad')],out/'03-fixed-visits.png')
    examples=[];image_hashes={}
    for model in ('canonical','broad'):
        for kind in ('stable_gain','stable_harm'):
            rows=[r for r in lib['events'] if r['model']==model and r[kind]]
            rows.sort(key=lambda r:((-r['gain'] if kind=='stable_gain' else r['gain']),r['initial'],r['start']))
            e=rows[0];p=Path(e['asset']['images']);image_hashes[str(p)]=sha(p)
            meta=read(e['asset']['record'])
            if image_hashes[str(p)]!=meta['artifact_sha256']:raise ValueError('Image bank identity mismatch')
            with np.load(p) as z:images=z['external_images'].copy()
            examples.append(dict(event=e,images=images))
    example_images(examples,out/'04-visibility-cases.png')
    write(root/'visual-examples.json',[x['event'] for x in examples])
    source=root/'analysis-source';source.mkdir(exist_ok=False)
    repo=Path(__file__).resolve().parents[2]
    for name in ['AlphaBrain/research/dsol/analysis/learner_failure.py','AlphaBrain/research/dsol/plotting/learner_failure.py','reports/paper1/learner_failure.py','tests/dsol_paper1/analysis/test_learner_failure.py']:
        dest=source/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,dest)
    write(root/'presentation-receipt.json',dict(figures={str(p):sha(p) for p in out.glob('*.png')},image_banks=image_hashes,
        new_model_calls=0,fit_calls=0,source_snapshot=str(source)))
    print(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);run(p.parse_args().config)
