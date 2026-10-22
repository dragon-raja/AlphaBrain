"""Re-express existing evidence for a narrative explanation; no experiments."""
import argparse
from pathlib import Path
from AlphaBrain.research.dsol.data.artifacts import read,write,sha
from AlphaBrain.research.dsol.plotting.action_structure import style
from AlphaBrain.research.dsol.plotting.accel_story import choice_history,cabinet_progress,same_state_explained,outcome_overlap


def build(config):
    cfg=read(config);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    for p,h in cfg['input_sha256'].items():
        if sha(p)!=h:raise ValueError('Source changed: '+p)
    write(out/'config.json',cfg);style(cfg['font'])
    choice_history(cfg['choice_cases'],out/'01-why-this-question.png')
    cabinet_progress(read(cfg['physical_summary'])['original_stage_tables'],out/'02-closer-but-penalized.png')
    r=next(r for r in read(cfg['same_state_summary'])['contrasts'] if r['initial']=='39dd9788d2b8a223cfc09850' and r['step']==48 and r['metric']=='baseline__score')
    same_state_explained(r['cells'],out/'03-same-state-plain.png')
    outcome_overlap(read(cfg['transitions']),out/'04-smoothness-counterexamples.png')
    write(out/'receipt.json',dict(status='COMPLETE',scope='Presentation only; existing means and fixed comparisons, no new score or selection.',new_model_calls=0,new_actions=0,new_renderings=0,
        aliases=cfg['aliases'],figures={p.name:sha(p) for p in out.glob('*.png')},source_sha256=sha(__file__)))
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);build(p.parse_args().config)
