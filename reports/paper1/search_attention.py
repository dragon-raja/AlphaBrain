"""Build figures and a compact CSV from completed native search outputs."""

import argparse
import csv
import json
from pathlib import Path

from AlphaBrain.research.dsol.plotting.search_attention import style,training_curves,replay_curves,live_task_figure
from AlphaBrain.research.dsol.analysis.search_live_summary import summarize_live


def build(training,output,font,live=None,audit=None):
    training,output=Path(training),Path(output)
    output.mkdir(parents=True,exist_ok=False);style(font)
    summary=json.loads((training/'summary.json').read_text())
    updates={seed:[json.loads(line) for line in (training/f'seed-{seed}'/'updates.jsonl').read_text().splitlines()]
             for seed in (0,1,2)}
    reference=None
    if audit:
        evidence=json.loads(Path(audit).read_text())
        reference=evidence['reference']['native_batch1_two_noise_97_query_success']
    figures={'training.png':training_curves(updates),'cached-evaluation.png':replay_curves(summary,reference)}
    if live:
        receipt=json.loads((Path(live)/'receipt.json').read_text())
        if receipt['status']!='COMPLETE':raise ValueError('Live results incomplete')
        figures['live-tasks.png']=live_task_figure(receipt)
        live_summary=summarize_live(receipt)
        (output/'live-comparison.json').write_text(json.dumps(live_summary,indent=2)+'\n')
        with (output/'live-comparison.csv').open('x') as stream:
            fields=['method','trials','live_successes','historical_successes','agreement',
                    'historical_fail_live_pass','historical_pass_live_fail','mean_queries','mean_steps']
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
            writer.writeheader();writer.writerows(live_summary['rows'])
    for name,figure in figures.items():figure.savefig(output/name,dpi=160)
    with (output/'comparison.csv').open('x') as stream:
        writer=csv.writer(stream);writer.writerow(['budget','method','success','queries','harmed_fraction','rescued_fraction'])
        for row in summary['rows']:
            writer.writerow([row['budget'],row['method']]+[row['metrics'][key]['mean'] for key in
                            ('success','queries','harmed_fraction','rescued_fraction')])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--font',required=True);parser.add_argument('--live');parser.add_argument('--audit');args=parser.parse_args()
    build(args.training,args.output,args.font,args.live,args.audit)
