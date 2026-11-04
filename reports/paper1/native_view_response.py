"""Validate every response-panel block before reporting all prescribed views."""
import argparse
import json
from pathlib import Path

from AlphaBrain.research.dsol.data.native_view_panel import views
from AlphaBrain.research.dsol.analysis.native_view_response import analyze
from AlphaBrain.research.dsol.plotting.native_view_response import figures
from benchmarks.LIBERO.native_view_panel import validate_block


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    root=args.run;read=lambda p:json.loads(Path(p).read_text());binding=read(root/'protocol/model-binding.json');campaign=read(binding['campaign'])
    reused=read(campaign['reuse']);tasks=list(reused['tasks']);scores=list(reused['scores']);proofs=[]
    for spec in binding['tasks'].values():
        proof=validate_block(spec['request'],spec['output']);proofs.append(proof)
        value=read(Path(spec['output'])/'results.json');tasks.extend(value['tasks']);scores.extend(value['scores'])
    result=analyze(campaign['states'],views(),tasks,scores);result['block_proofs']=proofs
    out=root/'analysis';out.mkdir(exist_ok=True)
    (out/'surface-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    figures(result,out)
    text=['# Native view response panel','',result['interpretation'],'',
          '| Task / initial | Nominal / 8 matched references | Old perturbed stay / 8 | Global lowest Accel / 8 | Rescue / harm |',
          '| --- | ---: | ---: | ---: | ---: |']
    for state in result['states']:
        s=state['original_starts'];text.append(f"| {state['task']} / {state['initial']} | {s['nominal_success']} | {s['stay_success']} | {s['selected_success']} | {s['rescues']} / {s['harms']} |")
    text+=['','Nominal references are reused for two starts. The global selector uses all 27 poses and is not an eight-query online method.',
           'All measured view scores and success fractions are in surface-summary.json. Red stars mark nominal poses in the figures. No training was performed.']
    (out/'README.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
