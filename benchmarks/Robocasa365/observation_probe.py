"""Explicit forward-only fixed-panel diagnostic; no simulator construction."""
import argparse
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--config',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if not args.execute:p.error('Explicit --execute required')
    from AlphaBrain.research.dsol.analysis.observation_evidence import run_probe
    from AlphaBrain.research.dsol.data.artifacts import sha
    from openpi_client import image_tools
    from .evaluation_policy import ResidentPolicy
    policy=ResidentPolicy(args.checkpoint,args.config)
    checked=False
    def predict(images,state,record,noise):
        nonlocal checked
        request={'prompt':record['prompt'],'observation/state':state}
        for key,image in zip(('image','right_image','wrist_image'),images):
            request['observation/'+key]=image_tools.convert_to_uint8(image_tools.resize_with_pad(image,224,224))
        traced=policy.trace(request,noise)
        if not checked:
            np.testing.assert_allclose(traced['actions'],policy.infer(request,noise),rtol=1e-5,atol=1e-5)
            checked=True
        return traced['actions'],traced['velocity'][0]
    identity=dict(checkpoint=str(args.checkpoint),config=args.config,
        norm_sha256=sha(args.checkpoint/'assets/norm_stats.json'),adapter_sha256=sha(__file__))
    run_probe(args.panel,'robocasa',args.output,identity,predict,(50,32))


if __name__=='__main__':main()
