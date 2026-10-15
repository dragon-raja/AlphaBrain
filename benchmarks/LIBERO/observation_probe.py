"""AlphaBrain released LIBERO checkpoints, explicit forward-only diagnostic."""
import argparse
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    if not args.execute:p.error('Explicit --execute required')
    import torch
    from AlphaBrain.model.framework.base_framework import BaseFramework
    from AlphaBrain.research.dsol.analysis.observation_evidence import run_probe
    from AlphaBrain.research.dsol.data.artifacts import sha
    torch.set_num_threads(1)
    model=BaseFramework.from_pretrained(str(args.checkpoint),strict_checkpoint=True)
    model=model.to(torch.bfloat16).to('cuda').eval();model.gripper_remap=False
    for parameter in model.parameters():parameter.requires_grad_(False)
    def predict(images,state,record,noise):
        example={'image':list(images),'lang':record['prompt'],'language':record['prompt'],'state':state}
        result=model.predict_action(examples=[example],noise=noise[None],return_flow_trace=True)
        np.testing.assert_array_equal(np.asarray(result['flow_initial_noise'],np.float32),noise[None])
        # Historical key is named normalized_actions but API already decodes mean/std.
        return np.asarray(result['normalized_actions'])[0],np.asarray(result['flow_velocity_trace'])[0]
    identity=dict(checkpoint=str(args.checkpoint),weights_sha256=sha(args.checkpoint/'model.safetensors'),
        config_sha256=sha(args.checkpoint/'framework_config.yaml'),adapter_sha256=sha(__file__),state_input=False)
    run_probe(args.panel,'libero',args.output,identity,predict,(10,7))


if __name__=='__main__':main()
