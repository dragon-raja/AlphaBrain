"""One explicit CPU-only PPO boundary; collector batches use center authority.

Each invocation either initializes a run or ingests one complete published
version and performs one unchanged PPO update. No VLA or simulator is imported.
"""
import argparse
from pathlib import Path
import json
from AlphaBrain.research.dsol.data.artifacts import read
from AlphaBrain.research.dsol.execution.manifest import file_digest
from AlphaBrain.research.dsol.data import reward_pilot, domain_adaptation
from AlphaBrain.research.dsol.data import native_training


def step(config_path,*,records_path=None,execute_cpu=False):
    if not execute_cpu:raise ValueError('Explicit CPU optimizer action required')
    cfg=read(config_path)
    if cfg.get('schema') in ('PREFIX_PILOT_LEARNER_V1','PREFIX_MAINLINE_LEARNER_V1'):
        from .prefix_pilot import learner_step
        return learner_step(config_path,records_path=records_path)
    if cfg.get('schema') in ('PREFIX_PILOT_LEARNER_V1','PREFIX_MAINLINE_LEARNER_V1'):
        from .prefix_pilot import learner_step
        return learner_step(config_path,records_path=records_path)
    if cfg.get('schema')=='CONFIRMED_CHANGE_PILOT_LEARNER_V1':
        from .confirmed_pilot import learner_step
        return learner_step(config_path,records_path=records_path)
    domain=cfg['schema']==domain_adaptation.LEARNER_SCHEMA
    adaptive=cfg['schema']==reward_pilot.LEARNER_SCHEMA
    native=cfg['schema']==native_training.LEARNER_SCHEMA
    if native:
        protocol=native_training.from_campaign(cfg)
        if (cfg['arm']!='G' or cfg['seed'] not in (0,1,2) or cfg['updates']!=64 or cfg['episodes_per_update']!=32
                or cfg['schedule']!=protocol['schedules'][str(cfg['seed'])]
                or cfg['initial_checkpoint']!=protocol['initial_checkpoint'] or cfg['run']!=f"G-seed-{cfg['seed']}"):
            raise ValueError('Native G learner contract differs')
    elif domain:
        protocol=domain_adaptation.from_campaign(cfg)
        if (cfg['arm'] not in domain_adaptation.ARMS or cfg['seed'] not in (0,1,2)
                or cfg['updates']!=protocol['updates'] or cfg['episodes_per_update']!=8
                or cfg['schedule']!=protocol['schedules'][str(cfg['seed'])]
                or cfg['initial_checkpoint']!=protocol['source_checkpoints'][str(cfg['seed'])]['path']
                or cfg['run']!=f"{cfg['arm']}-seed-{cfg['seed']}"):
            raise ValueError('Wrong domain adaptation learner contract')
    elif adaptive:
        protocol=reward_pilot.from_campaign(cfg)
        if (cfg['arm'] not in protocol['arms'] or cfg['seed'] not in (0,1,2)
                or cfg['updates']!=protocol['updates'] or cfg['episodes_per_update']!=32
                or cfg['schedule']!=protocol['schedules'][str(cfg['seed'])]
                or cfg['initial_checkpoint']!=protocol['initial_checkpoints'][cfg['seed']]['path']
                or cfg['run']!=f"{cfg['arm']}-seed-{cfg['seed']}"):
            raise ValueError('Wrong reward adaptation learner contract')
    elif (cfg['schema']!='CONTROLLED_PPO_LEARNER_V1' or cfg['arm'] not in ('A','B')
            or cfg['seed'] not in (0,1,2) or cfg['updates']!=256 or cfg['episodes_per_update']!=32):
        raise ValueError('Matched A/B learner contract required')
    for path,stamp in cfg['input_hashes'].items():
        if file_digest(path)!=stamp:raise ValueError('Learner input changed')
    if not {cfg['schedule'],cfg['initial_checkpoint'],cfg['protocol']}.issubset(cfg['input_hashes']):
        raise ValueError('Unbound learner inputs')
    import torch
    from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
    from AlphaBrain.research.dsol.selectors.continuous_ppo import ContinuousPPO,ContinuousPPOConfig
    from AlphaBrain.research.dsol.selectors.continuous_parallel import ingest
    from AlphaBrain.research.dsol.data.continuous_training_store import atomic_file,write_json
    torch.set_num_threads(2)
    if domain:torch.manual_seed(cfg['seed'])
    if native:
        from AlphaBrain.research.dsol.selectors.search_policy_factory import create_search_policy
        policy=create_search_policy(cfg['native_policy'])
        policy.operation_model_hashes=protocol['model_hashes']
        policy.training_protocol_sha256=file_digest(cfg['protocol'])
        origin=native_training.publication_origin(protocol,policy.training_protocol_sha256,cfg['seed'])
    else:policy=ContinuousActorCritic(4608)
    initial=torch.load(cfg['initial_checkpoint'],map_location='cpu',weights_only=True)
    warm = adaptive and protocol['initial_update'] != 0
    initialization=None
    if native:
        if initial.get('optimizer_updates')!=0 or initial.get('feature_contract_sha256')!=policy.feature_contract_sha256:
            raise ValueError('Wrong native random initialization/feature contract')
        policy.load_state_dict(initial['state_dict'],strict=True)
    elif domain:
        from AlphaBrain.research.dsol.selectors.continuous_ppo import initialize_adaptation_actor
        if domain_adaptation.ARMS[cfg['arm']][0]=='source':
            initialization=initialize_adaptation_actor(policy,initial['policy'])
        else:
            initialization=dict(copied_actor_keys=[],critic_copied=False,optimizer_copied=False,
                reward_statistics_copied=False,feature_alignment_verified=False)
    elif warm:
        if initial['update']!=64 or initial['cursor']!=2048:
            raise ValueError('Wrong full warm-start checkpoint')
        policy.load_state_dict(initial['policy'],strict=True)
    else:
        if initial['optimizer_updates']!=0 or initial['visual_dim']!=4608:
            raise ValueError('Wrong original actor initialization')
        policy.load_state_dict(initial['state_dict'],strict=True)
    root=Path(cfg['output']);resume=records_path is not None
    if root.exists()!=resume:raise ValueError('Init/update lifecycle differs; no automatic recovery')
    def no_live_backend(*args):raise RuntimeError('CPU learner cannot render or score')
    with ContinuousPPO(policy,ContinuousPPOConfig(updates=cfg['updates'],seed=cfg['seed'],
            episodes_per_update=cfg['episodes_per_update']),read(cfg['schedule']),
            no_live_backend,root,binding=dict(config_sha256=file_digest(config_path),
                source_kind='LIVE_RENDER_BATCH1_TWO_NOISES'),resume=resume,
            reward_config=(native_training.reward_config() if native else domain_adaptation.reward_config(protocol,cfg['arm']) if domain else
                reward_pilot.reward_config(cfg['arm']) if adaptive else None),
            warm_start=initial if warm and not resume else None) as trainer:
        if records_path is not None:
            batch=read(records_path)
            if (batch['run']!=cfg['run'] or batch['version']!=trainer.update
                    or batch['policy_sha256']!=file_digest(root/'publications'/f'policy-{trainer.update:04d}.pt')):
                raise ValueError('Stale/mixed policy update')
            ingest(trainer,{int(k):v for k,v in batch['records'].items()})
            trainer.optimize()
        publications=root/'publications';publications.mkdir(exist_ok=True)
        target=publications/f'policy-{trainer.update:04d}.pt'
        if target.exists():raise ValueError('Published version cannot be overwritten or replayed')
        publication=dict(version=trainer.update,state_dict=trainer.policy.state_dict())
        if native:
            publication.update(feature_contract_sha256=policy.feature_contract_sha256,
                training_protocol_sha256=policy.training_protocol_sha256,origin=origin)
        atomic_file(target,lambda stream:torch.save(publication,stream))
        result=dict(status='COMPLETE' if trainer.update==cfg['updates'] else 'PAUSED_AT_UPDATE_BOUNDARY',
            version=trainer.update,episodes=trainer.cursor,queries=trainer.queries,
            policy=str(target),policy_sha256=file_digest(target),run=cfg['run'],
            operation_model_updates=0,optimizer_device='cpu')
        if domain:
            result.update(domain_sha256=file_digest(protocol['domain']),
                episodes_per_update=8,initialization=initialization)
        write_json(root/'publication.json',result)
        if trainer.update==cfg['updates']:
            write_json(root/'status.json',dict(status='COMPLETE',updates=trainer.update,
                episodes=trainer.cursor,queries=trainer.queries,scoring_trajectories=2*trainer.queries))
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--records')
    parser.add_argument('--execute-cpu',action='store_true')
    args=parser.parse_args()
    print(json.dumps(step(args.config,records_path=args.records,execute_cpu=args.execute_cpu)))
