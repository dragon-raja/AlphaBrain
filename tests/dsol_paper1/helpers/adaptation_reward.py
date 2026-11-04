"""Synthetic domain-bound reward inputs; no scientific model or labels."""
from AlphaBrain.research.dsol.data.adaptation_reward import SCHEMA


def config(mode='window', domain='a'):
    return dict(schema=SCHEMA, domain_sha256=domain * 64, mode=mode,
                batch_size=8, window_size=32, noise_multiplier=2.,
                max_scale_ratio=2., query_cost=.01)


def batch(mean=1., std=.01, update=0):
    return [dict(status='COMPLETE', policy_update=update,
                 raw_reward=dict(schema='FORMAL_BATCH_REWARD_V1', initial_mean=mean,
                                 initial_std=std, score_members=2, collector_scale=1.),
                 transitions=[dict(best_previous=mean, best_current=.8 * mean,
                                   delta_queries=2, reward=mean - .8 * mean - .02)])
            for _ in range(8)]
