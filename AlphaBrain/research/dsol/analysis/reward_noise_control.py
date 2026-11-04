"""Selection-bias controls for a proposed reward, not a model efficacy test."""
import numpy as np


def independent_terminal_correction(*, search_gain, confirmation_initial,
                                    confirmation_selected, scale):
    """Cancel search-minimum shaping and replace it with signed held-out gain.

    The caller must establish that confirmation noise is independent of ALL
    acquisition/selection/scale decisions. Different array names or two-member
    agreement do not establish that condition. This primitive does not decide
    whether the selected view is accepted, or infer task success from Accel.
    """
    a, b = np.asarray(confirmation_initial, float), np.asarray(confirmation_selected, float)
    gain, scale = np.asarray(search_gain, float), np.asarray(scale, float)
    if (a.shape != b.shape or a.ndim < 1 or a.shape[-1] != 2
            or gain.shape != a.shape[:-1] or scale.ndim != 0
            or not all(np.isfinite(v).all() for v in (a,b,gain,scale))
            or (a < 0).any() or (b < 0).any() or (gain < 0).any() or scale <= 0):
        raise ValueError('Finite paired confirmation scores, nonnegative search gain and positive fixed scale required')
    confirmed = (a-b).mean(axis=-1)
    return (confirmed-gain)/scale


def noise_control(*, seed, episodes=4096, sigma=.001, signal=0., query_cost=.01):
    """Five visited poses plus return and two confirmations fit eight queries.

    Latent candidate scores decrease by ``signal`` per pose; zero is a no-view-
    benefit null. These synthetic controls are not robot/model observations.
    Scale and selection use search noise exclusively; confirmation is withheld.
    """
    if (type(episodes) is not int or episodes < 2 or not 0 < sigma <= .01
            or not 0 <= signal <= .01 or query_cost <= 0):
        raise ValueError('Bounded synthetic control parameters required')
    rng = np.random.default_rng(seed)
    truth = .5 - signal*np.arange(5)
    scores = truth[None,:,None] + rng.normal(0,sigma,(episodes,5,2))
    members = np.diff(scores,axis=1)
    change = np.abs(members.mean(axis=-1))
    disagreement = np.abs(members[...,0]-members[...,1])/2
    scale = max(float(np.quantile(change,.75)), 2*float(np.quantile(disagreement,.75)))
    selected = scores.mean(-1).argmin(-1)
    initial = scores[:,0].mean(-1)
    best = scores[np.arange(episodes),selected].mean(-1)
    gain = initial-best
    # Start + 4 candidate queries, optional return, 2 independent confirmations.
    queries = 7 + (selected != 4).astype(int)
    cost = query_cost*(queries-1)
    held_initial = truth[0] + rng.normal(0,sigma,(episodes,2))
    held_selected = truth[selected,None] + rng.normal(0,sigma,(episodes,2))
    naive = gain/scale-cost
    corrected = naive + independent_terminal_correction(search_gain=gain,
        confirmation_initial=held_initial, confirmation_selected=held_selected, scale=scale)
    def summary(values):
        mean=float(values.mean());se=float(values.std(ddof=1)/np.sqrt(episodes))
        return dict(mean=mean, descriptive_mean_interval=[mean-1.96*se,mean+1.96*se])
    return dict(scope='SYNTHETIC_NOISE_CONTROL_NOT_OPERATION_SUCCESS', seed=seed,
        episodes=episodes,sigma=sigma,signal=signal,scale=scale,
        max_queries=int(queries.max()),naive=summary(naive),corrected=summary(corrected),
        latent_mean_improvement=float(np.mean(truth[0]-truth[selected])),new_model_calls=0)
