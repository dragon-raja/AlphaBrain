"""Complete-search GAE and minibatch indices; never queries an environment."""
import numpy as np


def episode_targets(rewards, values, *, terminal, gamma=1., gae_lambda=.95):
    rewards, values = np.asarray(rewards, dtype=float), np.asarray(values, dtype=float)
    if (terminal is not True or rewards.ndim != 1 or rewards.shape != values.shape
            or not 1 <= len(rewards) <= 6 or not np.isfinite([rewards, values]).all()
            or not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1):
        raise ValueError('One complete finite search episode (one to six decisions) required')
    advantages = np.zeros_like(rewards)
    carry = 0.
    for step in reversed(range(len(rewards))):
        next_value = 0. if step == len(rewards) - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_value - values[step]
        carry = delta + gamma * gae_lambda * carry
        advantages[step] = carry
    return advantages.astype(np.float32), (advantages + values).astype(np.float32)


def complete_update(episodes, *, episodes_per_update=32):
    """Snapshot pre-action rows only after exactly 32 successful complete searches.

    Engineering failure aborts the update, never replaces a failed episode.
    Initial query cost is a logged constant, not an extra decision transition.
    """
    if len(episodes) != episodes_per_update:
        raise ValueError('Exact complete episode count required; do not fill with extra queries')
    rows = []
    for episode_id, episode in enumerate(episodes):
        transitions = episode['transitions']
        if episode['status'] != 'COMPLETE':
            raise ValueError('Incomplete or failed search must stop the update')
        advantages, returns = episode_targets([t['reward'] for t in transitions],
            [t['value'] for t in transitions], terminal=True)
        for step, (transition, advantage, target) in enumerate(zip(transitions, advantages, returns)):
            obs = {k: np.array(v, copy=True) for k, v in transition['observation'].items()}
            action = transition['action']
            mask = obs['action_mask']
            if isinstance(action, dict):
                stop = action.get('stop')
                latent = np.asarray(action.get('latent'), dtype=float)
                delta = np.asarray(action.get('delta_deg'), dtype=float)
                valid = (type(stop) is bool and latent.shape == delta.shape == (2,)
                         and np.isfinite([latent, delta]).all())
                if valid:
                    low, high = obs['delta_low'], obs['delta_high']
                    mapped = (high+low)/2 + (high-low)/2*np.tanh(latent)
                    valid = (not delta.any()) if stop else bool(np.allclose(delta, mapped, atol=1e-12, rtol=0))
                action = dict(stop=stop, latent=latent.copy(), delta_deg=delta.copy())
            else:
                valid = isinstance(action, (int, np.integer)) and 0 <= action < len(mask) and mask[action]
            if (not valid or not np.isfinite(transition['log_prob'])
                    or any(not np.isfinite(v).all() for v in obs.values())):
                raise ValueError('Invalid pre-action snapshot/action/log probability')
            rows.append(dict(observation=obs, action=action, old_log_prob=transition['log_prob'],
                advantage=float(advantage), returns=float(target), episode=episode_id, step=step))
    return rows


def minibatch_indices(count, *, seed, epochs=4, batch_size=64):
    if any(type(x) is not int or x < 1 for x in (count, epochs, batch_size)):
        raise ValueError('Positive batch sizes and epoch count required')
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(count)
        yield [order[i:i + batch_size] for i in range(0, count, batch_size)]
