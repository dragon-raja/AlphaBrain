"""Descriptive structure of saved LIBERO control commands, never a selector."""
import numpy as np


FEATURES = ('close_fraction', 'gripper_switch', 'gripper_variation',
            'rotation_magnitude', 'rotation_variation', 'rotation_reversal')


def action_structure(actions, horizon):
    """Keep leading axes. Clip exactly as the historical execution adapter.

    Rotation is OSC input, not a measured angle. Zero gripper is hold; count
    only direct open/close sign reversals. Undefined rotation directions remain
    NaN. No threshold is estimated from outcomes.
    """
    raw = np.asarray(actions, dtype=float)
    if raw.shape[-1] != 7 or not 2 <= horizon <= raw.shape[-2] or not np.isfinite(raw).all():
        raise ValueError('Invalid action block or horizon')
    x = np.clip(raw[..., :horizon, :], -1, 1)
    g, r = x[..., 6], x[..., 3:6]
    norm = np.linalg.norm(r, axis=-1)
    valid = (norm[..., 1:] > 1e-12) & (norm[..., :-1] > 1e-12)
    reverse = (r[..., 1:, :] * r[..., :-1, :]).sum(-1) < 0
    n = valid.sum(-1)
    return dict(close_fraction=(g > 0).mean(-1),
                gripper_switch=(g[..., 1:] * g[..., :-1] < 0).mean(-1),
                gripper_variation=np.abs(np.diff(g, axis=-1)).mean(-1),
                rotation_magnitude=norm.mean(-1),
                rotation_variation=np.linalg.norm(np.diff(r, axis=-2), axis=-1).mean(-1),
                rotation_reversal=np.divide((reverse & valid).sum(-1), n,
                    out=np.full(n.shape, np.nan), where=n > 0))


def outcome_label(delta):
    if delta is None or not np.isfinite(delta):
        return 'unknown'
    return 'improved' if delta > 1e-12 else 'degraded' if delta < -1e-12 else 'equal'


def ordinal_concordance(positive, negative):
    """P(feature positive > negative) + half ties; no fitted direction/cutoff."""
    a, b = np.asarray(positive, float), np.asarray(negative, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return None
    d = a[:, None] - b[None, :]
    return float(((d > 0) + .5 * (d == 0)).mean())
