"""Fixed, label-free decomposition of saved denoising velocities (CPU only).

Directions refer to normalized model coordinates, not physical motion. Radial
and angular scores are separate diagnostics, not additive parts of Accel.
"""
import numpy as np

from .accel import compute_accel_scores


def components(velocity, prefix=3, epsilon=1e-12):
    """Accept [..., denoise, horizon, dimension]; retain all leading axes.

    The squared increment identity is reconstructed before taking square roots.
    Undefined directions are NaN, never rewarded as zero. Magnitude diagnostics
    remain available at zero norm; zero total denominators remain invalid.
    """
    v = np.asarray(velocity, dtype=np.float64)
    if v.ndim < 3 or v.shape[-3] < prefix or not np.isfinite(v).all():
        raise ValueError('Invalid saved velocity trace')
    lead = v.shape[:-3]
    v = v[..., :prefix, :, :].reshape(-1, prefix, v.shape[-2] * v.shape[-1])
    if not v.shape[-1]:
        raise ValueError('Empty subspace')
    m = compute_accel_scores(v, prefixes=(prefix,), epsilon=epsilon)[prefix]
    r = np.linalg.norm(v, axis=-1)
    good = r > epsilon
    u = np.divide(v, r[..., None], out=np.zeros_like(v), where=good[..., None])
    pair_good = good[:, 1:] & good[:, :-1]
    radial = np.diff(r, axis=1) ** 2
    angular = r[:, 1:] * r[:, :-1] * np.sum(np.diff(u, axis=1) ** 2, axis=-1)
    energy = np.sum(np.diff(v, axis=1) ** 2, axis=-1)
    # Undefined angular coordinates do not prevent checking the raw identity:
    # when either radius is zero, the angular energy is exactly zero.
    if not np.allclose(radial + angular, energy, atol=1e-12, rtol=1e-11):
        raise ValueError('Squared increment identity failed')
    denominator = np.where(m['degenerate'], np.nan, m['denominator'])
    direction_valid = pair_good.all(1)
    result = dict(score=np.where(m['degenerate'], np.nan, m['score']),
        radial=prefix * np.sqrt(radial).sum(1) / denominator,
        angular=np.where(direction_valid, prefix * np.sqrt(angular).sum(1) / denominator, np.nan),
        radial_energy=radial.sum(1),
        angular_energy=np.where(direction_valid, angular.sum(1), np.nan),
        change_energy=energy.sum(1), denominator=m['denominator'],
        direction_valid=direction_valid, degenerate=m['degenerate'],
        reconstruction_error=np.abs(prefix*np.sqrt(radial+angular).sum(1)/denominator-m['score']))
    return {k: x.reshape(lead) for k, x in result.items()}


def structure(velocity, model):
    """Fixed scopes, with no use of outcome labels or fitted scales."""
    v = np.asarray(velocity)
    if model not in ('libero', 'official', 'xr1'):
        raise ValueError('Unknown model coordinate convention')
    horizon, dim = (5, 7) if model == 'libero' else ((25, 12) if model == 'official' else (16, 12))
    baseline = v[..., :16, :12] if model == 'xr1' else v
    near = v[..., :horizon, :dim]
    pieces = {'baseline': baseline, 'near': near}
    if baseline.shape[-2] > horizon:
        pieces['far'] = baseline[..., horizon:, :dim]
    groups = {'translation': slice(0, 3), 'rotation': slice(3, 6), 'gripper': slice(6, 7)}
    if dim == 12:
        groups.update(base=slice(7, 11), control_mode=slice(11, 12))
    if baseline.shape[-1] > dim:
        groups['inactive'] = slice(dim, baseline.shape[-1])
    result = {}
    for scope, x in pieces.items():
        result.update({scope+'__'+k: a for k, a in components(x).items()})
    # Channel contributions add back to the original numerator after dividing
    # each channel's squared increment by the full increment norm, pairwise.
    for scope, x in [('baseline', baseline), ('near', near)]:
        dv = np.diff(np.asarray(x[..., :3, :, :], dtype=np.float64), axis=-3)
        length = np.sqrt(np.sum(dv**2, axis=(-2, -1)))
        denominator = result[scope+'__denominator']
        contributions = []
        for name, sl in groups.items():
            if sl.start >= x.shape[-1]:
                continue
            key = scope+'_'+name
            result.update({key+'__'+k: a for k, a in components(x[..., sl]).items()})
            e = np.sum(dv[..., sl]**2, axis=(-2, -1))
            n = np.divide(e, length, out=np.zeros_like(e), where=length > 0).sum(-1)
            c = np.divide(3*n, denominator, out=np.full_like(n, np.nan), where=denominator > 1e-12)
            result[key+'__accel_contribution'] = c
            contributions.append(c)
        if not np.allclose(np.sum(contributions, axis=0), result[scope+'__score'], atol=1e-12, rtol=1e-11, equal_nan=True):
            raise ValueError('Channel attribution failed to reconstruct Accel')
    return result
