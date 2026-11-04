"""Descriptive geometry of already executed searches; no policy selection."""
from collections import defaultdict
import numpy as np


def rays(angles):
    """Relative optical directions as (right, forward, up), NOT camera positions."""
    a = np.asarray(angles, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
        raise ValueError('Finite yaw/pitch pairs required')
    yaw, pitch = np.deg2rad(a).T
    return np.column_stack((np.sin(yaw)*np.cos(pitch), np.cos(yaw)*np.cos(pitch), np.sin(pitch)))


def distance(angles):
    return np.rad2deg(np.arccos(np.clip(rays(angles)[:, 1], -1., 1.)))


def summarize(records, panel):
    panel = np.asarray(panel, dtype=float)
    rays(panel)
    groups = defaultdict(list); seen = set()
    for r in records:
        key = (r['state_key'], r['view_id'], r['method'])
        if key in seen: raise ValueError('Repeated search')
        seen.add(key)
        for field in ('success', 'stay_success', 'nominal_success'):
            if len(r[field]) != 2 or any(type(x) is not bool for x in r[field]):
                raise ValueError('Two paired Boolean outcomes required')
        for field in ('start_angles', 'selected_angles'):
            a = np.asarray(r[field], dtype=float)
            if a.shape != (2,) or not np.isfinite(a).all() or np.any(np.abs(a) > [30., 20.]):
                raise ValueError('Pose outside frozen pan/tilt domain')
        if not np.allclose(r['start_angles'], panel[r['view_id']], atol=1e-8, rtol=0):
            raise ValueError('Start differs from panel')
        groups[(r['split'], r['method'])].append(r)
    result = {}
    for (split, method), rows in sorted(groups.items()):
        start = np.asarray([r['start_angles'] for r in rows]); end = np.asarray([r['selected_angles'] for r in rows])
        ds, de = distance(start), distance(end)
        on_panel = np.any(np.all(np.abs(end[:, None]-panel[None]) <= 1e-7, axis=2), axis=1)
        y = np.asarray([r['success'] for r in rows]); stay = np.asarray([r['stay_success'] for r in rows]); nominal = np.asarray([r['nominal_success'] for r in rows])
        result[split+'/'+method] = dict(searches=len(rows), tasks=len({r['task'] for r in rows}),
            off_panel_count=int((~on_panel).sum()), off_panel_fraction=float((~on_panel).mean()),
            unchanged_fraction=float(np.all(np.abs(end-start)<=1e-7, axis=1).mean()),
            start_distance_median_deg=float(np.median(ds)), final_distance_median_deg=float(np.median(de)),
            final_distance_quartiles_deg=np.quantile(de, [.25,.5,.75]).tolist(),
            closer_fraction=float((de<ds-1e-7).mean()), farther_fraction=float((de>ds+1e-7).mean()),
            final_within_5deg_fraction=float((de<=5).mean()), exact_nominal_count=int(np.all(np.abs(end)<=1e-7,axis=1).sum()),
            selected_yaw_quantiles=np.quantile(end[:,0],[0,.25,.5,.75,1]).tolist(),
            selected_pitch_quantiles=np.quantile(end[:,1],[0,.25,.5,.75,1]).tolist(),
            success_rate=float(y.mean()), stay_rate=float(stay.mean()), nominal_rate=float(nominal.mean()),
            rescue_vs_stay=int((y & ~stay).sum()), harm_vs_stay=int((~y & stay).sum()),
            loss_vs_nominal_pairs=int((~y & nominal).sum()), gain_vs_nominal_pairs=int((y & ~nominal).sum()),
            comparison_scope='Search FROM perturbation versus paired nominal execution; NOT search starting at nominal.')
    return dict(groups=result, searches=len(records), nominal_start_searches=sum(r['view_id']==0 for r in records),
        interpretation='Fixed optical centre; two orientation DOFs. Off-panel is not outside angular limits. Descriptive, not proof of the causal reason for recovery.')
