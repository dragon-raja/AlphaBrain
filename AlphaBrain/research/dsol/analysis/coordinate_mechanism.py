"""Paired mounting interventions and endpoint sensitivity; no outcome simulation.

For the admitted virtual mount R_mount Ry(yaw) Rx(pitch), changing the mounting
yaw by b and the joint yaw by -b preserves the actual camera pose exactly.
This identity does not hold for arbitrary additive pitch/Euler offsets or an
unmodelled URDF chain. Only interior cases with identical action support qualify.
"""
import numpy as np


def mounting_intervention(observation, original_deg, mount_yaw_deg, *, bounds_deg=(30., 20.), step_deg=10.):
    q = np.asarray(original_deg, dtype=np.float64)
    bounds = np.asarray(bounds_deg, dtype=np.float64)
    if (q.shape != (2,) or bounds.shape != (2,) or not np.isfinite(q).all()
            or not np.isfinite(bounds).all() or np.any(bounds <= 0)
            or not np.isfinite([mount_yaw_deg, step_deg]).all() or step_deg <= 0):
        raise ValueError('Finite two-joint mount configuration required')
    moved = q - [mount_yaw_deg, 0.]
    if np.any(np.abs(q) + step_deg > bounds + 1e-12) or np.any(np.abs(moved) + step_deg > bounds + 1e-12):
        raise ValueError('Intervention would change the one-step action support')
    old_joint = np.concatenate((q / bounds, (q + bounds) / (2 * bounds), (bounds - q) / (2 * bounds))).astype(np.float32)
    if not np.array_equal(observation['joint_state'], old_joint):
        raise ValueError('Observation and declared mechanical joint coordinates differ')
    if not (np.array_equal(observation['delta_low'], [-step_deg]*2)
            and np.array_equal(observation['delta_high'], [step_deg]*2)):
        raise ValueError('Existing action interval is not the common interior support')
    result = {k: np.array(v, copy=True) for k, v in observation.items()}
    result['joint_state'] = np.concatenate((moved / bounds, (moved + bounds) / (2 * bounds),
                                           (bounds - moved) / (2 * bounds))).astype(np.float32)
    return result


def aggregate(rows):
    """Describe sensitivity; no success, improvement, or causal share claims."""
    summary = {}
    for seed in sorted({r['seed'] for r in rows}):
        rs = [r for r in rows if r['seed'] == seed]
        by_case = {}
        for r in rs:
            key = (r['task'], r['source_episode'])
            variants = by_case.setdefault(key, {})
            if r['mount_yaw_deg'] in variants:
                raise ValueError('Duplicate paired intervention')
            variants[r['mount_yaw_deg']] = r
        if any(set(v) != {-5., 0., 5.} for v in by_case.values()):
            raise ValueError('Incomplete paired mounting intervention')
        pairs = list(by_case.values())
        slopes = [(v[5.]['conditional_delta_deg'][0] - v[-5.]['conditional_delta_deg'][0])/10 for v in pairs]
        shifts = [np.linalg.norm(np.asarray(v[b]['conditional_delta_deg']) - v[0.]['conditional_delta_deg'])
                  for v in pairs for b in (-5., 5.)]
        flips = [v[b]['stop'] != v[0.]['stop'] for v in pairs for b in (-5., 5.)]
        probability_shifts = [abs(v[b]['stop_probability'] - v[0.]['stop_probability'])
                              for v in pairs for b in (-5., 5.)]
        summary[str(seed)] = dict(cases=len(pairs), tasks=len({r['task'] for r in rs}),
            median_conditional_action_shift_deg=float(np.median(shifts)),
            maximum_conditional_action_shift_deg=float(np.max(shifts)),
            median_yaw_response_per_mount_degree=float(np.median(slopes)),
            positive_yaw_response_percent=float(100*np.mean(np.asarray(slopes) > 1e-6)),
            deterministic_stop_flip_percent=float(100*np.mean(flips)),
            median_stop_probability_change=float(np.median(probability_shifts)),
            maximum_stop_probability_change=float(np.max(probability_shifts)))
    return summary


def endpoint_response(searches):
    """Paired physical endpoint response, not a causal shortcut percentage.

    A yaw response of one describes following the shifted mount one-for-one;
    zero describes invariant physical endpoints. Selection discontinuities and
    changed reachable sets prevent interpreting the slope as a mixture weight.
    """
    groups={}
    for row in searches:
        actor=row['method'].split('-mount-')[0]
        key=(row['state_key'],row['view_id'],actor)
        offset=row['mount_yaw_offset_deg']
        physical=np.asarray(row['selected_angles'],dtype=float)
        mechanical=np.asarray(row['mechanical_selected_angles'],dtype=float)
        if (physical.shape!=(2,) or not np.isfinite(physical).all()
                or not np.array_equal(physical,mechanical+[offset,0.])):
            raise ValueError('Endpoint frames disagree')
        variants=groups.setdefault(key,{})
        if offset in variants:raise ValueError('Duplicate endpoint pair')
        variants[offset]=physical
    if not groups or any(set(v)!={-5.,0.,5.} for v in groups.values()):
        raise ValueError('Complete endpoint triplets required')
    summary={}
    for actor in sorted({k[2] for k in groups}):
        pairs=[v for k,v in groups.items() if k[2]==actor]
        slopes=[(v[5.][0]-v[-5.][0])/10. for v in pairs]
        shifts=[float(np.linalg.norm(v[b]-v[0.])) for v in pairs for b in (-5.,5.)]
        summary[actor]=dict(paired_starts=len(pairs),
            median_yaw_response_per_mount_degree=float(np.median(slopes)),
            mean_yaw_response_per_mount_degree=float(np.mean(slopes)),
            median_endpoint_shift_joint_degrees=float(np.median(shifts)),
            maximum_endpoint_shift_joint_degrees=float(np.max(shifts)))
    return summary
