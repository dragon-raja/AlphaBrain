"""Predeclared, bounded render-tolerance proposal; never edits old receipts."""


MEAN_CANDIDATES = (0.002, 0.005, 0.01)
FRACTION_CANDIDATES = (0.0001, 0.0002, 0.0005)


def fraction_calibration_population(rows, excluded_layouts):
    calibration = confirmation_population(rows, excluded_layouts)
    validation = confirmation_population(rows, excluded_layouts | {r['layout_id'] for r in calibration})
    return ([dict(r, role='calibration') for r in calibration]
            + [dict(r, role='validation') for r in validation])


def propose_fraction_gate(calibration, validation):
    if not calibration or not validation:
        raise ValueError('Independent calibration and validation required')
    maximum = max(c['fraction_gt1'] for c in calibration)
    target = maximum if maximum <= FRACTION_CANDIDATES[0] else maximum * 1.25
    proposed = next((t for t in FRACTION_CANDIDATES if t >= target), None)
    fixed_ok = lambda c: c['max_abs'] <= 2 and c['mean_abs'] <= .005 and c['fraction_gt2'] == 0.
    passed = (proposed is not None and all(fixed_ok(c) for c in calibration + validation)
              and all(c['fraction_gt1'] <= proposed for c in validation))
    return dict(status='VALIDATED_FRACTION_PROPOSAL_NOT_APPLIED' if passed else 'HOLD',
        proposed_fraction_gt1=proposed, calibration_max_fraction_gt1=maximum,
        validation_max_fraction_gt1=max(c['fraction_gt1'] for c in validation),
        fixed_gate_failures=sum(not fixed_ok(c) for c in calibration + validation),
        unchanged_limits=dict(max_abs=2, mean_abs=.005, fraction_gt2=0.),
        candidates=list(FRACTION_CANDIDATES), margin=1.25,
        scope='Seen-only engineering proposal; cannot auto-admit quarantined target samples')


def calibration_population(rows):
    tasks = sorted({r['task'] for r in rows})
    selected = [dict(r, role='calibration' if r['scene_seed'] == 0 else 'validation')
                for r in rows if r['split'] == 'pretrain' and r['scene_seed'] in (0, 1)]
    if len(selected) != 2 * len(tasks) or any(
            sum(r['task'] == task and r['scene_seed'] == seed for r in selected) != 1
            for task in tasks for seed in (0, 1)):
        raise ValueError('Exactly two predetermined seen initials per task required')
    layouts = {role: {r['layout_id'] for r in selected if r['role'] == role}
               for role in ('calibration', 'validation')}
    if layouts['calibration'] & layouts['validation']:
        raise ValueError('Calibration and validation layouts overlap; do not silently reselect')
    return selected


def hard_pixel_gate(check):
    return (check['max_abs'] <= 2 and check['fraction_gt1'] <= .0001
            and check['fraction_gt2'] == 0.)


def confirmation_population(rows, excluded_layouts):
    """Fresh confirmation: first eligible later seen seed, metadata selection only."""
    selected = []
    for task in sorted({r['task'] for r in rows}):
        eligible = sorted((r for r in rows if r['task'] == task and r['split'] == 'pretrain'
                           and r['scene_seed'] >= 2 and r['layout_id'] not in excluded_layouts),
                          key=lambda r: r['scene_seed'])
        if not eligible:
            raise ValueError('No unused confirmation layout for ' + task)
        selected.append(dict(eligible[0], role='confirmation'))
    return selected


def validate_fixed_gate(checks, mean_limit):
    if not checks or mean_limit not in MEAN_CANDIDATES:
        raise ValueError('Checks and a predeclared fixed candidate required')
    return dict(status='VALIDATED_FIXED_PROPOSAL_NOT_APPLIED' if all(
        hard_pixel_gate(c) and c['mean_abs'] <= mean_limit for c in checks) else 'HOLD',
        proposed_mean_abs=mean_limit, validation_max_mean_abs=max(c['mean_abs'] for c in checks),
        validation_hard_failures=sum(not hard_pixel_gate(c) for c in checks),
        unchanged_limits=dict(max_abs=2, fraction_gt1=.0001, fraction_gt2=0.),
        scope='Fixed proposal confirmed on fresh initials; not a VLA action equivalence claim')


def propose_mean_gate(calibration, validation):
    """Choose on calibration only; validation failure cannot increase the gate."""
    if not calibration or not validation:
        raise ValueError('Both independent groups required')
    maximum = max(c['mean_abs'] for c in calibration)
    target = maximum if maximum <= MEAN_CANDIDATES[0] else maximum * 1.25
    proposed = next((t for t in MEAN_CANDIDATES if t >= target), None)
    return dict(status='VALIDATED_PROPOSAL_NOT_APPLIED' if proposed is not None
                and all(hard_pixel_gate(c) for c in calibration + validation)
                and all(c['mean_abs'] <= proposed for c in validation) else 'HOLD',
                proposed_mean_abs=proposed, calibration_max_mean_abs=maximum,
                validation_max_mean_abs=max(c['mean_abs'] for c in validation),
                calibration_hard_failures=sum(not hard_pixel_gate(c) for c in calibration),
                validation_hard_failures=sum(not hard_pixel_gate(c) for c in validation),
                candidates=list(MEAN_CANDIDATES), margin=1.25,
                unchanged_limits=dict(max_abs=2, fraction_gt1=.0001, fraction_gt2=0.),
                scope='Renderer numerical admission only; no claim of unchanged VLA actions or success')
