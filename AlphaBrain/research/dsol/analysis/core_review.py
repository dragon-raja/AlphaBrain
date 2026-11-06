"""Check a presentation against already audited development outcomes.

This does not replace the source report's protocol/hash audit or recompute its
bootstrap. In particular, repeated policy seeds are not independent scenes.
"""
from collections import defaultdict

import numpy as np

from .core_decision import identity, summarize


def verify_presentation(rows, summary):
    panels = defaultdict(list)
    for row in rows:
        panels[row['panel']].append(row)
    if panels.keys() != summary['summaries'].keys():
        raise ValueError('Presentation panel coverage differs from audited report')
    baseline = None
    for name, records in panels.items():
        observed = summarize(records)
        observed.update(
            rescues=sum(r['success'] and not r['stay_success'] for r in records),
            harms=sum(not r['success'] and r['stay_success'] for r in records),
            stay_rate=float(np.mean([r['stay_success'] for r in records])),
            reference_rate=float(np.mean([r['nominal_success'] for r in records])))
        for key, value in observed.items():
            if not np.allclose(value, summary['summaries'][name][key], rtol=0, atol=1e-12):
                raise ValueError(f'Presentation mismatch: {name}/{key}')
        current = {identity(r): (r['stay_success'], r['nominal_success']) for r in records}
        if baseline is not None and current != baseline:
            raise ValueError('Presentation changed paired baseline or identities')
        baseline = current
    for name, comparison in summary['comparisons'].items():
        left, right = name.split(' minus ')
        a = {identity(r): r for r in panels[left]}
        b = {identity(r): r for r in panels[right]}
        if a.keys() != b.keys():
            raise ValueError('Contrast identity mismatch')
        computed = dict(
            mean_difference=np.mean([int(a[k]['success'])-int(b[k]['success']) for k in a]),
            query_difference=np.mean([a[k]['queries']-b[k]['queries'] for k in a]),
            rescues=sum(a[k]['success'] and not b[k]['success'] for k in a),
            harms=sum(b[k]['success'] and not a[k]['success'] for k in a))
        for key, value in computed.items():
            if not np.isclose(value, comparison[key], rtol=0, atol=1e-12):
                raise ValueError(f'Contrast mismatch: {name}/{key}')
    return dict(panels=len(panels), outcome_rows=len(rows),
                comparisons=len(summary['comparisons']),
                tolerance=1e-12, new_model_calls=0,
                scope='CSV aggregates crosschecked; upstream protocol audit reused; bootstrap not rerun')
