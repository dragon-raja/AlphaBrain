#!/usr/bin/env python3
"""Find the earliest observed differences in independent diagnostic traces."""
from __future__ import annotations

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[3]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import scripts.dsol_paper1.run_full64_view_landscape_v2 as control

ROOT = control.ROOT/'repeatability-root-cause-v1'


def difference(a, b):
    if a.shape != b.shape:
        return {'equal': False, 'shape_a': list(a.shape), 'shape_b': list(b.shape), 'max_abs': None}
    equal = a.dtype == b.dtype and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
    delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
    return {'equal': bool(equal), 'numeric_equal': bool(np.array_equal(a, b)), 'max_abs': float(delta.max()) if delta.size else 0.0,
            'mean_abs': float(delta.mean()) if delta.size else 0.0,
            'changed_elements': int(np.count_nonzero(a != b)), 'elements': int(a.size)}


def compare(a, b):
    results = {}
    for kind in ('call', 'step'):
        ap = sorted(a.glob(kind+'-*.npz')); bp = sorted(b.glob(kind+'-*.npz'))
        first = {}; largest = {}
        for index, (pa, pb) in enumerate(zip(ap, bp)):
            with np.load(pa, allow_pickle=False) as left, np.load(pb, allow_pickle=False) as right:
                for key in sorted(set(left.files) & set(right.files)):
                    if key.startswith('actions_probe_'): continue
                    d = difference(left[key], right[key])
                    if not d['equal'] and key not in first:
                        first[key] = {'index': index, **d}
                    if d['max_abs'] is not None:
                        largest[key] = max(largest.get(key, 0.0), d['max_abs'])
                for key in sorted(set(left.files) ^ set(right.files)):
                    if key.startswith('actions_probe_'): continue
                    if key not in first:
                        first[key] = {'index': index, 'missing_in': 'a' if key not in left.files else 'b'}
        results[kind] = {'lengths': [len(ap), len(bp)], 'first_differences': first, 'max_abs_over_common_prefix': largest}
    return results


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preview', action='store_true'); args = parser.parse_args()
    report = {'utc': control.stamp(), 'tasks': {}}
    for task in ('goal_wine_rack', 'libero10_mug_microwave', 'goal_cream_cheese_bowl'):
        traces = {name: ROOT/(task+'-'+name) for name in ('live-a','live-b','replay-a','replay-b')}
        completed = {name: path for name,path in traces.items() if (path/'completion.json').exists()}
        result = {'complete': list(completed), 'repeated_request_tests': {}}
        for name, path in completed.items():
            calls = control.read_json(path/'calls.json')
            if name.startswith('live'):
                values = [p for call in calls for p in call['repeated_request_comparisons'] if p['probe'] > 0]
                result['repeated_request_tests'][name] = {'comparisons': len(values),
                    'nonidentical': sum(not p['exact'] for p in values),
                    'max_abs': max([p['max_abs_error'] for p in values], default=0.0)}
        for label, left, right in [('closed_loop', 'live-a', 'live-b'),
                                   ('fixed_actions', 'replay-a', 'replay-b'),
                                   ('source_vs_replay', 'live-a', 'replay-a')]:
            if left in completed and right in completed:
                result[label] = compare(completed[left], completed[right])
        report['tasks'][task] = result
    report['status'] = ('PASS_ALL_DIAGNOSTICS_ANALYZED' if all(len(r['complete']) == 4 for r in report['tasks'].values())
                        else 'PARTIAL_DIAGNOSTIC_ANALYSIS')
    if not args.preview:
        control.require(report['status']=='PASS_ALL_DIAGNOSTICS_ANALYZED', 'Not all diagnostic traces complete')
        control.write_new_json(ROOT/'trace-analysis-v1.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
