"""Matched-panel Pearson comparison; frozen results, zero new model calls."""
import argparse
import json
from pathlib import Path
import numpy as np
from AlphaBrain.research.dsol.analysis.spatial_task_average import linear_comparison
from AlphaBrain.research.dsol.execution.manifest import file_digest
from AlphaBrain.research.dsol.plotting.spatial_task_average import linear_figures


def build(root, native_panel, output, font):
    root, native_panel, output = Path(root), Path(native_panel), Path(output)
    source = root/'report/rows.json'
    expected = json.loads((root/'p3-recovery-explanation/provenance.json').read_text())['original_rows_sha256']
    if file_digest(source) != expected:
        raise ValueError('Fixed spatial panel changed')
    rows = json.loads(source.read_text())
    identity = {(r['state_key'], r['view_id']): r['angles_deg'] for r in rows}
    native = []
    for r in json.loads(native_panel.read_text()):
        key = r['state_key'], r['view_id']
        if key not in identity: continue
        if not np.allclose(identity[key], [r['yaw_deg'], r['pitch_deg']], atol=1e-12, rtol=0):
            raise ValueError('Native panel geometry differs')
        # Native archived accel is already the mean of its two selection members.
        native.append(dict(state_key=r['state_key'], task=r['task'], view_id=r['view_id'],
            angles_deg=identity[key], p3=[r['accel']]*2, success=r['success']))
    if len(native) != len(rows) or {(r['state_key'], r['view_id']) for r in native} != set(identity):
        raise ValueError('Native comparison is not a complete matched panel')
    result = {f'fastwam_{score}': linear_comparison(rows, score=score) for score in ('p3', 'p9')}
    result['pi05_matched'] = linear_comparison(native)
    output.mkdir(parents=True, exist_ok=False)
    (output/'analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    linear_figures(result, output, font_path=font)
    (output/'provenance.json').write_text(json.dumps(dict(fastwam_panel=str(source), fastwam_sha256=file_digest(source),
        native_panel=str(native_panel), native_sha256=file_digest(native_panel), matched_states=16,
        matched_positions=len(rows), model_calls=0, training_updates=0,
        native_score='Archived mean of two scoring members; no independent-member information inferred from duplicated mean',
        limits='Matched task/state/angle identities; model-specific frozen execution protocols retained, not a causal training contrast.'), indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'native-panel', 'output', 'font'): p.add_argument('--'+name, required=True)
    a = p.parse_args(); build(a.root, a.native_panel, a.output, a.font)
