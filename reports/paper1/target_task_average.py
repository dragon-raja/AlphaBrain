"""CPU-only task-average supplement of a previously audited fixed spatial panel."""
import argparse
import csv
import json
from pathlib import Path

from AlphaBrain.research.dsol.analysis.spatial_task_average import summarize
from AlphaBrain.research.dsol.execution.manifest import file_digest
from AlphaBrain.research.dsol.plotting.spatial_task_average import figures
from reports.paper1.target_spatial_recovery import TASK_LABELS


def build(root, output, font_path):
    root, output = Path(root), Path(output)
    provenance_path = root/'p3-recovery-explanation/provenance.json'
    provenance = json.loads(provenance_path.read_text())
    rows_path = root/'report/rows.json'
    if file_digest(rows_path) != provenance['original_rows_sha256']:
        raise ValueError('Published rows changed since raw-data audit')
    summary = summarize(json.loads(rows_path.read_text()))
    output.mkdir(parents=True, exist_ok=False)
    (output/'analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (output/'task-average.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['view_id', 'yaw_deg', 'pitch_deg', 'mean_p3', 'mean_success', 'mean_within_state_rank'])
        for i, angles in enumerate(summary['angles_deg']):
            writer.writerow([i, *angles, *[summary['mean'][k][i] for k in ('score', 'success', 'rank')]])
    figures(summary, output, font_path=font_path, task_labels=TASK_LABELS)
    (output/'provenance.json').write_text(json.dumps(dict(rows=str(rows_path), rows_sha256=file_digest(rows_path),
        prior_audit_provenance_sha256=file_digest(provenance_path), font_sha256=file_digest(font_path),
        model_calls=0, training_updates=0, interpolation='piecewise linear display only; statistics use measured points'), indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--font', required=True)
    args = parser.parse_args()
    build(args.root, args.output, args.font)
