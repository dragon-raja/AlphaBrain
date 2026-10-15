"""Compare a recovered training render with an existing successful reference.

CPU-only diagnostic: exact source/mount identities and fixed pixel tolerances.
No policy, simulator, dataset changes or relaxation of historical thresholds.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from AlphaBrain.research.dsol.data.artifacts import read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    old, new = [read(root / 'summary.json') for root in (args.reference, args.candidate)]
    exact_fields = ['job_id', 'source_hashes', 'layout_id', 'style_id', 'language',
                    'state_shape', 'action_shape', 'anchor_frame', 'fixed_local_mounts']
    fields = {key: old[key] == new[key] for key in exact_fields}
    same_files = set(old['image_sha256']) == set(new['image_sha256'])
    rows = []
    if same_files:
        for relative in sorted(old['image_sha256']):
            arrays = []
            for root, receipt in [(args.reference, old), (args.candidate, new)]:
                path = root / relative
                if sha(path) != receipt['image_sha256'][relative]:
                    raise ValueError('Image changed since receipt: ' + str(path))
                with Image.open(path) as image:
                    arrays.append(np.asarray(image.convert('RGB')).astype(np.int16))
            delta = np.abs(arrays[0] - arrays[1])
            maximum, mean, fraction = int(delta.max()), float(delta.mean()), float(np.mean(delta > 1))
            rows.append(dict(file=relative, max_abs=maximum, mean_abs=mean, fraction_gt1=fraction,
                             pixel_pass=maximum <= 2 and mean <= .002 and fraction <= .0001))
    result = dict(status='PASS' if all(fields.values()) and same_files and rows
                  and all(r['pixel_pass'] for r in rows) else 'HOLD',
        reference_receipt_sha256=sha(args.reference / 'summary.json'),
        candidate_receipt_sha256=sha(args.candidate / 'summary.json'),
        exact_identity_fields=fields, same_relative_files=same_files, images=rows,
        tolerances=dict(max_abs=2, mean_abs=.002, fraction_gt1=.0001),
        scope='Recorded training-frame render equivalence, not closed-loop policy equivalence')
    write(args.output, result)
    print(dict(status=result['status'], images=len(rows), exact_identity_fields=fields,
               max_pixel_error=max((r['max_abs'] for r in rows), default=None)), flush=True)


if __name__ == '__main__':
    main()
