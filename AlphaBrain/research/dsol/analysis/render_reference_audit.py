"""Separate historical-image agreement from present-runtime repeatability.

This audit does not replace references, modify tolerances, or release data.
Only unmodified-renderer stages may participate in its paired comparisons.
"""

from pathlib import Path
import numpy as np

from ..data.artifacts import read, sha
from .render_calibration_results import pixel_difference


def summarize_pairs(first, second):
    if len(first) == 0 or len(second) == 0:
        raise ValueError('Two nonempty independently generated image groups required')
    checks = [pixel_difference(a, b) for a in first for b in second]
    passed = lambda c: (c['max_abs'] <= 2 and c['mean_abs'] <= .005
                        and c['fraction_gt1'] <= .0001 and c['fraction_gt2'] == 0)
    return dict(comparisons=len(checks), failures=sum(not passed(c) for c in checks),
                maximum={key: max(c[key] for c in checks) for key in checks[0]})


def audit_reference(localization, repetition):
    localization, repetition = Path(localization), Path(repetition)
    first, second = read(localization / 'summary.json'), read(repetition / 'summary.json')
    if first['status'] != 'DIAGNOSTIC_COMPLETE_NOT_ADMITTED' or first['snapshot'] != second['snapshot']:
        raise ValueError('Completed diagnostics for the same snapshot required')
    manifest_path = Path(first['snapshot']) / 'manifest.json'
    manifest = read(manifest_path)
    fingerprints = [x[key] for x in (first, second) for key in ('numeric_start', 'numeric_end')]
    if (len(set(fingerprints)) != 1 or fingerprints[0] != manifest['snapshot_numeric_sha256']
            or first['manifest_sha256'] != sha(manifest_path)):
        raise ValueError('Physical state or source manifest mismatch')
    if (first['files']['render_images.npz'] != sha(localization / 'render_images.npz')
            or second['images_sha256'] != sha(repetition / 'images.npz')):
        raise ValueError('Diagnostic image checksum mismatch')
    canonical = Path(first['snapshot']) / 'canonical_rgb.npz'
    if sha(canonical) != manifest['files']['canonical_rgb.npz']['sha256']:
        raise ValueError('Historical image checksum mismatch')
    cameras = {}
    with (np.load(localization / 'render_images.npz', allow_pickle=False) as a,
          np.load(repetition / 'images.npz', allow_pickle=False) as b,
          np.load(canonical, allow_pickle=False) as historical):
        for camera in first['cameras']:
            original = historical['video.' + camera]
            if not np.array_equal(original, a[camera + '_historical']):
                raise ValueError('Historical image was changed')
            first_group = np.concatenate([a[camera + '_current'][None], a[camera + '_repeats']])
            second_group = []
            for stage in ('wrapper', 'direct', 'restore_each'):
                receipt = second['stages'][stage]
                if not receipt['poses_exact'] or receipt['unique_geom_buffers'] != 1:
                    raise ValueError('Pose or scene geometry changed')
                second_group.extend(b[stage + '_' + camera])
            cameras[camera] = dict(
                historical_agreement=summarize_pairs([original], [*first_group, *second_group]),
                current_cross_process=summarize_pairs(first_group, second_group))
    return dict(status='AUDIT_ONLY_NO_RELEASE', snapshot=first['snapshot'],
                source_receipts={str(p / 'summary.json'): sha(p / 'summary.json')
                                 for p in (localization, repetition)},
                numeric_sha256=fingerprints[0], cameras=cameras,
                gate=dict(max_abs=2, mean_abs=.005, fraction_gt1=.0001, fraction_gt2=0),
                limitations=['Post-hoc diagnosis of quarantined initials; not independent tolerance calibration',
                             'No pixel-reference replacement and no admission authorization',
                             'Finite same-host comparison; not a closed-loop equivalence guarantee'])
