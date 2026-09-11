"""Extracted from the audited INITIAL32 analysis; preserve scientific semantics."""

import numpy as np


def metric_choices(acc, vis):
    top = np.argsort(acc, axis=1, kind="stable")[:, :10]
    return dict(
        min_accel=np.argmin(acc, axis=1),
        visibility=np.argmax(vis, axis=1),
        accel10_visibility=np.array([min(top[i], key=lambda c: (-vis[i, c], c)) for i in range(len(acc))]),
    )
