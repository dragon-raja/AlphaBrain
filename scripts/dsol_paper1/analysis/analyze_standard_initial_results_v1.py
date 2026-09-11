#!/usr/bin/env python3
"""Compatibility entrypoint; implementation lives in AlphaBrain.research.dsol."""

from pathlib import Path as _RepoPath
import sys as _sys
_REPOSITORY_ROOT = _RepoPath(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPOSITORY_ROOT))


from pathlib import Path
import sys

REPO = _REPOSITORY_ROOT
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.metrics.view_rules import metric_choices
from AlphaBrain.research.dsol.analysis.statistics import hierarchy, paired_interval
from AlphaBrain.research.dsol.selectors.ridge import pca_fit, feature_matrix, ridge_predict, fit_ranker
from AlphaBrain.research.dsol.selectors.specification import CONTRACT
from AlphaBrain.research.dsol.analysis.pipeline import load_config, main

_config = load_config(REPO / "configs/dsol_paper1/analysis/standard_initial_v1.json")
ROOT, OUTPUT, WEIGHTS = _config.root, _config.output, _config.encoder_weights
MODELS = ["canonical", "broad"]
if __name__ == "__main__":
    args = sys.argv[1:]
    if "--config" not in args:
        args = ["--config", str(REPO / "configs/dsol_paper1/analysis/standard_initial_v1.json"), *args]
    main(args)
