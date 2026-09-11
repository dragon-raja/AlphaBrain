from __future__ import annotations

from tests.dsol_paper1.paths import ROOT as REPOSITORY_ROOT
import re
from pathlib import Path


ROOT = REPOSITORY_ROOT
CONTROLLER = ROOT / "scripts/dsol_paper1/operations/launchers/run_view_value_expectation_post_calibration.sh"


def test_accel_render_exposes_openpi_client_to_sim_python() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    match = re.search(
        r'(?ms)if \[\[ "\$stage" == render \]\]; then(?P<body>.*?)^\s*else$',
        source,
    )
    assert match is not None
    render_branch = match.group("body")
    assert "/projects/openpi/packages/openpi-client/src" in render_branch
    assert '"$SIM_PYTHON"' in render_branch
