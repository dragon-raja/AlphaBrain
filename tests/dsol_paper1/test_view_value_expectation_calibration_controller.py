from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "scripts/dsol_paper1/operations/launchers/run_view_value_expectation_calibration_tail.sh"


def _count_rows_function() -> str:
    source = CONTROLLER.read_text(encoding="utf-8")
    match = re.search(r"(?ms)^count_rows\(\) \{.*?^\}\n", source)
    assert match is not None
    return match.group(0)


def test_count_rows_treats_a_new_stage_directory_as_empty(tmp_path: Path) -> None:
    missing = tmp_path / "stage-B"
    command = f"set -euo pipefail\n{_count_rows_function()}\ncount_rows \"$1\""
    result = subprocess.run(
        ["bash", "-c", command, "count-rows-test", str(missing)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == "0\n"


def test_run_stage_creates_output_before_counting() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    function = re.search(r"(?ms)^run_stage\(\) \{.*?^\}\n", source)
    assert function is not None
    body = function.group(0)
    assert body.index('mkdir -p "$output"') < body.index('actual=$(count_rows "$output")')
