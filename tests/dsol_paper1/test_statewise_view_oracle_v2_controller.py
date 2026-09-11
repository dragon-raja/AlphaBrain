from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "scripts/dsol_paper1/operations/launchers/run_statewise_view_oracle_v2_development_O.sh"


def test_controller_requires_smoke_before_formal_O() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    assert '"$ROOT/smoke/S/audit.json"' in source
    assert source.index('smoke.get("status") != "PASS_COMPLETE"') < source.index(
        "development_O_wave_start"
    )


def test_controller_runs_all_eight_four_repeat_waves_without_analysis() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    assert "for wave_index in $(seq 0 7)" in source
    assert "RUN_ANALYSIS=0" in source
    assert "WAIT_STEPS=0" in source
    assert "REQUIRE_EXPLICIT_NOISE=1" in source
    assert 'POLICY_SERVER_COPIES_PER_GPU=${POLICY_SERVER_COPIES_PER_GPU:-2}' in source
    assert 'POLICY_CPU_THREADS=${POLICY_CPU_THREADS:-2}' in source
    assert 'SIM_CPU_THREADS=${SIM_CPU_THREADS:-1}' in source
    assert "audit_statewise_view_oracle_v2_run.py" in source
