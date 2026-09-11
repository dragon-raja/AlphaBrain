from __future__ import annotations

from tests.dsol_paper1.paths import ROOT as REPOSITORY_ROOT
from pathlib import Path


ROOT = REPOSITORY_ROOT
CONTROLLER = ROOT / "scripts/dsol_paper1/operations/launchers/run_statewise_view_oracle_v2_tail.sh"


def test_tail_freezes_selector_before_test_dense_execution() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    selector = source.index("train_statewise_view_selector_v1.py")
    freeze_check = source.index('[[ -f "$SELECTOR_FREEZE" ]]')
    test_dense = source.index("test-O-wave-$wave")
    assert selector < freeze_check < test_dense


def test_tail_uses_independent_banks_in_order() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    p = source.index("run_matrix development-P")
    q = source.index("run_matrix development-Q")
    test_p = source.index("run_matrix test-P")
    test_q = source.index("run_matrix test-Q-seed41")
    assert p < q < test_p < test_q
    assert "bank_P.manifest.json" in source
    assert "bank_Q.manifest.json" in source
    assert 'POLICY_SERVER_COPIES_PER_GPU=${POLICY_SERVER_COPIES_PER_GPU:-2}' in source
    assert 'POLICY_CPU_THREADS=${POLICY_CPU_THREADS:-2}' in source
    assert 'SIM_CPU_THREADS=${SIM_CPU_THREADS:-1}' in source


def test_tail_cross_checkpoint_transfer_reuses_frozen_Q_protocol() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    assert 'run_matrix test-Q-seed42 "$Q_TEST_PROTOCOL"' in source
    assert 'run_matrix test-Q-seed43 "$Q_TEST_PROTOCOL"' in source
