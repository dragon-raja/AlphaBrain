from __future__ import annotations

import pytest

from scripts.dsol_paper1.summarize_dsol_libero_hdf5_closed_loop import (
    PHYSICS_STATE_STAGE,
    audit_paired_physics,
)


def _row(state_hash: str, *, stage: str = PHYSICS_STATE_STAGE) -> dict:
    return {
        "initial_metrics": {
            "physics_state_sha256": state_hash,
            "physics_state_stage": stage,
        }
    }


def test_paired_physics_audit_accepts_exact_restored_source_state() -> None:
    audit_paired_physics(
        {
            "pair": {
                "canonical": _row("same"),
                "broad": _row("same"),
            }
        }
    )


def test_paired_physics_audit_rejects_wrong_hash_stage() -> None:
    with pytest.raises(ValueError, match="invalid paired physics hash stage"):
        audit_paired_physics(
            {
                "pair": {
                    "canonical": _row("same", stage="after_wait"),
                    "broad": _row("same", stage="after_wait"),
                }
            }
        )


def test_paired_physics_audit_rejects_source_state_mismatch() -> None:
    with pytest.raises(ValueError, match="paired physics state mismatch"):
        audit_paired_physics(
            {
                "pair": {
                    "canonical": _row("first"),
                    "broad": _row("second"),
                }
            }
        )
