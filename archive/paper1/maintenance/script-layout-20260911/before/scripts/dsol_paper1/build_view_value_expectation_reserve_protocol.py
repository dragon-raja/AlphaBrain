#!/usr/bin/env python3
"""Materialize bank-F protocol identities after a machine reserve decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-protocol", type=Path, required=True)
    parser.add_argument("--reserve-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.primary_protocol.read_text())
    decision = json.loads(args.reserve_decision.read_text())
    if protocol.get("status") != "PASS" or protocol.get("bank_id") != "E":
        raise ValueError("reserve construction requires a complete primary bank-E protocol")
    if not decision.get("activate_bank_F") or decision.get("status") != "ACTIVATE_BANK_F":
        raise ValueError("bank F may only open after an ACTIVATE_BANK_F decision")
    specs = []
    for original in protocol["specs"]:
        spec = dict(original)
        identity = (
            f"expectation-v1::heldout-F::seed-{spec['checkpoint_seed']}::"
            f"{spec['pair_key']}::{spec['selector_method']}::"
            f"{spec['selected_candidate_id']}::{spec['policy_repeat_id']}"
        )
        spec.update(
            noise_bank_id="F",
            episode_id=hashlib.sha256(identity.encode()).hexdigest()[:24],
            diagnostic_role="heldout_precision_reserve_bank_F",
        )
        specs.append(spec)
    payload = {
        **protocol,
        "schema": "dsol_view_value_expectation_heldout_reserve_protocol_v1",
        "bank_id": "F",
        "episode_count": len(specs),
        "policy_noise_repeats": 32,
        "reserve_decision": str(args.reserve_decision.resolve()),
        "reserve_decision_sha256": hashlib.sha256(args.reserve_decision.read_bytes()).hexdigest(),
        "specs": specs,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"status": "PASS", "episodes": len(specs), "bank": "F"}, sort_keys=True))


if __name__ == "__main__":
    main()
