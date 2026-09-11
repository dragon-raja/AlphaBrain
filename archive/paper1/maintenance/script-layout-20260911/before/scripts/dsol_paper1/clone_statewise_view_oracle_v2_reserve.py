#!/usr/bin/env python3
"""Clone a frozen Q-test candidate matrix onto the pre-materialized R bank."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

try:
    from .build_statewise_view_oracle_v2 import freeze_json
    from .explicit_flow_noise import sha256_file
except ImportError:  # Direct script execution.
    from build_statewise_view_oracle_v2 import freeze_json
    from explicit_flow_noise import sha256_file


def clone(protocol_path: Path) -> dict[str, object]:
    source = json.loads(protocol_path.read_text(encoding="utf-8"))
    if source.get("phase") != "independent_confirmation" or source.get("noise_bank_id") != "Q":
        raise ValueError("reserve source must be the frozen Q confirmation protocol")
    result = copy.deepcopy(source)
    result["status"] = "PASS_FROZEN_RESERVE_SAME_CANDIDATES"
    result["phase"] = "independent_precision_reserve"
    result["diagnostic_role"] = "statewise_view_oracle_v2_R_test"
    result["episode_identity_prefix"] = "oracle-v2::R::test"
    result["noise_bank_id"] = "R"
    result["input_confirmation_protocol"] = str(protocol_path.resolve())
    result["input_confirmation_protocol_sha256"] = sha256_file(protocol_path)
    result["candidate_set_changed_from_Q"] = False
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-protocol", type=Path, required=True)
    parser.add_argument("--output-protocol", type=Path, required=True)
    args = parser.parse_args()
    result = clone(args.input_protocol.resolve())
    freeze_json(args.output_protocol.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "episodes": result["episode_count"],
                "candidate_set_changed": result["candidate_set_changed_from_Q"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
