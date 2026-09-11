"""Configuration-driven offline analysis. No simulator or policy service imports."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import numpy as np
from ..data.artifacts import read, sha, write
from ..data.initial_matrix import audit_and_load
from ..selectors.visual_features import encode_assets
from ..selectors.specification import CONTRACT
from .initial_study import summarize


@dataclass(frozen=True)
class AnalysisConfig:
    root: Path
    output: Path
    encoder_weights: Path


def load_config(path):
    payload = read(path)
    return AnalysisConfig(
        *(Path(payload[k]).expanduser().resolve() for k in ["experiment_root", "analysis_output", "encoder_weights"])
    )


def run(config, *, reuse_audited=None):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = config.output / "history" / stamp
    out.mkdir(parents=True, exist_ok=False)
    package = Path(__file__).resolve().parents[1]
    source_hashes = {str(p): sha(p) for p in sorted(package.rglob("*.py"))}
    write(
        out / "protocol.json",
        {
            **CONTRACT,
            "frozen_utc": stamp,
            "release_sha256": sha(config.root / "release.json"),
            "source_hashes": source_hashes,
        },
    )
    if reuse_audited is None:
        r, y, acc, vis, camera = audit_and_load(config.root, out)
        images, context = encode_assets(config.root, out, config.encoder_weights)
    else:
        prior = Path(reuse_audited)
        provenance = read(prior / "provenance.json")
        if provenance["release_sha256"] != sha(config.root / "release.json"):
            raise ValueError("Cannot reuse a different scientific release")
        for name in ["matrix.npz", "embeddings.npz", "states.json", "inputs.json", "record-hashes.json"]:
            if sha(prior / name) != provenance["artifacts"][name]:
                raise ValueError("Changed audited artifact: " + name)
            (out / name).symlink_to(os.path.relpath((prior / name).resolve(), out))
        r = read(config.root / "release.json")
        with np.load(prior / "matrix.npz") as a:
            y, acc, vis, camera = [a[k] for k in ["success", "accel_members", "visibility", "camera"]]
        with np.load(prior / "embeddings.npz") as a:
            images, context = a["external"], a["context"]
        # Explicit reference: reuse is not a second raw-record audit.
        write(out / "reused-inputs.json", {"archive": str(prior), "provenance_sha256": sha(prior / "provenance.json")})
    result = summarize(out, r, y, acc, vis, camera, images, context)
    receipt = {
        "status": "COMPLETE_OFFLINE_ANALYSIS",
        "archive": str(out),
        "episodes": int(y.size),
        "release_sha256": sha(config.root / "release.json"),
        "source_hashes": source_hashes,
        "encoder_sha256": sha(config.encoder_weights),
        "new_closed_loops": 0,
        "vla_training_steps": 0,
        "reused_audit": str(reuse_audited) if reuse_audited else None,
        "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
    }
    write(out / "provenance.json", receipt)
    write(config.output / "latest.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Explicit experiment/analysis/encoder paths JSON")
    parser.add_argument(
        "--reuse-audited", type=Path, help="Reuse hash-verified matrix/features; does not re-audit raw ledgers"
    )
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config), reuse_audited=args.reuse_audited), ensure_ascii=False))


if __name__ == "__main__":
    main()
