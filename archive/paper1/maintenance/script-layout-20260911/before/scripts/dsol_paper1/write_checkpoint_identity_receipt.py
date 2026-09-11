from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_manifest = args.run_manifest.resolve()
    checkpoint = args.checkpoint.resolve()
    weights = checkpoint / "model.safetensors"
    if not run_manifest.is_file() or not weights.is_file():
        parser.error("run manifest or checkpoint weights are missing")
    payload = {
        "schema": "dsol_pretrained_checkpoint_identity_receipt_v1",
        "status": "VERIFIED",
        "run_manifest": str(run_manifest),
        "run_manifest_sha256": sha256(run_manifest),
        "pretrained_checkpoint": str(checkpoint),
        "pretrained_checkpoint_model_sha256": sha256(weights),
        "purpose": (
            "records the explicit DSOL_PRETRAINED_CHECKPOINT for a run whose "
            "legacy manifest writer only recorded the default base manifest"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=args.output.parent,
        prefix=f".{args.output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(args.output)
    print(json.dumps({"status": "VERIFIED", "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
