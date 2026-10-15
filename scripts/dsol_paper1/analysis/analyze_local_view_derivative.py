"""Consume the explicit local RGB-D/gradient pilot; never starts GPU work."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from AlphaBrain.research.dsol.analysis.local_view_derivative import analyze_probes


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["output"])
    results = []
    for model in ("canonical", "broad"):
        rows = json.loads((root / f"{model}-rendered.json").read_text())["rows"]
        for state in config["state_indices"]:
            for candidate in config["candidate_indices"]:
                bank = np.load(Path(config.get("render_input", config["output"])) / f"render-{state}-{candidate}.npz")
                subset = [r for r in rows if r["state_index"] == state and r["candidate_index"] == candidate]
                scores, grads = [], []
                for seed in config["score_seeds"]:
                    selected = sorted([r for r in subset if r["seed"] == seed], key=lambda r: r["probe_index"])
                    if [r["probe_index"] for r in selected] != list(range(13)):
                        raise ValueError("Missing or duplicated probes")
                    scores.append([r["score"] for r in selected])
                    grads.append(np.load(selected[0]["gradient_file"])["gradient"])
                result = analyze_probes(bank["probe_images"], bank["depth_optical"][0],
                    bank["geometry_intrinsics"], bank["twists"], scores, grads)
                if len(scores) >= 4:
                    split = len(scores)//2
                    proposed = analyze_probes(bank["probe_images"], bank["depth_optical"][0],
                        bank["geometry_intrinsics"], bank["twists"], scores[:split], grads[:split])
                    heldout = np.mean(scores[split:], axis=0)
                    result["independent_noise_check"] = dict(
                        direction_seeds=config["score_seeds"][:split],
                        evaluation_seeds=config["score_seeds"][split:],
                        base_score=float(heldout[0]),
                        methods={name: dict(selected_probe_index=v["selected_probe_index"],
                            score=float(heldout[v["selected_probe_index"]]),
                            score_change=float(heldout[v["selected_probe_index"]]-heldout[0]))
                            for name,v in proposed["geometric_predictions"].items()})
                results.append(dict(model=model, state_index=state, candidate_index=candidate, **result))
    repo = Path(__file__).resolve().parents[3]
    sources = [repo / p for p in (
        "AlphaBrain/research/dsol/metrics/pi05_differentiable.py",
        "AlphaBrain/research/dsol/metrics/accel_torch.py",
        "AlphaBrain/research/dsol/acquisition/image_geometry.py",
        "AlphaBrain/research/dsol/analysis/local_view_derivative.py",
        "scripts/dsol_paper1/runtime/local_view_diagnostic.py",
        "scripts/dsol_paper1/analysis/analyze_local_view_derivative.py")]
    evidence = [Path(config["release"]), Path(args.config), *root.glob("*-rendered.json")]
    render_root = Path(config.get("render_input", config["output"]))
    evidence.extend(render_root.glob("render-*.npz"))
    evidence.extend(root.glob("gradient-*.npz"))
    provenance = dict(source_hashes_at_analysis={str(p):sha256(p) for p in sources},
                      input_hashes={str(p.resolve()):sha256(p) for p in evidence})
    with (root / "analysis.json").open("x") as stream:
        json.dump(dict(config=config, provenance=provenance, results=results), stream, indent=2, allow_nan=False)
    for r in results:
        print(r["model"], r["state_index"], r["candidate_index"], "base", round(r["base_score"],6),
              "cosines", [round(v["scaled_direction_cosine"],3) for v in r["geometric_predictions"].values()],
              "changes", [round(v["selected_score_change"],6) for v in r["geometric_predictions"].values()])


if __name__ == "__main__":
    main()
