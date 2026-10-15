"""Publish template diagnostics, preserving previous artifacts and current PDF."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from AlphaBrain.research.dsol.analysis.accel_field import normalize_coordinates
from AlphaBrain.research.dsol.analysis.accel_template import analyze_templates, template_for
from AlphaBrain.research.dsol.plotting.accel_template import coverage_figure, decomposition_figure
from AlphaBrain.research.dsol.plotting.view_space import candidate_coordinates


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def build(config_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = read(config_path)
    root = Path(config["audited_analysis"])
    matrix = root / "matrix.npz"
    if sha(matrix) != config["expected_matrix_sha256"] or config["primary_low_count"] != 24:
        raise ValueError("Frozen input identity or primary region changed")
    release = read(config["release"])
    if read(root / "inputs.json")[config["release"]] != sha(config["release"]):
        raise ValueError("Frozen release changed")
    states = read(root / "states.json")
    candidates = release["protocol"]["state_blocks"][0]["candidates"]
    if not all(b["candidates"] == candidates for b in release["protocol"]["state_blocks"]):
        raise ValueError("Catalog geometry mismatch")
    x, lo, span = normalize_coordinates(candidate_coordinates(candidates))
    with np.load(matrix) as data:
        a = data["accel_members"].copy()
        ids = data["candidate_ids"].copy()
    if a.shape != (2, 32, 97, 8) or list(ids) != [c["selected_candidate_id"] for c in candidates]:
        raise ValueError("Unexpected data order/shape")
    tasks = np.array([s["task_ordinal"] for s in states])
    coverage, decomposition, curves = analyze_templates(a[:, :, 1:], tasks, x, config["low_fractions"])
    out = Path(config["output"]) / "history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "protocol.json", config)
    write(out / "coverage.json", coverage)
    write(out / "decomposition.json", decomposition)
    selections = []
    for m in range(2):
        for kind in ["same_task", "other_tasks"]:
            for s in range(32):
                template = template_for(a[m, :, 1:].mean(-1), tasks, s, kind)
                ranked = np.argsort(template, kind="stable")[:5] + 1
                selections.append({"model": m, "template": kind, "initial": s,
                                   "task_id": states[s]["task_id"],
                                   "top5": [candidates[int(c)] for c in ranked]})
    write(out / "template-selections.json", selections)
    np.savez_compressed(out / "coverage-curves.npz", **curves)
    summary = {"scope": config["scope"], "new_rollouts": 0, "current_pdf_modified": False, "models": []}
    for m in range(2):
        model = {"model": ["canonical", "broad"][m], "templates": []}
        for kind in ["same_task", "other_tasks"]:
            c = [r for r in coverage if r["model"] == m and r["template"] == kind and r["low_count"] == 24]
            item = {"template": kind, "coverage": {}, "decomposition": []}
            ys_primary = curves[f"m{m}_{kind}_low24"]
            item["coverage"]["top1_in_low24_count"] = int(np.sum(ys_primary[:, 0] > 0))
            item["coverage"]["top5_contains_low24_count"] = int(np.sum(ys_primary[:, 4] > 0))
            item["coverage"]["initial_count"] = len(ys_primary)
            item["coverage"]["at_least80_at32_count"] = int(np.sum(ys_primary[:, 31] >= .8))
            item["coverage"]["at_least80_at48_count"] = int(np.sum(ys_primary[:, 47] >= .8))
            item["coverage"]["sensitivity_at32"] = {
                str(k): float(np.median(curves[f"m{m}_{kind}_low{k}"][:, 31]))
                for k in [int(np.ceil(96*f)) for f in config["low_fractions"]]}
            for key in ["capture_at_24", "capture_at_32", "capture_at_48", "min_sites_capture80", "min_sites_capture90", "min_sites_capture100"]:
                vals = np.array([r[key] for r in c])
                item["coverage"][key] = {"median": float(np.median(vals)), "q25": float(np.quantile(vals, .25)), "q75": float(np.quantile(vals, .75)), "min": float(vals.min()), "max": float(vals.max())}
            for count in [24, 32, 48]:
                ys = np.concatenate([curves[f"m{m}_{kind}_noise{d}"] for d in range(2)], axis=0)
                item["coverage"][f"cross_noise_capture_at_{count}"] = {"median": float(np.median(ys[:, count-1]))}
            item["coverage"]["cross_noise_top1_hits_by_direction"] = [int(np.sum(curves[f"m{m}_{kind}_noise{d}"][:,0]>0)) for d in range(2)]
            for split in ["full_map_descriptive", "random", "spatial"]:
                for method in ["constant", "affine", "affine_linear", "affine_quadratic"]:
                    rows = [r for r in decomposition if r["model"] == m and r["template"] == kind and r["split"] == split and r["method"] == method]
                    item["decomposition"].append({"split": split, "method": method,
                        **{key: float(np.median([r[key] for r in rows if r.get(key) is not None])) if any(r.get(key) is not None for r in rows) else None for key in ["r2", "rmse", "low_quartile_recall"]}})
            model["templates"].append(item)
        summary["models"].append(model)
    write(out / "summary.json", summary)
    for name, fig in [("coverage.png", coverage_figure(curves)), ("decomposition.png", decomposition_figure(decomposition))]:
        fig.savefig(out / name, dpi=160)
        plt.close(fig)
    sources = [Path(config_path), Path(__file__), Path("AlphaBrain/research/dsol/analysis/accel_template.py"),
               Path("AlphaBrain/research/dsol/analysis/accel_field.py"), Path("AlphaBrain/research/dsol/plotting/accel_template.py")]
    if sha(matrix) != config["expected_matrix_sha256"]:
        raise ValueError("Matrix changed while running")
    write(out / "provenance.json", {"inputs": {str(matrix): sha(matrix), config["release"]: sha(config["release"]), str(root / "states.json"): sha(root / "states.json")},
        "sources": {str(p.resolve()): sha(p) for p in sources}, "numpy": np.__version__,
        "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}, "no_new_rollouts": True})
    write(Path(config["output"]) / "latest.json", {"archive": str(out), "summary": str(out / "summary.json")})
    print(json.dumps({"archive": str(out), "models": summary["models"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    build(parser.parse_args().config)
