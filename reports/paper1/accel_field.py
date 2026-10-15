"""Build an auditable CPU analysis and PNG atlas without replacing the current PDF."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from AlphaBrain.research.dsol.analysis.accel_field import (
    dense_field, normalize_coordinates, quadratic_curvature,
    structure_diagnostics, validate_interpolation,
)
from AlphaBrain.research.dsol.plotting.accel_field import diagnostic_heatmaps, field_slices, sampling_support
from AlphaBrain.research.dsol.plotting.view_space import candidate_coordinates


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def median(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return float(np.median(values)) if values else None


def build(config_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scipy

    config = read(config_path)
    if config["display_method"] != "rbf_smooth" or config["random_fold_seed"] != 20260915:
        raise ValueError("This protocol freezes display method and fold seed")
    root = Path(config["audited_analysis"])
    release_path = Path(config["release"])
    inputs = read(root / "inputs.json")
    if inputs[str(release_path)] != sha(release_path):
        raise ValueError("Release identity changed")
    release, states = read(release_path), read(root / "states.json")
    candidates = release["protocol"]["state_blocks"][0]["candidates"]
    if not all(b["candidates"] == candidates for b in release["protocol"]["state_blocks"]):
        raise ValueError("Camera catalogs differ across initials")
    coordinates = candidate_coordinates(candidates)
    x, lo, span = normalize_coordinates(coordinates)
    matrix_hash = sha(root / "matrix.npz")
    with np.load(root / "matrix.npz") as data:
        archived = data["accel_members"].copy()
        candidate_ids = data["candidate_ids"].copy()
    if archived.shape != (2, 32, 97, 8) or not np.isfinite(archived).all():
        raise ValueError("Unexpected Accel archive")
    if list(candidate_ids) != [c["selected_candidate_id"] for c in candidates]:
        raise ValueError("Candidate order mismatch")
    audit = []
    seeds = []
    for m, model in enumerate(["canonical", "broad"]):
        for s in range(32):
            path = release_path.parent / "scores" / model / "aa0-a" / f"{s:02d}" / "ranking.json"
            if sha(path) != inputs[str(path)]:
                raise ValueError("Ranking input identity changed")
            record = read(path)
            rows = {r["candidate_id"]: r for r in record["ranking"]}
            restored = np.array([rows[c]["member_accel_3"] for c in candidate_ids])
            np.testing.assert_array_equal(restored, archived[m, s])
            if len(record["ensemble_seeds"]) != 8:
                raise ValueError("Noise seed metadata missing")
            seeds.append(record["ensemble_seeds"])
            audit.append({"model": model, "initial": s, "path": str(path), "sha256": inputs[str(path)], "ensemble_seeds": record["ensemble_seeds"]})
    tasks = np.array([s["task_ordinal"] for s in states])
    initials = np.array([s["initialization"]["init_state_index"] for s in states])
    if not np.array_equal(tasks, np.repeat(np.arange(8), 4)) or not np.array_equal(initials, np.tile(np.arange(4), 8)):
        raise ValueError("Unexpected state ordering")
    members = archived[:, :, 1:]
    mean = members.mean(-1)
    fields = mean.reshape(64, 96)
    out = Path(config["output"]) / "history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "protocol.json", config)
    write(out / "noise-audit.json", audit)
    diagnostics = structure_diagnostics(members, tasks, initials)
    write(out / "structure.json", diagnostics)
    validation = []
    noise_validation = []
    predictions = {}
    for mode in ["random", "spatial"]:
        p, rows = validate_interpolation(x, fields, config["methods"], mode)
        validation.extend(rows)
        predictions.update({mode + "_" + k: v for k, v in p.items()})
        for direction in [0, 1]:
            halves = [members[..., :4].mean(-1).reshape(64, 96), members[..., 4:].mean(-1).reshape(64, 96)]
            p, rows = validate_interpolation(x, halves[direction], ["quadratic", "rbf_smooth"], mode,
                                             target_fields=halves[1-direction])
            noise_validation.extend([{**r, "noise_direction": direction} for r in rows])
            predictions.update({f"{mode}_noise{direction}_" + k: v for k, v in p.items()})
    np.savez_compressed(out / "heldout-predictions.npz", **predictions)
    write(out / "validation.json", validation)
    write(out / "cross-noise-validation.json", noise_validation)
    curvature = quadratic_curvature(x, fields)
    write(out / "quadratic-curvature.json", curvature)
    dense = dense_field(x, fields, config["dense_resolution"])
    np.savez_compressed(out / "estimated-volume.npz", **dense, physical_coordinates=dense["normalized_query"]*span+lo,
                        measured_coordinates=coordinates, measured_accel_members=members,
                        model_order=np.array(["canonical", "broad"]))
    summary = {"matrix_sha256": matrix_hash, "archived_score_shape": list(archived.shape),
               "individual_scores": archived.size, "spatial_views": len(x),
               "noise_seeds_identical_across_states_and_models": all(s == seeds[0] for s in seeds),
               "noise_seeds_paired_across_models_within_initial": all(seeds[s] == seeds[s+32] for s in range(32)),
               "unique_initial_seed_lists": len({tuple(s) for s in seeds}),
               "bounds": {"minimum": lo.tolist(), "maximum": (lo+span).tolist()},
               "dense_hull_fraction": float(dense["inside_hull"].mean()),
               "new_rollouts": 0, "vla_training_steps": 0, "models": []}
    for m in range(2):
        item = {"model": ["canonical", "broad"][m], "validation": [], "cross_noise_validation": []}
        for key in ["noise", "within_task", "other_task_template"]:
            rows = [r for r in diagnostics[key] if r["model"] == m]
            item[key] = {k: median(rows, k) for k in ["rho", "low_quartile_recall", "mean_noise_variance_to_spatial_variance", "offset_removed_noise_variance_to_spatial_variance"]}
        for mode in ["random", "spatial"]:
            for method in config["methods"]:
                rows = [r for r in validation if r["field"]//32 == m and r["split"] == mode and r["method"] == method]
                item["validation"].append({"split": mode, "method": method,
                    **{k: median(rows, k) for k in ["coverage", "r2", "rmse", "rho", "low_quartile_recall", "selected_range_regret"]},
                    "interior_r2": median([r["interior"] for r in rows], "r2")})
            for method in ["quadratic", "rbf_smooth"]:
                rows = [r for r in noise_validation if r["field"]//32 == m and r["split"] == mode and r["method"] == method]
                item["cross_noise_validation"].append({"split": mode, "method": method,
                    **{k: median(rows, k) for k in ["r2", "rho", "low_quartile_recall"]}})
        item["positive_quadratic_count"] = sum(r["positive_definite"] for r in curvature[m*32:(m+1)*32])
        item["interior_quadratic_min_count"] = sum(r["stationary_in_box"] for r in curvature[m*32:(m+1)*32])
        summary["models"].append(item)
    summary["cross_model"] = {k: median(diagnostics["cross_model"], k) for k in ["rho", "low_quartile_recall"]}
    write(out / "summary.json", summary)

    def save(fig, name):
        fig.savefig(out / name, dpi=150)
        plt.close(fig)

    save(sampling_support(coordinates, x), "sampling-support.png")
    labels = [s["task_id"] for s in states[::4]]
    save(diagnostic_heatmaps(diagnostics, [f"T{i}" for i in range(8)]), "structure-overlap.png")
    save(field_slices(coordinates, x, mean.mean(1), "All 32 initials averaged (not a universal field)"), "mean-slices.png")
    for task in range(8):
        save(field_slices(coordinates, x, mean[:, tasks == task].mean(1), f"T{task}: {labels[task]} | average of 4 initials"), f"task-{task}-slices.png")
    for s, state in enumerate(states):
        save(field_slices(coordinates, x, mean[:, s], f"T{tasks[s]} initial {initials[s]}: {state['task_id']} | mean of 8 scoring noises"), f"initial-{s:02d}-slices.png")
    sources = [Path(config_path), Path(__file__),
               Path("AlphaBrain/research/dsol/analysis/accel_field.py"), Path("AlphaBrain/research/dsol/plotting/accel_field.py")]
    provenance = {"inputs": {str(root / "matrix.npz"): matrix_hash, str(release_path): sha(release_path),
                              str(root / "states.json"): sha(root / "states.json")},
                  "sources": {str(p.resolve()): sha(p) for p in sources},
                  "numpy": np.__version__, "scipy": scipy.__version__,
                  "new_rollouts": 0, "current_pdf_modified": False,
                  "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    if sha(root / "matrix.npz") != matrix_hash:
        raise ValueError("Input changed during analysis")
    write(out / "provenance.json", provenance)
    write(Path(config["output"]) / "latest.json", {"archive": str(out), "summary": str(out / "summary.json")})
    print(json.dumps({"archive": str(out), "summary": summary}, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    build(parser.parse_args().config)
