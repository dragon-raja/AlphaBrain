"""Freeze score-only decisions before reading success; archive an outcome audit."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from AlphaBrain.research.dsol.analysis.template_outcomes import choose_endpoints, summarize_endpoints
from AlphaBrain.research.dsol.data.artifacts import read, write, sha
from AlphaBrain.research.dsol.plotting.template_outcomes import outcome_figure


def build(config_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = read(config_path)
    source = Path(config["audited_analysis"])
    previous = Path(config["previous_template_analysis"])
    if sha(source / "matrix.npz") != config["expected_matrix_sha256"]:
        raise ValueError("Input matrix identity changed")
    states = read(source / "states.json")
    tasks = np.array([s["task_ordinal"] for s in states])
    initials = np.array([s["initialization"]["init_state_index"] for s in states])
    with np.load(source / "matrix.npz") as data:
        a = data["accel_members"].copy()
        ids = data["candidate_ids"].copy()
    choices, references = choose_endpoints(a, tasks, initials)
    old = read(previous / "template-selections.json")
    for row in old:
        selected = ids[choices["previous_"+row["template"]][row["model"], row["initial"]]]
        if selected != row["top5"][0]["selected_candidate_id"]:
            raise ValueError("Prior template decision changed")
    out = Path(config["output"]) / "history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "protocol.json", config)
    write(out / "score-only-selections.json", {"freeze_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_indices": {k:v.tolist() for k,v in choices.items()},
        "candidate_ids": {k:ids[v].tolist() for k,v in choices.items()}, "references": references,
        "success_used_for_selection": False})
    selection_hash = sha(out / "score-only-selections.json")
    # Open success only after the immutable score-only selection receipt is written.
    with np.load(source / "matrix.npz") as data:
        success = data["success"].copy()
    result = summarize_endpoints(success, choices, ids, tasks, initials)
    write(out / "outcomes.json", result)
    fig = outcome_figure(result)
    fig.savefig(out / "outcomes.png", dpi=160)
    plt.close(fig)
    if sha(out / "score-only-selections.json") != selection_hash or sha(source / "matrix.npz") != config["expected_matrix_sha256"]:
        raise ValueError("Inputs or choices changed during outcome join")
    files = [Path(config_path), Path(__file__), Path("AlphaBrain/research/dsol/analysis/template_outcomes.py"),
             Path("AlphaBrain/research/dsol/analysis/accel_template.py"), Path("AlphaBrain/research/dsol/analysis/statistics.py"),
             Path("AlphaBrain/research/dsol/plotting/template_outcomes.py")]
    write(out / "provenance.json", {"inputs": {str(source / "matrix.npz"): sha(source / "matrix.npz"), str(source / "states.json"): sha(source / "states.json"), str(previous / "template-selections.json"): sha(previous / "template-selections.json")},
        "sources": {str(p.resolve()): sha(p) for p in files},
        "artifacts": {p.name:sha(p) for p in out.iterdir() if p.is_file()},
        "new_rollouts": 0, "vla_training_steps": 0, "current_pdf_modified": False})
    write(Path(config["output"]) / "latest.json", {"archive": str(out), "outcomes": str(out / "outcomes.json")})
    compact = {m["model"]:{pop:{k:round(v["success"]["mean"]*100, 4) for k,v in rows.items()} for pop,rows in m["populations"].items()} for m in result["models"]}
    print(json.dumps({"archive": str(out), "results": compact}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    build(parser.parse_args().config)
