"""Freeze Accel-only template endpoints, then join archived full-task outcomes."""

import numpy as np

from .accel_template import template_for
from .statistics import paired_interval


def choose_endpoints(accel, tasks, initials):
    """No success labels accepted. All choices index original canonical-first catalog.

    Previous templates use leave-initial/task-out all32 reference fields.
    Dev templates ONLY use official initial indices 0 and 1, and are evaluated
    on indices 2 and 3. Canonical excluded from every template, but included in
    the separately named min-Accel97 and known-canonical references.
    """
    a = np.asarray(accel)
    tasks, initials = np.asarray(tasks), np.asarray(initials)
    if a.shape != (2, 32, 97, 8) or not np.isfinite(a).all():
        raise ValueError("Expected finite 2 x 32 x 97 x 8 Accel scores")
    if not np.array_equal(tasks, np.repeat(np.arange(8), 4)) or not np.array_equal(initials, np.tile(np.arange(4), 8)):
        raise ValueError("Expected task-major official initial ordering")
    means = a.mean(-1)
    choices = {"known_canonical": np.zeros((2, 32), dtype=int),
               "min_accel97": means.argmin(-1),
               "min_accel96": means[:, :, 1:].argmin(-1)+1}
    kinds = ["previous_same_task", "previous_other_tasks", "dev_global", "dev_same_task", "dev_other_tasks"]
    choices.update({k: np.zeros((2, 32), dtype=int) for k in kinds})
    references = []
    for m in range(2):
        fields = means[m, :, 1:]
        for s in range(32):
            for kind in kinds:
                if kind.startswith("previous_"):
                    k = kind.removeprefix("previous_")
                    t = template_for(fields, tasks, s, k)
                    mask = ((tasks == tasks[s]) & (np.arange(32) != s)) if k == "same_task" else tasks != tasks[s]
                else:
                    mask = initials < 2
                    if kind == "dev_same_task":
                        mask &= tasks == tasks[s]
                    elif kind == "dev_other_tasks":
                        mask &= tasks != tasks[s]
                    ref = fields[mask]
                    if kind != "dev_same_task":
                        ref = (ref-ref.mean(1, keepdims=True))/np.maximum(ref.std(1, keepdims=True), 1e-12)
                    t = ref.mean(0)
                choices[kind][m, s] = int(np.argmin(t))+1
                references.append({"model": m, "initial": s, "method": kind,
                                   "reference_initials": np.flatnonzero(mask).tolist()})
    return choices, references


def summarize_endpoints(success, choices, candidate_ids, tasks, initials):
    y = np.asarray(success)
    if y.shape != (2, 32, 97, 32) or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected binary full-task outcomes, 2 x 32 x 97 x 32")
    q = y.mean(-1)
    heldout = np.flatnonzero(np.char.startswith(np.asarray(candidate_ids, dtype=str), "broad_heldout_"))
    if len(heldout) != 32:
        raise ValueError("Expected 32 heldout camera starts")
    evaluation = np.flatnonzero(np.asarray(initials) >= 2)
    results = {"models": [], "definition": {
        "outcome": "32 archived full-task execution noises per endpoint; no mid-task view switching",
        "main_population": "16 initials 2/3; stay averaged over 32 heldout-catalog start cameras",
        "all32": "Previous cross-fitted templates and metric references only; dev-template training initials not evaluated",
        "interval": "Paired task then initial bootstrap, 10000 draws; conditional on observed execution-noise means and fixed templates, not a training-procedure uncertainty estimate",
        "scope": "Exploratory; previously inspected scores/outcomes, no blind confirmation, acquisition motion or new rollout",
    }}
    for m in range(2):
        values = {k: q[m, np.arange(32), c[m]] for k, c in choices.items()}
        values["stay_heldout32"] = q[m][:, heldout].mean(1)
        values["stay_noncanonical96"] = q[m, :, 1:].mean(1)
        model = {"model": ["canonical", "broad"][m], "populations": {}}
        for population, ix in [("all32", np.arange(32)), ("evaluation16", evaluation)]:
            rows = {}
            for method, v in values.items():
                if population == "all32" and method.startswith("dev_"):
                    continue
                delta = v-values["stay_heldout32"]
                rows[method] = {"success": paired_interval(v, ix),
                    "gain_vs_stay": paired_interval(delta, ix),
                    "delta_vs_min_accel97": paired_interval(v-values["min_accel97"], ix),
                    "delta_vs_min_accel96": paired_interval(v-values["min_accel96"], ix),
                    "delta_vs_canonical": paired_interval(v-values["known_canonical"], ix),
                    "per_initial_success": v[ix].tolist(),
                    "per_task_success": v[ix].reshape(8, -1).mean(1).tolist()}
                if method in choices:
                    endpoints = choices[method][m, ix]
                    rows[method]["candidate_ids"] = [str(candidate_ids[c]) for c in endpoints]
                    # Same endpoint contrasted with each starting view, average execution-noise first.
                    diffs = v[ix, None] - q[m][ix][:, heldout]
                    rows[method]["start_comparison"] = {
                        "improved_fraction": float(np.mean(diffs > 1e-12)),
                        "worsened_fraction": float(np.mean(diffs < -1e-12)),
                        "equal_fraction": float(np.mean(abs(diffs) <= 1e-12))}
            model["populations"][population] = rows
        results["models"].append(model)
    return results
