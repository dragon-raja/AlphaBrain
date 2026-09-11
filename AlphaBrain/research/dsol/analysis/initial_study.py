"""Extracted from the audited INITIAL32 analysis; preserve scientific semantics."""

import numpy as np
from scipy.stats import spearmanr
from ..data.artifacts import write
from ..metrics.view_rules import metric_choices
from ..selectors.specification import CONTRACT
from ..selectors.ridge import fit_ranker
from .statistics import hierarchy, paired_interval

MODELS = ["canonical", "broad"]


def summarize(out, r, Y, acc, vis, camera, images, context):
    q = Y.mean(axis=-1)
    am = acc.mean(axis=-1)
    initials = np.array([s["initialization"]["init_state_index"] for s in r["states"]])
    tasks = np.array([s["task_ordinal"] for s in r["states"]])
    dev = np.flatnonzero(initials < 2)
    test = np.flatnonzero(initials >= 2)
    # Actual camera geometry: displacement and relative rotation from the canonical view.
    delta = camera[:, :, :3, 3] - camera[:, :1, :3, 3]
    rotation = np.einsum("sji,scjk->scik", camera[:, 0, :3, :3], camera[:, :, :3, :3])
    canonical_flag = np.broadcast_to((np.arange(97) == 0)[None, :, None], (32, 97, 1))
    geom = np.concatenate([delta, rotation.reshape(32, 97, 9), canonical_flag], axis=-1)
    results = {
        "contract": CONTRACT,
        "episodes": int(Y.size),
        "models": {},
        "task_ids": [r["states"][i * 4]["task_id"] for i in range(8)],
    }
    selected_all = {}
    per_initial = {}
    for m, model in enumerate(MODELS):
        choices = {"canonical": np.zeros(32, dtype=int), **metric_choices(am[m], vis)}
        global_c = q[m, dev].mean(axis=0).argmax()
        task_c = q[m, dev].reshape(8, 2, 97).mean(axis=1).argmax(axis=1)
        choices["dev_global_fixed"] = np.full(32, global_c)
        choices["dev_task_fixed"] = task_c[tasks]
        learning = {}
        for family in CONTRACT["families"]:
            pred, weights, info = fit_ranker(geom, context, images, tasks, initials, q[m], family)
            choices[family] = pred.argmax(axis=1)
            learning[family] = info
            np.savez_compressed(
                out / f"{model}-{family}.npz",
                **weights,
                predictions=pred,
                choices=choices[family],
                development_indices=dev,
                test_indices=test,
            )
        methods = {name: q[m, np.arange(32), c] for name, c in choices.items()}
        methods["uniform"] = q[m].mean(axis=1)
        methods["initial_oracle"] = q[m].max(axis=1)
        task_upper = q[m].reshape(8, 4, 97).mean(axis=1).argmax(axis=1)
        methods["all_task_oracle"] = q[m, np.arange(32), task_upper[tasks]]
        scopes = {}
        for label, subset in [("all32", np.arange(32)), ("heldout16", test), ("development16", dev)]:
            scopes[label] = {"hierarchy": hierarchy(q[m, subset].reshape(8, -1, 97)), "methods": {}}
            for key, value in methods.items():
                if key == "all_task_oracle" and label != "all32":
                    continue
                scopes[label]["methods"][key] = {
                    "success": paired_interval(value, subset),
                    "gain_vs_canonical": paired_interval(value - methods["canonical"], subset),
                    "per_task_success": value[subset].reshape(8, -1).mean(axis=1).tolist(),
                }
        correlation = []
        stability = []
        for s in range(32):
            aa = spearmanr(-am[m, s], q[m, s]).statistic
            vv = spearmanr(vis[s], q[m, s]).statistic
            win = acc[m, s].argmin(axis=0)
            stability.append(
                {
                    "initial": s,
                    "distinct_single_noise_winners": int(len(set(win.tolist()))),
                    "agreement_with_ensemble_winner": float(np.mean(win == choices["min_accel"][s])),
                }
            )
            correlation.append(
                {
                    "initial": s,
                    "minus_accel_spearman": float(aa) if np.isfinite(aa) else None,
                    "visibility_spearman": float(vv) if np.isfinite(vv) else None,
                }
            )
        results["models"][model] = {
            "scopes": scopes,
            "learners": learning,
            "rank_correlations": correlation,
            "score_noise_stability": stability,
            "candidate_groups": {
                "canonical": float(q[m, :, 0].mean()),
                "training_catalog_64": float(q[m, :, 1:65].mean()),
                "heldout_catalog_32": float(q[m, :, 65:].mean()),
            },
            "fraction_within_5pp_of_canonical": float((q[m] >= q[m, :, :1] - 0.05).mean()),
        }
        selected_all[model] = {k: v.tolist() for k, v in choices.items()}
        per_initial[model] = {k: v.tolist() for k, v in methods.items()}
    results["training_effect"] = {
        k: paired_interval(np.asarray(per_initial["broad"][k]) - np.asarray(per_initial["canonical"][k]), np.arange(32))
        for k in ["canonical", "uniform", "initial_oracle"]
    }
    write(out / "selections.json", selected_all)
    write(out / "per-initial.json", per_initial)
    write(out / "summary.json", results)
    return results
