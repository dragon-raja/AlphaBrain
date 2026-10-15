"""Reproduce a five-figure local-gradient atlas from immutable diagnostic data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from AlphaBrain.research.dsol.analysis.local_view_derivative import (
    analyze_probes, local_noise_controls, paired_score_matrix,
)
from AlphaBrain.research.dsol.plotting.local_view_derivative import (
    process_figure, direction_figure, noise_figure, transfer_figure, axis_curves_figure,
)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):
            value.update(chunk)
    return value.hexdigest()


def build(config_path, output, *, font_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    config_path, output=Path(config_path),Path(output)
    # New derived artifacts only; never rewrite diagnostic results or the PDF.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if font_path:
        font_manager.fontManager.addfont(font_path)
    fonts={f.name for f in font_manager.fontManager.ttflist}
    if "Noto Sans CJK JP" not in fonts:
        raise RuntimeError("A CJK font is required; do not silently produce missing glyphs")
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"Noto Sans CJK JP","axes.unicode_minus":False,
        "font.size":10,"figure.facecolor":"white","savefig.facecolor":"white"})
    config=json.loads(config_path.read_text())
    root=Path(config["output"])
    render=Path(config.get("render_input",config["output"]))
    analysis_path=root/"analysis.json"
    archived=json.loads(analysis_path.read_text())
    for p,expected in archived["provenance"]["input_hashes"].items():
        if sha(p)!=expected:
            raise ValueError("Frozen input changed: "+p)
    cases=[]
    for model in ["canonical","broad"]:
        rows=json.loads((root/f"{model}-rendered.json").read_text())["rows"]
        for state in config["state_indices"]:
            for candidate in config["candidate_indices"]:
                bank=dict(np.load(render/f"render-{state}-{candidate}.npz"))
                scores=paired_score_matrix(rows,state_index=state,candidate_index=candidate,seeds=config["score_seeds"])
                gradients=np.array([np.load(root/f"gradient-{model}-{state}-{candidate}-{seed}.npz")["gradient"]
                    for seed in config["score_seeds"]])
                record=next(r for r in archived["results"] if (r["model"],r["state_index"],r["candidate_index"])==(model,state,candidate))
                np.testing.assert_allclose(scores.mean(0),record["mean_probe_scores"],atol=1e-15,rtol=0)
                first=analyze_probes(bank["probe_images"],bank["depth_optical"][0],
                    bank["geometry_intrinsics"],bank["twists"],scores[:4],gradients[:4])
                controls=local_noise_controls(scores,first,split=4)
                np.testing.assert_allclose(controls["geometry_chain"]["mean_test_change"],
                    record["independent_noise_check"]["methods"]["all_valid"]["score_change"],atol=1e-15,rtol=0)
                cases.append(dict(model=model,state_index=state,candidate_index=candidate,
                    bank=bank,scores=scores,record=record,controls=controls))
    fixed=next(c for c in cases if c["state_index"]==4 and c["candidate_index"]==1)
    figures=[("01_process",process_figure(fixed["bank"])),
             ("02_directions",direction_figure([c["record"] for c in cases])),
             ("03_noise",noise_figure(cases,config["score_seeds"])),
             ("04_independent_noise",transfer_figure(cases)),
             ("05_axis_profiles",axis_curves_figure(cases))]
    for name,figure in figures:
        figure.savefig(output/(name+".png"),dpi=160)
        figure.savefig(output/(name+".svg"))
        plt.close(figure)
    summary=[]
    for c in cases:
        summary.append({k:c[k] for k in ("model","state_index","candidate_index","controls")})
    sources=[Path(__file__),Path(__file__).resolve().parents[2]/"AlphaBrain/research/dsol/plotting/local_view_derivative.py",
             Path(__file__).resolve().parents[2]/"AlphaBrain/research/dsol/analysis/local_view_derivative.py"]
    payload=dict(scope="Exploratory offline mechanism diagnostics, not task outcomes or no-query algorithms",
        config=str(config_path.resolve()),config_sha256=sha(config_path),
        analysis=str(analysis_path),analysis_sha256=sha(analysis_path),
        source_hashes={str(p):sha(p) for p in sources},
        artifacts={p.name:sha(p) for p in output.iterdir()},results=summary)
    (output/"visualization.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"output":str(output),"figures":len(figures),"cases":len(cases)}))
    for model in ("canonical","broad"):
        selected=[c for c in cases if c["model"]==model]
        print(model,{m:sum(c["controls"][m]["mean_test_change"]<0 for c in selected)
                    for m in selected[0]["controls"]})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("output")
    parser.add_argument("--font",help="Explicit local Noto Sans CJK font file, if not in matplotlib cache")
    args=parser.parse_args()
    build(args.config,args.output,font_path=args.font)
