"""Compose four verified suite panels, preserving native four-step GR00T scores."""
from pathlib import Path
import argparse,json
import numpy as np
from reports.paper1.target_spatial_panel import load_verified_panel
from AlphaBrain.research.dsol.analysis.spatial_score_validity import summarize
from AlphaBrain.research.dsol.analysis.spatial_task_average import linear_comparison
from AlphaBrain.research.dsol.plotting.spatial_score_validity import native_panel_figures


def build(root, output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=False)
    all_rows=[];per_suite={}
    for suite in ('libero_spatial','libero_object','libero_goal','libero_10'):
        rows,_=load_verified_panel(root/suite,prefixes=(3,4),expected_states=4)
        all_rows.extend(rows)
        per_suite[suite]={name:linear_comparison(rows,score=name) for name in ('p3','p4')}
        native_panel_figures(per_suite[suite],output/suite,model_label='GR00T N1.5 / '+suite)
    if len(all_rows)!=432 or len({r['state_key'] for r in all_rows})!=16:raise ValueError('Incomplete four-suite panel')
    comparison={name:linear_comparison(all_rows,score=name) for name in ('p3','p4')}
    selection=summarize(all_rows,score_names=('p3','p4'))
    reference=np.asarray([r['success'] for r in all_rows if r['view_id']==0])
    perturbed=np.asarray([r['success'] for r in all_rows if r['view_id']!=0])
    outcomes=dict(reference_success=float(reference.mean()),reference_successes=int(reference.sum()),reference_episodes=reference.size,
                  perturbed_success=float(perturbed.mean()),perturbed_successes=int(perturbed.sum()),perturbed_episodes=perturbed.size)
    for name,value in [('rows',all_rows),('linear-comparison',comparison),('per-suite-linear',per_suite),('selection',selection),('outcomes',outcomes)]:
        (output/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    native_panel_figures(comparison,output,model_label='GR00T N1.5 / eight-task diagnostic')
    lines=['GR00T N1.5 fixed-view diagnostic — complete 8 tasks / 16 initials / 27 poses / 864 episodes.',
        'Native four-step flow: compare prefix 3 and full 4. Neither is FastWAM p9; sampler unchanged.',
        'All outcomes retained. No PPO, feature-transfer training or outcome-based pose selection.',
        'Each suite uses its own checkpoint. Native 10 settling actions; common LIBERO 1.4.1/MuJoCo 3.2.3 runtime, not an official 1.5 benchmark reproduction.',
        'Plots show task-equal raw-value Pearson fields. The combined field mixes suite-specific checkpoints; per-suite and within-state results are also supplied.',
        'Minimum-Accel selection observes all 27 candidates and is an offline diagnostic, not the eight-query deployed search policy.',
        f"Reference {100*outcomes['reference_success']:.2f}% ({outcomes['reference_successes']}/{outcomes['reference_episodes']}); perturbed {100*outcomes['perturbed_success']:.2f}% ({outcomes['perturbed_successes']}/{outcomes['perturbed_episodes']})."]
    for name,c in comparison.items():
        lines.append(f"{name}: nonreference Pearson(-Accel,success)={c['scopes']['nonreference']['favorable_r']}; within-state centered={c['scopes']['nonreference']['state_centered_favorable_r']}.")
    for a in selection['aggregate']:
        lines.append(f"{a['score']}, exclude_reference={a['exclude_reference']}: argmin success={a['argmin_success_state_mean']:.6f}; low-seven gain={a['low_group_gain_task_mean']:.6f}, task-cluster interval={a['task_cluster_95_interval']}.")
    (output/'README.txt').write_text('\n'.join(lines)+'\n')
    return dict(outcomes=outcomes,linear={n:c['scopes'] for n,c in comparison.items()},selection=selection['aggregate'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(build(a.root,a.output)))
