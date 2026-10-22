"""Descriptive evidence tables; never fit phase cutoffs or a selection rule."""
from collections import Counter
from pathlib import Path
import numpy as np
from ..data.artifacts import read,write,sha


def distribution(values):
    x=np.asarray(values,float)
    return dict(mean=float(x.mean()),min=float(x.min()),max=float(x.max()),values=x.tolist())


def summarize(root):
    root=Path(root);target=root/'evidence-summary.json'
    if target.exists():raise ValueError('Refuse to overwrite evidence summary')
    files=['physical-timelines.json','original-nodes.json','same-state-alignment.json','view-contrasts.json','flip-intervals.json','positive-controls.json']
    physical,original,same,contrasts,flips,positive=[read(root/f) for f in files]
    branches=[];stages=[];view_effects=[];intervals=[]
    for case in sorted({b['case'] for b in physical}):
        for view in sorted({b['view'] for b in physical if b['case']==case}):
            bs=[b for b in physical if b['case']==case and b['view']==view]
            targets=bs[0]['frames'][0]['target_distance_m'];contacts={}
            for t in targets:
                parts=[{v for f in b['frames'] for v in f['contact_bodies'].get(t,[])} for b in bs]
                left=lambda s:any(v.endswith(('leftfinger','finger_joint1_tip')) for v in s)
                right=lambda s:any(v.endswith(('rightfinger','finger_joint2_tip')) for v in s)
                both=[any(left(f['contact_bodies'].get(t,[])) and right(f['contact_bodies'].get(t,[])) for f in b['frames']) for b in bs]
                contacts[t]=dict(first=[b['first_target_contacts'][t] for b in bs],repeats=sum(bool(s) for s in parts),
                    palm_repeats=sum(any(v.endswith('right_gripper') for v in s) for s in parts),
                    any_finger_repeats=sum(left(s) or right(s) for s in parts),bilateral_same_frame_repeats=sum(both),body_names_by_repeat=[sorted(s) for s in parts])
            branches.append(dict(case=case,view=view,n=len(bs),first_closing_execution=[b['first_closing_execution'] for b in bs],
                closing_execution_repeats=sum(b['first_closing_execution'] is not None for b in bs),contacts=contacts,
                final_joint_separation_m=distribution([b['frames'][-1]['raw_joint_separation_m'] for b in bs]),
                max_target_displacement_m={t:distribution([max(f['target_displacement_m'][t] for f in b['frames']) for b in bs]) for t in targets}))
            for step in sorted({r['step'] for r in original if r['case']==case}):
                rr=[r for r in original if r['case']==case and r['view']==view and r['step']==step];frames=[b['frames'][step] for b in bs]
                stages.append(dict(case=case,view=view,step=step,
                    joint_separation_m=distribution([f['raw_joint_separation_m'] for f in frames]),
                    distance_m={t:distribution([f['target_distance_m'][t] for f in frames]) for t in targets},
                    metrics={k:distribution([r['metrics'][k] for r in rr]) for k in rr[0]['metrics'] if k=='baseline__score' or (k.startswith('baseline_') and k.endswith('__accel_contribution'))},
                    predicted_closing_counts=[r['predicted_gripper']['closing_count'] for r in rr],
                    observed_contact_repeats={t:sum(r['physical']['target_contact_seen'][t] for r in rr) for t in targets},
                    executed_prefix_length=rr[0]['executed_prefix_length']))
    for initial in sorted({r['initial'] for r in contrasts}):
        for origin in sorted({r['origin'] for r in contrasts if r['initial']==initial}):
            for step in sorted({r['step'] for r in contrasts if r['initial']==initial}):
                rr=[r for r in contrasts if (r['initial'],r['origin'],r['step'])==(initial,origin,step)]
                deltas=[r['good_minus_bad']['baseline__score'] for r in rr]
                view_effects.append(dict(initial=initial,origin=origin,step=step,
                    good_view_lower=sum(x<0 for x in deltas),bad_view_lower=sum(x>0 for x in deltas),
                    channels={k:distribution([r['good_minus_bad'][k] for r in rr]) for k in rr[0]['good_minus_bad']},
                    good_view_predicted_closing_counts=[r['good_view_prediction']['closing_count'] for r in rr],
                    bad_view_predicted_closing_counts=[r['bad_view_prediction']['closing_count'] for r in rr],
                    contact_seen=[r['physical']['target_contact_seen'] for r in rr]))
            for lo,hi in sorted({(f['after'],f['by']) for f in flips if f['initial']==initial and f['origin']==origin}):
                ff=[f for f in flips if (f['initial'],f['origin'],f['after'],f['by'])==(initial,origin,lo,hi)]
                intervals.append(dict(initial=initial,origin=origin,after=lo,by=hi,n=len(ff),repeats=[f['repeat'] for f in ff],
                    without_closing_execution=sum(f['closing_executions_in_interval']==0 for f in ff),
                    without_target_contact_by_end=sum(all(x is None for x in f['first_contact_by_interval_end'].values()) for f in ff)))
    result=dict(status='COMPLETE',branches=branches,original_stage_tables=stages,same_state_view_effects=view_effects,
        observed_flip_intervals=intervals,libero_plus_groups=positive['groups'],window_transition_groups=positive['existing_window_transitions'],
        limitations=['Outcome labels belong to original initial-view execution, not new labels for intermediate cross-view interventions.',
        'No physical stage evidence for Plus positives in the analyzed telemetry; score-only controls cannot establish the phase mechanism.',
        'Raw finger separation and render geometry are separately recorded times; no inverse integration or continuous score interpolation.',
        'Joint-coordinate separation is an opening proxy, not calibrated pad-to-pad clearance.',
        'Eight noises in selected examples are descriptive repeats, not independent task replications.'],
        input_sha256={f:sha(root/f) for f in files},source_sha256=sha(__file__))
    write(target,result)
    return target
