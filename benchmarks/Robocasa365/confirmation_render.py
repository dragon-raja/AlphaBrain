"""Current-reference and 97-candidate rendering for an approved fresh initial."""
import argparse
import os
from pathlib import Path
import time

from AlphaBrain.research.dsol.data.artifacts import read, write, sha
from AlphaBrain.research.dsol.data.render_reference import (CAMERAS, LIMITS,
    save_reference, assert_validation, validate_candidate)
from .snapshot_bundle import load_snapshot_bundle
from .reference_render import inspect_reference
from .evaluation_render import render_initial


def run(plan_file, initial, output):
    plan=read(plan_file); output=Path(output); output.mkdir(parents=True,exist_ok=False)
    jobs=[j for j in plan['jobs'] if j['initial_id']==initial]
    if len(jobs)!=1 or plan['pixel_gate']!=LIMITS:raise ValueError('Unregistered initial or changed gates')
    job=jobs[0]
    state=dict(status='RUNNING',pid=os.getpid(),initial_id=initial,plan_sha256=sha(plan_file),
               source_sha256=sha(__file__),started_unix=time.time(),policy_inferences=0)
    write(output/'status.json',state)
    try:
        bundle=load_snapshot_bundle(Path(job['path']))
        if sha(Path(job['path'])/'manifest.json')!=job['snapshot_manifest_sha256']:
            raise ValueError('Snapshot identity changed')
        save_reference(job,{key:bundle.canonical_images['video.'+key] for key in CAMERAS},
            output/'reference',provenance=dict(kind='first_fresh_snapshot_capture_no_pixel_cherry_picking',
                capture_manifest_sha256=job['snapshot_manifest_sha256']))
        reference=output/'reference/reference.json'
        validation=output/'validation.json'
        receipt=inspect_reference(job,validation,reference_file=reference)
        assert_validation(job,reference,receipt)
        render_initial(job,plan['catalog'],output/'candidates',sha(plan_file),
                       pixel_gate=LIMITS,reference_file=reference)
        validate_candidate(job,plan['catalog'],output/'candidates',sha(plan_file))
        state.update(status='COMPLETE',candidate_count=len(plan['catalog']['views']),
            reference_file=str(reference),reference_sha256=sha(reference),
            validation_file=str(validation),validation_sha256=sha(validation),
            candidate_root=str(output/'candidates'),candidate_summary_sha256=sha(output/'candidates/summary.json'))
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc));raise
    finally:
        state['finished_unix']=time.time();write(output/'status.json',state)
    return state


def validate_output(plan_file,initial,root):
    root=Path(root);state=read(root/'status.json');plan=read(plan_file)
    if (state['status']!='COMPLETE' or state['initial_id']!=initial
            or state['plan_sha256']!=sha(plan_file) or state['candidate_count']!=97
            or state['source_sha256']!=sha(__file__)):
        raise ValueError('Rendering worker identity/coverage mismatch')
    job=next(j for j in plan['jobs'] if j['initial_id']==initial)
    for key in ('reference','validation'):
        if sha(state[key+'_file'])!=state[key+'_sha256']:raise ValueError('Render artifact changed')
    assert_validation(job,state['reference_file'],read(state['validation_file']))
    validate_candidate(job,plan['catalog'],state['candidate_root'],sha(plan_file))
    if sha(Path(state['candidate_root'])/'summary.json')!=state['candidate_summary_sha256']:
        raise ValueError('Candidate summary changed')
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--initial',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.plan,args.initial,args.output)
