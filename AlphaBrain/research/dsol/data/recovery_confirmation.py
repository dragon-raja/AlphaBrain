"""Bounded confirmation release identities; no simulator or model imports."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil

from .artifacts import read, sha, write, path_mapping


def planned_seed(namespace, task, split, index):
    message = f'{namespace}|{task}|{split}|{index}'
    return int(hashlib.sha256(message.encode()).hexdigest()[:8], 16) % 2147000000 + 20000


def validate_initial_plan(config, planned):
    population = config['population']
    expected = {(task, split, index)
                for task in population['tasks'] for split in population['environment_splits']
                for index in range(population['new_initials_per_task_split'])}
    rows = planned['records']
    actual = [(r['task'], r['split'], r['index']) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Initial population differs from frozen factorial scope')
    if len(rows) != population['new_physical_initials']:
        raise ValueError('Initial budget mismatch')
    for row in rows:
        if row['scene_seed'] != planned_seed(population['new_seed_namespace'],
                                             row['task'], row['split'], row['index']):
            raise ValueError('Frozen seed rule mismatch')
    if len({r['scene_seed'] for r in rows}) != len(rows):
        raise ValueError('Duplicate scene seeds')
    return rows


def authorize(config_file, initials_file, groups_file, output, *, path_mappings=()):
    """Record explicit user approval without claiming runtime admission."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    config, planned, groups = read(config_file), read(initials_file), read(groups_file)
    if planned['config_sha256'] != sha(config_file):
        raise ValueError('Initial plan references another proposal')
    if (groups['config_sha256'] != sha(config_file)
            or groups['endpoint_sha256'] != sha(config['sources']['development_endpoints'])):
        raise ValueError('Development strata source changed')
    rows = validate_initial_plan(config, planned)
    with path_mapping(path_mappings):
        p0 = read(config['sources']['original_p0_plan'])
        old = read(p0['reference_plan'])
        old_identities = {(r['task'], r['split'], r['scene_seed']) for r in old['jobs']}
        if old_identities & {(r['task'], r['split'], r['scene_seed']) for r in rows}:
            raise ValueError('New seed overlaps registered prior initial')
        prior = []
        for job in old['jobs']:
            manifest_file = Path(job['path']) / 'manifest.json'
            manifest = read(manifest_file)
            prior.append(dict(task=job['task'], split=job['split'], scene_seed=job['scene_seed'],
                path=job['path'], manifest_sha256=sha(manifest_file),
                snapshot_numeric_sha256=manifest['snapshot_numeric_sha256']))
    output.mkdir(parents=True, exist_ok=False)
    # Copy exact input bytes, preserving the draft's original status and hash.
    for source, name in ((config_file, 'proposal.json'), (initials_file, 'planned-initials.json'),
                         (groups_file, 'development-strata.json')):
        (output / name).write_bytes(Path(source).read_bytes())
    write(output / 'prior-snapshots.json', prior)
    release = dict(status='AUTHORIZED_PREPARATION_RUNTIME_ADMISSION_PENDING',
        authorized_at_utc=datetime.now(timezone.utc).isoformat(),
        authorization='User explicitly approved planning and experiment execution, with bounded scope.',
        proposal_sha256=sha(config_file), initials_sha256=sha(initials_file),
        development_strata_sha256=sha(groups_file), prior_snapshots_sha256=sha(output/'prior-snapshots.json'),
        counts=dict(initials=len(rows), by_split=dict(Counter(r['split'] for r in rows)),
                    models=2, short_branches=124160, max_full_rollouts=48640),
        primary_window=3, windows=[1, 2, 3, 4, 5], execution_repeats=8,
        stop_rule='End this confirmation at the approved budget, including null or inconclusive results. '
                  'No success-based reruns, task replacement, best-window selection or automatic enlargement.',
        correction_rule='Only missing valid trials after diagnosed engineering failure may be rerun; retain attempts.',
        libero_plus_scope='CPU resource/protocol analysis only; no new training or GPU rollouts authorized here.',
        novelty='New initials/noise within known tasks and official splits; not new-task/unseen-kitchen proof.')
    write(output / 'authorization.json', release)
    return release


def freeze_preparation_source(repository, destination):
    """Freeze preparation dependencies; later protocol workers need own admission."""
    repository,destination=Path(repository).resolve(),Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=False)
    for folder in ('AlphaBrain','benchmarks'):
        shutil.copytree(repository/folder,destination/folder,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    files={str(p.relative_to(destination)):sha(p) for p in sorted(destination.rglob('*.py'))}
    receipt=dict(status='PREPARATION_SOURCE_FROZEN',files=files,
                 scope='Fresh snapshot collection only; no formal inference admission claimed')
    write(destination/'source-manifest.json',receipt)
    return receipt


def prepare_render_plan(release, population, catalog_plan, destination):
    """Freeze all 80 admitted initials and the unchanged original view catalog."""
    from .render_reference import LIMITS
    release,population,destination=map(Path,(release,population,destination))
    if destination.exists():raise FileExistsError(destination)
    approved=read(release/'authorization.json')
    proposal=read(release/'proposal.json')
    planned=validate_initial_plan(proposal,read(release/'planned-initials.json'))
    status=read(population/'status.json');jobs=read(population/'population.json')
    if (status['status']!='COMPLETE' or status['completed']!=80 or len(jobs)!=80
            or status['authorization_sha256']!=sha(release/'authorization.json')):
        raise ValueError('Approved population is not complete')
    keys=lambda rows:{(r['task'],r['split'],r['scene_seed']) for r in rows}
    if keys(jobs)!=keys(planned) or len({j['initial_id'] for j in jobs})!=80:
        raise ValueError('Population coverage changed')
    prior={r['snapshot_numeric_sha256'] for r in read(release/'prior-snapshots.json')}
    for record,job in zip(status['records'],jobs):
        if record['job']!=job or sha(record['status_file'])!=record['status_sha256']:
            raise ValueError('Population evidence changed')
        receipt=read(record['status_file']);manifest=Path(job['path'])/'manifest.json'
        if (receipt['status']!='PASS' or sha(manifest)!=job['snapshot_manifest_sha256']
                or receipt['snapshot_numeric_sha256']!=read(manifest)['snapshot_numeric_sha256']
                or receipt['snapshot_numeric_sha256'] in prior):
            raise ValueError('Physical identity or initial admission invalid')
        prior.add(receipt['snapshot_numeric_sha256'])
    catalog=read(catalog_plan)['catalog']
    if Counter(v['role'] for v in catalog['views'])!={'reference':1,'train':64,'extra_eval':32}:
        raise ValueError('Original 97-view catalog required')
    plan=dict(format='robocasa_recovery_confirmation_render_v1',jobs=jobs,catalog=catalog,pixel_gate=LIMITS,
        authorization_sha256=sha(release/'authorization.json'),population_status_sha256=sha(population/'status.json'),
        population_file=str(population/'population.json'),population_sha256=sha(population/'population.json'),
        catalog_source=str(catalog_plan),catalog_source_sha256=sha(catalog_plan),
        counts=dict(initials=80,candidates=97,images=80*97*3),
        scope='Observation cache only; no policy execution, method selection or outcome filtering',
        initial_novelty=approved['novelty'])
    write(destination,plan)
    return plan


def build_render_queue(runtime_file, queue_root):
    from ..execution.manifest import task_key
    from ..execution.queue import prepare
    runtime_file=Path(runtime_file).resolve();runtime=read(runtime_file)
    plan=read(runtime['plan']);fingerprint=sha(runtime['plan'])
    admission=read(runtime['admission'])
    if (admission['status']!='COMPLETE' or admission['plan_sha256']!=fingerprint
            or admission['candidate_count']!=97):raise ValueError('Real render pilot not admitted')
    if admission['initial_id']!=plan['jobs'][0]['initial_id']:
        raise ValueError('Pilot must be the first predeclared initial')
    name='recovery-confirmation-render-20261009';tasks=[]
    root=Path(runtime['result_root']);root.mkdir(parents=True,exist_ok=False)
    for job in plan['jobs'][1:]:
        initial=job['initial_id']
        addresses=[dict(model='observation_cache',task=job['task'],initial=initial,view=v['view_id'],
                        stream='score',repeat=0,noise='no_policy_noise_snapshot_render') for v in plan['catalog']['views']]
        tasks.append(dict(key=task_key(name,fingerprint,addresses),addresses=addresses,
            output=str(root/'workers'/initial),payload=dict(initial=initial)))
    inputs=[Path(runtime['plan']),runtime_file,Path(runtime['admission'])]
    inputs += list(Path(runtime['source']).rglob('*.py'))
    manifest=dict(format='dsol_work_queue_v1',experiment_id=name,protocol_sha256=fingerprint,
        slots=[dict(host=runtime['host'],id=str(i)) for i in runtime['gpu_ids']],tasks=tasks,
        artifacts=[dict(path=str(p),sha256=sha(p)) for p in inputs if '__pycache__' not in p.parts],
        runtime=runtime,backend=dict(module='benchmarks.Robocasa365.render_queue',factory='factory'),
        reused_first_initial=dict(path=runtime['admission'],sha256=sha(runtime['admission']),
                                  initial_id=admission['initial_id']))
    write(root/'manifest.json',manifest);prepare(manifest,queue_root)
    return dict(status='PREPARED',blocks=len(tasks),reused_initials=1,queue=str(queue_root))


def finalize_render(runtime_file, queue_root, destination):
    from ..execution.queue import summary
    from .render_reference import assert_validation,validate_candidate
    runtime=read(runtime_file);plan=read(runtime['plan']);destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    state=summary(queue_root)
    if state['status']!='COMPLETE' or state['counts']['COMPLETE']!=79:
        raise ValueError('Rendering queue not complete')
    records=[]
    for index,job in enumerate(plan['jobs']):
        path=(Path(runtime['admission']) if index==0 else
              Path(runtime['result_root'])/'workers'/job['initial_id']/'status.json')
        receipt=read(path)
        if (receipt['status']!='COMPLETE' or receipt['initial_id']!=job['initial_id']
                or receipt['plan_sha256']!=sha(runtime['plan']) or receipt['candidate_count']!=97):
            raise ValueError('Renderer receipt mismatch')
        assert_validation(job,receipt['reference_file'],read(receipt['validation_file']))
        validate_candidate(job,plan['catalog'],receipt['candidate_root'],sha(runtime['plan']))
        records.append(dict(initial_id=job['initial_id'],status='PASS',
            **{key:receipt[key] for key in ('reference_file','reference_sha256','validation_file',
                'validation_sha256','candidate_root','candidate_summary_sha256')}))
    result=dict(complete=True,eligible_for_formal_release=True,total=80,counts={'PASS':80},
        plan_sha256=sha(runtime['plan']),records=records,
        scope='New-population render cache; no policy or closed-loop outcome admission implied')
    write(destination,result);return result


def prepare_probe_config(release,reference_plan,reference_release,official_plan,xr1_plan,output):
    """Bind fresh observations to unchanged, audited model identities."""
    release,output=Path(release),Path(output)
    if output.exists():raise FileExistsError(output)
    ref=read(reference_plan);admitted=read(reference_release)
    if (not admitted['complete'] or admitted['counts']!={'PASS':80}
            or admitted['plan_sha256']!=sha(reference_plan)):
        raise ValueError('Fresh rendered population not released')
    official=read(official_plan)['models']['official'];xr=read(xr1_plan)
    for name,expected in official['files'].items():
        stat=(Path(official['checkpoint'])/name).stat()
        if stat.st_size!=expected['bytes'] or stat.st_mtime_ns!=expected['mtime_ns']:
            raise ValueError('Official checkpoint changed since full byte audit')
    xr_receipt=Path(xr['checkpoint'])/'IMPORT_RECEIPT.json'
    if sha(xr_receipt)!=xr['checkpoint_receipt_sha256']:
        raise ValueError('XR-1 checkpoint identity changed')
    output.mkdir(parents=True,exist_ok=False)
    plan=dict(format='fresh_recovery_confirmation_probe_v1',
        reference_plan=str(reference_plan),reference_release=str(reference_release),
        reference_plan_sha256=sha(reference_plan),reference_release_sha256=sha(reference_release),
        authorization=str(release/'authorization.json'),authorization_sha256=sha(release/'authorization.json'),
        initial_ids=[j['initial_id'] for j in ref['jobs']],all_views=[v['view_id'] for v in ref['catalog']['views']],
        models={'official':official},pixel_mean_limit=.005,
        score_repeats=8,windows=[1,2,3,4,5],primary_window=3,
        source_models={str(official_plan):sha(official_plan),str(xr1_plan):sha(xr1_plan)})
    write(output/'plan.json',plan)
    cfg=dict(population_mode='fresh_confirmation',panel_plan=str(output/'plan.json'),
        xr1_model={key:xr[key] for key in ('checkpoint','model_python','upstream','checkpoint_receipt_sha256')})
    write(output/'config.json',cfg)
    return cfg


def build_fresh_probe_queue(runtime_file,queue_root,chunk_size=32):
    from ..execution.manifest import task_key
    from ..execution.queue import prepare
    from .short_view_program import partition_missing
    runtime_file=Path(runtime_file).resolve();runtime=read(runtime_file);cfg=read(runtime['config'])
    if cfg.get('population_mode')!='fresh_confirmation':raise ValueError('Fresh-population configuration required')
    plan=read(cfg['panel_plan']);reference=read(plan['reference_plan'])
    initials,views=plan['initial_ids'],plan['all_views'];jobs={j['initial_id']:j for j in reference['jobs']}
    if len(initials)!=80 or len(views)!=97:raise ValueError('Frozen confirmation scope required')
    admission_path=Path(runtime['admission']);admission=read(admission_path)
    identity=read(admission_path.parent/'identity.json')
    if (admission['status']!='COMPLETE' or admission['completed_cells']!=2
            or identity['initials']!=initials[:1] or identity['views']!=['canonical','extra_002']
            or identity['model']!=runtime['model'] or identity['config_sha256']!=sha(runtime['config'])
            or identity['source_sha256']!=sha(Path(runtime['source'])/'benchmarks/Robocasa365/short_view_probe.py')
            or identity.get('population_mode')!='fresh_confirmation'):
        raise ValueError('New model/source/config actual admission required')
    reused={(initials[0],v) for v in identity['views']}
    chunks=partition_missing(initials,views,reused,chunk_size)
    result=Path(runtime['result_root']);result.mkdir(parents=True,exist_ok=False)
    protocol=dict(format='fresh_recovery_confirmation_short_probes_v1',model=runtime['model'],
        initials=initials,views=views,score_repeats=8,windows=[1,2,3,4,5],primary_window=3,
        config_sha256=sha(runtime['config']),panel_plan_sha256=sha(cfg['panel_plan']),
        admission_sha256=sha(admission_path),source=runtime['source'],
        full_task_evaluations=0,stopping='Success logged; true termination fails closed; no score imputation',
        noise='New initial identity; original explicit score/replan addressing; execution stream disjoint',
        first_window='Independent initial-only eight-draw anchor, exactly replayed in probe window1',
        chunk_size=chunk_size)
    write(result/'protocol.json',protocol);fingerprint=sha(result/'protocol.json')
    name='recovery-confirmation-probes-'+runtime['model']+'-20261009';tasks=[]
    ownership=[dict(initial=initials[0],view=v,cell=str(admission_path.parent/'cells'/initials[0]/v))
               for v in identity['views']]
    for index,chunk in enumerate(chunks):
        initial=chunk['initial'];selected=chunk['views'];output=result/'workers'/f'{initial}-{index:03d}'
        addresses=[dict(model=runtime['model'],task=jobs[initial]['task'],initial=initial,view=v,
            stream='score',repeat=r,noise=f'fresh_score/{initial}/{r}/replans0-4') for v in selected for r in range(8)]
        tasks.append(dict(key=task_key(name,fingerprint,addresses),addresses=addresses,
                          output=str(output),payload=dict(initial=initial,views=selected)))
        ownership += [dict(initial=initial,view=v,cell=str(output/'cells'/initial/v)) for v in selected]
    if len(ownership)!=80*97 or len({(r['initial'],r['view']) for r in ownership})!=80*97:
        raise ValueError('Probe coverage is incomplete or duplicated')
    write(result/'ownership.json',dict(records=ownership,reused_admission_cells=2))
    inputs=[runtime_file,Path(runtime['config']),Path(cfg['panel_plan']),Path(plan['reference_plan']),
            Path(plan['reference_release']),admission_path,admission_path.parent/'identity.json',
            result/'protocol.json',result/'ownership.json']
    inputs += list(Path(runtime['source']).rglob('*.py'))
    manifest=dict(format='dsol_work_queue_v1',experiment_id=name,protocol_sha256=fingerprint,
        slots=[dict(host=runtime['host'],id=str(i)) for i in runtime['gpu_ids']],tasks=tasks,
        artifacts=[dict(path=str(p),sha256=sha(p)) for p in inputs if '__pycache__' not in p.parts],
        runtime=runtime,backend=dict(module='benchmarks.Robocasa365.probe_queue',factory='factory'))
    write(result/'manifest.json',manifest);prepare(manifest,queue_root)
    return dict(status='PREPARED',blocks=len(tasks),reused_cells=2,new_cells=80*97-2,short_branches=80*97*8)


def prepare_execution_plan(release, probe_config, selections, output, *, path_mappings=(),
                           model_partition=None):
    """Freeze complete populations, optionally in an explicitly authorized model partition."""
    release, output = Path(release), Path(output)
    if output.exists():
        raise FileExistsError(output)
    required = {'official', 'xr1'} if model_partition is None else {model_partition}
    if not required <= {'official', 'xr1'} or set(selections) != required:
        raise ValueError('Both approved models or one explicit model partition required')
    cfg = read(probe_config); population = read(cfg['panel_plan'])
    if sha(population['authorization']) != sha(release/'authorization.json'):
        raise ValueError('Population belongs to another authorization')
    approved = read(release/'authorization.json')
    if approved['counts']['initials'] != 80 or approved['counts']['max_full_rollouts'] != 48640:
        raise ValueError('Approved budget changed')
    chosen = {}; inputs = {}; total = 0
    for model, root in selections.items():
        root = Path(root); status = read(root/'status.json'); binding = read(root/'binding.json')
        if (status['status'] != 'COMPLETE' or status['model'] != model or binding['model'] != model
                or status['completed_initials'] != 80 or status['pending_initials']
                or len(status['records']) != 80
                or {r['initial'] for r in status['records']} != set(population['initial_ids'])):
            raise ValueError('All 80 initial choices must be frozen before endpoint release')
        chosen[model] = {}
        for entry in status['records']:
            if sha(entry['path']) != entry['sha256']:
                raise ValueError('Selected-view manifest changed')
            record = read(entry['path'])
            if (record['model'] != model or record['initial'] != entry['initial']
                    or record['binding_sha256'] != sha(root/'binding.json')
                    or record['execution_repeats'] != list(range(8))
                    or not 33 <= len(record['evaluation_views']) <= 38
                    or record['new_rollouts'] != len(record['evaluation_views'])*8):
                raise ValueError('Selected-view execution budget changed')
            chosen[model][entry['initial']] = dict(path=entry['path'], sha256=entry['sha256'])
            total += record['new_rollouts']
        inputs[str(root/'binding.json')] = sha(root/'binding.json')
        inputs[str(root/'status.json')] = sha(root/'status.json')
    partition_cap = approved['counts']['max_full_rollouts'] * len(required) // 2
    if total > partition_cap:
        raise ValueError('Confirmation rollout cap exceeded')
    proposal = read(release/'proposal.json')
    with path_mapping(path_mappings):
        p0 = read(proposal['sources']['original_p0_plan'])
    plan = dict(format='recovery_confirmation_execution_v1',
                probe_config=str(probe_config), probe_config_sha256=sha(probe_config),
                probe_plan_sha256=sha(cfg['panel_plan']), initial_ids=population['initial_ids'],
                selections=chosen, selection_sources=inputs, execution_repeats=list(range(8)),
                task_horizons={t: p0['task_horizons'][t] for t in proposal['population']['tasks']},
                rollout_count=total, max_rollouts=partition_cap, aggregate_max_rollouts=48640,
                model_partition=model_partition,
                authorization=str(release/'authorization.json'), authorization_sha256=sha(release/'authorization.json'),
                noise='Original per-model addressing, execution stream, new initial IDs, paired across views',
                no_new_oracle_claim=True, actual_runtime_admission_pending=True)
    write(output, plan)
    return plan


def after_probe_preparation(config_file):
    """CPU-only stage hook: wait, audit choices, prepare the full-task plan.

    Does not start/cancel GPU workers. Real endpoint admission remains mandatory.
    A failed worker is an engineering stop, never an automatic retry.
    """
    import time
    from ..analysis import recovery_confirmation as selection_module
    cfg = read(config_file); status_file = Path(cfg['status_file'])
    if status_file.exists():
        raise FileExistsError(status_file)
    config_sha = sha(config_file)
    sources = {__file__: sha(__file__), selection_module.__file__: sha(selection_module.__file__)}
    started = time.time()
    try:
        while True:
            progress = {}; finished = True
            for model, root in cfg['probes'].items():
                tasks = read(Path(root)/'manifest.json')['tasks']
                complete = 0
                for task in tasks:
                    path = Path(task['output'])/'status.json'
                    if path.exists():
                        state = read(path)
                        if state['status'] == 'FAILED':
                            raise RuntimeError(f'Failed probe retained: {path}')
                        complete += state['status'] == 'COMPLETE'
                progress[model] = dict(completed_blocks=complete, total_blocks=len(tasks))
                finished &= complete == len(tasks)
            write(status_file, dict(status='WAITING_FOR_PROBES', progress=progress,
                                    started_unix=started, checked_unix=time.time(),
                                    config_sha256=config_sha, sources=sources, gpu_jobs_started=0))
            if finished:
                break
            if time.time()-started > cfg['timeout_seconds']:
                raise TimeoutError('Probe preparation wait budget exceeded; inspect, do not retry automatically')
            time.sleep(min(60, cfg.get('poll_seconds', 60)))
        if sha(config_file) != config_sha or any(sha(p) != digest for p, digest in sources.items()):
            raise ValueError('Preparation source/config changed while waiting')
        for model, root in cfg['probes'].items():
            result = selection_module.freeze_ready_initials(root, cfg['selections'][model])
            if result['status'] != 'COMPLETE':
                raise ValueError('Workers finished without complete scoring population')
        plan = prepare_execution_plan(cfg['release'], cfg['probe_config'], cfg['selections'],
                                      cfg['execution_plan'], path_mappings=cfg.get('path_mappings', ()))
        write(status_file, dict(status='READY_FOR_ENDPOINT_RUNTIME_ADMISSION', started_unix=started,
                                finished_unix=time.time(), execution_plan=cfg['execution_plan'],
                                rollout_count=plan['rollout_count'], gpu_jobs_started=0,
                                config_sha256=config_sha, sources=sources))
        return plan
    except BaseException as exc:
        write(status_file, dict(status='FAILED', error=repr(exc), started_unix=started,
                                finished_unix=time.time(), config_sha256=config_sha, gpu_jobs_started=0))
        raise
