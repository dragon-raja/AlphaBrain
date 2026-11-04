"""Explicit PPO boundary resumption over frozen controlled phases.

This CPU composition never changes a running release or recreates unknown work.
The original learner, optimizer, RNG and reward state remain canonical. Evaluation
uses its separately admitted frozen release and preallocated leaf budget.
"""
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import RLock, Thread, Event
import time

from AlphaBrain.research.dsol.data.artifacts import read, write
from AlphaBrain.research.dsol.execution.manifest import file_digest as sha
from AlphaBrain.research.dsol.execution.phase_budget import PhaseBudget
from AlphaBrain.research.dsol.execution.fleet_slots import FleetSlots
from AlphaBrain.research.dsol.execution.queue import exclusive
from .prefix_campaign import PrefixCampaign
from .prefix_continuation import frozen_stage, process_alive
from .prefix_admission import peer_numerical_proof
from .controlled_campaign import collection_records, phase_evidence


def verify_zero_attempt(audit, *, expected_sha256):
    if sha(audit) != expected_sha256:
        raise ValueError('Changed explicit zero-call audit')
    a = read(audit)
    if (a['status'] != 'VERIFIED_PRE_DISPATCH_ZERO_MODEL_CALLS'
            or any(a[k] != 0 for k in ('model_calls', 'episodes', 'optimizer_updates'))
            or a['permit_created'] or a['local_center_created']
            or a['processes'] or a['research_apps_on_target_cards']):
        raise ValueError('Unknown or dispatched work cannot be retried')
    return a


def accepted_version(root, name):
    files = sorted((Path(root)/'publications'/name).glob('[0-9][0-9][0-9][0-9].json'))
    if not files:
        raise ValueError('Canonical optimizer publication required')
    p = read(files[-1])
    if (p['run'] != name or p['episodes'] != 32*p['version']
            or sha(p['policy']) != p['policy_sha256']):
        raise ValueError('Changed accepted optimizer publication')
    return p['version']


class PrefixResumption:
    def __init__(self, path):
        self.config = read(path); self.root = Path(self.config['output'])
        self.old = Path(self.config['original']); self.lock = RLock()
        self.state = {}; self.errors = {}; self.stop = Event()
        self.pool = FleetSlots(self.config['capacity'], initial_capacity=self.config['initial_capacity'])
        self.train = PrefixCampaign(self.config['training_config'])
        self.evaluate = PrefixCampaign(self.config['evaluation_config'])
        self.train.pool = self.evaluate.pool = self.pool
        self.protocol = self.train.protocol
        original = read(self.old/'coordinator-config.json')
        self.train.budget = PhaseBudget(Path(original['local_root'])/'budget', self.old/'budget',
            self.protocol['authorization'], authorization_sha256=sha(self.protocol['authorization']))
        old_eval = read(self.config['evaluation_budget_config'])
        self.evaluate.budget = PhaseBudget(Path(old_eval['local_root'])/'budget',
            Path(old_eval['output'])/'budget', self.protocol['authorization'],
            authorization_sha256=sha(self.protocol['authorization']), allocation=old_eval['budget_allocation'])

    def status(self, name, **values):
        with self.lock:
            self.state.setdefault(name, {}).update(values, updated_unix=time.time())
            write(self.root/'status.json', dict(status='PARTIAL_HOLD' if self.errors else 'RUNNING',
                cells=self.state, errors=self.errors, resources=self.pool.snapshot(),
                budgets=dict(training=self.train.budget.status(), evaluation=self.evaluate.budget.status()),
                automatic_retry=False, updated_unix=time.time()))

    def preflight(self):
        if process_alive(read(self.old/'coordinator-process.json')):
            raise RuntimeError('Original owner is still alive')
        verify_zero_attempt(self.config['zero_audit'], expected_sha256=self.config['zero_audit_sha256'])
        for name, version in self.config['start_versions'].items():
            if accepted_version(self.old, name) != version or not 0 < version < 64:
                raise ValueError('Resumption boundary changed')
            for intent in (self.old/'publications'/name).glob('*.attempt.json'):
                if not intent.with_name(intent.name.replace('.attempt.json', '.json')).exists():
                    raise RuntimeError('Unknown CPU optimizer outcome')
            for v in range(version, 64):
                prior = self.old/'campaigns'/name/f'train-{v:04d}.json'
                if prior.exists() and str(prior) != self.config['verified_zero_campaign']:
                    raise RuntimeError('Unreviewed old collection exists')
        for f in self.train.budget.root.glob('*.json'):
            r = read(f)
            if read(r['campaign']).get('kind') != 'EXPLICIT_DELEGATED_CUMULATIVE_BUDGET' and r['status'] != 'COMPLETE':
                raise RuntimeError('Unsettled old scientific phase')

    def phase(self, ctx, name, kind, schedule, *, version=None, budget=None, host=None, stay=False):
        family = ctx.protocol['cells'][name]['family']
        hosts = [host] if host else ctx.cfg['hosts'][family]
        if kind == 'evaluation':
            suites = {r['suite'] for r in read(schedule)}
            hosts = [h for h in hosts if all(h+'/'+s in ctx.proofs.get(family, {})
                     and read(ctx.proofs[family][h+'/'+s])['source_identity'] == ctx.manifest['source_identity'] for s in suites)]
        count = 4 if kind in ('training', 'admission') else 2
        with self.pool.lease(name+'-'+kind, hosts, count=count,
                             max_count=count if kind != 'evaluation' else 8) as (node, ids):
            path, label = ctx.build(name, kind, schedule, node, version=version, stay=stay)
            cfg = ctx.execution(name, family, node, ids)
            cp = ctx.root/'bindings'/name/(label+'.json'); write(cp, cfg)
            ledger = budget or ctx.budget
            # Separate attempted phase, while retaining the original zero-call reservation.
            key = name+'-'+label+'-'+self.config['reservation_namespace']
            ledger.reserve(key, path)
            self.status(name, state=kind.upper(), phase=label, host=node, gpu_ids=ids)
            result = frozen_stage(ctx.manifest['release'], cp, label, path,
                self.root/'controller-logs'/name/(kind+'-'+label+'.log'),
                kind='training' if kind in ('training', 'admission') else 'target_recovery')
            ledger.settle(key, result)
            write(ctx.root/'settlements'/name/(label+'.json'), result)
            return cfg, path, label

    def admit(self, ctx, family, host_only=None):
        name = next(n for n, c in ctx.protocol['cells'].items() if c['family'] == family and c['seed'] == 0)
        for host in ([host_only] if host_only else ctx.cfg['hosts'][family]):
            cfg, path, label = self.phase(ctx, name, 'admission', ctx.cfg['engineering_schedules'][family],
                                          budget=self.train.budget, host=host)
            for b in read(path)['blocks']:
                suite = b['rows'][0]['suite']
                p = phase_evidence(cfg, label)/'results'/f"block-{b['index']:04d}"/'admission.json'
                proof = peer_numerical_proof({family: {host+'/'+suite: str(p)}},
                    {family+'/'+suite: self.config['reference_proofs'][family+'/'+suite]})
                write(ctx.root/'peer-proofs'/(family+'-'+host+'-'+suite+'.json'), proof)
                ctx.proofs.setdefault(family, {})[host+'/'+suite] = str(p)

    def reclaim_finished_evaluator(self):
        """Accept idle cards only after the old evaluation owner has finished."""
        from .prefix_evaluation import PrefixEvaluation
        old = Path(self.config['previous_evaluator'])
        receipt = read(old/'coordinator-process.json')
        normalized = dict(pid=receipt['pid'], start_ticks=receipt.get('birth_ticks', receipt.get('start_ticks')))
        while process_alive(normalized):
            if self.stop.wait(10): return
        done = read(old/'completion.json')
        allowed = {'settled-continuation', *self.config['start_versions']}
        if set(done['errors']) - allowed:
            raise RuntimeError('Unreviewed previous evaluation failure')
        expected = set(self.protocol['cells']) - set(self.config['start_versions'])
        if (any(done['completed'].get(n) != 'COMPLETE_FINAL_EVALUATION' for n in expected)
                or done['completed'].get('baseline') != 'COMPLETE_STAY_REFERENCE'):
            raise RuntimeError('Old evaluation scope incomplete')
        for ctx_path in read(old/'evaluation-config.json')['contexts'].values():
            cfg = read(ctx_path)
            if any(read(f)['status'] != 'COMPLETE' for f in (Path(cfg['local_root'])/'budget').glob('*.json')):
                raise RuntimeError('Unknown evaluation consumption prevents card handoff')
        probe = PrefixEvaluation.__new__(PrefixEvaluation)
        probe.config = read(old/'evaluation-config.json')
        for host, planned in self.config['capacity'].items():
            ids = [i for i in planned if i not in self.pool.activated[host]]
            while ids and not probe.idle_cards(host, ids):
                if self.stop.wait(10): return
            if not ids: continue
            proof = self.root/'resource-grants'/(host+'.json')
            write(proof, dict(status='VERIFIED_STOPPED_SETTLED_OLD_EVALUATOR_AND_LIVE_IDLE',
                old_completion_sha256=sha(old/'completion.json'), host=host, gpu_ids=ids, checked_unix=time.time()))
            self.pool.grant(host, ids, evidence_sha256=sha(proof))
            # The new source must independently pass actual host/suite admission.
            for family in self.config['evaluation_admissions_after_handoff']:
                if host in self.evaluate.cfg['hosts'][family]: self.admit(self.evaluate, family, host_only=host)

    def cell(self, name, start):
        try:
            ctx = self.train; cell = ctx.protocol['cells'][name]
            for version in range(start, 64):
                if accepted_version(self.old, name) != version:
                    raise RuntimeError('Canonical learner changed during collection')
                cfg, path, label = self.phase(ctx, name, 'training', cell['training_schedule'], version=version)
                batch = ctx.root/'batches'/(name+f'-{version:04d}.json')
                write(batch, dict(cell=name, version=version, policy_sha256=sha(ctx.publication(name, version)),
                                 records=collection_records(cfg, label, path)))
                # cpu_update uses the ORIGINAL learner config/source and accepted receipt root.
                canonical = PrefixCampaign(self.old/'coordinator-config.json')
                canonical.cpu_update(name, version+1, batch)
                self.status(name, state='UPDATE_COMPLETE', version=version+1, episodes=32*(version+1))
            self.status(name, state='TRAINING_COMPLETE', version=64, episodes=2048)
        except BaseException as error:
            with self.lock: self.errors[name] = repr(error)
            self.status(name, state='HOLD_NO_AUTOMATIC_REPLAY'); raise

    def final(self, name):
        if accepted_version(self.old, name) != 64:
            raise RuntimeError('Final evaluation requires accepted v64 actor')
        cell = self.evaluate.protocol['cells'][name]
        for schedule, meta in cell['evaluation_schedules'].items():
            if meta['purpose'] == 'final':
                self.phase(self.evaluate, name, 'evaluation', schedule, version=64)
        self.status(name, state='COMPLETE_FINAL_EVALUATION')

    def run(self):
        owner = Path(self.config['owner']); owner.mkdir(parents=True, exist_ok=False)
        with exclusive(owner):
            self.preflight()
            def reclaim():
                try: self.reclaim_finished_evaluator()
                except BaseException as error:
                    with self.lock: self.errors['resource-handoff'] = repr(error)
                    self.status('resource-handoff', state='HOLD_NO_AUTOMATIC_REPLAY')
            thread = Thread(target=reclaim, daemon=True); thread.start()
            self.train.proofs = read(self.old/'admission-proofs.json')
            # Only a new host/model pair needs a new old-release admission.
            for family in self.config['training_admissions']: self.admit(self.train, family)
            with self.lock:
                for family, paths in self.train.proofs.items():
                    self.evaluate.proofs.setdefault(family, {}).update({key: value for key, value in paths.items()
                        if read(value)['source_identity'] == self.train.manifest['source_identity']})
            with ThreadPoolExecutor(3) as pool:
                for f in as_completed([pool.submit(self.cell, n, v) for n, v in self.config['start_versions'].items()]):
                    f.result()
            for family in self.config['evaluation_admissions']: self.admit(self.evaluate, family)
            with ThreadPoolExecutor(3) as pool:
                for f in as_completed([pool.submit(self.final, n) for n in self.config['start_versions']]): f.result()
            self.stop.set(); thread.join(timeout=40)
            write(self.root/'completion.json', dict(status='COMPLETE' if not self.errors else 'PARTIAL_HOLD', errors=self.errors,
                dispatch_enabled=False, optimizer_resets=0, replayed_completed_episodes=0))


class EvaluationHandoff(PrefixResumption):
    """Only never-dispatched evaluation schedules after an audited owner handoff.

    This CPU composition continues to call the original frozen GPU phase source.
    It has no training loop and cannot replay a partially completed phase. The
    exact scope and immutable evidence hashes must be frozen before construction.
    """

    def preflight(self):
        c = self.config
        for path, expected in c['handoff_input_hashes'].items():
            if sha(path) != expected: raise ValueError('Changed handoff evidence')
        audit = read(c['handoff_audit'])
        if audit['status'] != 'SETTLED_OWNERS_RETIRED_NO_REPLAY':
            raise ValueError('Explicit settled handoff required')
        for path in c['retired_owners']:
            r = read(path)
            if process_alive(dict(pid=r['pid'], start_ticks=r.get('birth_ticks', r.get('start_ticks')))):
                raise RuntimeError('Prior owner is still alive')
        if set(c['capacity']) & set(c['excluded_hosts']):
            raise ValueError('Transferred host remains in capacity')
        if any(set(hs) - set(c['capacity']) for hs in self.evaluate.cfg['hosts'].values()):
            raise ValueError('Execution host bypasses capacity partition')
        if c['jobs'] != audit['remaining_jobs'] or not c['jobs']:
            raise ValueError('Continuation differs from audited remaining scope')
        identities = set()
        for job in c['jobs']:
            name, schedule = job['name'], job['schedule']
            key = (name, schedule)
            if key in identities: raise ValueError('Repeated remaining evaluation')
            identities.add(key)
            cell = self.protocol['cells'][name]
            purpose = cell['evaluation_schedules'][schedule]['purpose']
            if (purpose not in ('baseline', 'final') or type(job['stay']) is not bool
                    or job['stay'] != (purpose == 'baseline') or accepted_version(self.old, name) != 64):
                raise ValueError('Changed frozen evaluation semantics or actor')
            if sha(schedule) != c['handoff_input_hashes'][schedule]:
                raise ValueError('Changed frozen schedule')
        for family, refs in c['restored_proofs'].items():
            for key, path in refs.items():
                if (key.split('/')[0] not in c['capacity']
                        or read(path)['source_identity'] != self.evaluate.manifest['source_identity']):
                    raise ValueError('Stale or excluded host admission')

    def run(self):
        owner = Path(self.config['owner']); owner.mkdir(parents=True, exist_ok=False)
        with exclusive(owner):
            self.preflight()
            self.evaluate.proofs = deepcopy(self.config['restored_proofs'])
            pairs = self.config['admit_pairs']
            if pairs:
                with ThreadPoolExecutor(len(pairs)) as pool:
                    for future in as_completed([pool.submit(self.admit, self.evaluate, f, h) for f, h in pairs]):
                        future.result()
            write(self.root/'admission-completion.json', dict(status='PASS_REMAINING_HOSTS', proofs=self.evaluate.proofs))
            groups = {}
            for job in self.config['jobs']:
                groups.setdefault((job['name'], job['stay']), []).append(job)

            def group(jobs):
                for job in jobs:
                    name = job['name']
                    try:
                        self.phase(self.evaluate, name, 'evaluation', job['schedule'], version=64, stay=job['stay'])
                    except BaseException as error:
                        with self.lock: self.errors[name+'-'+Path(job['schedule']).stem] = repr(error)
                        self.status(name, state='HOLD_NO_AUTOMATIC_REPLAY')
                        raise
                self.status(jobs[0]['name'], state='COMPLETE_REMAINING_BASELINE' if jobs[0]['stay'] else 'COMPLETE_FINAL_EVALUATION')

            with ThreadPoolExecutor(len(groups)) as pool:
                futures = [pool.submit(group, jobs) for jobs in groups.values()]
                for future in as_completed(futures):
                    try: future.result()
                    except BaseException: pass  # Preserve other disjoint, already approved groups.
            write(self.root/'completion.json', dict(status='COMPLETE' if not self.errors else 'PARTIAL_HOLD', errors=self.errors,
                dispatch_enabled=False, optimizer_resets=0, replayed_completed_episodes=0,
                excluded_hosts=self.config['excluded_hosts']))
