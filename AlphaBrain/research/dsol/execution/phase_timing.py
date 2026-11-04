"""Injectable monotonic phase timings; no GPU synchronization or scientific changes."""
from contextlib import contextmanager
import time
import math


class PhaseTiming:
    def __init__(self,clock=time.perf_counter):self.clock=clock;self.rows={}

    def wrap(self, name, callback):
        def measured(*args, **kwargs):
            with self.measure(name):
                return callback(*args, **kwargs)
        return measured

    @contextmanager
    def measure(self,name):
        begin=self.clock()
        try:yield
        finally:
            row=self.rows.setdefault(name,dict(seconds=0.,count=0))
            row['seconds']+=self.clock()-begin;row['count']+=1

    @contextmanager
    def context(self,context,name):
        began=self.clock();closing=None
        try:
            with context as value:
                self.rows.setdefault(name+'_start',dict(seconds=0.,count=0))
                self.rows[name+'_start']['seconds']+=self.clock()-began
                self.rows[name+'_start']['count']+=1
                try:yield value
                finally:closing=self.clock()
        finally:
            if closing is not None:
                row=self.rows.setdefault(name+'_close',dict(seconds=0.,count=0))
                row['seconds']+=self.clock()-closing;row['count']+=1


def summarize_completed_stages(stages):
    """Describe one caller-selected cohort of settled production stages.

    Pure CPU analysis: no filesystem, ledger, network or dispatch access. Callers
    must separate models/releases/scientific protocols. Timers may overlap each
    other and worker computation; their sums are not recoverable wall time.
    This is collection-stage telemetry, not end-to-end training or an ETA.
    Missing timers remain missing instead of being interpreted as zero.
    """
    rows = list(stages)
    if not rows:
        raise ValueError('At least one completed stage required')
    phases = set()
    seconds = calls = episodes = 0
    timers = {}
    for row in rows:
        phase = row.get('phase')
        if not isinstance(phase, str) or not phase or phase in phases:
            raise ValueError('Unique nonempty phase identities required')
        phases.add(phase)
        if (row.get('status') != 'COMPLETE' or row.get('cpu') is not False
                or row.get('dispatch_enabled') is not False):
            raise ValueError('Only settled, dispatch-closed production stages allowed')
        elapsed = row.get('seconds')
        if (type(elapsed) not in (int, float) or not math.isfinite(elapsed)
                or elapsed <= 0):
            raise ValueError('Positive finite elapsed time required')
        for key in ('actual_model_calls', 'actual_episodes'):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError('Nonnegative integer work counts required')
        seconds += elapsed
        calls += row['actual_model_calls']
        episodes += row['actual_episodes']
        for name, value in row.get('metrics', {}).items():
            if not name.endswith('_seconds'):
                continue
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value < 0):
                raise ValueError('Nonnegative finite timer required')
            timer = timers.setdefault(name, dict(seconds=0., stages=0,
                                                covered_stage_seconds=0.))
            timer['seconds'] += value
            timer['stages'] += 1
            timer['covered_stage_seconds'] += elapsed
    for timer in timers.values():
        timer['seconds_per_covered_stage'] = timer['seconds'] / timer['stages']
        timer['timer_to_covered_wall_ratio'] = (
            timer['seconds'] / timer['covered_stage_seconds'])
    return dict(
        schema='COMPLETED_STAGE_TELEMETRY_V1', stage_count=len(rows),
        phases=sorted(phases), stage_wall_seconds=seconds,
        mean_stage_seconds=seconds / len(rows), actual_model_calls=calls,
        actual_episodes=episodes, episodes_per_stage_hour=episodes * 3600 / seconds,
        calls_per_stage_second=calls / seconds, timers=timers,
        dispatch_authority=False,
        scope='Collection stages only; excludes setup between stages and learner updates',
        limitation='Timers overlap; ratios are not removable wall time or causal speedup',
    )
