"""Engineering capacity estimates, separate from task-success statistics."""
import math


def host_capacity(results, supervisors, workers):
    if workers <= 0 or len(results) != len(supervisors) or not results:
        raise ValueError('Completed, paired benchmark records are required')
    if any(r.get('status') != 'PASS' or 'rollout' not in r for r in results):
        raise ValueError('Failed or incomplete benchmark cannot estimate capacity')
    if any(s.get('status') != 'COMPLETE' or s.get('returncode') != 0 for s in supervisors):
        raise ValueError('Supervisor did not confirm completion')
    rollout_seconds = sum(r['rollout']['seconds'] for r in results)
    steps = sum(r['rollout']['steps'] for r in results)
    makespan = max(s['finished_unix'] for s in supervisors) - min(s['started_unix'] for s in supervisors)
    if steps <= 0 or not math.isfinite(rollout_seconds) or rollout_seconds <= 0 or makespan <= 0:
        raise ValueError('Invalid timing record')
    return dict(workers=workers, episodes=len(results), executed_steps=steps,
        mean_rollout_seconds=rollout_seconds / len(results),
        seconds_per_executed_step=rollout_seconds / steps,
        observed_cold_batch_episodes_per_hour=len(results) / makespan * 3600,
        modeled_resident_steps_per_second=workers * steps / rollout_seconds,
        mean_startup_and_gold_seconds=sum(r['seconds'] - r['rollout']['seconds'] for r in results) / len(results),
        batch_makespan_seconds=makespan,
        caveat='Resident rate is extrapolated from concurrent probes, not a sustained worker-queue measurement')


def population_work(hosts, initial_horizons, candidates, models, repeats):
    if not hosts or not initial_horizons or any(x <= 0 for x in initial_horizons):
        raise ValueError('Host capacities and positive per-initial horizons required')
    if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in (candidates, models, repeats)):
        raise ValueError('Positive integer budgets required')
    episodes = len(initial_horizons) * candidates * models * repeats
    total_steps = sum(initial_horizons) * candidates * models * repeats
    resident_rate = sum(h['modeled_resident_steps_per_second'] for h in hosts)
    cold_rate = sum(h['observed_cold_batch_episodes_per_hour'] for h in hosts)
    return dict(episodes=episodes, full_horizon_steps=total_steps,
        modeled_resident_days=total_steps / resident_rate / 86400,
        cold_probe_pattern_days=episodes / cold_rate / 24,
        assumptions='Resident estimate requires model/environment reuse; full native horizons, measured task cost extrapolated. Neither estimate is a confidence bound or committed finish time.')
