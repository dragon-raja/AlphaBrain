"""CPU-only descriptive audits; observed trajectories are not counterfactuals."""
from collections import Counter
import math
import statistics


def paired_recovery(rows):
    """Nominal references may repeat across starts; count paired outcomes explicitly."""
    keys = [(r['state_key'], r['repeat'], r['start']) for r in rows]
    if not rows or len(keys) != len(set(keys)):
        raise ValueError('Empty or duplicate paired outcomes')
    lost = [r for r in rows if r['nominal'] and not r['stay']]
    rescued = sum(r['selected'] for r in lost)
    harmed = sum(r['nominal'] and r['stay'] and not r['selected'] for r in rows)
    return dict(paired_outcomes=len(rows),
                unique_nominal_references=len({(r['state_key'], r['repeat']) for r in rows}),
                nominal_success=sum(r['nominal'] for r in rows),
                stay_success=sum(r['stay'] for r in rows),
                selected_success=sum(r['selected'] for r in rows),
                nominal_success_lost_to_perturbation=len(lost),
                nominal_failure_helped_by_perturbation=sum(not r['nominal'] and r['stay'] for r in rows),
                lost_success_recovered=rescued,
                conditional_recovery_fraction=rescued/len(lost) if lost else None,
                preserved_nominal_success_harmed=harmed,
                net_recovery_on_nominal_success=rescued-harmed,
                all_rescues=sum(not r['stay'] and r['selected'] for r in rows),
                all_harms=sum(r['stay'] and not r['selected'] for r in rows))


def distribution(values):
    values = sorted(float(x) for x in values)
    if not values:
        return dict(n=0)
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Nonfinite audit input')
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                minimum=values[0], maximum=values[-1])


def training_summary(records, *, scale, cost):
    """Check the telescoping reward including the charged initial observation."""
    if not records or scale <= 0 or cost < 0:
        raise ValueError('Explicit records and positive reward scale required')
    rows = [r['metrics'] for r in records]
    if len({r['episode'] for r in rows}) != len(rows):
        raise ValueError('Duplicate episode')
    errors = [abs(r['total_reward'] - ((r['initial_accel']-r['best_accel'])/scale
                                     - cost*(r['queries']-1))) for r in rows]
    initial_errors = [abs(r['total_reward_including_initial']-(r['total_reward']-cost)) for r in rows]
    if max(errors + initial_errors) > 1e-8:
        raise ValueError('Stored reward differs from frozen formula')
    moved = [r for r in rows if r['queries'] > 1]
    def group(items):
        return dict(episodes=len(items), queries=sum(r['queries'] for r in items),
                    immediate_stop=sum(r['queries']==1 for r in items),
                    reasons=dict(Counter(r['reason'] for r in items)),
                    initial_accel=distribution(r['initial_accel'] for r in items),
                    absolute_improvement=distribution(r['initial_accel']-r['best_accel'] for r in items),
                    return_after_initial=distribution(r['total_reward'] for r in items),
                    positive_return=sum(r['total_reward']>1e-10 for r in items))
    return dict(all=group(rows), continued=group(moved), reward_error_max=max(errors),
                initial_charge_error_max=max(initial_errors),
                updates={str(u):group([r for r in rows if r['policy_update']==u])
                         for u in sorted({r['policy_update'] for r in rows})},
                tasks={t:group([r for r in rows if r['initial_identity']['task']==t])
                       for t in sorted({r['initial_identity']['task'] for r in rows})},
                hindsight_cost_sensitivity={str(c):dict(
                    observed_continuations=len(moved),
                    positive=sum((r['initial_accel']-r['best_accel'])/scale-c*(r['queries']-1)>1e-10 for r in moved),
                    mean_return=statistics.mean((r['initial_accel']-r['best_accel'])/scale-c*(r['queries']-1) for r in moved) if moved else None)
                    for c in (0., .0025, .005, cost)},
                cost_sensitivity_is_off_policy_descriptive_only=True)
