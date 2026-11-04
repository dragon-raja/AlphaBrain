"""CPU-only accounting for a native response panel and paired online search.

Nominal return and stay reuse the panel outcomes only after input-equivalence
validation. Online selected poses always receive their own execution budget.
This accounting is a proposal, never execution authorization.
"""
import math


def comparison_budget(horizons, *, initials, views, repeats, query_cap=8):
    horizons = list(horizons)
    values = [*horizons, initials, views, repeats, query_cap]
    if (not horizons or any(type(v) is not int or v <= 0 for v in values)
            or views < 2 or query_cap != 8):
        raise ValueError('Positive integer coverage and frozen eight-query cap required')
    states = len(horizons) * initials
    starts = states * (views - 1)
    execution_per_view = sum(math.ceil(h / 5) for h in horizons) * initials * repeats
    panel = dict(episodes=states * views * repeats,
                 execution_calls=execution_per_view * views,
                 scoring_calls=states * views * 4)
    online = dict(searches=starts, episodes=starts * repeats,
                  execution_calls=execution_per_view * (views - 1),
                  selection_scoring_calls=starts * query_cap * 2,
                  validation_scoring_calls=starts * 2,
                  scoring_calls=starts * (query_cap * 2 + 2))
    methods = dict(panel=panel, local_near_best_accel=online.copy(),
                   continuous_ppo=online.copy())
    for row in methods.values():
        row['total_calls'] = row['execution_calls'] + row['scoring_calls']
    return dict(status='PROPOSED_NOT_AUTHORIZED', tasks=len(horizons), states=states,
                perturbed_starts=starts, methods=methods,
                episodes=sum(row['episodes'] for row in methods.values()),
                total_calls=sum(row['total_calls'] for row in methods.values()),
                nominal_return_additional_calls=0, training_updates=0)
