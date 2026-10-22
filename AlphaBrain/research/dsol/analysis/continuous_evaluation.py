"""Paired continuous-view task statistics, never cached labels or training reward."""
from collections import defaultdict

METHODS = ('stay', 'local_near_best_accel', 'continuous_ppo')


def summarize(tasks, searches):
    """Only complete three-method seed pairs enter comparisons.

    Harm = stay success / method failure divided by ALL complete paired seeds;
    conditional harm has its own explicit stay-success denominator. Query means
    count search sessions once, not four reused execution seeds.
    """
    groups = defaultdict(dict)
    for row in tasks:
        key = (row['cohort'], row['initial'], row['start'], row['repeat'])
        if row['method'] not in METHODS or row['method'] in groups[key] or type(row['success']) is not bool:
            raise ValueError('Invalid/duplicate evaluation result')
        groups[key][row['method']] = row
    complete = [g for g in groups.values() if set(g) == set(METHODS)]
    search_map = {}
    for row in searches:
        key = (row['cohort'], row['initial'], row['start'], row['method'])
        if key in search_map:
            raise ValueError('Duplicate search session')
        search_map[key] = row
    scopes = {'all': complete}
    for task in sorted({g['stay']['task'] for g in complete}):
        scopes[task] = [g for g in complete if g['stay']['task'] == task]
    result = dict(completed_tasks=len(tasks), complete_pairs=len(complete),
                  incomplete_pairs=len(groups)-len(complete), scopes={})
    for scope, pairs in scopes.items():
        n = len(pairs)
        methods = {}
        for method in METHODS:
            success = sum(g[method]['success'] for g in pairs)
            harm = sum(g['stay']['success'] and not g[method]['success'] for g in pairs)
            rescue = sum(not g['stay']['success'] and g[method]['success'] for g in pairs)
            stay_success = sum(g['stay']['success'] for g in pairs)
            sessions = {(g[method]['cohort'], g[method]['initial'], g[method]['start'], method) for g in pairs}
            if any(k not in search_map for k in sessions):
                raise ValueError('Task without its real search record')
            queries = [search_map[k]['queries'] for k in sessions]
            methods[method] = dict(n=n, success=success, success_rate=success/n if n else None,
                harm=harm, rescue=rescue, harm_rate=harm/n if n else None,
                rescue_rate=rescue/n if n else None, stay_success=stay_success,
                conditional_harm_rate=harm/stay_success if stay_success else None,
                mean_queries=sum(queries)/len(queries) if queries else None, search_sessions=len(queries))
        result['scopes'][scope] = methods
    return result
