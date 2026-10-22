"""Validate local-search splits and compute budgets in explicit units."""
import math


def validate_plan(plan):
    train, evaluation = plan['training'], plan['evaluation']
    if plan['automatic_execution'] is not False:
        raise ValueError('Design review cannot authorize execution')
    indices = train['initial_indices']
    if len(indices) != len(set(indices)) or train['episodes'] != len(indices) * train['episodes_per_initial']:
        raise ValueError('Duplicate training initial or inconsistent episode count')
    seen = set(train['tasks']); eval_addresses = set()
    sessions = scoring = episodes = actions = execution = 0
    starts = len(plan['camera']['evaluation_start_angles_deg'])
    seeds = evaluation['execution_repeat_ids']
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError('Unique paired execution seeds required')
    methods = evaluation['methods']
    if len(methods) != len(set(methods)):
        raise ValueError('Duplicate evaluation method')
    for task in evaluation['tasks']:
        if task['seen_by_actor'] != (task['task'] in seen):
            raise ValueError('Incorrect seen/heldout task label')
        for initial in task['indices']:
            address = (task['cohort'], initial)
            if address in eval_addresses or (task['cohort'] == train['cohort'] and initial in indices):
                raise ValueError('Evaluation initial overlap')
            eval_addresses.add(address)
        horizon = evaluation['horizons'][task['suite']]
        cells = len(task['indices']) * starts
        sessions += cells * len(methods)
        scoring += cells * sum(evaluation['method_query_caps'][m] for m in methods) * len(plan['search']['score_noise_members'])
        task_episodes = cells * len(methods) * len(seeds)
        episodes += task_episodes; actions += task_episodes * horizon
        execution += task_episodes * math.ceil(horizon / plan['operation_policy']['executed_actions_per_replan'])
    training = train['episodes'] * train['query_budget'] * len(plan['search']['score_noise_members'])
    computed = dict(training_search_trajectories=training, evaluation_search_sessions=sessions,
        evaluation_search_trajectories=scoring, evaluation_episodes=episodes,
        evaluation_max_actions=actions, evaluation_execution_trajectories=execution,
        future_trajectories=training + scoring + execution + plan['budget']['additional_admission_trajectories'])
    computed['total_including_completed_admission'] = computed['future_trajectories'] + plan['budget']['completed_admission_trajectories']
    if any(plan['budget'][key] != value for key, value in computed.items()):
        raise ValueError('Declared budget differs from task/seed/query/horizon arithmetic')
    return computed
