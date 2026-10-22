"""Validate local-search splits and compute budgets in explicit units."""
import math
from collections import Counter
import numpy as np


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


def validate_schedules(plan, training_rows, evaluation_rows, inputs):
    """Resolve indices to actual task/state identities; reject incomplete pairing.

    `inputs` maps cohort names to their input manifest records. Only identity
    metadata is inspected, never cached outcomes or new pose scores.
    """
    budget = validate_plan(plan)
    train, evaluation = plan['training'], plan['evaluation']
    if [r['episode'] for r in training_rows] != list(range(train['episodes'])):
        raise ValueError('Training schedule must contain every episode exactly once in order')
    expected = Counter({i: train['episodes_per_initial'] for i in train['initial_indices']})
    if Counter(r['initial'] for r in training_rows) != expected:
        raise ValueError('Training initial multiplicity differs from design')
    train_states, train_tasks = set(), set()
    for row in training_rows:
        state = inputs[train['cohort']][row['initial']]['state']
        if (row['state_key'] != state['pair_key'] or row['task'] != state['task_id']
                or type(row['proposal_seed']) is not int or row['proposal_seed'] < 0):
            raise ValueError('Training state/task/seed identity mismatch')
        angles = np.asarray(row['angles_deg'])
        if (angles.shape != (2,) or not np.isfinite(angles).all()
                or np.any(np.abs(angles) > [25, 15]) or np.max(np.abs(angles)) < 10):
            raise ValueError('Training start violates frozen angle distribution support')
        train_states.add(state['pair_key']); train_tasks.add(state['task_id'])
    if train_tasks != set(train['tasks']) or len(train_states) != len(train['initial_indices']):
        raise ValueError('Training tasks/physical initial identities are duplicated or missing')
    expected_cells = {(t['cohort'], i, s, method): t for t in evaluation['tasks']
        for i in t['indices'] for s in range(len(plan['camera']['evaluation_start_angles_deg']))
        for method in evaluation['methods']}
    seen, seeds, eval_states = set(), {}, set()
    for row in evaluation_rows:
        key = (row['cohort'], row['initial'], row['start'], row['method'])
        if key not in expected_cells or key in seen:
            raise ValueError('Unexpected or duplicate evaluation cell')
        seen.add(key); task = expected_cells[key]
        state = inputs[row['cohort']][row['initial']]['state']
        if (row['task'] != task['task'] or state['task_id'] != row['task']
                or state['suite'] != task['suite'] or row['state_key'] != state['pair_key']
                or row['max_steps'] != evaluation['horizons'][task['suite']]
                or row['query_cap'] != evaluation['method_query_caps'][row['method']]
                or row['execution_repeats'] != evaluation['execution_repeat_ids']
                or row['angles_deg'] != plan['camera']['evaluation_start_angles_deg'][row['start']]):
            raise ValueError('Evaluation task/state/horizon/seed/budget contract mismatch')
        pair = key[:3]
        if type(row['proposal_seed']) is not int or row['proposal_seed'] < 0:
            raise ValueError('Nonnegative proposal seed required')
        if seeds.setdefault(pair, row['proposal_seed']) != row['proposal_seed']:
            raise ValueError('Methods must share the same proposal seed')
        eval_states.add(state['pair_key'])
    expected_states = sum(len(t['indices']) for t in evaluation['tasks'])
    if seen != set(expected_cells) or len(eval_states) != expected_states or train_states & eval_states:
        raise ValueError('Missing evaluation cells, repeated physical states or train/eval leakage')
    return dict(status='CPU_SCHEDULE_PASS', training_episodes=len(training_rows),
                training_initials=len(train_states), evaluation_initials=len(eval_states),
                evaluation_search_sessions=len(evaluation_rows), budget=budget)
