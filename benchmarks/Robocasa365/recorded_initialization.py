"""Bounded warm-up reset retries before restoring an immutable demonstration.

This is not an evaluation reset strategy. The sampled state must subsequently
be replaced by the recorded XML and state, with exact restoration checks.
"""


PLACEMENT_EXHAUSTED = 'Ran _load_model() 50 times but could not initialize task!'


def reset_for_recorded_replay(base, max_attempts=1):
    if not isinstance(max_attempts, int) or not 1 <= max_attempts <= 5:
        raise ValueError('Initialization attempts must be in 1..5')
    errors = []
    for attempt in range(1, max_attempts + 1):
        try:
            base.reset()
            return dict(attempts=attempt, prior_errors=errors,
                        scope='Warm-up only; recorded XML/state must replace sampled scene')
        except RuntimeError as exc:
            if str(exc) != PLACEMENT_EXHAUSTED:
                raise
            errors.append(str(exc))
            print(f'Recorded-scene initialization attempt {attempt}/{max_attempts}: placement exhausted', flush=True)
            if attempt == max_attempts:
                raise
            # Continue the existing placement RNG; reseeding would repeat the
            # same failed warm-up. No demonstration selection or labels change.
